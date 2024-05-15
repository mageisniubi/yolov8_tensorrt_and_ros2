#include "rclcpp/rclcpp.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include "target_bbox_msgs/msg/bounding_box.hpp"
#include "target_bbox_msgs/msg/bounding_boxes.hpp"
#include <names.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void serialize_engine(std::string &wts_name, std::string &engine_name, int &is_p, std::string &sub_type, float &gd,
                      float &gw, int &max_channels)
{
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    IHostMemory *serialized_engine = nullptr;

    if (is_p == 6)
    {
        serialized_engine = buildEngineYolov8DetP6(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels);
    }
    else if (is_p == 2)
    {
        serialized_engine = buildEngineYolov8DetP2(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels);
    }
    else
    {
        serialized_engine = buildEngineYolov8Det(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels);
    }

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p)
    {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    delete serialized_engine;
    delete config;
    delete builder;
}

void infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, int batchsize,
           float *decode_ptr_host, float *decode_ptr_device, int model_bboxes, std::string cuda_post_process)
{
    context.enqueue(batchsize, buffers, stream, nullptr);
    if (cuda_post_process == "c")
    {
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream));
    }
    else if (cuda_post_process == "g")
    {
        CUDA_CHECK(
            cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float *)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream); // cuda nms
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                                   sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost,
                                   stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine,
                        IExecutionContext **context)
{
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char *serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}
void prepare_buffer(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                    float **output_buffer_host, float **decode_ptr_host, float **decode_ptr_device,
                    std::string cuda_post_process)
{
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void **)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    if (cuda_post_process == "c")
    {
        *output_buffer_host = new float[kBatchSize * kOutputSize];
    }
    else if (cuda_post_process == "g")
    {
        if (kBatchSize > 1)
        {
            std::cerr << "Do not yet support GPU post processing for multiple batches" << std::endl;
            exit(0);
        }
        // Allocate memory for decode_ptr_host and copy to device
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void **)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, int &is_p, std::string &img_dir,
                std::string &sub_type, std::string &cuda_post_process, float &gd, float &gw, int &max_channels)
{
    if (argc < 4)
        return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7))
    {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto sub_type = std::string(argv[4]);

        if (sub_type[0] == 'n')
        {
            gd = 0.33;
            gw = 0.25;
            max_channels = 1024;
        }
        else if (sub_type[0] == 's')
        {
            gd = 0.33;
            gw = 0.50;
            max_channels = 1024;
        }
        else if (sub_type[0] == 'm')
        {
            gd = 0.67;
            gw = 0.75;
            max_channels = 576;
        }
        else if (sub_type[0] == 'l')
        {
            gd = 1.0;
            gw = 1.0;
            max_channels = 512;
        }
        else if (sub_type[0] == 'x')
        {
            gd = 1.0;
            gw = 1.25;
            max_channels = 640;
        }
        else
        {
            return false;
        }
        if (sub_type.size() == 2 && sub_type[1] == '6')
        {
            is_p = 6;
        }
        else if (sub_type.size() == 2 && sub_type[1] == '2')
        {
            is_p = 2;
        }
    }
    else if (std::string(argv[1]) == "-d" && argc == 5)
    {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        cuda_post_process = std::string(argv[4]);
    }
    else
    {
        return false;
    }
    return true;
}

class yolov8_ros2 : public rclcpp::Node
{
public:
    yolov8_ros2(std::string name) : Node(name)
    {
        cudaSetDevice(kGpuId);
        this->declare_parameter<std::string>("engine_name", "");
        this->declare_parameter<std::string>("img_sub_topic", "");
        this->declare_parameter<std::string>("img_pub_topic", "");
        this->declare_parameter<std::string>("box_pub_topic", "");
        this->declare_parameter<std::string>("cuda_post_process", "");
        
        this->get_parameter("engine_name", engine_name);
        this->get_parameter("img_sub_topic", img_sub_topic);
        this->get_parameter("img_pub_topic", img_pub_topic);
        this->get_parameter("box_pub_topic", box_pub_topic);
        this->get_parameter("cuda_post_process", cuda_post_process);

        RCLCPP_INFO(this->get_logger(), "%s节点已经启动.", name.c_str());
        deserialize_engine(engine_name, &runtime, &engine, &context);
        CUDA_CHECK(cudaStreamCreate(&stream));
        cuda_preprocess_init(kMaxInputImageSize);
        auto out_dims = engine->getBindingDimensions(1);
        model_bboxes = out_dims.d[0];
        prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host,
                       &decode_ptr_device, cuda_post_process);

        img_subscribe = this->create_subscription<sensor_msgs::msg::Image>(img_sub_topic.c_str(), 10, std::bind(&yolov8_ros2::img_callback, this, std::placeholders::_1));

        bboxs_pub = this->create_publisher<target_bbox_msgs::msg::BoundingBoxes>(
            box_pub_topic.c_str(), 1000);

        obj_image_pub = this->create_publisher<sensor_msgs::msg::Image>(
            img_pub_topic.c_str(), 10);
    }

    ~yolov8_ros2()
    {
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(device_buffers[0]));
        CUDA_CHECK(cudaFree(device_buffers[1]));
        CUDA_CHECK(cudaFree(decode_ptr_device));
        delete[] decode_ptr_host;
        delete[] output_buffer_host;
        cuda_preprocess_destroy();
        // Destroy the engine
        delete context;
        delete engine;
        delete runtime;
    }

private:
    void img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // 接收到图像的形式要转成BGR的形式
        cv_bridge::CvImagePtr cv_color_ptr;
        cv::Mat color_img, BGR;
        cv_color_ptr = cv_bridge::toCvCopy(msg, "8UC3");
        cv_color_ptr->image.copyTo(color_img);
        cvtColor(color_img, BGR, cv::COLOR_RGB2BGR);
        std::vector<cv::Mat> img_batch;
        img_batch.push_back(BGR);
        // Preprocess
        cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);

        // Run inference
        infer(*context, stream, (void **)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host,
              decode_ptr_device, model_bboxes, cuda_post_process);
        std::vector<std::vector<Detection>> res_batch;
        if (cuda_post_process == "c")
        {
            // NMS
            batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
        }
        else if (cuda_post_process == "g")
        {
            // Process gpu decode and nms results
            batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
        }

        auto &res = res_batch[0];
        target_bbox_msgs::msg::BoundingBoxes boxes;

        for (size_t j = 0; j < res.size(); j++)
        {
            target_bbox_msgs::msg::BoundingBox box;
            cv::Rect r = get_rect(BGR, res[j].bbox);
            cv::Scalar color = cv::Scalar(names::color_list[(int)res[j].class_id][0],
                                          names::color_list[(int)res[j].class_id][1],
                                          names::color_list[(int)res[j].class_id][2]);
            cv::rectangle(BGR, r, color, 6); // 框的大小
            std::string label = names::CLASSES[(int)res[j].class_id];
            cv::putText(BGR, label, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            box.probability = res[j].conf;
            box.class_id = names::CLASSES[(int)res[j].class_id];
            box.xmin = r.x;
            box.ymin = r.y;
            box.xmax = r.x + r.width;
            box.xmax = r.y + r.height;
            box.img_width = BGR.cols;
            box.img_height = BGR.rows;
            boxes.bounding_boxes.emplace_back(box);
        }
        bboxs_pub->publish(boxes);
        sensor_msgs::msg::Image img_msg;
        cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", BGR).toImageMsg(img_msg);
        img_msg.header.frame_id = "camera";
        obj_image_pub->publish(img_msg);
    }

    void infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, int batchsize,
               float *decode_ptr_host, float *decode_ptr_device, int model_bboxes, std::string cuda_post_process)
    {
        context.enqueue(batchsize, buffers, stream, nullptr);
        if (cuda_post_process == "c")
        {
            CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                                       stream));
        }
        else if (cuda_post_process == "g")
        {
            CUDA_CHECK(
                cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
            cuda_decode((float *)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
            cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream); // cuda nms
            CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                                       sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost,
                                       stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine,
                            IExecutionContext **context)
    {
        std::ifstream file(engine_name, std::ios::binary);
        if (!file.good())
        {
            std::cerr << "read " << engine_name << " error!" << std::endl;
            assert(false);
        }
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        char *serialized_engine = new char[size];
        assert(serialized_engine);
        file.read(serialized_engine, size);
        file.close();

        *runtime = createInferRuntime(gLogger);
        assert(*runtime);
        *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
        assert(*engine);
        *context = (*engine)->createExecutionContext();
        assert(*context);
        delete[] serialized_engine;
    }
    void prepare_buffer(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                        float **output_buffer_host, float **decode_ptr_host, float **decode_ptr_device,
                        std::string cuda_post_process)
    {
        assert(engine->getNbBindings() == 2);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine->getBindingIndex(kInputTensorName);
        const int outputIndex = engine->getBindingIndex(kOutputTensorName);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc((void **)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
        if (cuda_post_process == "c")
        {
            *output_buffer_host = new float[kBatchSize * kOutputSize];
        }
        else if (cuda_post_process == "g")
        {
            if (kBatchSize > 1)
            {
                std::cerr << "Do not yet support GPU post processing for multiple batches" << std::endl;
                exit(0);
            }
            // Allocate memory for decode_ptr_host and copy to device
            *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
            CUDA_CHECK(cudaMalloc((void **)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_subscribe;
    rclcpp::Publisher<target_bbox_msgs::msg::BoundingBoxes>::SharedPtr bboxs_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr obj_image_pub;

    // Deserialize the engine from file
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    std::string engine_name = "yolov8s.engine";
    std::string img_sub_topic = "/rs_d435/image_raw";
    std::string img_pub_topic = "/image/obj_detection";
    std::string box_pub_topic = "/targets/bboxs";
    std::string cuda_post_process = "g"; // gpu  如果用cpu写c
    cudaStream_t stream;
    int model_bboxes;
    float *device_buffers[2];
    float *output_buffer_host = nullptr;
    float *decode_ptr_host = nullptr;
    float *decode_ptr_device = nullptr;
};

int main(int argc, char **argv)
{

    std::string wts_name = "";
    std::string engine_name = "";
    std::string img_dir;
    std::string sub_type = "";
    std::string cuda_post_process = "";
    int model_bboxes;
    int is_p = 0;
    float gd = 0.0f, gw = 0.0f;
    int max_channels = 0;

    if (!parse_args(argc, argv, wts_name, engine_name, is_p, img_dir, sub_type, cuda_post_process, gd, gw,
                    max_channels))
    {
        rclcpp::init(argc, argv);
        rclcpp::spin(std::make_shared<yolov8_ros2>("yolov8_ros2"));
        rclcpp::shutdown();
        return 0;
    }

    cudaSetDevice(kGpuId);

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty())
    {
        serialize_engine(wts_name, engine_name, is_p, sub_type, gd, gw, max_channels);
        return 0;
    }

    // Deserialize the engine from file
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = engine->getBindingDimensions(1);
    model_bboxes = out_dims.d[0];
    // Prepare cpu and gpu buffers
    float *device_buffers[2];
    float *output_buffer_host = nullptr;
    float *decode_ptr_host = nullptr;
    float *decode_ptr_device = nullptr;

    // Read images from directory
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0)
    {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host,
                   &decode_ptr_device, cuda_post_process);

    // batch predict
    for (size_t i = 0; i < file_names.size(); i += kBatchSize)
    {
        // Get a batch of images
        std::vector<cv::Mat> img_batch;
        std::vector<std::string> img_name_batch;
        for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++)
        {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
            img_batch.push_back(img);
            img_name_batch.push_back(file_names[j]);
        }
        // Preprocess
        cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
        // Run inference
        infer(*context, stream, (void **)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host,
              decode_ptr_device, model_bboxes, cuda_post_process);
        std::vector<std::vector<Detection>> res_batch;
        if (cuda_post_process == "c")
        {
            // NMS
            batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
        }
        else if (cuda_post_process == "g")
        {
            // Process gpu decode and nms results
            batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
        }
        // Draw bounding boxes
        draw_bbox(img_batch, res_batch);
        // Save images
        for (size_t j = 0; j < img_batch.size(); j++)
        {
            cv::imwrite("_" + img_name_batch[j], img_batch[j]);
        }
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    CUDA_CHECK(cudaFree(decode_ptr_device));
    delete[] decode_ptr_host;
    delete[] output_buffer_host;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
