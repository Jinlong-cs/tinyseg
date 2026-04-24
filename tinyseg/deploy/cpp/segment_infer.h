#ifndef SEGMENT_INFER_H
#define SEGMENT_INFER_H

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include <dnn/hb_dnn.h>
#include <opencv2/opencv.hpp>

#define SEG_CHECK_SUCCESS(ret_code, errmsg)                                                           \
    do                                                                                                \
    {                                                                                                 \
        if ((ret_code) != 0)                                                                         \
        {                                                                                             \
            std::cout << "=> [SEG ERROR] " << (errmsg) << ", error code: " << (ret_code)           \
                      << "!" << std::endl;                                                            \
            return ret_code;                                                                          \
        }                                                                                             \
    } while (0)

struct SegmentOutput
{
    cv::Mat class_mask; // full-res class mask (0=background, 1=drivable, 2=stairs)
};

class SegmentInfer
{
public:
    SegmentInfer() = default;
    ~SegmentInfer();

    int init(const std::string &model_file_name,
             float score_threshold = 0.25f,
             float nms_threshold = 0.70f,
             float mask_threshold = 0.50f,
             bool use_letterbox = true);

    int infer(const cv::Mat &image_bgr, SegmentOutput &output);

    int input_width() const
    {
        return input_w_;
    }

    int input_height() const
    {
        return input_h_;
    }

private:
    struct Candidate
    {
        int class_id = -1;
        float score = 0.0f;
        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;
        std::array<float, 32> mask_coeff{};
    };

    int prepare_input_tensor();
    int prepare_output_tensors();
    int bgr_to_nv12(const cv::Mat &bgr, cv::Mat &nv12) const;
    cv::Mat preprocess(const cv::Mat &image_bgr);
    int postprocess(const cv::Mat &orig_bgr, SegmentOutput &output);
    float sigmoid(float x) const;

    // DNN handles
    hbPackedDNNHandle_t packed_dnn_handle_ = nullptr;
    hbDNNHandle_t dnn_handle_ = nullptr;

    hbDNNTensor input_tensor_{};
    std::vector<hbDNNTensor> output_tensors_;
    hbDNNTensorProperties input_properties_{};

    bool initialized_ = false;
    std::string model_file_name_;

    // runtime params
    float score_threshold_ = 0.25f;
    float nms_threshold_ = 0.70f;
    float mask_threshold_ = 0.50f;
    float conf_threshold_raw_ = 0.0f;
    bool use_letterbox_ = true;

    // model config
    int input_w_ = 0;
    int input_h_ = 0;
    std::array<int, 3> strides_{{8, 16, 32}};
    static constexpr int num_classes_ = 2;
    static constexpr int box_channels_ = 64;  // DFL: 4 * reg_max(16)
    static constexpr int reg_max_ = 16;
    static constexpr int mask_channels_ = 32;

    // preprocessing state per frame
    float x_scale_ = 1.0f;
    float y_scale_ = 1.0f;
    int x_shift_ = 0;
    int y_shift_ = 0;

};

#endif
