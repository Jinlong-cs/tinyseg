#include "segment_infer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>

#include <dnn/hb_dnn_ext.h>
#include <dnn/hb_sys.h>
#include <opencv2/dnn.hpp>

namespace
{

struct ProtoShape
{
    int h = 0;
    int w = 0;
    int c = 0;
    bool chw_layout = false;
};

bool all_finite(std::initializer_list<float> values)
{
    for (const float value : values)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    return true;
}

template <size_t N>
bool copy_finite_values(const float *src, std::array<float, N> &dst)
{
    for (size_t i = 0; i < N; ++i)
    {
        if (!std::isfinite(src[i]))
        {
            return false;
        }
        dst[i] = src[i];
    }
    return true;
}

} // namespace

SegmentInfer::~SegmentInfer()
{
    int ret_code = 0;

    if (input_tensor_.sysMem[0].virAddr != nullptr)
    {
        ret_code = hbSysFreeMem(&input_tensor_.sysMem[0]);
        if (ret_code != 0)
        {
            std::cout << "=> [SEG WARN] hbSysFreeMem input failed: " << ret_code << std::endl;
        }
    }

    for (auto &tensor : output_tensors_)
    {
        if (tensor.sysMem[0].virAddr != nullptr)
        {
            ret_code = hbSysFreeMem(&tensor.sysMem[0]);
            if (ret_code != 0)
            {
                std::cout << "=> [SEG WARN] hbSysFreeMem output failed: " << ret_code << std::endl;
            }
        }
    }
    output_tensors_.clear();

    if (packed_dnn_handle_ != nullptr)
    {
        ret_code = hbDNNRelease(packed_dnn_handle_);
        if (ret_code != 0)
        {
            std::cout << "=> [SEG WARN] hbDNNRelease failed: " << ret_code << std::endl;
        }
    }
}

int SegmentInfer::init(const std::string &model_file_name,
                       float score_threshold,
                       float nms_threshold,
                       float mask_threshold,
                       bool use_letterbox)
{
    model_file_name_ = model_file_name;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    mask_threshold_ = mask_threshold;
    use_letterbox_ = use_letterbox;

    if (score_threshold_ <= 0.0f || score_threshold_ >= 1.0f)
    {
        std::cout << "=> [SEG ERROR] score_threshold must be in (0,1), got "
                  << score_threshold_ << std::endl;
        return -1;
    }
    conf_threshold_raw_ = -std::log(1.0f / score_threshold_ - 1.0f);

    int ret_code = 0;
    int32_t model_count = 0;
    const char *model_file = model_file_name_.c_str();

    ret_code = hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_file, 1);
    SEG_CHECK_SUCCESS(ret_code, "hbDNNInitializeFromFiles failed");

    const char **model_name_list = nullptr;
    ret_code = hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle_);
    SEG_CHECK_SUCCESS(ret_code, "hbDNNGetModelNameList failed");
    if (model_count <= 0)
    {
        std::cout << "=> [SEG ERROR] no model found in bin file." << std::endl;
        return -1;
    }

    ret_code = hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_list[0]);
    SEG_CHECK_SUCCESS(ret_code, "hbDNNGetModelHandle failed");

    ret_code = hbDNNGetInputTensorProperties(&input_properties_, dnn_handle_, 0);
    SEG_CHECK_SUCCESS(ret_code, "hbDNNGetInputTensorProperties failed");

    if (input_properties_.tensorLayout == HB_DNN_LAYOUT_NHWC)
    {
        input_h_ = input_properties_.validShape.dimensionSize[1];
        input_w_ = input_properties_.validShape.dimensionSize[2];
    }
    else if (input_properties_.tensorLayout == HB_DNN_LAYOUT_NCHW)
    {
        input_h_ = input_properties_.validShape.dimensionSize[2];
        input_w_ = input_properties_.validShape.dimensionSize[3];
    }
    else
    {
        std::cout << "=> [SEG ERROR] unsupported input tensor layout: "
                  << input_properties_.tensorLayout << std::endl;
        return -1;
    }

    ret_code = prepare_input_tensor();
    SEG_CHECK_SUCCESS(ret_code, "prepare_input_tensor failed");

    ret_code = prepare_output_tensors();
    SEG_CHECK_SUCCESS(ret_code, "prepare_output_tensors failed");

    if (output_tensors_.size() != 10)
    {
        std::cout << "=> [SEG ERROR] expected 10 outputs for YOLO-Seg, got "
                  << output_tensors_.size() << std::endl;
        return -1;
    }

    const int num_classes = output_tensors_[0].properties.validShape.dimensionSize[3];
    if (num_classes != num_classes_)
    {
        std::cout << "=> [SEG ERROR] invalid class count from output tensor: "
                  << num_classes << std::endl;
        return -1;
    }
    const int box_channels = output_tensors_[1].properties.validShape.dimensionSize[3];
    if (box_channels != box_channels_)
    {
        std::cout << "=> [SEG ERROR] invalid box channel count from output tensor: "
                  << box_channels << std::endl;
        return -1;
    }
    const int mask_channels = output_tensors_[2].properties.validShape.dimensionSize[3];
    if (mask_channels != mask_channels_)
    {
        std::cout << "=> [SEG ERROR] invalid mask channel count from output tensor: "
                  << mask_channels << std::endl;
        return -1;
    }

    initialized_ = true;
    std::cout << "=> [SEG INFO] model loaded: " << model_name_list[0]
              << ", input shape = (1,3," << input_h_ << "," << input_w_ << ")" << std::endl;
    return 0;
}

int SegmentInfer::prepare_input_tensor()
{
    int ret_code = 0;
    std::memset(&input_tensor_, 0, sizeof(hbDNNTensor));
    input_tensor_.properties = input_properties_;
    input_tensor_.properties.tensorType = HB_DNN_IMG_TYPE_NV12;

    ret_code = hbSysAllocCachedMem(&input_tensor_.sysMem[0], input_properties_.alignedByteSize);
    SEG_CHECK_SUCCESS(ret_code, "hbSysAllocCachedMem input failed");

    return 0;
}

int SegmentInfer::prepare_output_tensors()
{
    int ret_code = 0;
    int32_t output_count = 0;
    ret_code = hbDNNGetOutputCount(&output_count, dnn_handle_);
    SEG_CHECK_SUCCESS(ret_code, "hbDNNGetOutputCount failed");

    output_tensors_.resize(output_count);
    for (int i = 0; i < output_count; ++i)
    {
        std::memset(&output_tensors_[i], 0, sizeof(hbDNNTensor));
        ret_code = hbDNNGetOutputTensorProperties(&output_tensors_[i].properties, dnn_handle_, i);
        SEG_CHECK_SUCCESS(ret_code, "hbDNNGetOutputTensorProperties failed");

        ret_code = hbSysAllocCachedMem(&output_tensors_[i].sysMem[0],
                                       output_tensors_[i].properties.alignedByteSize);
        SEG_CHECK_SUCCESS(ret_code, "hbSysAllocCachedMem output failed");
    }

    return 0;
}

int SegmentInfer::bgr_to_nv12(const cv::Mat &bgr, cv::Mat &nv12) const
{
    if (bgr.empty() || bgr.type() != CV_8UC3)
    {
        std::cout << "=> [SEG ERROR] invalid input image in bgr_to_nv12" << std::endl;
        return -1;
    }

    const int h = bgr.rows;
    const int w = bgr.cols;
    const int area = h * w;

    cv::Mat yuv_i420;
    cv::cvtColor(bgr, yuv_i420, cv::COLOR_BGR2YUV_I420);
    if (!yuv_i420.isContinuous())
    {
        yuv_i420 = yuv_i420.clone();
    }

    nv12.create(h * 3 / 2, w, CV_8UC1);
    uint8_t *dst = nv12.ptr<uint8_t>();
    const uint8_t *src = yuv_i420.ptr<uint8_t>();

    std::memcpy(dst, src, area);
    const uint8_t *u_plane = src + area;
    const uint8_t *v_plane = u_plane + area / 4;
    uint8_t *uv_dst = dst + area;
    for (int i = 0; i < area / 4; ++i)
    {
        uv_dst[2 * i] = u_plane[i];
        uv_dst[2 * i + 1] = v_plane[i];
    }
    return 0;
}

cv::Mat SegmentInfer::preprocess(const cv::Mat &image_bgr)
{
    cv::Mat resized;
    if (!use_letterbox_)
    {
        cv::resize(image_bgr, resized, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_LINEAR);
        x_scale_ = static_cast<float>(input_w_) / static_cast<float>(image_bgr.cols);
        y_scale_ = static_cast<float>(input_h_) / static_cast<float>(image_bgr.rows);
        x_shift_ = 0;
        y_shift_ = 0;
        return resized;
    }

    x_scale_ = std::min(static_cast<float>(input_w_) / static_cast<float>(image_bgr.cols),
                        static_cast<float>(input_h_) / static_cast<float>(image_bgr.rows));
    y_scale_ = x_scale_;

    const int new_w = static_cast<int>(std::round(image_bgr.cols * x_scale_));
    const int new_h = static_cast<int>(std::round(image_bgr.rows * y_scale_));

    x_shift_ = (input_w_ - new_w) / 2;
    y_shift_ = (input_h_ - new_h) / 2;

    const int right_pad = input_w_ - new_w - x_shift_;
    const int bottom_pad = input_h_ - new_h - y_shift_;

    cv::resize(image_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, resized, y_shift_, bottom_pad, x_shift_, right_pad,
                       cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    return resized;
}

float SegmentInfer::sigmoid(float x) const
{
    return 1.0f / (1.0f + std::exp(-x));
}

int SegmentInfer::postprocess(const cv::Mat &orig_bgr, SegmentOutput &output)
{
    std::vector<Candidate> candidates;
    candidates.reserve(1024);

    for (int scale_idx = 0; scale_idx < 3; ++scale_idx)
    {
        const int cls_idx = scale_idx * 3 + 0;
        const int box_idx = scale_idx * 3 + 1;
        const int mce_idx = scale_idx * 3 + 2;
        const int stride = strides_[scale_idx];

        const auto &cls_prop = output_tensors_[cls_idx].properties;
        const auto &box_prop = output_tensors_[box_idx].properties;
        const auto &mce_prop = output_tensors_[mce_idx].properties;

        const int h = cls_prop.validShape.dimensionSize[1];
        const int w = cls_prop.validShape.dimensionSize[2];
        const int cls_c = cls_prop.validShape.dimensionSize[3];
        const int box_c = box_prop.validShape.dimensionSize[3];
        const int mce_c = mce_prop.validShape.dimensionSize[3];

        if (cls_c != num_classes_ || box_c != box_channels_ || mce_c != mask_channels_)
        {
            std::cout << "=> [SEG ERROR] output shape mismatch at scale " << scale_idx << std::endl;
            return -1;
        }

        const float *cls_raw = reinterpret_cast<float *>(output_tensors_[cls_idx].sysMem[0].virAddr);
        const float *box_raw = reinterpret_cast<float *>(output_tensors_[box_idx].sysMem[0].virAddr);
        const float *mce_raw = reinterpret_cast<float *>(output_tensors_[mce_idx].sysMem[0].virAddr);

        for (int iy = 0; iy < h; ++iy)
        {
            for (int ix = 0; ix < w; ++ix)
            {
                const int offset = iy * w + ix;
                const float *cls_ptr = cls_raw + offset * cls_c;
                const float *box_ptr = box_raw + offset * box_c;
                const float *mce_ptr = mce_raw + offset * mce_c;

                int best_cid = 0;
                float best_logit = cls_ptr[0];
                for (int c = 1; c < cls_c; ++c)
                {
                    if (cls_ptr[c] > best_logit)
                    {
                        best_logit = cls_ptr[c];
                        best_cid = c;
                    }
                }
                if (!std::isfinite(best_logit) || best_logit < conf_threshold_raw_)
                {
                    continue;
                }

                // DFL decode: softmax over reg_max bins then expectation
                float ltrb[4] = {0.f, 0.f, 0.f, 0.f};
                for (int d = 0; d < 4; ++d)
                {
                    const float *dfl_ptr = box_ptr + d * reg_max_;
                    float max_val = dfl_ptr[0];
                    for (int r = 1; r < reg_max_; ++r)
                        if (dfl_ptr[r] > max_val) max_val = dfl_ptr[r];
                    float sum_exp = 0.f;
                    float weighted = 0.f;
                    for (int r = 0; r < reg_max_; ++r)
                    {
                        float e = std::exp(dfl_ptr[r] - max_val);
                        sum_exp += e;
                        weighted += e * static_cast<float>(r);
                    }
                    ltrb[d] = (sum_exp > 0.f) ? (weighted / sum_exp) : 0.f;
                }
                const float left = ltrb[0];
                const float top = ltrb[1];
                const float right = ltrb[2];
                const float bottom = ltrb[3];
                if (!all_finite({left, top, right, bottom}))
                {
                    continue;
                }

                const float cx = (static_cast<float>(ix) + 0.5f) * static_cast<float>(stride);
                const float cy = (static_cast<float>(iy) + 0.5f) * static_cast<float>(stride);

                Candidate cand;
                cand.class_id = best_cid;
                cand.score = sigmoid(best_logit);
                cand.x1 = cx - left * static_cast<float>(stride);
                cand.y1 = cy - top * static_cast<float>(stride);
                cand.x2 = cx + right * static_cast<float>(stride);
                cand.y2 = cy + bottom * static_cast<float>(stride);
                if (!all_finite({cand.score, cand.x1, cand.y1, cand.x2, cand.y2}))
                {
                    continue;
                }
                if (!copy_finite_values(mce_ptr, cand.mask_coeff))
                {
                    continue;
                }
                candidates.push_back(cand);
            }
        }
    }

    std::vector<int> selected_candidate_ids;
    selected_candidate_ids.reserve(candidates.size());
    for (int cid = 0; cid < num_classes_; ++cid)
    {
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        std::vector<int> reverse_map;
        boxes.reserve(candidates.size());
        scores.reserve(candidates.size());
        reverse_map.reserve(candidates.size());

        for (size_t i = 0; i < candidates.size(); ++i)
        {
            if (candidates[i].class_id != cid)
            {
                continue;
            }
            if (!all_finite({candidates[i].score,
                             candidates[i].x1,
                             candidates[i].y1,
                             candidates[i].x2,
                             candidates[i].y2}))
            {
                continue;
            }
            const float w = candidates[i].x2 - candidates[i].x1;
            const float h = candidates[i].y2 - candidates[i].y1;
            if (!all_finite({w, h}) || w <= 1.0f || h <= 1.0f)
            {
                continue;
            }
            boxes.emplace_back(candidates[i].x1, candidates[i].y1, w, h);
            scores.push_back(candidates[i].score);
            reverse_map.push_back(static_cast<int>(i));
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, score_threshold_, nms_threshold_, indices);
        for (const int idx : indices)
        {
            selected_candidate_ids.push_back(reverse_map[idx]);
        }
    }

    std::sort(selected_candidate_ids.begin(), selected_candidate_ids.end(),
              [&](const int lhs, const int rhs)
              { return candidates[lhs].score > candidates[rhs].score; });

    const auto &proto_prop = output_tensors_[9].properties;
    const float *proto_ptr = reinterpret_cast<float *>(output_tensors_[9].sysMem[0].virAddr);
    const int p1 = proto_prop.validShape.dimensionSize[1];
    const int p2 = proto_prop.validShape.dimensionSize[2];
    const int p3 = proto_prop.validShape.dimensionSize[3];

    ProtoShape proto_shape;
    if (proto_prop.tensorLayout == HB_DNN_LAYOUT_NCHW && p1 == mask_channels_)
    {
        proto_shape = ProtoShape{p2, p3, p1, true};
    }
    else if (p3 == mask_channels_)
    {
        // Some converted models report NCHW but keep data in HWC memory order.
        proto_shape = ProtoShape{p1, p2, p3, false};
    }
    else if (p2 == mask_channels_)
    {
        proto_shape = ProtoShape{p1, p3, p2, false};
    }
    else
    {
        std::cout << "=> [SEG ERROR] cannot parse proto shape: (" << p1 << ", "
                  << p2 << ", " << p3 << ")" << std::endl;
        return -1;
    }

    auto proto_value = [&](int y, int x, int c) -> float
    {
        if (proto_shape.chw_layout)
        {
            const size_t offset = (static_cast<size_t>(c) * proto_shape.h + static_cast<size_t>(y)) * proto_shape.w + static_cast<size_t>(x);
            return proto_ptr[offset];
        }
        const size_t offset = (static_cast<size_t>(y) * proto_shape.w + static_cast<size_t>(x)) * proto_shape.c + static_cast<size_t>(c);
        return proto_ptr[offset];
    };

    output.class_mask = cv::Mat::zeros(orig_bgr.rows, orig_bgr.cols, CV_8UC1);

    for (const int selected_id : selected_candidate_ids)
    {
        const Candidate &cand = candidates[selected_id];

        const int ox1 = std::clamp(static_cast<int>(std::floor((cand.x1 - static_cast<float>(x_shift_)) / x_scale_)), 0, orig_bgr.cols - 1);
        const int oy1 = std::clamp(static_cast<int>(std::floor((cand.y1 - static_cast<float>(y_shift_)) / y_scale_)), 0, orig_bgr.rows - 1);
        const int ox2 = std::clamp(static_cast<int>(std::ceil((cand.x2 - static_cast<float>(x_shift_)) / x_scale_)), 0, orig_bgr.cols);
        const int oy2 = std::clamp(static_cast<int>(std::ceil((cand.y2 - static_cast<float>(y_shift_)) / y_scale_)), 0, orig_bgr.rows);
        if (ox2 <= ox1 || oy2 <= oy1)
        {
            continue;
        }

        const int px1 = std::clamp(static_cast<int>(std::floor(cand.x1 * static_cast<float>(proto_shape.w) / static_cast<float>(input_w_))), 0, proto_shape.w - 1);
        const int py1 = std::clamp(static_cast<int>(std::floor(cand.y1 * static_cast<float>(proto_shape.h) / static_cast<float>(input_h_))), 0, proto_shape.h - 1);
        const int px2 = std::clamp(static_cast<int>(std::ceil(cand.x2 * static_cast<float>(proto_shape.w) / static_cast<float>(input_w_))), 0, proto_shape.w);
        const int py2 = std::clamp(static_cast<int>(std::ceil(cand.y2 * static_cast<float>(proto_shape.h) / static_cast<float>(input_h_))), 0, proto_shape.h);
        if (px2 <= px1 || py2 <= py1)
        {
            continue;
        }

        cv::Mat mask_small(py2 - py1, px2 - px1, CV_8UC1, cv::Scalar(0));
        for (int py = py1; py < py2; ++py)
        {
            uint8_t *row_ptr = mask_small.ptr<uint8_t>(py - py1);
            for (int px = px1; px < px2; ++px)
            {
                float sum = 0.0f;
                for (int c = 0; c < mask_channels_; ++c)
                {
                    sum += cand.mask_coeff[static_cast<size_t>(c)] * proto_value(py, px, c);
                }
                row_ptr[px - px1] = (sum > mask_threshold_) ? 255 : 0;
            }
        }

        const cv::Rect roi_rect(ox1, oy1, ox2 - ox1, oy2 - oy1);
        cv::Mat mask_roi;
        cv::resize(mask_small, mask_roi, roi_rect.size(), 0.0, 0.0, cv::INTER_NEAREST);

        output.class_mask(roi_rect).setTo(static_cast<uint8_t>(cand.class_id + 1), mask_roi);
    }

    return 0;
}

int SegmentInfer::infer(const cv::Mat &image_bgr, SegmentOutput &output)
{
    if (!initialized_)
    {
        std::cout << "=> [SEG ERROR] model is not initialized." << std::endl;
        return -1;
    }
    if (image_bgr.empty())
    {
        std::cout << "=> [SEG ERROR] empty input image." << std::endl;
        return -1;
    }

    cv::Mat preprocessed = preprocess(image_bgr);
    cv::Mat input_nv12;
    int ret_code = bgr_to_nv12(preprocessed, input_nv12);
    SEG_CHECK_SUCCESS(ret_code, "bgr_to_nv12 failed");

    ret_code = hbSysWriteMem(&input_tensor_.sysMem[0],
                             reinterpret_cast<char *>(input_nv12.data),
                             input_tensor_.sysMem[0].memSize);
    SEG_CHECK_SUCCESS(ret_code, "hbSysWriteMem input failed");

    ret_code = hbSysFlushMem(&input_tensor_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    SEG_CHECK_SUCCESS(ret_code, "hbSysFlushMem input failed");

    hbDNNTensor *output_ptr = output_tensors_.data();
    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    ret_code = hbDNNInfer(&task_handle, &output_ptr, &input_tensor_, dnn_handle_, &infer_ctrl_param);
    SEG_CHECK_SUCCESS(ret_code, "hbDNNInfer failed");

    ret_code = hbDNNWaitTaskDone(task_handle, 0);
    SEG_CHECK_SUCCESS(ret_code, "hbDNNWaitTaskDone failed");

    ret_code = hbDNNReleaseTask(task_handle);
    SEG_CHECK_SUCCESS(ret_code, "hbDNNReleaseTask failed");
    for (auto &tensor : output_tensors_)
    {
        ret_code = hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        SEG_CHECK_SUCCESS(ret_code, "hbSysFlushMem output failed");
    }

    ret_code = postprocess(image_bgr, output);
    SEG_CHECK_SUCCESS(ret_code, "postprocess failed");
    return 0;
}
