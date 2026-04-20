#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "segment_infer.h"

namespace fs = std::filesystem;

namespace
{

struct Options
{
    fs::path model_path;
    fs::path input_dir;
    fs::path output_dir;
    float score_threshold = 0.25f;
    float nms_threshold = 0.70f;
    float mask_threshold = 0.50f;
    bool use_letterbox = true;
};

struct ImageResult
{
    std::string image_name;
    std::string overlay_path;
    std::string mask_color_path;
    double infer_ms = 0.0;
    std::string status = "ok";
};

static const cv::Vec3b kClassColorsBGR[] = {
    {0, 0, 0},       // background
    {113, 204, 46},  // drivable
    {60, 76, 231},   // stairs
};
static constexpr int kNumColors = sizeof(kClassColorsBGR) / sizeof(kClassColorsBGR[0]);

bool has_image_extension(const fs::path &path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp";
}

std::vector<fs::path> collect_images(const fs::path &input_dir)
{
    std::vector<fs::path> images;
    for (const auto &entry : fs::directory_iterator(input_dir))
    {
        if (!entry.is_regular_file()) continue;
        if (!has_image_extension(entry.path())) continue;
        images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    return images;
}

cv::Mat render_color_mask(const cv::Mat &class_mask)
{
    cv::Mat color_mask(class_mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int index = 0; index < kNumColors; ++index)
    {
        color_mask.setTo(kClassColorsBGR[index], class_mask == index);
    }
    return color_mask;
}

cv::Mat render_overlay(const cv::Mat &image_bgr, const cv::Mat &class_mask)
{
    cv::Mat color_mask = render_color_mask(class_mask);
    cv::Mat overlay;
    cv::addWeighted(image_bgr, 0.6, color_mask, 0.4, 0.0, overlay);
    return overlay;
}

std::string json_escape(const std::string &value)
{
    std::ostringstream escaped;
    for (const char ch : value)
    {
        switch (ch)
        {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                escaped << ch;
                break;
        }
    }
    return escaped.str();
}

void write_summary(const fs::path &summary_path,
                   const Options &options,
                   const std::vector<ImageResult> &results)
{
    double sum_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    int success_count = 0;
    for (const auto &result : results)
    {
        if (result.status != "ok") continue;
        ++success_count;
        sum_ms += result.infer_ms;
        min_ms = std::min(min_ms, result.infer_ms);
        max_ms = std::max(max_ms, result.infer_ms);
    }
    const double mean_ms = success_count > 0 ? (sum_ms / static_cast<double>(success_count)) : 0.0;
    if (success_count == 0)
    {
        min_ms = 0.0;
    }

    std::ofstream output(summary_path);
    output << "{\n";
    output << "  \"model\": \"" << json_escape(options.model_path.string()) << "\",\n";
    output << "  \"input_dir\": \"" << json_escape(options.input_dir.string()) << "\",\n";
    output << "  \"output_dir\": \"" << json_escape(options.output_dir.string()) << "\",\n";
    output << "  \"images_total\": " << results.size() << ",\n";
    output << "  \"images_succeeded\": " << success_count << ",\n";
    output << "  \"images_failed\": " << (results.size() - static_cast<size_t>(success_count)) << ",\n";
    output << "  \"latency_ms\": {\n";
    output << "    \"mean\": " << std::fixed << std::setprecision(3) << mean_ms << ",\n";
    output << "    \"min\": " << min_ms << ",\n";
    output << "    \"max\": " << max_ms << "\n";
    output << "  },\n";
    output << "  \"results\": [\n";
    for (size_t index = 0; index < results.size(); ++index)
    {
        const auto &result = results[index];
        output << "    {\n";
        output << "      \"image\": \"" << json_escape(result.image_name) << "\",\n";
        output << "      \"status\": \"" << json_escape(result.status) << "\",\n";
        output << "      \"infer_ms\": " << std::fixed << std::setprecision(3) << result.infer_ms << ",\n";
        output << "      \"overlay\": \"" << json_escape(result.overlay_path) << "\",\n";
        output << "      \"mask_color\": \"" << json_escape(result.mask_color_path) << "\"\n";
        output << "    }";
        if (index + 1 != results.size()) output << ",";
        output << "\n";
    }
    output << "  ]\n";
    output << "}\n";
}

void print_usage()
{
    std::cout
        << "Usage: segment_image_infer --model MODEL.bin --input-dir IMAGES --output-dir OUTPUT [options]\n"
        << "Options:\n"
        << "  --score-threshold FLOAT   default 0.25\n"
        << "  --nms-threshold FLOAT     default 0.70\n"
        << "  --mask-threshold FLOAT    default 0.50\n"
        << "  --no-letterbox            disable letterbox preprocessing\n";
}

bool parse_args(int argc, char **argv, Options &options)
{
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        auto require_value = [&](const char *name) -> const char * {
            if (index + 1 >= argc)
            {
                std::cerr << "Missing value for " << name << std::endl;
                std::exit(1);
            }
            return argv[++index];
        };

        if (arg == "--model")
        {
            options.model_path = require_value("--model");
        }
        else if (arg == "--input-dir")
        {
            options.input_dir = require_value("--input-dir");
        }
        else if (arg == "--output-dir")
        {
            options.output_dir = require_value("--output-dir");
        }
        else if (arg == "--score-threshold")
        {
            options.score_threshold = std::stof(require_value("--score-threshold"));
        }
        else if (arg == "--nms-threshold")
        {
            options.nms_threshold = std::stof(require_value("--nms-threshold"));
        }
        else if (arg == "--mask-threshold")
        {
            options.mask_threshold = std::stof(require_value("--mask-threshold"));
        }
        else if (arg == "--no-letterbox")
        {
            options.use_letterbox = false;
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage();
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage();
            return false;
        }
    }

    if (options.model_path.empty() || options.input_dir.empty() || options.output_dir.empty())
    {
        print_usage();
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    Options options;
    if (!parse_args(argc, argv, options))
    {
        return options.model_path.empty() || options.input_dir.empty() || options.output_dir.empty() ? 1 : 0;
    }

    if (!fs::is_regular_file(options.model_path))
    {
        std::cerr << "Model file not found: " << options.model_path << std::endl;
        return 1;
    }
    if (!fs::is_directory(options.input_dir))
    {
        std::cerr << "Input directory not found: " << options.input_dir << std::endl;
        return 1;
    }

    const std::vector<fs::path> image_paths = collect_images(options.input_dir);
    if (image_paths.empty())
    {
        std::cerr << "No images found in: " << options.input_dir << std::endl;
        return 1;
    }

    const fs::path overlay_dir = options.output_dir / "overlay";
    const fs::path mask_dir = options.output_dir / "mask_color";
    fs::create_directories(overlay_dir);
    fs::create_directories(mask_dir);

    SegmentInfer infer;
    const int init_ret = infer.init(options.model_path.string(),
                                    options.score_threshold,
                                    options.nms_threshold,
                                    options.mask_threshold,
                                    options.use_letterbox);
    if (init_ret != 0)
    {
        std::cerr << "Failed to initialize model: " << options.model_path << std::endl;
        return init_ret;
    }

    std::vector<ImageResult> results;
    results.reserve(image_paths.size());

    for (size_t index = 0; index < image_paths.size(); ++index)
    {
        const fs::path &image_path = image_paths[index];
        ImageResult result;
        result.image_name = image_path.filename().string();

        cv::Mat image_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image_bgr.empty())
        {
            result.status = "read_failed";
            results.push_back(result);
            std::cout << "[" << (index + 1) << "/" << image_paths.size() << "] "
                      << image_path.filename().string() << " read failed" << std::endl;
            continue;
        }

        SegmentOutput output;
        const auto start_time = std::chrono::steady_clock::now();
        const int infer_ret = infer.infer(image_bgr, output);
        const auto end_time = std::chrono::steady_clock::now();
        result.infer_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        if (infer_ret != 0)
        {
            result.status = "infer_failed";
            results.push_back(result);
            std::cout << "[" << (index + 1) << "/" << image_paths.size() << "] "
                      << image_path.filename().string() << " infer failed" << std::endl;
            continue;
        }

        cv::Mat overlay = render_overlay(image_bgr, output.class_mask);
        cv::Mat mask_color = render_color_mask(output.class_mask);

        const fs::path overlay_path = overlay_dir / (image_path.stem().string() + "_overlay.png");
        const fs::path mask_path = mask_dir / (image_path.stem().string() + "_mask.png");
        cv::imwrite(overlay_path.string(), overlay);
        cv::imwrite(mask_path.string(), mask_color);

        result.overlay_path = fs::relative(overlay_path, options.output_dir).string();
        result.mask_color_path = fs::relative(mask_path, options.output_dir).string();
        results.push_back(result);

        std::cout << "[" << (index + 1) << "/" << image_paths.size() << "] "
                  << image_path.filename().string() << " "
                  << std::fixed << std::setprecision(2) << result.infer_ms << " ms" << std::endl;
    }

    const fs::path summary_path = options.output_dir / "summary.json";
    write_summary(summary_path, options, results);
    std::cout << "Saved outputs to: " << options.output_dir << std::endl;
    std::cout << "Saved summary to: " << summary_path << std::endl;
    return 0;
}
