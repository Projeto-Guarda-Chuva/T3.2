#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#if __has_include(<onnxruntime_cxx_api.h>)
#  include <onnxruntime_cxx_api.h>
#elif __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#  include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#else
#  error "Cannot find onnxruntime_cxx_api.h – check ONNXRUNTIME_ROOT in CMakeLists"
#endif

// COCO-17 keypoint indices
namespace KP {
    constexpr int NOSE         = 0;
    constexpr int LEFT_EYE     = 1;
    constexpr int RIGHT_EYE    = 2;
    constexpr int LEFT_EAR     = 3;
    constexpr int RIGHT_EAR    = 4;
    constexpr int LEFT_SHOULDER  = 5;
    constexpr int RIGHT_SHOULDER = 6;
    constexpr int LEFT_ELBOW     = 7;
    constexpr int RIGHT_ELBOW    = 8;
    constexpr int LEFT_WRIST     = 9;
    constexpr int RIGHT_WRIST    = 10;
    constexpr int LEFT_HIP       = 11;
    constexpr int RIGHT_HIP      = 12;
    constexpr int LEFT_KNEE      = 13;
    constexpr int RIGHT_KNEE     = 14;
    constexpr int LEFT_ANKLE     = 15;
    constexpr int RIGHT_ANKLE    = 16;
    constexpr int NUM_KP         = 17;
}

// Esqueleto COCO
static const std::vector<std::pair<int,int>> SKELETON = {
    {KP::NOSE, KP::LEFT_EYE},    {KP::NOSE, KP::RIGHT_EYE},
    {KP::LEFT_EYE, KP::LEFT_EAR},{KP::RIGHT_EYE, KP::RIGHT_EAR},
    {KP::LEFT_SHOULDER, KP::RIGHT_SHOULDER},
    {KP::LEFT_SHOULDER, KP::LEFT_ELBOW},
    {KP::LEFT_ELBOW, KP::LEFT_WRIST},
    {KP::RIGHT_SHOULDER, KP::RIGHT_ELBOW},
    {KP::RIGHT_ELBOW, KP::RIGHT_WRIST},
    {KP::LEFT_SHOULDER, KP::LEFT_HIP},
    {KP::RIGHT_SHOULDER, KP::RIGHT_HIP},
    {KP::LEFT_HIP, KP::RIGHT_HIP},
    {KP::LEFT_HIP, KP::LEFT_KNEE},
    {KP::LEFT_KNEE, KP::LEFT_ANKLE},
    {KP::RIGHT_HIP, KP::RIGHT_KNEE},
    {KP::RIGHT_KNEE, KP::RIGHT_ANKLE}
};


struct Keypoint {
    float x = 0, y = 0, conf = 0;
};

struct Detection {
    cv::Rect2d bbox;
    float      score = 0;
    std::array<Keypoint, KP::NUM_KP> kps;
};

enum class Gesture { REPOUSO, SUBIR, DESCER };

static std::string gestureName(Gesture g) {
    switch (g) {
        case Gesture::SUBIR:   return "SUBIR";
        case Gesture::DESCER:  return "DESCER";
        default:               return "REPOUSO";
    }
}
static cv::Scalar gestureColor(Gesture g) {
    switch (g) {
        case Gesture::SUBIR:   return {0, 220, 0};
        case Gesture::DESCER:  return {0, 60, 220};
        default:               return {180, 180, 180};
    }
}

// Análise do gesto

struct GestureConfig {
    float confThreshold = 0.40f; // min keypoint confidence to use
};

class GestureAnalyzer {
public:
    explicit GestureAnalyzer(GestureConfig cfg = GestureConfig{}) : cfg_(cfg) {}
    // Pulso acima do ombro/abaixo do quadril = Subir/Descer, se nada disso então repouso
    Gesture update(const Detection& det, int /*imgH*/) {
        float thr = cfg_.confThreshold;
        float shoulderY = avg(det, {KP::LEFT_SHOULDER, KP::RIGHT_SHOULDER}, thr);
        float hipY      = avg(det, {KP::LEFT_HIP,      KP::RIGHT_HIP},      thr);
        float wristY    = avg(det, {KP::LEFT_WRIST,    KP::RIGHT_WRIST},    thr);

        if (wristY < 0) return Gesture::REPOUSO;
        if (shoulderY > 0 && wristY < shoulderY) return Gesture::SUBIR;
        if (hipY      > 0 && wristY > hipY)      return Gesture::DESCER;
        return Gesture::REPOUSO;
    }

    float getShoulderY(const Detection& det) const {
        return avg(det, {KP::LEFT_SHOULDER, KP::RIGHT_SHOULDER}, cfg_.confThreshold);
    }
    float getHipY(const Detection& det) const {
        return avg(det, {KP::LEFT_HIP, KP::RIGHT_HIP}, cfg_.confThreshold);
    }

private:
    GestureConfig cfg_;

    static float avg(const Detection& det, std::initializer_list<int> idxs, float thr) {
        float sum = 0; int n = 0;
        for (int i : idxs)
            if (det.kps[i].conf >= thr) { sum += det.kps[i].y; ++n; }
        return n > 0 ? sum / n : -1.f;
    }
};

// ONNX backend inference
// Pra rodar na ESP32-P4:
//   No lugar de run() colocar esp-dl Model::run() pra cada tensor de saída.
//   A lógica permanece igual.
class OnnxInferenceBackend {
public:
    struct Level {
        std::vector<float> data;
        int channels, gridH, gridW;
    };

    struct RunResult {
        std::vector<Level> levels;
    };

    explicit OnnxInferenceBackend(const std::string& modelPath,
                                  bool useGpu = false) {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "gesture");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        (void)useGpu;

        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), opts);
        Ort::AllocatorWithDefaultOptions alloc;

        // Input
        auto inName = session_->GetInputNameAllocated(0, alloc);
        inputName_  = inName.get();
        auto inShape = session_->GetInputTypeInfo(0)
                           .GetTensorTypeAndShapeInfo().GetShape();
        inputH_ = (int)inShape[2];
        inputW_ = (int)inShape[3];
        std::cout << "[Backend] Input: " << inputW_ << "x" << inputH_ << "\n";

        //Outputs
        numOutputs_ = session_->GetOutputCount();
        for (size_t i = 0; i < numOutputs_; ++i) {
            auto name = session_->GetOutputNameAllocated(i, alloc);
            outputNames_.push_back(name.get());
        }
        for (auto& n : outputNames_)
            outputNamePtrs_.push_back(n.c_str());
        std::cout << "[Backend] Outputs: " << numOutputs_ << " tensor(s)\n";
    }

    int inputH() const { return inputH_; }
    int inputW() const { return inputW_; }

    RunResult run(const std::vector<float>& inputTensor) {
        std::array<int64_t, 4> inShape = {1, 3, (int64_t)inputH_, (int64_t)inputW_};
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        auto inTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            const_cast<float*>(inputTensor.data()),
            inputTensor.size(),
            inShape.data(), inShape.size());

        const char* inNames[] = {inputName_.c_str()};
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            inNames, &inTensor, 1,
            outputNamePtrs_.data(), numOutputs_);

        RunResult res;
        for (size_t i = 0; i < numOutputs_; ++i) {
            auto  info  = outputs[i].GetTensorTypeAndShapeInfo();
            auto  shape = info.GetShape();  // [1, C, gH, gW]
            float* ptr  = outputs[i].GetTensorMutableData<float>();
            size_t n    = info.GetElementCount();

            if (!shapePrinted_) {
                std::cout << "[Backend] Output[" << i << "] shape: [";
                for (size_t d = 0; d < shape.size(); ++d)
                    std::cout << shape[d] << (d+1<shape.size()?",":"");
                std::cout << "]\n";
            }

            Level lv;
            lv.channels = (int)shape[1];
            lv.gridH    = shape.size() >= 4 ? (int)shape[2] : 1;
            lv.gridW    = shape.size() >= 4 ? (int)shape[3] : 1;
            lv.data     = std::vector<float>(ptr, ptr + n);
            res.levels.push_back(std::move(lv));
        }
        shapePrinted_ = true;
        return res;
    }

private:
    Ort::Env env_{nullptr};
    std::unique_ptr<Ort::Session> session_;
    std::string              inputName_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> outputNamePtrs_;
    size_t numOutputs_ = 0;
    int    inputH_ = 192, inputW_ = 192;
    bool   shapePrinted_ = false;
};

// Preprocessor
struct PrepareResult {
    std::vector<float> tensor;
    float scale;
    float padLeft, padTop;
};

static PrepareResult preprocess(const cv::Mat& frame, int netH, int netW) {
    int fh = frame.rows, fw = frame.cols;
    float scale = std::min((float)netH / fh, (float)netW / fw);
    int nh = (int)std::round(fh * scale);
    int nw = (int)std::round(fw * scale);

    cv::Mat resized;
    cv::resize(frame, resized, {nw, nh}, 0, 0, cv::INTER_LINEAR);

    // Letterbox padding
    cv::Mat padded(netH, netW, CV_8UC3, cv::Scalar(114,114,114));
    int top  = (netH - nh) / 2;
    int left = (netW - nw) / 2;
    resized.copyTo(padded(cv::Rect(left, top, nw, nh)));

    // BGR → RGB, HWC → CHW, /255
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0/255.0);

    std::vector<float> tensor(3 * netH * netW);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; ++c)
        std::memcpy(tensor.data() + c * netH * netW,
                    channels[c].ptr<float>(),
                    netH * netW * sizeof(float));

    return {tensor, scale, (float)left, (float)top};
}

// Postprocessor
static float dfl_decode(const float* logits, int reg_max) {
    // softmax
    float maxv = *std::max_element(logits, logits + reg_max);
    float sum  = 0;
    std::vector<float> sm(reg_max);
    for (int i = 0; i < reg_max; ++i) { sm[i] = std::exp(logits[i] - maxv); sum += sm[i]; }
    float val = 0;
    for (int i = 0; i < reg_max; ++i) val += sm[i]/sum * i;
    return val;
}

static std::vector<Detection> postprocess(
    const OnnxInferenceBackend::RunResult& res,
    const PrepareResult& prep,
    int origH, int origW,
    int netH,  int netW,
    float confThr = 0.40f,
    float nmsIou  = 0.45f)
{
    constexpr int REG_MAX = 16;
    constexpr int NUM_KP  = KP::NUM_KP;

    std::vector<cv::Rect2d> boxes;
    std::vector<float>      scores;
    std::vector<std::array<Keypoint, NUM_KP>> kpsList;

    auto toOrig = [&](float nx, float ny, float& ox, float& oy) {
        ox = (nx - prep.padLeft) / prep.scale;
        oy = (ny - prep.padTop)  / prep.scale;
    };

    for (auto& lv : res.levels) {
        // stride = input_size / grid_size
        float stride = (float)netW / lv.gridW;
        int   gH     = lv.gridH;
        int   gW     = lv.gridW;
        int   C      = lv.channels;

        // Accessor: lv.data[c * gH*gW + gy*gW + gx]
        auto at = [&](int c, int gy, int gx) -> float {
            return lv.data[c * gH * gW + gy * gW + gx];
        };

        for (int gy = 0; gy < gH; ++gy) {
            for (int gx = 0; gx < gW; ++gx) {

                // Class confidence (sigmoid)
                float cls_raw = at(4 * REG_MAX, gy, gx);
                float conf    = 1.f / (1.f + std::exp(-cls_raw));
                if (conf < confThr) continue;

                // DFL bbox decode
                // Anchor centre in net-input space
                float ax = (gx + 0.5f) * stride;
                float ay = (gy + 0.5f) * stride;

                // 4 distances: left, top, right, bottom
                float dist[4];
                for (int side = 0; side < 4; ++side) {
                    const float* logits = &lv.data[(side * REG_MAX) * gH * gW
                                                   + gy * gW + gx];
                    std::vector<float> buf(REG_MAX);
                    for (int r = 0; r < REG_MAX; ++r)
                        buf[r] = at(side * REG_MAX + r, gy, gx);
                    dist[side] = dfl_decode(buf.data(), REG_MAX) * stride;
                }
                // dist: [left, top, right, bottom] from anchor centre
                float nx0 = ax - dist[0], ny0 = ay - dist[1];
                float nx1 = ax + dist[2], ny1 = ay + dist[3];

                float x0, y0, x1, y1;
                toOrig(nx0, ny0, x0, y0);
                toOrig(nx1, ny1, x1, y1);
                x0 = std::clamp(x0, 0.f, (float)origW);
                y0 = std::clamp(y0, 0.f, (float)origH);
                x1 = std::clamp(x1, 0.f, (float)origW);
                y1 = std::clamp(y1, 0.f, (float)origH);
                if (x1 <= x0 || y1 <= y0) continue;

                //Keypoints
                std::array<Keypoint, NUM_KP> kps;
                int kp_base = 4 * REG_MAX + 1;
                for (int k = 0; k < NUM_KP; ++k) {
                    float kx_raw = at(kp_base + k*3 + 0, gy, gx);
                    float ky_raw = at(kp_base + k*3 + 1, gy, gx);
                    float kv_raw = at(kp_base + k*3 + 2, gy, gx);

                    // YOLO11 kp encoding: offset*2 + anchor_centre
                    float nkx = kx_raw * 2.f * stride + ax - stride;
                    float nky = ky_raw * 2.f * stride + ay - stride;
                    float kv  = 1.f / (1.f + std::exp(-kv_raw)); // sigmoid

                    float ox, oy;
                    toOrig(nkx, nky, ox, oy);
                    kps[k] = {ox, oy, kv};
                }

                boxes.push_back(cv::Rect2d(x0, y0, x1-x0, y1-y0));
                scores.push_back(conf);
                kpsList.push_back(kps);
            }
        }
    }

    std::vector<int> nmsIdx;
    cv::dnn::NMSBoxes(boxes, scores, confThr, nmsIou, nmsIdx);

    std::vector<Detection> dets;
    for (int i : nmsIdx) {
        Detection d;
        d.bbox  = boxes[i];
        d.score = scores[i];
        d.kps   = kpsList[i];
        dets.push_back(d);
    }
    return dets;
}

// Visualizer
namespace Viz {
    static cv::Scalar kpColor(int idx) {
        if (idx <= 4)  return {50, 220, 50};   // face  – green
        if (idx <= 8)  return {220, 180, 30};  // arms  – yellow
        if (idx <= 10) return {0, 120, 255};   // wrists– orange
        if (idx <= 12) return {180, 50, 180};  // hips  – purple
        return             {80, 200, 200};     // legs  – cyan
    }

    static void drawSkeleton(cv::Mat& img, const Detection& det,
                             float confThr = 0.3f)
    {
        for (auto& [a,b] : SKELETON) {
            const auto& ka = det.kps[a]; const auto& kb = det.kps[b];
            if (ka.conf < confThr || kb.conf < confThr) continue;
            cv::line(img, {(int)ka.x,(int)ka.y}, {(int)kb.x,(int)kb.y},
                     {200,200,200}, 2, cv::LINE_AA);
        }
        for (int k = 0; k < KP::NUM_KP; ++k) {
            const auto& kp = det.kps[k];
            if (kp.conf < confThr) continue;
            cv::circle(img, {(int)kp.x,(int)kp.y}, 5, kpColor(k), -1, cv::LINE_AA);
            cv::circle(img, {(int)kp.x,(int)kp.y}, 5, {0,0,0},    1, cv::LINE_AA);
        }
    }

    static void drawBbox(cv::Mat& img, const Detection& det,
                         Gesture g, int personIdx)
    {
        cv::Rect r((int)det.bbox.x,(int)det.bbox.y,
                   (int)det.bbox.width,(int)det.bbox.height);
        cv::Scalar col = gestureColor(g);
        cv::rectangle(img, r, col, 2, cv::LINE_AA);

        std::string label = "P" + std::to_string(personIdx) + " "
                          + gestureName(g)
                          + " " + std::to_string((int)(det.score*100)) + "%";
        int base = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &base);
        cv::Rect bg(r.x, r.y - ts.height - 6, ts.width + 6, ts.height + 6);
        bg &= cv::Rect(0,0,img.cols,img.rows);
        cv::rectangle(img, bg, col, -1);
        cv::putText(img, label, {r.x+3, r.y-4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 1, cv::LINE_AA);
    }

    static void drawRefLines(cv::Mat& img, const Detection& det,
                             const GestureAnalyzer& ga)
    {
        int xL = (int)det.bbox.x;
        int xR = (int)(det.bbox.x + det.bbox.width);

        float sy = ga.getShoulderY(det);
        float hy = ga.getHipY(det);

        if (sy > 0)
            cv::line(img, {xL,(int)sy}, {xR,(int)sy}, {0,200,255}, 1, cv::LINE_AA);
        if (hy > 0)
            cv::line(img, {xL,(int)hy}, {xR,(int)hy}, {255,100,0}, 1, cv::LINE_AA);
    }

    static void drawHUD(cv::Mat& img, int fps, int frameIdx) {
        std::string info = "Frame: " + std::to_string(frameIdx)
                         + "  FPS: " + std::to_string(fps);
        cv::putText(img, info, {10, 24},
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, {240,240,10}, 1, cv::LINE_AA);

        // Legend
        int y = img.rows - 60;
        for (auto& [g, name] : std::vector<std::pair<Gesture,std::string>>{
                {Gesture::SUBIR,"SUBIR"},{Gesture::DESCER,"DESCER"},{Gesture::REPOUSO,"REPOUSO"}}) {
            cv::rectangle(img, {10, y-14, 18, 16}, gestureColor(g), -1);
            cv::putText(img, name, {32, y},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, {240,240,240}, 1, cv::LINE_AA);
            y += 22;
        }
    }
}


// Main
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <video_path|0> <model.onnx>\n";
        return 1;
    }

    std::string src   = argv[1];
    std::string model = argv[2];

    //Open video
    cv::VideoCapture cap;
    if (src == "0") cap.open(0);
    else            cap.open(src);
    if (!cap.isOpened()) { std::cerr << "Cannot open video: " << src << "\n"; return 1; }

    int origW = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int origH = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps_src = cap.get(cv::CAP_PROP_FPS);
    if (fps_src <= 0) fps_src = 30;
    std::cout << "[Video] " << origW << "x" << origH << " @ " << fps_src << " fps\n";

    //Load model
    OnnxInferenceBackend backend(model, /*useGpu=*/false);
    int netH = backend.inputH(), netW = backend.inputW();

    //Até 4 pessoas pra não pesar
    constexpr int MAX_PERSONS = 4;
    std::vector<GestureAnalyzer> analyzers(MAX_PERSONS);

    cv::VideoWriter writer;
    {
        std::string outPath = "output_gesture.mp4";
        int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
        writer.open(outPath, fourcc, fps_src, {origW, origH});
        if (!writer.isOpened())
            std::cerr << "[Warn] Could not open output file. Display only.\n";
    }

    cv::Mat frame;
    int frameIdx = 0;
    auto tPrev = std::chrono::steady_clock::now();
    int displayFps = 0;

    while (cap.read(frame)) {
        ++frameIdx;

        //Inference
        auto prep   = preprocess(frame, netH, netW);
        auto result = backend.run(prep.tensor);
        auto dets   = postprocess(result, prep, origH, origW, netH, netW,
                                  /*confThr=*/0.40f, /*nmsIou=*/0.45f);

        //Gesture Analysis 
        cv::Mat vis = frame.clone();

        // Sort detections left-to-right
        std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b){
            return a.bbox.x < b.bbox.x;
        });

        int nPerson = std::min((int)dets.size(), MAX_PERSONS);
        for (int i = 0; i < nPerson; ++i) {
            const auto& det = dets[i];
            Gesture g = analyzers[i].update(det, origH);

            Viz::drawSkeleton(vis, det);
            Viz::drawRefLines(vis, det, analyzers[i]);
            Viz::drawBbox(vis, det, g, i+1);
        }

        //FPS
        auto tNow = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(tNow - tPrev).count();
        tPrev = tNow;
        displayFps = (int)(1.0 / (elapsed + 1e-9));

        Viz::drawHUD(vis, displayFps, frameIdx);

        // Output
        if (writer.isOpened()) writer.write(vis);
        cv::imshow("Gesture Detector", vis);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    std::cout << "[Done] Processed " << frameIdx << " frames.\n";
    return 0;
}
