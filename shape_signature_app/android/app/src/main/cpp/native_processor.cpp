#include "native_processor.h"

std::vector<double> computeFourierDescriptor(const std::vector<cv::Point>& contour);

cv::Mat pixelsToMat(JNIEnv* env, jintArray pixels, jint width, jint height) {
    jint* pixelData = env->GetIntArrayElements(pixels, nullptr);
    cv::Mat img(height, width, CV_8UC4);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int pixel = pixelData[idx];
            uchar r = (pixel >> 16) & 0xFF;
            uchar g = (pixel >> 8) & 0xFF;
            uchar b = pixel & 0xFF;
            img.at<cv::Vec4b>(y, x) = cv::Vec4b(b, g, r, 255);
        }
    }
    env->ReleaseIntArrayElements(pixels, pixelData, JNI_ABORT);
    return img;
}

std::vector<cv::Point> detectLargestContour(const cv::Mat& img) {
    cv::Mat gray, blurred, binary;
    cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty()) return {};

    auto largest = std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });

    return *largest;
}

std::vector<double> computeShapeSignature(const cv::Mat& img) {
    auto contour = detectLargestContour(img);
    if (contour.empty()) return {};
    return computeFourierDescriptor(contour);
}

std::vector<double> computeFourierDescriptor(const std::vector<cv::Point>& contour) {
    if (contour.size() < 3) return {};

    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) return {};

    double cx = m.m10 / m.m00;
    double cy = m.m01 / m.m00;

    int n = static_cast<int>(contour.size());
    cv::Mat input(1, n, CV_32FC2);
    for (int i = 0; i < n; i++) {
        input.at<cv::Vec2f>(0, i)[0] = static_cast<float>(contour[i].x - cx);
        input.at<cv::Vec2f>(0, i)[1] = static_cast<float>(contour[i].y - cy);
    }

    cv::Mat complexI;
    cv::dft(input, complexI, cv::DFT_COMPLEX_OUTPUT);

    float mag1 = std::sqrt(complexI.at<cv::Vec2f>(0, 1)[0] * complexI.at<cv::Vec2f>(0, 1)[0] +
                            complexI.at<cv::Vec2f>(0, 1)[1] * complexI.at<cv::Vec2f>(0, 1)[1]);
    if (mag1 < 1e-10f) return {};

    const int nFourier = N_DESCRIPTOR_COMPONENTS - 1;
    std::vector<double> descriptor;
    int maxIdx = std::min(n, nFourier);
    for (int i = 1; i <= maxIdx; i++) {
        float re = complexI.at<cv::Vec2f>(0, i)[0];
        float im = complexI.at<cv::Vec2f>(0, i)[1];
        float mag = std::sqrt(re * re + im * im);
        descriptor.push_back(static_cast<double>(mag / mag1));
    }

    while (descriptor.size() < static_cast<size_t>(nFourier)) {
        descriptor.push_back(0.0);
    }

    double area = cv::contourArea(contour);
    cv::Rect rect = cv::boundingRect(contour);
    double compactness = (rect.width * rect.height > 0)
        ? area / static_cast<double>(rect.width * rect.height)
        : 0.0;
    descriptor.push_back(compactness);

    return descriptor;
}

extern "C" {

JNIEXPORT jfloatArray JNICALL
Java_com_shapesignature_app_MainActivity_nativeProcessImage(
    JNIEnv* env, jobject thiz,
    jintArray pixels, jint width, jint height) {

    cv::Mat img = pixelsToMat(env, pixels, width, height);
    std::vector<double> descriptor = computeShapeSignature(img);

    if (descriptor.empty()) return nullptr;

    jfloatArray result = env->NewFloatArray(N_DESCRIPTOR_COMPONENTS);
    if (result == nullptr) return nullptr;

    jfloat temp[N_DESCRIPTOR_COMPONENTS];
    for (int i = 0; i < N_DESCRIPTOR_COMPONENTS; i++) {
        temp[i] = static_cast<jfloat>(descriptor[i]);
    }
    env->SetFloatArrayRegion(result, 0, N_DESCRIPTOR_COMPONENTS, temp);
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_shapesignature_app_MainActivity_nativeProcessContour(
    JNIEnv* env, jobject thiz,
    jfloatArray xPoints, jfloatArray yPoints) {

    jsize len = env->GetArrayLength(xPoints);
    if (len < 3) return nullptr;

    jfloat* xArr = env->GetFloatArrayElements(xPoints, nullptr);
    jfloat* yArr = env->GetFloatArrayElements(yPoints, nullptr);

    std::vector<cv::Point> contour(len);
    for (int i = 0; i < len; i++) {
        contour[i] = cv::Point(static_cast<int>(xArr[i]), static_cast<int>(yArr[i]));
    }

    env->ReleaseFloatArrayElements(xPoints, xArr, JNI_ABORT);
    env->ReleaseFloatArrayElements(yPoints, yArr, JNI_ABORT);

    std::vector<double> descriptor = computeFourierDescriptor(contour);

    if (descriptor.empty()) return nullptr;

    jfloatArray result = env->NewFloatArray(N_DESCRIPTOR_COMPONENTS);
    if (result == nullptr) return nullptr;

    jfloat temp[N_DESCRIPTOR_COMPONENTS];
    for (int i = 0; i < N_DESCRIPTOR_COMPONENTS; i++) {
        temp[i] = static_cast<jfloat>(descriptor[i]);
    }
    env->SetFloatArrayRegion(result, 0, N_DESCRIPTOR_COMPONENTS, temp);
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_shapesignature_app_MainActivity_nativeDetectContour(
    JNIEnv* env, jobject thiz,
    jintArray pixels, jint width, jint height) {

    cv::Mat img = pixelsToMat(env, pixels, width, height);
    auto contour = detectLargestContour(img);

    if (contour.empty()) return nullptr;

    jsize nPts = static_cast<jsize>(contour.size());
    jsize totalLen = nPts * 2;
    jfloatArray result = env->NewFloatArray(totalLen);
    if (result == nullptr) return nullptr;

    // Reserve en el heap para evitar stack overflow con contornos grandes
    std::vector<jfloat> buffer(totalLen);
    for (int i = 0; i < nPts; i++) {
        buffer[i * 2] = static_cast<jfloat>(contour[i].x);
        buffer[i * 2 + 1] = static_cast<jfloat>(contour[i].y);
    }
    env->SetFloatArrayRegion(result, 0, totalLen, buffer.data());
    return result;
}

}
