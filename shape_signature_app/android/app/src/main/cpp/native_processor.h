#ifndef NATIVE_PROCESSOR_H
#define NATIVE_PROCESSOR_H

#include <jni.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define N_DESCRIPTOR_COMPONENTS 13

extern "C" {

JNIEXPORT jfloatArray JNICALL
Java_com_shapesignature_app_MainActivity_nativeProcessImage(
    JNIEnv* env, jobject thiz,
    jintArray pixels, jint width, jint height);

JNIEXPORT jfloatArray JNICALL
Java_com_shapesignature_app_MainActivity_nativeProcessContour(
    JNIEnv* env, jobject thiz,
    jfloatArray xPoints, jfloatArray yPoints);

JNIEXPORT jfloatArray JNICALL
Java_com_shapesignature_app_MainActivity_nativeDetectContour(
    JNIEnv* env, jobject thiz,
    jintArray pixels, jint width, jint height);

}

#endif
