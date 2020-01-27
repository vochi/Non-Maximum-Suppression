#pragma once
#include <assert.h>
#include <opencv2/opencv.hpp>

/**
 * @brief nms
 * Non maximum suppression
 * @param srcRects
 * @param resRects
 * @param thresh
 * @param neighbors
 */
inline std::pair<std::vector<cv::Rect>, std::vector<size_t>>
nms(const std::vector<cv::Rect>& srcRects, float thresh, int neighbors = 0)
{
    std::vector<cv::Rect> resRects;
    std::vector<size_t> resIndices;

    const size_t size = srcRects.size();
    if (!size) {
        return {resRects, resIndices};
    }

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i) {
        idxs.insert(std::pair<int, size_t>(srcRects[i].br().y, i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0) {
        // grab the last rectangle
        auto lastElem         = --std::end(idxs);
        size_t rectIdx        = lastElem->second;
        const cv::Rect& rect1 = srcRects[rectIdx];

        int neigborsCount = 0;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs);) {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

            float intArea   = (rect1 & rect2).area();
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap   = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh) {
                pos = idxs.erase(pos);
                ++neigborsCount;
            } else {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors) {
            resRects.push_back(rect1);
            resIndices.push_back(rectIdx);
        }
    }

    return {resRects, resIndices};
}

/**
 * @brief nms2
 * Non maximum suppression with detection scores
 * @param srcRects
 * @param scores
 * @param resRects
 * @param thresh
 * @param neighbors
 */
inline std::pair<std::vector<cv::Rect>, std::vector<size_t>> nms2(
    const std::vector<cv::Rect>& srcRects,
    const std::vector<float>& scores,
    float thresh,
    int neighbors      = 0,
    float minScoresSum = 0.f)
{
    std::vector<cv::Rect> resRects;
    std::vector<size_t> resIndices;

    const size_t size = srcRects.size();
    if (!size) {
        return {resRects, resIndices};
    }

    assert(srcRects.size() == scores.size());

    // Sort the bounding boxes by the detection score
    std::multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i) {
        idxs.insert(std::pair<float, size_t>(scores[i], i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0) {
        // grab the last rectangle
        auto lastElem         = --std::end(idxs);
        size_t rectIdx = lastElem->second;
        const cv::Rect& rect1 = srcRects[rectIdx];

        int neigborsCount = 0;
        float scoresSum   = lastElem->first;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs);) {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

            float intArea   = (rect1 & rect2).area();
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap   = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh) {
                scoresSum += pos->first;
                pos = idxs.erase(pos);
                ++neigborsCount;
            } else {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors && scoresSum >= minScoresSum) {
            resRects.push_back(rect1);
            resIndices.push_back(rectIdx);
        }
    }

    return {resRects, resIndices};
}
