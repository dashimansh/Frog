#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "KalmanTracker.h"

enum class ETrackerType
{
    CSRT,
    KCF,
    CAMSHIFT
};

enum class ETrackState
{
    Idle,
    Tracking,
    Occluded,
    Lost
};

class ObjectTracker
{
public:
    ObjectTracker();

    bool Init(
        const cv::Mat& Frame,
        const cv::Rect& BBox,
        ETrackerType Type =
            ETrackerType::CSRT);

    cv::Rect Update(
        const cv::Mat& Frame);

    bool Reinit(
        const cv::Mat& Frame,
        const cv::Rect& NewBBox);

    void Reset();
    void Draw(cv::Mat& Frame);

    ETrackState GetState() const
    {
        return State;
    }
    cv::Rect GetCurrentBox() const
    {
        return CurrentBox;
    }
    cv::Rect GetPredictedBox() const
    {
        return KFilter.GetPredictedBox();
    }
    int GetLostCount() const
    {
        return LostFrameCount;
    }
    bool IsTracking() const
    {
        return State == ETrackState::Tracking;
    }
    bool IsOccluded() const
    {
        return State == ETrackState::Occluded;
    }
    bool IsLost() const
    {
        return State == ETrackState::Lost;
    }

private:
    KalmanTracker KFilter;
    ETrackState   State       = ETrackState::Idle;
    ETrackerType  TrackerType = ETrackerType::CSRT;
    cv::Rect      CurrentBox;
    int           LostFrameCount = 0;

    // Primary CSRT tracker
    cv::Ptr<cv::Tracker> PrimaryTracker;

    // Backup KCF tracker
    cv::Ptr<cv::Tracker> BackupTracker;
    bool bBackupActive = false;

    // CamShift
    cv::Mat  RoiHist;
    cv::Rect TrackWindow;

    // Original template
    cv::Mat ObjTemplate;

    // Color histogram
    cv::Mat ObjHistogram;

    // Background subtractor
    cv::Ptr<cv::BackgroundSubtractorMOG2>
        BGSub;

    // Face detector
    cv::CascadeClassifier FaceDetector;
    bool bFaceDetectorLoaded = false;

    // Scale factors
    std::vector<double> Scales;

    // Internal methods
    bool UpdatePrimary(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool UpdateBackup(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool UpdateCamShift(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool ValidateBox(
        const cv::Mat& Frame,
        const cv::Rect& Box);

    cv::Rect AdaptScale(
        const cv::Mat& Frame,
        const cv::Rect& Box);

    double CompareHistogram(
        const cv::Mat& Frame,
        const cv::Rect& Box);

    // Re-acquisition methods
    bool FindByFace(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool FindByMotion(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool FindByTemplateAndColor(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool TryReacquire(
        const cv::Mat& Frame);

    void ReinitTrackers(
        const cv::Mat& Frame,
        const cv::Rect& Box);

    void LoadFaceDetector();
};
