#include "opencv2/core.hpp"
#include <iostream>
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/videoio.hpp>

#include <chrono>

class Timer {
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start =
      std::chrono::high_resolution_clock::now();
  const char *m_name = "Timer";

public:
  Timer();

  Timer(const char *name);

  ~Timer();
};

//////////////////////////////////////////////

Timer::Timer() { std::cout << "starting timer...\n"; }

Timer::Timer(const char *name) : m_name(name) {
  std::cout << "Starting timer: " << m_name << '\n';
}

Timer::~Timer() {
  std::chrono::duration<float> duration =
      std::chrono::high_resolution_clock::now() - start;
  float ms = duration.count() * 1000.0f;
  std::cout << m_name << " took " << ms << " ms" << std::endl;
}

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
const char *keys = "{ help h |                          | Print help message. }"
                   "{ input1 | /home/afterburner/Downloads/lvlian/object.png"
                   "  | Path to input image 1. }"
                   "{ input2 | ../lvlian.mp4"
                   "| Path to input image 2. }"
                   "{NUM_IMAGES|20|the number of scene images}"
                   "{scale|0.4|the scale to resize scene images}"
                   "{USE_CAMERA|false|whether or not to use camera 0, this will take over images and video}"
                   ;
int main(int argc, char *argv[]) {
  CommandLineParser parser(argc, argv, keys);
  Mat img_object = imread(parser.get<String>("input1"), IMREAD_GRAYSCALE);
  std::string scene_path = parser.get<std::string>("input2");
  const auto USE_CAMERA = parser.get<bool>("USE_CAMERA");
  const bool is_video{scene_path.find("mp4") != std::string::npos && ! USE_CAMERA};
  if (img_object.empty()) {
    cout << "Could not open or find the image!\n" << endl;
    parser.printMessage();
    return -1;
  }
  const auto scale = parser.get<double>("scale");
  imshow("object to search", img_object);
  //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
  const int minHessian = 100;
  Ptr<SURF> detector = SURF::create(minHessian);

  Mat descriptors_object, descriptors_scene;
  Mat image_scene, image_scene_bw, img_scene;
  std::vector<KeyPoint> keypoints_object;

  std::vector<KeyPoint> keypoints_scene;
  std::vector<Point2f> obj, scene;
  std::vector<DMatch> good_matches;
  std::vector<std::vector<DMatch>> knn_matches;

  VideoCapture cap;
  const int max_image = parser.get<int>("NUM_IMAGES");
  std::string base_name;
  std::string format;
  std::vector<std::string> image_paths;

  const int wait_time = is_video||USE_CAMERA ? 1 : 0;

  if (is_video) {
    cap.open(scene_path);
    if (!cap.isOpened()) {
      cout << "Failed to open camera\n";
      return 1;
    }
  } else if(!USE_CAMERA) {
    base_name = scene_path.substr(0, scene_path.find("0."));
    format = scene_path.substr(scene_path.find("0.") + 2);
    for (int i = 0; i < max_image; i++) {
      image_paths.push_back(base_name + std::to_string(i) + "." + format);
      cout << image_paths[i] << '\n';
    }
  }else{
    cap.open(0);
    if (!cap.isOpened()) {
      cout << "Failed to open camera\n";
      return 1;
    }
  }

  detector->detectAndCompute(img_object, noArray(), keypoints_object,
                             descriptors_object);

  int index = 0;
  while (true) {
    Timer t("frame_duration");
    keypoints_scene.clear();
    obj.clear();
    scene.clear();
    good_matches.clear();
    knn_matches.clear();

    if (is_video||USE_CAMERA) {
      cap >> image_scene;
      if(image_scene.empty()){
        break;
      }
      resize(image_scene, image_scene, {}, scale, scale);
    } else {
      if (index == max_image) {
        break;
      }
      image_scene = imread(image_paths[index], IMREAD_COLOR);
      if (image_scene.empty()) {
        cout << "image_" << index << " empty, exit program...\n";
        break;
      }
      resize(image_scene, image_scene, {}, scale, scale);
      index++;
    }
    cvtColor(image_scene, image_scene_bw, COLOR_BGR2GRAY);
    detector->detectAndCompute(image_scene_bw, noArray(), keypoints_scene,
                               descriptors_scene);
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher =
        DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    matcher->knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < knn_matches.size(); i++) {
      if (knn_matches[i][0].distance <
          ratio_thresh * knn_matches[i][1].distance) {
        good_matches.push_back(knn_matches[i][0]);
      }
    }
    //-- Draw matches
    //-- Localize the object

    for (size_t i = 0; i < good_matches.size(); i++) {
      //-- Get the keypoints from the good matches
      obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
      scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }
    Mat H = findHomography(obj, scene, RANSAC);
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f(img_object.cols - 1, 0);
    obj_corners[2] = Point2f(img_object.cols - 1, img_object.rows - 1);
    obj_corners[3] = Point2f(0, img_object.rows - 1);
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform(obj_corners, scene_corners, H);
    //-- Draw lines between the corners (the mapped object in the scene -
    // image_2
    //)

    line(image_scene, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
    line(image_scene, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
    line(image_scene, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
    line(image_scene, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);
    //-- Show detected matches
    imshow("Good Matches & Object detection", image_scene);
    const char key = static_cast<char>(waitKey(wait_time));
    if (key == 'q') {
      break;
    }
  }
  return 0;
}
#else
int main() {
  std::cout
      << "This tutorial code needs the xfeatures2d contrib module to be run."
      << std::endl;
  return 0;
}
#endif