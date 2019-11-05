
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

// Default command line args
//vector<String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features_type = "surf";
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = false;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
string warp_type = "spherical";
float match_conf = 0.3f;
int range_width = -1;

int main(int argc, char* argv[])
{
#if ENABLE_LOG
	int64 app_start_time = getTickCount();
#endif

	vector<cv::String> videoNames{"D:/video/212.flv","D:/video/211.flv"};
	int num_images = static_cast<int>(videoNames.size());
	vector<UMat> src(num_images);
	vector<VideoCapture> captures(num_images);
	for (int i = 0; i < num_images; i++) {
		captures[i] = VideoCapture(videoNames[i]);
	}
	cv::UMat frame;
	for (int j = 0; j < num_images; j++)
	{
		if (!captures[j].read(frame))
			return -1;
		frame.copyTo(src[j]);
	}
	// Check if have enough images
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return -1;
	}

	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

	LOGLN("Finding features...");
#if ENABLE_LOG
	int64 t = getTickCount();
#endif

	Ptr<FeaturesFinder> finder;
	if (features_type == "surf")
	{
		finder = makePtr<SurfFeaturesFinder>();
	}
	else if (features_type == "orb")
	{
		finder = makePtr<OrbFeaturesFinder>();
	}
	else
	{
		cout << "Unknown 2D features type: '" << features_type << "'.\n";
		return -1;
	}

	UMat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<UMat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;

	for (int i = 0; i < num_images; ++i)
	{

		full_img = src[i];
		full_img_sizes[i] = full_img.size();

		if (full_img.empty())
		{
			LOGLN("Can't open image " << videoNames[i]);
			return -1;
		}
		if (work_megapix < 0)
		{
			img = full_img;
			work_scale = 1;
			is_work_scale_set = true;
		}
		else
		{
			if (!is_work_scale_set)
			{
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
				is_work_scale_set = true;
			}
			cv::resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
		}
		if (!is_seam_scale_set)
		{
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}

		(*finder)(img, features[i]);
		features[i].img_idx = i;
		LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

		cv::resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
		images[i] = img.clone();
	}

	finder->collectGarbage();
	full_img.release();
	img.release();

	LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	LOG("Pairwise matching");
#if ENABLE_LOG
	t = getTickCount();
#endif
	vector<MatchesInfo> pairwise_matches;
	Ptr<FeaturesMatcher> matcher;
	if (matcher_type == "affine")
		matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
	else if (range_width == -1)
		matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
	else
		matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();
	
	LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	// Leave only images we are sure are from the same panorama
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector<UMat> img_subset;
	vector<String> img_names_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		img_names_subset.push_back(videoNames[indices[i]]);
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	videoNames = img_names_subset;
	full_img_sizes = full_img_sizes_subset;

	// Check if we still have enough images
	num_images = static_cast<int>(videoNames.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return -1;
	}

	Ptr<Estimator> estimator;
	if (estimator_type == "affine")
		estimator = makePtr<AffineBasedEstimator>();
	else
		estimator = makePtr<HomographyBasedEstimator>();

	vector<CameraParams> cameras;
	if (!(*estimator)(features, pairwise_matches, cameras))
	{
		cout << "Homography estimation failed.\n";
		return -1;
	}

	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
	}

	Ptr<detail::BundleAdjusterBase> adjuster;
	if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
	else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
	else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
	else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
	else
	{
		cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
		return -1;
	}
	adjuster->setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);
	if (!(*adjuster)(features, pairwise_matches, cameras))
	{
		cout << "Camera parameters adjusting failed.\n";
		return -1;
	}

	// Find median focal length

	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		LOGLN("Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
		focals.push_back(cameras[i].focal);
	}

	sort(focals.begin(), focals.end());
	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	if (do_wave_correct)
	{
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R.clone());
		waveCorrect(rmats, wave_correct);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}

	LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
	t = getTickCount();
#endif

	vector<Point> corners(num_images);
	vector<UMat> masks_warped(num_images);
	vector<UMat> images_warped(num_images);
	vector<Size> sizes(num_images);
	vector<UMat> masks(num_images);

	// Prepare images masks
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks

	Ptr<WarperCreator> warper_creator;
	if (warp_type == "plane")
		warper_creator = makePtr<cv::PlaneWarper>();
	else if (warp_type == "affine")
		warper_creator = makePtr<cv::AffineWarper>();
	else if (warp_type == "cylindrical")
		warper_creator = makePtr<cv::CylindricalWarper>();
	else if (warp_type == "spherical")
		warper_creator = makePtr<cv::SphericalWarper>();
	else if (warp_type == "fisheye")
		warper_creator = makePtr<cv::FisheyeWarper>();
	else if (warp_type == "stereographic")
		warper_creator = makePtr<cv::StereographicWarper>();
	else if (warp_type == "compressedPlaneA2B1")
		warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
	else if (warp_type == "compressedPlaneA1.5B1")
		warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
	else if (warp_type == "compressedPlanePortraitA2B1")
		warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
	else if (warp_type == "compressedPlanePortraitA1.5B1")
		warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
	else if (warp_type == "paniniA2B1")
		warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
	else if (warp_type == "paniniA1.5B1")
		warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
	else if (warp_type == "paniniPortraitA2B1")
		warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
	else if (warp_type == "paniniPortraitA1.5B1")
		warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
	else if (warp_type == "mercator")
		warper_creator = makePtr<cv::MercatorWarper>();
	else if (warp_type == "transverseMercator")
		warper_creator = makePtr<cv::TransverseMercatorWarper>();

	if (!warper_creator)
	{
		cout << "Can't create the following warper '" << warp_type << "'\n";
		return 1;
	}

	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();
	}

	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	// Release unused memory
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();
	img.release();

	LOGLN("Compositing...");
#if ENABLE_LOG
	
#endif
	//double compose_seam_aspect = 1;
	double compose_work_aspect = 1;
	bool firstFlag = true;
	vector<UMat> vecK;
	vector<cv::UMat> vecWarpedImg;
	// Compute relative scales
	compose_work_aspect = compose_scale / work_scale;

	// Update warped image scale
	warped_image_scale *= static_cast<float>(compose_work_aspect);
	warper = warper_creator->create(warped_image_scale);

	// Update corners and sizes
	for (int i = 0; i < num_images; ++i)
	{
		// Update intrinsics
		cameras[i].focal *= compose_work_aspect;
		cameras[i].ppx *= compose_work_aspect;
		cameras[i].ppy *= compose_work_aspect;

		// Update corner and size
		Size sz = full_img_sizes[i];
		cv::UMat kt;
		cameras[i].K().convertTo(kt, CV_32F);
		vecK.push_back(kt);
		Rect roi = warper->warpRoi(sz, kt, cameras[i].R);
		corners[i] = roi.tl();
		sizes[i] = roi.size();
	}

	while (1) {
		t = getTickCount();
		for (int img_idx = 0; img_idx < num_images; ++img_idx)
		{
			//LOGLN("Compositing image #" << indices[img_idx] + 1);

			// Read image and resize it if necessary
			full_img = src[img_idx];
			UMat img_warped, img_warped_s;
			double t1 = getTickCount();
			// Warp the current image
			warper->warp(full_img, vecK[img_idx], cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
			//LOGLN("warp, time: " << ((getTickCount() - t1) / getTickFrequency()) << " sec");
			// Compensate exposure
			//img_warped.convertTo(img_warped_s, CV_16S);
			img_warped.convertTo(img_warped_s, CV_8UC3);
			img_warped.release();
			full_img.release();
			vecWarpedImg.push_back(img_warped_s);
		}
		UMat result;
		
		int y_dv, x_dv;
		bool first_left = true, first_top = true;
		/*vector<float> gains;
		cv::UMat temp1;
		cv::UMat temp2;*/

		if (corners[0].y > corners[1].y) {
			y_dv = corners[0].y - corners[1].y;
			first_top = false;
		}
		else {
			y_dv = corners[1].y - corners[0].y;
			first_top = true;
		}

		if (corners[0].x > corners[1].x) {
			x_dv = corners[1].x + sizes[1].width - corners[0].x;
			vecWarpedImg[1] = vecWarpedImg[1](cv::Rect(0, 0, vecWarpedImg[1].cols - x_dv / 2, vecWarpedImg[1].rows));
			vecWarpedImg[0] = vecWarpedImg[0](cv::Rect(x_dv/2, 0, vecWarpedImg[0].cols - x_dv / 2, vecWarpedImg[0].rows));
			first_left = false;
		}
		else {
			x_dv = corners[0].x + sizes[0].width - corners[1].x;
			/*temp1 = vecWarpedImg[0](cv::Rect(sizes[0].width - x_dv, corners[1].y - corners[0].y + sizes[0].height / 3, x_dv, sizes[0].height / 3));
			temp2 = vecWarpedImg[1](cv::Rect(0, sizes[0].height / 3, x_dv, sizes[0].height / 3));
			cvtColor(temp1, temp1, CV_BGR2YUV);
			cvtColor(temp2, temp2, CV_BGR2YUV);
			Scalar tmp1Mean = mean(temp1);
			Scalar tmp2Mean = mean(temp2);
			gains.push_back(tmp1Mean.val[0] / tmp2Mean.val[0]);
			gains.push_back(tmp1Mean.val[1] / tmp2Mean.val[1]);
			gains.push_back(tmp1Mean.val[2] / tmp2Mean.val[2]);*/
			vecWarpedImg[0] = vecWarpedImg[0](cv::Rect(0, 0, vecWarpedImg[0].cols - x_dv / 2, vecWarpedImg[0].rows));
			vecWarpedImg[1] = vecWarpedImg[1](cv::Rect(x_dv / 2, 0, vecWarpedImg[1].cols - x_dv / 2, vecWarpedImg[1].rows));
			first_left = true;
		}
		
		if (vecWarpedImg[0].rows > vecWarpedImg[1].rows) {
			int h = vecWarpedImg[0].rows - vecWarpedImg[1].rows;
			int w = vecWarpedImg[0].cols - vecWarpedImg[1].cols;
			
			if (first_top) {
				copyMakeBorder(vecWarpedImg[1], vecWarpedImg[1], y_dv + h, 0, 0, 0, BORDER_CONSTANT);
				copyMakeBorder(vecWarpedImg[0], vecWarpedImg[0], 0, y_dv, 0, 0, BORDER_CONSTANT);
			}
			else {
				copyMakeBorder(vecWarpedImg[1], vecWarpedImg[1], 0, y_dv + h, 0, 0, BORDER_CONSTANT);
				copyMakeBorder(vecWarpedImg[0], vecWarpedImg[0], y_dv, 0, 0, 0, BORDER_CONSTANT);
			}
		}
		else {
			int h = vecWarpedImg[1].rows - vecWarpedImg[0].rows;
			int w = vecWarpedImg[1].cols - vecWarpedImg[0].cols;
			
			if (first_top) {
				copyMakeBorder(vecWarpedImg[1], vecWarpedImg[1], y_dv, 0, 0, 0, BORDER_CONSTANT);
				copyMakeBorder(vecWarpedImg[0], vecWarpedImg[0], 0, y_dv + h, 0, 0, BORDER_CONSTANT);
			}
			else {
				copyMakeBorder(vecWarpedImg[1], vecWarpedImg[1], 0, y_dv, 0, 0, BORDER_CONSTANT);
				copyMakeBorder(vecWarpedImg[0], vecWarpedImg[0], y_dv + h, 0, 0, 0, BORDER_CONSTANT);
			}
		}
		/*cv::UMat yuvImg;
		cvtColor(vecWarpedImg[1], yuvImg, CV_BGR2YUV);
		multiply(yuvImg, Scalar(gains[0], gains[1], gains[2]), yuvImg);
		cvtColor(yuvImg, vecWarpedImg[1], CV_YUV2BGR);*/
		if (first_left) {
			cv::hconcat(vecWarpedImg[0], vecWarpedImg[1], result);
		}
		else {
			cv::hconcat(vecWarpedImg[1], vecWarpedImg[0], result);
		}
		
		vecWarpedImg.clear();
		LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

		
		cv::namedWindow("result123", WINDOW_NORMAL);
		cv::imshow("result123", result);
		cv::waitKey(1);
		for (int j = 0; j < num_images; j++)
		{
			if (!captures[j].read(frame))
				return -1;
			frame.copyTo(src[j]);
		}
	}
	LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
	return 0;
}
