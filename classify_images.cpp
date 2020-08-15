#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iostream>
#include <typeinfo>
#include <stdio.h>
#include <algorithm>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace boost::filesystem;

int find_pos(vector<int> v, int x){
	for(int i=0; i<v.size(); i++){
		if(v[i]==x) return i;
	}
	return -1;
}

int find_max(vector<int> v){
	int maxval=v[0];
	int maxval_pos=0;
	for(int i=1; i<v.size(); i++){
		if(v[i]>maxval){
			maxval=v[i];
			maxval_pos=i;
		}
	}
	return maxval;
}

bool check_overlap(vector<Point2f> corners, int a, int b){ // Returns true if two rectangles overlap
	Point2f l1=corners[(4*a)+0];
	Point2f r1=corners[(4*a)+2];
	Point2f l2=corners[(4*b)+0];
	Point2f r2=corners[(4*b)+2];
	// If one rectangle is on left side of other
    if (l1.x >= r2.x || l2.x >= r1.x)
        return false;
    // If one rectangle is above other
    if (l1.y >= r2.y || l2.y >= r1.y)
        return false;
	return true;
}

vector<int> sort_descending(vector<int> v){
	vector<int> order;
	for(int i=0; i<v.size(); i++){
		int max_pos = find_pos(v,find_max(v));
		order.push_back(max_pos);
		v[max_pos]=-1;
	}
	return order;
}

int main(int argc, char** argv  )
{
	path objDir(argv[1]);
	directory_iterator end_itr_obj;
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create( minHessian );
	Ptr<SURF> extractor = SURF::create();

	// Get training images of the object
	vector<Mat> img_objects;
	for(directory_iterator itr_obj(objDir); itr_obj!=end_itr_obj; ++itr_obj){
		if(is_regular_file(itr_obj->path())){
			Mat image_obj = imread(itr_obj->path().string(), CV_LOAD_IMAGE_GRAYSCALE);
			if( !image_obj.data ){
				cout<< " --(!) Error reading image1 " << endl;
				return -1;
			}
			img_objects.push_back(image_obj);
		}
	}

	path targetDir(argv[2]);
	directory_iterator end_itr;
	vector<string> filename;
	for(directory_iterator itr(targetDir); itr!=end_itr; ++itr){
		if(is_regular_file(itr->path())) filename.push_back(itr->path().string());
	}
	sort(filename.begin(),filename.end());

	for(int i=0; i<filename.size(); i++){
		Mat image_scene_RGB = imread(filename[i]); // Get scene
		Mat image_scene;
		cvtColor(image_scene_RGB,image_scene, CV_BGR2GRAY);
		if( !image_scene.data){
			cout<< " --(!) Error reading image2 " << endl;
			return -1;
		}
		vector<KeyPoint> keypoints_scene;
		detector->detect( image_scene, keypoints_scene ); // Detect keypoints of scene
		Mat descriptors_scene; // Calculate scene descriptors
		extractor->compute( image_scene, keypoints_scene, descriptors_scene );
		vector<Point2f> matched_corners;
		vector <int> matched_score;
		// Matching descriptor vectors using FLANN matcher
		for(int i = 0; i < img_objects.size(); i++){// Matching for each object
			vector<KeyPoint> keypoints_obj;
			detector->detect( img_objects[i], keypoints_obj ); // Detect keypoints of img_objects
			Mat descriptors_obj;
			extractor->compute( img_objects[i], keypoints_obj, descriptors_obj ); // Calculate object descriptors
			FlannBasedMatcher matcher;
			vector<DMatch> matches;
			matcher.match( descriptors_obj, descriptors_scene, matches );
			Mat img_matches;
			drawMatches( img_objects[i], keypoints_obj, image_scene, keypoints_scene,
				matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
			//-- Step 4: Localize the object
			vector<Point2f> obj;
			vector<Point2f> scene;
			for( int i = 0; i < matches.size(); i++ )
			{//-- Step 5: Get the keypoints from the  matches
				obj.push_back( keypoints_obj [matches[i].queryIdx ].pt );
				scene.push_back( keypoints_scene[ matches[i].trainIdx ].pt );
			}
			//-- Step 6:FindHomography
			Mat H = findHomography( obj, scene, CV_RANSAC );
			//-- Step 7: Get the corners of the object which needs to be detected.
			vector<Point2f> obj_corners(4);
			obj_corners[0] = cvPoint(0,0);
			obj_corners[1] = cvPoint( img_objects[i].cols, 0 );
			obj_corners[2] = cvPoint( img_objects[i].cols, img_objects[i].rows );
			obj_corners[3] = cvPoint( 0, img_objects[i].rows );
			//-- Step 8: Get the corners of the object form the scene(background image)
			vector<Point2f> scene_corners(4);
			//-- Step 9:Get the perspectiveTransform
			perspectiveTransform( obj_corners, scene_corners, H);
			//-- Rectify edges to largest possible rectangle
			if(scene_corners[0].y > scene_corners[1].y) scene_corners[0].y = scene_corners[1].y; //top (min)
			else scene_corners[1].y = scene_corners[0].y;
			if(scene_corners[0].x > scene_corners[3].x) scene_corners[0].x = scene_corners[3].x; //left (min)
			else scene_corners[3].x = scene_corners[0].x;
			if(scene_corners[1].x > scene_corners[2].x) scene_corners[2].x = scene_corners[1].x; //right (max)
			else scene_corners[1].x = scene_corners[2].x;
			if(scene_corners[2].y > scene_corners[3].y) scene_corners[3].y = scene_corners[2].y; //bottom (max)
			else scene_corners[2].y = scene_corners[3].y;
			//-- Ignore excessive small/big rectangle
			if(	scene_corners[0].x < 0 || scene_corners[0].y < 0 || scene_corners[1].x < 0 || scene_corners[1].y < 0||
					scene_corners[2].x < 0 || scene_corners[2].y < 0 || scene_corners[3].x < 0 || scene_corners[3].y < 0||
					abs(scene_corners[0].x-scene_corners[1].x)<100 ||
					abs(scene_corners[0].x-scene_corners[1].x)>400 ||
					abs(scene_corners[3].y-scene_corners[0].y)<50 ||
					abs(scene_corners[3].y-scene_corners[0].y)>200){
			}
			else{
				int matched_keypoints = 0; // count keypoints within the rectangle
				for( int i = 0; i < scene.size(); i++ ){
					if (scene[i].x > scene_corners[0].x && scene[i].x < scene_corners[1].x &&
							scene[i].y > scene_corners[0].y && scene[i].y < scene_corners[3].y) matched_keypoints++;
				}
				int score = (matched_keypoints*100) /scene.size();
				if(score > 20){
					for(int i=0; i<scene_corners.size(); i++){
						matched_corners.push_back(scene_corners[i]);
					}
					matched_score.push_back(score);
				}
			}
		}//end for loop of each object

		// If detected boxes overlap each other, keep the one with higher score
		int nObject=0;
		if(!matched_score.empty()){
			if(matched_score.size() == 1){// there's only one detected box
				line( image_scene_RGB, matched_corners[0], matched_corners[1], Scalar(0, 255, 0), 4 );
				line( image_scene_RGB, matched_corners[1], matched_corners[2], Scalar( 0, 255, 0), 4 );
				line( image_scene_RGB, matched_corners[2], matched_corners[3], Scalar( 0, 255, 0), 4 );
				line( image_scene_RGB, matched_corners[3], matched_corners[0], Scalar( 0, 255, 0), 4 );
				nObject++;
			}
			else{
				vector<int> order = sort_descending(matched_score);
				for(int i=0; i<order.size(); i++){
					int temp;
					if(order[i]>=0){
						line( image_scene_RGB, matched_corners[(order[i]*4)+0], matched_corners[(order[i]*4)+1], Scalar(0, 255, 0), 4);
						line( image_scene_RGB, matched_corners[(order[i]*4)+1], matched_corners[(order[i]*4)+2], Scalar(0, 255, 0), 4);
						line( image_scene_RGB, matched_corners[(order[i]*4)+2], matched_corners[(order[i]*4)+3], Scalar(0, 255, 0), 4);
						line( image_scene_RGB, matched_corners[(order[i]*4)+3], matched_corners[(order[i]*4)+0], Scalar(0, 255, 0), 4);
						temp=order[i];
						order[i]=-1;
						nObject++;
					}
					if(find_max(order)>=0){
						for(int j=i+1; j<order.size(); j++){
							if(check_overlap(matched_corners,temp,order[j])) order[j]=-1;
						}
					}
				}
			}
			cout << filename[i] << "," << nObject;
			for(int i=0; i<nObject; i++)
				cout << "," << int(matched_corners[(4*i)+0].y) << ","<< int(matched_corners[(4*i)+0].x) << "," << int(matched_corners[(4*i)+2].x-matched_corners[(4*i)+0].x) << ","<< int(matched_corners[(4*i)+2].y-matched_corners[(4*i)+0].y) ;
			cout << ""<<endl;
			// imshow("DetectedImage", image_scene_RGB);
			// waitKey(0);
		}
		else{
			cout<< filename[i] << "," << nObject << endl;
		}
	}
  return 0;
}
