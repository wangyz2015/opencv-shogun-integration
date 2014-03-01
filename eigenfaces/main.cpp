
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include "facerecognizer_eigenfaces.h"

#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/preprocessor/PCA.h>

using namespace cv;
using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Read
// Input - filename
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            Mat im = imread(path, 0);
            cv::resize(im, im, cv::Size(25, 25));
            images.push_back(im);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[]) {

    shogun::init_shogun_with_defaults();

    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
        exit(1);
    }
    string output_folder = ".";
    if (argc == 3) {
        output_folder = string(argv[2]);
    }
    // Get the path to your CSV.
    string fn_csv = string(argv[1]);
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;
	// The following lines simply get the last images from
	// your dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();

	// The following lines create an Eigenfaces model for
	// face recognition and train it with the images and
	// labels read from the given CSV file.
	// This here is a full PCA, if you just want to keep
	// 10 principal components (read Eigenfaces), then call
	// the factory method like this:
	//
	//      cv::createEigenFaceRecognizer(10);
	//
	// If you want to create a FaceRecognizer with a
	// confidence threshold (e.g. 123.0), call it with:
	//
	//      cv::createEigenFaceRecognizer(10, 123.0);
	//
	// If you want to use _all_ Eigenfaces and have a threshold,
	// then call the method like this:
	//
	//      cv::createEigenFaceRecognizer(0, 123.0);
	//

	int _num_components = 23;

	FaceRecognizer_EigenFaces* recog = new FaceRecognizer_EigenFaces(_num_components);

	Ptr<FaceRecognizer> model = createEigenFaceRecognizer(_num_components);
	model->train(images, labels);
	recog->train(images, labels);

	// The following line predicts the label of a given
	// test image:
	int predictedLabel = model->predict(testSample);
	int predictedLabelshogun = recog->predict(testSample);

	std::cout << "predictedLabel: " << predictedLabel<< std::endl;
	std::cout << "predictedLabel_shogun: " << predictedLabelshogun<< std::endl;

	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	string result_message_shogun = format("Predicted (Shogun) class = %d / Actual class = %d.", predictedLabelshogun, testLabel);

	shogun::exit_shogun();

	return 0;
}

