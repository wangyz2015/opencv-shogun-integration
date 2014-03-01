#ifndef FACERECOGNIZER_EIGENFACES_H
#define FACERECOGNIZER_EIGENFACES_H

//Standard C++
#include <vector>
#include <iostream>

//shogun
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Features.h>
#include <shogun/features/DummyFeatures.h>
#include <shogun/preprocessor/PCA.h>

//opencv
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace shogun;
using namespace cv;

class FaceRecognizer_EigenFaces
{
public:
    FaceRecognizer_EigenFaces(int num_components);

    void train(vector<Mat> images, std::vector<int>& labels);
    int predict(Mat image);

    SGMatrix<float64_t> subspaceProject( SGMatrix<float64_t> _W, SGVector<float64_t> _mean, SGVector<float64_t> _src);

private:
    int _num_components;
    int _threshold;
    SGMatrix<float64_t> eigenvectors_mainComponents;
    std::vector<SGMatrix<float64_t> > _projections;
    SGVector<float64_t> values_means2;
    SGVector<float64_t> _labels;


};

#endif // FACERECOGNIZER_EIGENFACES_H
