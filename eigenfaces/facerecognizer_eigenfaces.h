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

/**
 * this class implements eigenfaces.
 *
 * \author Alejandro Hernández
 */

class FaceRecognizer_EigenFaces
{
public:
    /**
      * Constructor
      */
    FaceRecognizer_EigenFaces(int num_components);
    /**
      * Train step
      * \param images vector with the faces
      * \param labels vector with the labels
      */
    void train(vector<Mat> images, std::vector<int>& labels);

    /**
      * predict step
      * \param image the image to recognize
      * @return predicted label
      */
    int predict(Mat image);


    /**
      * projection
      * \param _w
      * \param _mean
      * \param _src
      */
    SGMatrix<float64_t> subspaceProject( SGMatrix<float64_t> _W, SGVector<float64_t> _mean, SGVector<float64_t> _src);

private:
    /**
      * number of main components >0 < features
      */
    int _num_components;
    /**
      * Threshold PCA
      */
    int _threshold;
    /**
      * EigenVector main components
      */
    SGMatrix<float64_t> eigenvectors_mainComponents;
    /**
      * Vector with projections
      */
    std::vector<SGMatrix<float64_t> > _projections;
    /**
      * Vector with the means of the features
      */
    SGVector<float64_t> values_means2;

    /**
      * Labels of each column
      */
    SGVector<float64_t> _labels;


};

#endif // FACERECOGNIZER_EIGENFACES_H
