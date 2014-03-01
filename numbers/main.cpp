#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/labels/Labels.h>
using namespace cv;

int readFlippedInteger(FILE *fp)
{
    int ret = 0;
    unsigned char *temp;

    temp = (unsigned char*)(&ret);
    fread(&temp[3], sizeof(unsigned char), 1, fp);
    fread(&temp[2], sizeof(unsigned char), 1, fp);
    fread(&temp[1], sizeof(unsigned char), 1, fp);
    fread(&temp[0], sizeof(unsigned char), 1, fp);

    return ret;
}

int main()
{

    shogun::init_shogun_with_defaults();

    FILE *fp = fopen("../train-images.idx3-ubyte", "rb");
    FILE *fp2 = fopen("../train-labels.idx1-ubyte", "rb");
    if(!fp || !fp2)
        return 0;

    int magicNumber = readFlippedInteger(fp);
    int numImages = readFlippedInteger(fp);
    int numRows = readFlippedInteger(fp);
    int numCols = readFlippedInteger(fp);

    fseek(fp2, 0x08, SEEK_SET);
    int size = numRows*numCols;
    CvMat *trainingVectors = cvCreateMat(numImages, size, CV_32FC1);
    CvMat *trainingLabels = cvCreateMat(numImages, 1, CV_32FC1);

    // Labels and features containers
    shogun::SGVector<float64_t> lab(numImages);
    shogun::SGMatrix<float64_t> feat(size, numImages);


    unsigned char *temp = new unsigned char[size];
    unsigned char tempClass=0;
    for(int i=0;i<numImages;i++)
    {
        fread((void*)temp, size, 1, fp);

        fread((void*)(&tempClass), sizeof(unsigned char), 1, fp2);

        trainingLabels->data.fl[i] = tempClass;
        lab.set_element(tempClass, i);

        for(int k=0;k<size;k++){
            trainingVectors->data.fl[i*size+k] = temp[k];
            feat(k, i)=temp[k];
        }
    }

//    feat.display_matrix();
    std::cout << "feat: " << feat.num_cols << " " << feat.num_rows << std::endl;

    shogun::CMulticlassLabels* labels = new shogun::CMulticlassLabels(lab);
    shogun::CDenseFeatures<float64_t>* features = new shogun::CDenseFeatures<float64_t>(feat);
    shogun::CKNN* knn_shogun = new shogun::CKNN(1, new shogun::CEuclideanDistance(features, features), labels);
    knn_shogun->train();

    KNearest knn(trainingVectors, trainingLabels);

    printf("Maximum k: %d\n", knn.get_max_k());

    fclose(fp);
    fclose(fp2);

    fp = fopen("../t10k-images.idx3-ubyte", "rb");
    fp2 = fopen("../t10k-labels.idx1-ubyte", "rb");


    magicNumber = readFlippedInteger(fp);
    numImages = readFlippedInteger(fp);
    numRows = readFlippedInteger(fp);
    numCols = readFlippedInteger(fp);

    fseek(fp2, 0x08, SEEK_SET);

    CvMat *testVectors = cvCreateMat(numImages, size, CV_32FC1);
    CvMat *testLabels = cvCreateMat(numImages, 1, CV_32FC1);
    CvMat *actualLabels = cvCreateMat(numImages, 1, CV_32FC1);

    temp = new unsigned char[size];
    tempClass=1;
    CvMat *currentTest = cvCreateMat(1, size, CV_32FC1);
    CvMat *currentLabel = cvCreateMat(1, 1, CV_32FC1);
    shogun::SGMatrix<float64_t> currentTest_shogun(size, 1);

    int totalCorrect=0;
    int totalCorrect_shogun=0;

    numImages = 1000;

    for(int i=0;i<numImages;i++) {
        fread((void*)temp, size, 1, fp);
        fread((void*)(&tempClass), sizeof(unsigned char), 1, fp2);

        actualLabels->data.fl[i] = (float)tempClass;

        for(int k=0;k<size;k++){
              testVectors->data.fl[i*size+k] = temp[k];
              currentTest->data.fl[k] = temp[k];
              currentTest_shogun(k, 0) =temp[k];
        }

        shogun::CDenseFeatures<float64_t>* fe = new shogun::CDenseFeatures<float64_t>(currentTest_shogun);

        knn_shogun->apply_multiclass(fe);
        shogun::SGMatrix<int32_t> multiple_k_output = knn_shogun->classify_for_multiple_k();

        float response = knn.find_nearest(currentTest, 5, currentLabel);
        testLabels->data.fl[i] = currentLabel->data.fl[0];

        if(currentLabel->data.fl[0]==actualLabels->data.fl[i])
            totalCorrect++;
        if(multiple_k_output(0, 0)==actualLabels->data.fl[i])
            totalCorrect_shogun++;
    }


    // Free memory
    SG_UNREF(knn_shogun)

    printf("TotalCorrect: %i\n", totalCorrect);

    printf("Time: %d\nAccuracy: %f\n\n", (int)time, (double)totalCorrect*100/numImages);
    printf("Time: %d\nAccuracy shogun: %f\n\n", (int)time, (double)totalCorrect_shogun*100/numImages);
    shogun::exit_shogun();
    return 0;
}
