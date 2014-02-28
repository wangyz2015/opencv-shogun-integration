/* Using OpenCV and Shogun Machine Learning
 * gist by Kevin Hughes
 */
 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
 
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
 
#include <iostream>
 
using namespace cv;
using namespace shogun;
 
int main()
{
	init_shogun_with_defaults();
	
	/* OpenCV 2 Shogun
	 * ---------------
	 * To use OpenCV with the Shogun Machine Learning we need to move data
	 * between these 2 libraries. The Shogun classes we'll be interested
	 * in will take CDenseFeatures objects. To create a CDenseFeatures we 
	 * first need an SGMatrix:
	 */
	 
	 CDenseFeatures<float64_t>* input = new CDenseFeatures<float64_t>(X);
	 /* where X is an SGMatrix. */
	 
	 /* Thus the basic task is how to move data between cv::Mat and
	  * shogun::SGMatrix.
	  */
	
	/* OpenCV 2 Shogun using SGMatrix Constructor
	 * ------------------------------------------
	 * We can pass an array of data to the SGMatrix constructor and initialize
	 * the SGMatrix with the data from the cv::Mat object. We need to set the 
	 * ref_counting flag to false otherwise Shogun will attempt to free memory
	 * that is being managed by OpenCV. The cv::Mat.data always returns
	 * unsigned char* regardless of the type, therefore you'll need
	 * to remember to manually cast when working with non unsigned char
	 * types. Also note that using this method the SGMatrix will be transposed
	 * with respect to the cv::Mat. See the two examples below and the 
	 * full OpenCV -> Shogun type conversion chart. 
	 */
	
	Mat cvMat = Mat::eye(3,3,CV_8UC1);
	cvMat.at<unsigned char>(0,1) = 3;
	SGMatrix<uint8_t> sgMat(cvMat.data,3,3,false);
	
	Mat cvMat = Mat::eye(3,3,CV_64FC1);
	cvMat.at<double>(0,1) = 3;
	SGMatrix<float64_t> sgMat((float64_t*)cvMat.data,3,3,false);
	
	// OpenCV   | Shogun
	// CV_8SC1  | int8_t
	// CV_8UC1  | uint8_t
	// CV_16SC1 | int16_t
	// CV_16UC1 | uint16_t 
	// CV_32SC1 | int32_t
	// CV_32UC1 | uint32_t
	// CV_32FC1 | float32_t
	// CV_64FC1 | float64_t
	
	/* Note that the "C1" is optional as this is the default assumption 
	 * eg. CV_8UC1 == CV_8U
	 */ 
	
	/* cv::Mat::convertTo may be usefule in some cases.
	 * for example:
	 */
	Mat cvMat = Mat::eye(3,3,CV_32FC1);
	cvMat.at<float>(0,1) = 3;
	cvMat.convertTo(cvMat,CV_64FC1);
	SGMatrix<float64_t> sgMat((float64_t*)cvMat.data,3,3,false);
	
	std::cout << cvMat << std::endl << std::endl;
	// output:
	// [1, 3, 0;
	//  0, 1, 0;
	//  0, 0, 1]
 
	sgMat.display_matrix();
	// output:
	// matrix=[
	// [	1,	0,	0],
	// [	3,	1,	0],
	// [	0,	0,	1]
	// ]
 
	/* OpenCV 2 Shogun using manual copy
	 * ---------------------------------
	 * Another way to copy from cv::Mat to shogun::SGMatrix is to 
	 * initialize and empty SGMatrix and copy the data manually using
	 * for loops (or iteraters if you like). Note that this method
	 * may or may not transpose the result depending on the exact
	 * technique.
	 */
 
	Mat cvMat = Mat::eye(3,3,CV_64FC1);
	cvMat.at<double>(0,1) = 3;
	SGMatrix<float64_t> sgMat(3,3);
 
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			sgMat(i,j) = cvMat.at<double>(i,j); // not transposed
	
	std::cout << cvMat << std::endl << std::endl;
	// output:
	// [1, 3, 0;
	//  0, 1, 0;
	//  0, 0, 1]
 
	sgMat.display_matrix();
	// output:
	// matrix=[
	// [	1,	3,	0],
	// [	0,	1,	0],
	// [	0,	0,	1]
	// ]
	
	for(int i = 0; i < 9; i++)
		sgMat[i] = ((double*)cvMat.data)[i]; // transposed
	
	/* and */
	
	MatConstIterator_<double> it = cvMat.begin<double>();
	for(int i = 0; it != cvMat.end<double>(); it++, i++)
		sgMat[i] = *it; // transposed
	
	std::cout << cvMat << std::endl << std::endl;
	// output:
	// [1, 3, 0;
	//  0, 1, 0;
	//  0, 0, 1]
 
	sgMat.display_matrix();
	// output:
	// matrix=[
	// [	1,	0,	0],
	// [	3,	1,	0],
	// [	0,	0,	1]
	// ]
	
	/* OpenCV 2 Shogun using memcpy
	 * ----------------------------
	 * using memcpy is another option rather than manually copying.
	 * In this case the result will be transposed.
	 */
	
	Mat cvMat = Mat::eye(3,3,CV_64FC1);
	cvMat.at<double>(0,1) = 3;
	SGMatrix<float64_t> sgMat(3,3);
	
	memcpy(sgMat.matrix,(double*)cvMat.data,3*3*sizeof(double));
	
	std::cout << cvMat << std::endl << std::endl;
	// output:
	// [1, 3, 0;
	//  0, 1, 0;
	//  0, 0, 1]
 
	sgMat.display_matrix();
	// output:
	// matrix=[
	// [	1,	0,	0],
	// [	3,	1,	0],
	// [	0,	0,	1]
	// ]
	
	
	
	/* Shogun 2 OpenCV
	 * ---------------
	 * Usually if the Shogun class modifies or returns data to us it 
	 * will also be a CDenseFeatures object (or a CFeatures object, in
	 * which case cast it first). We can get an SGMatrix using the following:
	 */
	 
	 SGMatrix<float64_t> Y = 
	            ((CDenseFeatures<float64_t>*)output)->get_feature_matrix(); 
	  
	 /* Thus the task is again simplified to moving data between SGMatrix
	  * and cv::Mat.
	  */
	
	/* Shogun 2 OpenCV using cv::Mat constructor
	 * -----------------------------------------
	 * cv::Mat can be initialized with existing data by passing
	 * the size of the matrix and a void* to the array. Note that using this
	 * technique will transpose the results. 
	 */
	
	SGMatrix<float64_t> sgMat; 
	sgMat = SGMatrix<float64_t>::create_identity_matrix(3,1);
	sgMat(0,1) = 3;
	Mat cvMat(3,3,CV_64FC1,(void*)sgMat.matrix);
	
	sgMat.display_matrix();
	// output:
	// matrix=[
	// [	1,	3,	0],
	// [	0,	1,	0],
	// [	0,	0,	1]
	// ]
	
	std::cout << cvMat << std::endl << std::endl;
	// output:
	// [1, 0, 0;
	//  3, 1, 0;
	//  0, 0, 1]
	
	/* Shogun 2 OpenCV using manual copy
	 * ---------------------------------
	 * The opposite of the manual copy methods for OpenCV 2 Shogun.
	 * Depending on the exact method the result may or may not be transposed.
	 */
	
	SGMatrix<float64_t> sgMat; 
	sgMat = SGMatrix<float64_t>::create_identity_matrix(3,1);
	sgMat(0,1) = 3;
	Mat cvMat(3,3,CV_64FC1);
	
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			cvMat.at<double>(i,j) = sgMat(i,j); // not transposed
	
	sgMat.display_matrix();
	// output:
	// matrix=[
	// [	1,	3,	0],
	// [	0,	1,	0],
	// [	0,	0,	1]
	// ]
	
	std::cout << cvMat << std::endl << std::endl;
	// output:
	// [1, 3, 0;
	//  0, 1, 0;
	//  0, 0, 1]
	
	for(int i = 0; i < 9; i++)
		((double*)cvMat.data)[i] = sgMat[i]; // transposed
	
	sgMat.display_matrix();
	// output:
	// matrix=[
	// [	1,	3,	0],
	// [	0,	1,	0],
	// [	0,	0,	1]
	// ]
	
	std::cout << cvMat << std::endl << std::endl;
	// output:
	// [1, 0, 0;
	//  3, 1, 0;
	//  0, 0, 1]
	
	/* Shogun 2 OpenCV using memcpy
	 * ----------------------------
	 * an alternative to performing a manul copy is to use memcpy.
	 * The result will be transposed.
	 */
	
	SGMatrix<float64_t> sgMat; 
	sgMat = SGMatrix<float64_t>::create_identity_matrix(3,1);
	sgMat(0,1) = 3;
	Mat cvMat(3,3,CV_64FC1);
	
	memcpy((double*)cvMat.data,sgMat.matrix,3*3*sizeof(double));
	
	sgMat.display_matrix();
	// output:
	// matrix=[
	// [	1,	3,	0],
	// [	0,	1,	0],
	// [	0,	0,	1]
	// ]
	
	std::cout << cvMat << std::endl << std::endl;
	// output:
	// [1, 0, 0;
	//  3, 1, 0;
	//  0, 0, 1]
	
	
	/* Note for production code you'll obviously want to use
	 * methods for the number of rows, columns and total elements etc.
	 * SGMatrix::num_rows, SGMatrix::num_cols
	 * cv::Mat::rows, cv::Mat::cols, cv::Mat::total()
	 */
	
	/* Note: Shogun CLabels and SGVector
	 * Shogun also has Labels classes and a SGVector class which you may 
	 * need to work with. The procedures in this document should be 
	 * extensible to these classes. I'll refer you to the OpenCV and Shogun
	 * docs to figure this one out.
	 *
	 * http://docs.opencv.org/index.html
	 * http://www.shogun-toolbox.org/doc/en/current/
	 */
	
	exit_shogun();
	
	return 0;
}
