#include "facerecognizer_eigenfaces.h"

FaceRecognizer_EigenFaces::FaceRecognizer_EigenFaces(int num_components)
{
    this->_num_components= num_components;
    _threshold= 1e100;

}

// Calculate the euclidean distance between two vector of the same size
double norm(SGMatrix<float64_t> m1, SGMatrix<float64_t> m2){

    double result=0;

    for(int i = 0; i< m1.num_cols;i++){
        result += (m1(0, i)-m2(0, i))*(m1(0, i)-m2(0, i));
    }

    return std::sqrt(result);
}

int FaceRecognizer_EigenFaces::predict(Mat image)
{
    // image as a rows
    SGVector<float64_t> sgMat(image.cols*image.rows);
    cv::Mat src;
    image.convertTo(src,CV_64FC1);
    //resize image -> shogun PCA takes a lot time features>>vectors
    cv::resize(src, src,  cv::Size(25, 25));

    MatConstIterator_<double> it = src.begin<double>();
    for(int i = 0; it != src.end<double>(); it++, i++)
        sgMat[i] = *it;

    SGMatrix<float64_t> q = subspaceProject(eigenvectors_mainComponents, values_means2, sgMat);
    double minDist = DBL_MAX;
    int minClass = -1;

    // Search the closest face
    for(size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
        double dist = norm(_projections[sampleIdx], q);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels[sampleIdx];
        }
    }
    return minClass;

}

void FaceRecognizer_EigenFaces::train(vector<Mat> images, std::vector<int>& labels)
{
    if(images.size()==0)
        return;

    _labels = SGVector<float64_t>(labels.size());
    for(int i = 0; i < labels.size();i++){
        _labels.set_element(labels[i], i);
    }

    //ROW matrix
    SGMatrix<float64_t> sgMat(images[0].cols*images[0].rows, images.size());
    for(int _num_images =0; _num_images  < images.size();  _num_images++ ){
        Mat image = images[_num_images];
        image.convertTo(image,CV_64FC1);

        MatConstIterator_<double> it = image.begin<double>();
        for(int i = 0; it != image.end<double>(); it++, i++)
            sgMat[i+(_num_images*images[0].cols*images[0].rows)] = *it;
    }

    //calculate PCA
    CDenseFeatures<float64_t>* input = new CDenseFeatures<float64_t>(sgMat);
    shogun::CPCA* pca = new shogun::CPCA(false, THRESHOLD);
    pca->init(input);

    SGVector<float64_t> values = pca->get_eigenvalues();
    SGVector<float64_t> eigenvalues(_num_components);
    SGVector<float64_t> eigenvalues_all(values.size());

    SGVector<float64_t> values_means = pca->get_mean();
    values_means2=  SGVector<float64_t>(values.size());

    SGMatrix<float64_t> eigenvectors = pca->get_transformation_matrix();
    eigenvectors_mainComponents =  SGMatrix<float64_t>(_num_components, values.size());

//    for(int i =0 ; i < eigenvectors.num_cols; i++){
//        for(int j =0 ; j < _num_components; j++){
//            std::cout <<i << " " <<  j  <<std::endl;
//            eigenvectors_mainComponents(j, i) = eigenvectors(eigenvectors.num_rows - j -1, i);
//        }
//    }

//    std::cout << eigenvectors.num_cols << " "<< eigenvectors.num_rows  << std::endl;
//    std::cout << eigenvectors_mainComponents.num_cols << " "<< eigenvectors_mainComponents.num_rows  << std::endl;
    for(int i =0 ; i < eigenvectors.num_rows; i++){
        for(int j =0 ; j < _num_components; j++){
            eigenvectors_mainComponents( j, i)= eigenvectors(i, eigenvectors.num_cols - j -1);
        }
    }



    int t= 0;
    for(int i = values.size()-1; i > values.size()-_num_components-1;i--){
        eigenvalues.set_element(values.get_element(i), t) ;
    }

    for(int i = 0; i < values.size();i++){
        values_means2.set_element((float64_t)(values_means.get_element(i)*-1), t) ;
        eigenvalues_all.set_element((float64_t)(values.get_element(i)), t++) ;
    }

    // save projections
    for(int sampleIdx = 0; sampleIdx < input->get_num_vectors(); sampleIdx++) {
        SGVector<float64_t> v = input->get_feature_vector(sampleIdx);
        SGMatrix<float64_t> p = subspaceProject(eigenvectors_mainComponents, values_means2, v);
        _projections.push_back(p);
    }
}

SGMatrix<float64_t> FaceRecognizer_EigenFaces::subspaceProject( SGMatrix<float64_t> _W, SGVector<float64_t> _mean, SGVector<float64_t> _src)
{
    SGVector<float64_t> X(_mean.size());

    shogun::SGVector<float64_t>::add(X, 1, _src, 1, _mean,_mean.size());
    SGMatrix<float64_t> X_matrix(1 , _W.num_cols);

    int indice = 0;
    for(int c = 0; c< X_matrix.num_cols; c++){
        for(int r = 0; r< X_matrix.num_rows; r++){
            X_matrix(r, c) = X.get_element(indice++);
        }
    }
    SGMatrix<float64_t> Y = shogun::SGMatrix<float64_t>::matrix_multiply(X_matrix, _W, false, true);
    return Y;
}
