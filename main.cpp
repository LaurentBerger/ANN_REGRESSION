#include <opencv2/opencv.hpp> 
#include <opencv2/core/ocl.hpp> 
#include <opencv2/ml/ml.hpp> 
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

float xFunction(float x, float y)
{
    return  2 * x*x - 4 * y*y + 8;
}
float yFunction(float x, float y)
{
    return  sin(x + y) ;
}

int main (int argc,char **argv)
{
    Mat samples(16, 2, CV_32FC1);
    Mat responses(16, 2, CV_32FC1);
// Init data
    int nb=0;
    for (int i=-2;i<2;i++)
        for (int j = -2; j < 2; j++, nb++)
        {
            samples.at<float>(nb, 0) = static_cast<float>(i);
            samples.at<float>(nb, 1) = static_cast<float>(j);
            responses.at<float>(nb, 0) = xFunction(samples.at<float>(nb, 0), samples.at<float>(nb, 1));
            responses.at<float>(nb, 1) = yFunction(samples.at<float>(nb, 0), samples.at<float>(nb, 1));
        }
// Init ANN 4 layers : input 2 neurons (2 var), 2 hidden with 2xvar, and output layer 2 neurons
    Ptr<ml::TrainData> baseDonnees=ml::TrainData::create(samples,ml::ROW_SAMPLE,responses);
    Ptr<ml::ANN_MLP> a = ml::ANN_MLP_ANNEAL::create();
    Mat_<int> layerSizes(1, 4);
    layerSizes(0, 0) = baseDonnees->getNVars();
    layerSizes(0, 1) = 2 * baseDonnees->getNVars();
    layerSizes(0, 2) = 2 * baseDonnees->getNVars();
    layerSizes(0, 3) = baseDonnees->getResponses().cols;
    a->setLayerSizes(layerSizes);
// Activation is SIGMOID_SYM
    a->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
    a->setTrainMethod(ml::ANN_MLP::RPROP, 0.001);
    a->setTermCriteria(TermCriteria(TermCriteria::COUNT, 100000, 0.001));
    a->train(baseDonnees);
    cout <<"Classifier : "<< a->isClassifier()<<"\n";
    double err=0;
    nb=0;
    for (float x = -2; x<=2; x += 0.5)
        for (float y = -2; y <= 2; y += 0.5,nb++)
        {
            Mat res;
            Mat data=(Mat_<float>(1,2)<<x,y);
            float p=a->predict(data, res);
            double e  = norm(res- (Mat_<float>(1, 2) << xFunction(x, y), yFunction(x, y)), NORM_L2SQR);
            err += e;
            std::cout.unsetf(std::ios::floatfield);
            std::cout.precision(4);
            int fieldLength=7;
            cout << setw(fieldLength) <<x<<"\t" << setw(fieldLength) <<y<<"\t" << setw(fieldLength) <<p<< "\t";
            cout << setw(fieldLength) << xFunction(x,y) << "->" << setw(fieldLength) << res.at<float>(0, 0) <<  "\t" ;
            cout << setw(fieldLength) << yFunction(x, y)<< "->" << setw(fieldLength) << res.at<float>(0,1)<<"\t" << setw(fieldLength) <<e<<"\n";
        }
    cout << " MSE = " << sqrt(err / nb);
    return 0.0;
}