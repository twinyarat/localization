#include<iostream>
#include<vector>
#include<matplot/matplot.h>
#include "preprocess.h"


using namespace std;
using namespace preprocess;
using namespace matplot;
using namespace Eigen;
using namespace kalman;
int main(){

	const double DT = 0.1;
	vector<Control> controls;
	vector<State> groundTruths;
	vector<Measurement> measurements;

	//Preprocess data 
	readStatesFromFile("data/Robot3_Groundtruth.dat", groundTruths);
	readControlsFromFile("data/Robot3_Odometry.dat", controls);
	readMeasuresFromFile("data/Robot3_Measurement.dat", measurements, DT);
	alignTime(groundTruths,controls, measurements, DT);

	//Make barcode-landmarkPosition correspondence
	map<int, int> lmToB = landmarkToBarcode("data/Barcodes.dat");
	map<int, Position> bToPos = barcodeToPosition("data/Landmark_Groundtruth.dat", lmToB);

	//trim trajectories
	const int BEGIN = 400;
	const int END = 3500;
	groundTruths = vector<State>(groundTruths.begin() + BEGIN, groundTruths.begin()+ END);
	controls = vector<Control>(controls.begin() + BEGIN, controls.begin() + END);
	measurements = vector<Measurement>(measurements.begin() + BEGIN,measurements.begin()+ END);


	//Learn control-noise parameters and measure noise covariance from data.
	VectorXd alphas = learnControlNoiseParams(controls, groundTruths, DT);
	MatrixXd Q = learnMeasureCovariance(measurements, groundTruths, bToPos);

	/////////////////////////////////
	//////TEST EKF LOCALIZATION   ///
	/////////////////////////////////
	auto ctrlIter = controls.begin();
	auto measrIter = measurements.begin();

	//1. Initialize state distribution
	VectorXd mean = groundTruths[0].toVector();
	MatrixXd covar(S_DIMENSION, S_DIMENSION);
	covar << 0.1,0,0,
			 0,0.1,0,
			 0,0,0.1;
	StateDistribution gaussian(mean, covar);
	vector<VectorXd> stateEstimates;


	//2. EKF Localize Iterations
	while(ctrlIter != controls.end()){
		gaussian =  EKF_known_correspondence(gaussian, *ctrlIter, *measrIter, bToPos, alphas, Q, DT);
		stateEstimates.push_back(gaussian.mean);
		++ctrlIter;
		++measrIter;
	}
	
	//3. Collect data for plotting ground truth
	vector<double> XPos;
	vector<double> YPos;
	for(auto g: groundTruths){
		XPos.push_back(g.x);
		YPos.push_back(g.y);
	}

	//4. Collect means of EKF distributions for plotting 
	vector<double> XEstimate;
	vector<double> YEstimate;
	for(auto g: stateEstimates){
		XEstimate.push_back(g[0]);
		YEstimate.push_back(g[1]);
	}

	//5. plot result
	auto fig = figure();
	auto ax1 = fig->current_axes();
	plot(XEstimate, YEstimate); 
	hold(on);
	auto g = ax1->scatter(XPos, YPos, 0.5); 
	fig->show();
	return 0;
	
}