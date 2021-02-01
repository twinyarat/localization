#include<iostream>
#include<fstream>
#include<vector>
#include<matplot/matplot.h>
#include <random>
#include "preprocess.h"
#include "particleFilter.h"
#include "kalman.h"
#include<sstream>
using namespace preprocess;
using namespace matplot;
using namespace Eigen;
using namespace kalman;




void testEKF(){

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
	const int END = 2500;
	groundTruths = vector<State>(groundTruths.begin() + BEGIN, groundTruths.begin()+ END);
	controls = vector<Control>(controls.begin() + BEGIN, controls.begin() + END);
	measurements = vector<Measurement>(measurements.begin() + BEGIN,measurements.begin()+ END);


	//Learn control-noise parameters and measure noise covariance from data.
	VectorXd alphas = learnControlNoiseParams(controls, groundTruths, DT);
	MatrixXd Q = learnMeasureNoiseCovariance(measurements, groundTruths, bToPos);

	/////////////////////////////////
	//////TEST EKF LOCALIZATION   ///
	/////////////////////////////////
	auto ctrlIter = controls.begin();
	auto measrIter = measurements.begin();

	//1. Initialize state distribution
	VectorXd mean = groundTruths[0].toVector();
	MatrixXd covar(S_DIMENSION, S_DIMENSION);
	covar << 0.2,0,0,
			 0,0.2,0,
			 0,0,0.2;
	NormalDistribution gaussian(mean, covar);
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

}

void testUKF(){
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
	const int END = 2500;
	groundTruths = vector<State>(groundTruths.begin() + BEGIN, groundTruths.begin()+ END);
	controls = vector<Control>(controls.begin() + BEGIN, controls.begin() + END);
	measurements = vector<Measurement>(measurements.begin() + BEGIN,measurements.begin()+ END);

	
	VectorXd mean = groundTruths[0].toVector();
	MatrixXd covar(S_DIMENSION, S_DIMENSION);
	covar << 0.2,0,0,
			 0,0.2,0,
			 0,0,0.2;

	NormalDistribution gaussian(mean, covar);
	auto Q = learnMeasureNoiseCovariance(measurements, groundTruths, bToPos);
	auto alphas = learnControlNoiseParams(controls, groundTruths, DT);
	auto ctrlIter = controls.begin();
	auto measrIter = measurements.begin();


	vector<double> XEstimate;
	vector<double> YEstimate;

	// 2. UKF Localize Iterations
	while(ctrlIter != controls.end()){

		gaussian = UKF_known_correspondence(gaussian, *ctrlIter, *measrIter, bToPos, alphas, Q, DT, 0.8);
		mean = gaussian.mean;
		XEstimate.push_back(mean[0]);
		YEstimate.push_back(mean[1]);


		++ctrlIter;
		++measrIter;

	}
	
	// 3. Collect data for plotting ground truth
	vector<double> XPos;
	vector<double> YPos;
	for(auto g: groundTruths){
		XPos.push_back(g.x);
		YPos.push_back(g.y);
	}

	//4. plot result
	auto fig = figure();
	auto ax1 = fig->current_axes();
	plot(XEstimate, YEstimate); 
	hold(on);
	auto g = ax1->scatter(XPos, YPos, 0.5); 
	fig->show();


}

void saveParticlesToFile(vector<Particle> parts, string fname){
	ofstream outfile;
	outfile.open(fname);
	if(outfile){
		for(auto p: parts){
			for(int i = 0; i < p.vec.size(); ++i){
				outfile << p.vec[i] << '\t';
			}
			outfile << '\n';
		}		
		outfile.close();
	}
}

vector<Particle> readParticlesFromFile(string fname){
	string line;
	vector<Particle> out;
	ifstream infile(fname);
	if(infile.is_open()){
		while(getline(infile,line)){
	 		istringstream ss(line);
	 		VectorXd vec(S_DIMENSION);
	 		double weight = 1.0;
	 		double x,y,rad;
			ss >> x >> y >> rad;
			vec << x,y,rad;
			Particle freshParticle(vec,weight);
			out.push_back(freshParticle);
	 	}
		infile.close();
	}

	else{
		cout << "Unable to open file \n"; 
	}

	return out;
}	


int main(){
	testUKF();

	// auto parts = readParticlesFromFile("particles.dat");
	// vector<double> XPos;
	// vector<double> YPos;
	// for(auto p: parts){
	// 	XPos.push_back(p.vec[0]);
	// 	YPos.push_back(p.vec[1]);
	// }
	// auto fig = figure();
	// auto ax1 = fig->current_axes();
	// auto g = ax1->scatter(XPos, YPos, 0.5); 
	// show();

	// const double DT = 0.1;
	// vector<Control> controls;
	// vector<State> groundTruths;
	// vector<Measurement> measurements;

	// //Preprocess data 
	// readStatesFromFile("data/Robot3_Groundtruth.dat", groundTruths);
	// readControlsFromFile("data/Robot3_Odometry.dat", controls);
	// readMeasuresFromFile("data/Robot3_Measurement.dat", measurements, DT);
	// alignTime(groundTruths,controls, measurements, DT);

	// //Make barcode-landmarkPosition correspondence
	// map<int, int> lmToB = landmarkToBarcode("data/Barcodes.dat");
	// map<int, Position> bToPos = barcodeToPosition("data/Landmark_Groundtruth.dat", lmToB);

	// //trim trajectories
	// const int BEGIN = 400;
	// const int END = 2500;
	// groundTruths = vector<State>(groundTruths.begin() + BEGIN, groundTruths.begin()+ END);
	// controls = vector<Control>(controls.begin() + BEGIN, controls.begin() + END);
	// measurements = vector<Measurement>(measurements.begin() + BEGIN,measurements.begin()+ END);

	// VectorXd mean = groundTruths[0].toVector();
	// auto Q = learnMeasureNoiseCovariance(measurements, groundTruths, bToPos);
	// auto alphas = learnControlNoiseParams(controls, groundTruths, DT);
	// auto ctrlIter = controls.begin();
	// auto measrIter = measurements.begin();

	// VectorXd c(C_DIMENSION); 


	// vector<vector<double>> Xhistory;
	// vector<vector<double>> Yhistory;
	// vector<double> Xmean;
	// vector<double> Ymean;
	// vector<token::Particle> newsamples;

	// vector<token::Particle> particles = particleFilter::uniformlyDistributeParticles(mean[0] -2, mean[0] +2,mean[1] -2,mean[1]+2,mean[2]-0.2, mean[2]+0.2);

	// for(int i = 0; i < 1000; ++i){
	// 	c =(*ctrlIter).toVector();
	// 	newsamples.clear();
	// 	for(auto& p: particles){
	// 		auto pos = particleFilter::sampleFromMotion(p.vec,c, alphas, DT);
	// 		auto weight = particleFilter::sampleFromMeasurement(pos, *measrIter, Q, bToPos);
	// 		// double weight = 1.0;	
	// 		newsamples.push_back(token::Particle(pos, weight));
	// 	}
	// 	//normalize their weights obtained from measurement model
	// 	particleFilter::normalizeWeights(newsamples);
	// 	particles = particleFilter::lowVarianceSampling(newsamples);


	// 	if(i == 200){
	// 		//save particles to file for cluster testing
	// 		saveParticlesToFile(particles, "particles.dat");
	// 	}
	// 	// particles = newsamples;

	// 	vector<double> XPos;
	// 	vector<double> YPos;

	// 	// for(auto p: particles){
	// 	// 	XPos.push_back(p.vec[0]);
	// 	// 	YPos.push_back(p.vec[1]);
	// 	// }
	// 	// Xhistory.push_back(XPos);
	// 	// Yhistory.push_back(YPos);

	// 	auto distribution = particleFilter::distributionFromStatistics(particles);
	// 	Xmean.push_back(distribution.mean[0]);
	// 	Ymean.push_back(distribution.mean[1]);

	// 	++ctrlIter;
	// 	++measrIter;
	// }


	
	// // 3. Collect data for plotting ground truth
	// vector<double> XPos;
	// vector<double> YPos;
	// for(auto g: groundTruths){
	// 	XPos.push_back(g.x);
	// 	YPos.push_back(g.y);
	// }
	
	// // for(int i = 0; i < Xhistory.size(); ++i){
	// 	auto fig = figure();
	// 	auto ax1 = fig->current_axes();
	// 	// axis({0, 3.5, -1, 4});
	// 	auto g = ax1->scatter(XPos, YPos, 0.5); 
	// 	hold(on);
	// 	ax1->scatter(Xmean, Ymean, 0.5);
	// 	show();
	// 	// ax1->
	// 	// scatter(Xhistory[i], Yhistory[i], 0.75);
	// 	// string name = "particlesImages/image_"+ to_string(i) + ".jpg";
	// 	// save(name);
	// // }

	

	return 0;
	
	
}





