#include"particleFilter.h"
#include<iostream>
using namespace std;

//generate a sample state of the subsequent time step using a perturbed control
VectorXd particleFilter::sampleFromMotion(VectorXd& state, VectorXd& control, VectorXd& alphas, double dt){
	VectorXd sample(S_DIMENSION);
	double x = state[0];
	double y = state[1];
	double th = state[2];
	double v = control[0];
	double w = control[1]  == 0? W_MIN: control[1] ; //avoid 0-division;
	double bearingPerturb = 0;

	//compute deviations of control noises
	double linVelocityDev = abs(alphas[0]*v + alphas[1]*w);
	double angVelocityDev = abs(alphas[2]*v + alphas[3]*w);
	double bearingPerturbation = abs(alphas[4]*v + alphas[5]*w);
	
	//construct normal distributions of perturbations
	std::default_random_engine generator(std::random_device{}());
	std::normal_distribution<double> dLinVel(0,linVelocityDev);
	std::normal_distribution<double> dAngVel(0,angVelocityDev);
	std::normal_distribution<double> dBearing(0,bearingPerturbation);

	//perturb control input
	v += dLinVel(generator);
	w += dAngVel(generator);
	bearingPerturb = dBearing(generator)*dt;
	

	//update state using perturbed control
	w = (w == 0? W_MIN: w); //avoid 0-division
	double radius = v/w;
	x += (radius*(-sin(th) + sin(th + w*dt)));
	y += (radius*(cos(th) - cos(th + w*dt)));
	th +=  (w*dt + bearingPerturb);
	th = atan2(sin(th), cos(th)); //wrap within [-pi, pi]

	sample << x, y, th;
	return sample;
}

//compute the likelihood of drawing x from a univariate normal distribution parameterized by mean and variance
double particleFilter::gaussianDensity1D(double x,double mean, double variance){
	double MINCONST = 0.0000000001;
	variance = (variance == 0 ? MINCONST: variance); //avoid 0-division
	return  (1/(sqrt(2*M_PI* variance))) * exp( -pow(x-mean, 2)/(2*variance));
}

//normalize weights of particles.****IN-PLACE Operation.****
void particleFilter::normalizeWeights(std::vector<token::Particle>& particles){
	//sum up weights of particles
	double norm = std::accumulate(particles.begin(), particles.end(), 0.0, [&](double sum,token::Particle& p){ return sum + p.weight;} );
	for(auto& p: particles){
		p.weight /= norm;
	}
}

//computes the likelihood of taking measurement measure given state and known correspondence
double particleFilter::sampleFromMeasurement(VectorXd& state, token::Measurement& measure, MatrixXd& noiseCovar, std::map<int, Position>& correspondence){
	double sampleWeight = 1.0;
	double rangeVar = noiseCovar(0,0);
	double bearingVar = noiseCovar(1,1);
	//process each reading in a measurement
	VectorXd expectedReading(M_DIMENSION);
	for(auto reading: measure.readings){
		//if barcode is of landmark 
		if(correspondence.find(reading.barcode) != correspondence.end()){
			expectedReading = kalman::expectedReading(state, reading, correspondence);
			double expectedRange = expectedReading[0];
			double expectedBearing = expectedReading[1];
			sampleWeight *= particleFilter::gaussianDensity1D(reading.range, expectedRange,rangeVar)*particleFilter::gaussianDensity1D(reading.bearing, expectedBearing, bearingVar);
		}
	}
	return sampleWeight;
}

//draw particle from a set of particles with probability proportional to its weight
std::vector<token::Particle> particleFilter::lowVarianceSampling(std::vector<token::Particle>& particles){
	std::vector<token::Particle> resampled;
	double sampleSize = particles.size();
	
	//uniformly sample from [0, 1/sampleSize] for a left sentinel
	std::random_device rd;
	std::default_random_engine generator(rd()); 
	std::uniform_real_distribution<double> distribution(0,1/double(sampleSize));	
	double leftSentinel = distribution(generator);

	//resample
	int pIndex = 0;
	double weightOffset = particles[0].weight;
	for(int i = 0; i < particles.size(); ++i){		
		double threshold = leftSentinel + double(i/sampleSize); //evenly increment threshold
		while(weightOffset < threshold){
			++pIndex;
			weightOffset += particles[pIndex].weight;
		}
		//all resampled particles have the same weight
		resampled.push_back(token::Particle(particles[pIndex].vec, 1/sampleSize));
		// resampled.push_back(particles[pIndex]);
	}

	return resampled;
}

//uniformly distribute particles across the state space
std::vector<token::Particle> particleFilter::uniformlyDistributeParticles(double xmin, double xmax, double ymin, double ymax, double thmin, double thmax){
	std::vector<token::Particle> samples;
	//generate uniform distributions
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_real_distribution<double> Xdistribution(xmin,xmax);
	std::uniform_real_distribution<double> Ydistribution(ymin,ymax);
	std::uniform_real_distribution<double> THdistribution(thmin,thmax);

	//sampling
	double weight = 1/double(NUM_PARTICLES);
	for(int i = 0; i < NUM_PARTICLES; ++i){
		VectorXd vec(S_DIMENSION);
		vec << Xdistribution(generator), Ydistribution(generator), THdistribution(generator);
		samples.push_back(token::Particle(vec,weight));
	}
	return samples;
}

//compute the mean and covariance of the particles' distribution
token::NormalDistribution particleFilter::distributionFromStatistics(std::vector<token::Particle >& particles){
	int DIM = particles[0].vec.rows();
	VectorXd mean = VectorXd::Zero(DIM);
	MatrixXd covariance = MatrixXd::Zero(DIM,DIM);
	
	//compute mean of distribution
	for(auto p: particles){
		mean += p.weight * p.vec;
	}
	//compute covariance of distribution
	for(auto p: particles){
		covariance += p.weight * (p.vec - mean) * (p.vec -mean).transpose();
	}

	return NormalDistribution(mean, covariance);
}







