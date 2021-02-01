#include "kalman.h"
using MAP = std::map<int,token::Position>;

//construct the jacobian of the dynamics/motion model with respect to state
MatrixXd kalman::motionJacobianState(const VectorXd& state, const Control& ctrl, double dt){
	double w = (ctrl.ang_vel == 0)? W_MIN:ctrl.ang_vel;//avoid 0-division
	double v = ctrl.lin_vel; 
	double radius = v/w; 
	double th = state[2];

	MatrixXd A(S_DIMENSION,S_DIMENSION);

	A << 1,0, radius*(-cos(th)+cos(th + w*dt)),
		 0,1, radius*(-sin(th)+sin(th + w*dt)),
		 0,0, 1;

	return A;
}

//construct the jacobian of the dynamics/motion model with respect to control
MatrixXd kalman::motionJacobianControl(const VectorXd state, const Control& ctrl, double dt){
	double w = (ctrl.ang_vel == 0)? W_MIN:ctrl.ang_vel;
	double v = ctrl.lin_vel;
	double th = state[2]; 

	MatrixXd B(S_DIMENSION, C_DIMENSION);
	
	B << (-sin(th) + sin(th + w*dt))/w, (v/(pow(w,2)))*(sin(th) -sin(th + w*dt)) + dt*v*cos(th + w*dt)/w,
		 (cos(th) - cos(th + w*dt))/w , (v/(pow(w,2)))*(-cos(th)+cos(th + w*dt)) + dt*v*sin(th + w*dt)/w,
		 0							  , dt;
			
	return B;
}

//construct the covariance of motion-related noise from the covariance of control-related noise
MatrixXd kalman::motionCovariance(const MatrixXd& controlCovar, const MatrixXd& motionJacobianControl){
	
	return motionJacobianControl*controlCovar*(motionJacobianControl.transpose());
}

//construct the jacobian of the measurement model with respect to state (with known barcode-landmark correspondence)
MatrixXd kalman::measurementJacobianState(const VectorXd& state, const Reading& m, MAP& correspondence){
	MatrixXd H(M_DIMENSION, S_DIMENSION);
	double landmarkXPos = correspondence[m.barcode].x;
	double landmarkYPos = correspondence[m.barcode].y;
	double x = state[0];
	double y = state[1];
	double th = state[2];

	double dy = landmarkYPos-y;
	double dx = landmarkXPos-x;
	double expected_squared_range = pow(dx,2) + pow(dy,2);

	H << -dx/sqrt(expected_squared_range), -dy/sqrt(expected_squared_range), 0,
		 dy/expected_squared_range		 , -dx/expected_squared_range	   ,-1;

	return H;
}

//construct the expected result of motion
VectorXd kalman::expectedMotion(const VectorXd& state, const VectorXd& ctrl, double dt){
	double v = ctrl[0]; 
	double w = (ctrl[1] == 0)? W_MIN:ctrl[1];//avoid 0-division
	double radius = v/w; 
	double x = state[0];
	double y = state[1];
	double th = state[2];
	double dth = dt*w;
	double theta_bar = th+dth;

	VectorXd mean(S_DIMENSION);

	mean << x + radius*(-sin(th) +sin(theta_bar)),
			y + radius*(cos(th) - cos(theta_bar)),
			theta_bar;

	return mean;
}

//construct the expected reading vector from state and correspondence
VectorXd kalman::expectedReading(const VectorXd& s, const Reading& r, MAP& correspondence){
	VectorXd vec(M_DIMENSION);
	double landmarkXPos = correspondence[r.barcode].x;
	double landmarkYPos = correspondence[r.barcode].y;
	double x = s[0];
	double y = s[1];
	double th = atan2(sin(s[2]),cos(s[2])); //wrap[-pi, pi]
	
	//compute expected measurement
	double dy = landmarkYPos - y;
	double dx = landmarkXPos - x;
	double expected_squared_range = pow(dx,2) + pow(dy,2);
	double expected_bearing = atan2(dy,dx)- th;

	vec << sqrt(expected_squared_range),
		   atan2(sin(expected_bearing),cos(expected_bearing)); //wrap within [-pi, pi]

	return vec;
}

//computes the expected control (linear and angular velocities) that would take curState to nextState
VectorXd kalman::expectedControl(const State& curState, const State& nextState, double dt){
	VectorXd u_bar(C_DIMENSION);
	double w_bar = 0;
	double velocity_bar = 0;
	double curTheta = curState.theta;

	w_bar = (nextState.theta - curTheta)/dt;
	w_bar = atan2(sin(w_bar), cos(w_bar)); //wrap within [-pi,pi]
	velocity_bar = w_bar*(nextState.y - curState.y)/(cos(curTheta)-cos(curTheta+w_bar*dt));

	u_bar << velocity_bar, w_bar;
	return u_bar;
}

//learn the intrinsic parameters a of the control noise covariance M.
//where M is of the form [{a(0)v+a(1)w}^2,	{a(4)v+a(5)w}^2
//						  {a(4)v+a(5)w}^2,  {a(2)v+a(3)w}^2]
VectorXd kalman::learnControlNoiseParams(std::vector<Control> controls, std::vector<State> groundTruths, double dt){
	int TIMESTEPS = groundTruths.size() - 1; //compute expected controls using state-pairs
	
	MatrixXd sampleCovarTemp(C_DIMENSION, C_DIMENSION);
	VectorXd ctrlVec(C_DIMENSION);
	VectorXd expectedCtrlVec(C_DIMENSION);

	std::vector<double> samplecov00;
	std::vector<double> samplecov11;
	std::vector<double> samplecov01;
	std::vector<double> samplectrlv;
	std::vector<double> samplectrlw;
	
	//collect all control samples and variance samples into vectors
	for(int i = 0; i < TIMESTEPS; ++i){
		ctrlVec = controls[i].toVector();
		expectedCtrlVec = expectedControl(groundTruths[i], groundTruths[i+1], dt);
		sampleCovarTemp = (ctrlVec -  expectedCtrlVec)*((ctrlVec - expectedCtrlVec).transpose());
		if(sampleCovarTemp(0,0) > 0 && sampleCovarTemp(1,1) > 0 && sampleCovarTemp(0,1) > 0) { //ensure positive definiteness
			samplectrlv.push_back(abs(controls[i].lin_vel));
			samplectrlw.push_back(abs(controls[i].ang_vel == 0? W_MIN: controls[i].ang_vel));		
			samplecov01.push_back((sampleCovarTemp(0,1)));
			samplecov00.push_back((sampleCovarTemp(0,0)));
			samplecov11.push_back((sampleCovarTemp(1,1)));
		}		
	}

	//collect all covariance and control samples into matrices
	MatrixXd sampleCovar(samplecov00.size(),3);
	MatrixXd controlSamples(samplectrlv.size(),C_DIMENSION);
	for(int i = 0; i < samplecov00.size(); ++i){
		
		sampleCovar(i,0) = sqrt(std::accumulate(samplecov00.begin(),samplecov00.end(), 0.0))/samplecov00.size();
		sampleCovar(i,1) = sqrt(std::accumulate(samplecov11.begin(), samplecov11.end(), 0.0))/samplecov11.size();
		sampleCovar(i,2) = sqrt(std::accumulate(samplecov01.begin(), samplecov01.end(), 0.0))/samplecov01.size();

		controlSamples(i,0) = samplectrlv[i];
		controlSamples(i,1) = samplectrlw[i];
	}

	//Solve A'Ax=A'b and reshape solution, alphas, into row vector
	MatrixXd params = controlSamples.colPivHouseholderQr().solve(sampleCovar); 
	Map<VectorXd> alphas(params.data(), params.size());
	return alphas;
}

//perform Maximum Likelihood estimate of covariance of measurement-related noise 
MatrixXd kalman::learnMeasureNoiseCovariance(std::vector<Measurement>& measurements, std::vector<State>& groundTruths, MAP& correspondence){
	MatrixXd Q = MatrixXd::Zero(M_DIMENSION, M_DIMENSION);

	auto mIter = measurements.begin();
	auto gIter = groundTruths.begin();
	int sampleSize = 0;
	VectorXd expected_reading;

	while(gIter != groundTruths.end()){
		for(auto reading : mIter->readings){
			if(correspondence.find(reading.barcode) != correspondence.end()){
				expected_reading =  expectedReading((*gIter).toVector(), reading, correspondence);
				Q = Q+ (reading.toVector()-expected_reading)*((reading.toVector()-expected_reading).transpose());
				++sampleSize;
			}
		}
		++mIter;
		++gIter;		
	}

	return Q/sampleSize;
}

//incorporate measurement(all sensor reading) into the state distribution update
NormalDistribution kalman::processObservationsEKF(VectorXd& mu_bar, MatrixXd& sigma_bar, MatrixXd& noiseCovar, Measurement& measure, MAP& correspondence ){
	MatrixXd H; 
	MatrixXd S;
	MatrixXd K; 
	VectorXd innovation;

	for(auto reading: measure.readings){
		//Process a reading if it has a barcode associated with landmark and not other moving objects
		if( correspondence.find(reading.barcode) != correspondence.end() ) {
			H = measurementJacobianState(mu_bar, reading, correspondence); //compute the jacobian of the measurement model wrt to state
			S = H*sigma_bar*(H.transpose()) + noiseCovar; //incorporate measurement-related noise
			K = sigma_bar*(H.transpose())*(S.inverse()); //kalman gain
			innovation = reading.toVector() - expectedReading(mu_bar, reading, correspondence);
			mu_bar = mu_bar + K*innovation;
			sigma_bar = sigma_bar - K*H*sigma_bar;
		}
	}
	return NormalDistribution(mu_bar, sigma_bar);
}


//perform one iteration of the Extended Kalman Filter Localization algorithm (w/ known correspondence). 
//return a Gaussian distribution over states of the subsequent time step.
NormalDistribution kalman::EKF_known_correspondence(const NormalDistribution belief, Control& ctrl, Measurement& measure, MAP& correspondence, VectorXd& alphas, MatrixXd measrCovar, double dt){
	auto mu = belief.mean;
	auto sigma = belief.covar;
	
	//1. Compute Motion-related Jacobians
	auto G = motionJacobianState(mu,ctrl,dt);
	auto V = motionJacobianControl(mu,ctrl,dt);
	
	//2. Construct control-noise covariance M 	
	MatrixXd M(C_DIMENSION, C_DIMENSION);
	double v = abs((ctrl.lin_vel));
	double w = abs((ctrl.ang_vel == 0 ? W_MIN:ctrl.ang_vel));
	M << pow(alphas[0]*v + alphas[1]*w, 2),   0,
           0 , 			pow(alphas[2]*v + alphas[3]*w,2);

	//3. PREDICTION: Update gaussian using control 
	VectorXd mu_bar = expectedMotion(mu, ctrl.toVector(), dt);
	MatrixXd sigma_bar = G*sigma*(G.transpose()) + motionCovariance(M,V);

	//4. MEASUREMENT UPDATE: Incorporate all measurement readings at current timestep
	NormalDistribution distr = processObservationsEKF(mu_bar, sigma_bar, measrCovar, measure, correspondence);

	return distr;
}




////////////////////////////////////////////////
//////////////// UKF Functions /////////////////
////////////////////////////////////////////////

// //generate sigma points of the distribution parameterized by mean and covar
// //parameters lambda, alpha, and beta are used to compute the weights associated with each sigma point
// std::vector<SigmaPoint>  kalman::makeSigmaPoints(const VectorXd& mean, const MatrixXd& covar, double lambda){
// 	std::vector<SigmaPoint> SigmaPts;
// 	double gamma = 0;
	
// 	//compute the first statistics
// 	int DIM = mean.rows();
// 	double meanW = lambda/(DIM + lambda); //weight for recovering mean
// 	double covarW = meanW; //+ (1 - pow(a,2)+b); //weight for recovering covariance
// 	SigmaPts.push_back(SigmaPoint(mean, meanW, covarW));
	
// 	//Cholesky decompose sigma matrix
// 	LLT<MatrixXd> lltOfA(covar); 
// 	MatrixXd sqrtCovar = lltOfA.matrixL(); 

// 	//collect the remaining statistics
// 	for(int i = 0; i < sqrtCovar.cols(); ++i){
// 		meanW = 1/(2*(DIM+lambda));
// 		covarW = meanW;
// 		gamma = sqrt(DIM + lambda);
// 		SigmaPts.push_back(SigmaPoint(mean + gamma*sqrtCovar.col(i), meanW, covarW ));
// 		SigmaPts.push_back(SigmaPoint(mean - gamma*sqrtCovar.col(i),meanW, covarW ));
// 	}
// 	return SigmaPts;
// } 



//generate sigma points of the distribution parameterized by mean and covar.
//parameter m must satisfy 0.5 < m < 1 based on reference paper.
//REFERENCE: "A New Method for Generating Sigma Points and Weights for Nonlinear Filtering" by Rahul Radhakrishnan
std::vector<SigmaPoint>  kalman::makeSigmaPoints(const VectorXd& mean, const MatrixXd& covar, double m){
	std::vector<SigmaPoint> SigmaPts;
	
	//0. Cholesky decompose covar
	LLT<MatrixXd> lltOfA(covar); 
	MatrixXd sqrtCovar = lltOfA.matrixL(); 
	
	//1. compute alphas
	std::vector<double> alphas;
	for(int i = 0; i < covar.cols(); ++i){
		auto p = covar.col(i);
		double a = abs(mean.dot(p))/(mean.norm()*p.norm());
		alphas.push_back(a);
	}

	//2. compute auxillary variables
	double alphaSum = accumulate(alphas.begin(), alphas.end(), 0.0);
	double beta = 0.25*m*(*max_element(alphas.begin(), alphas.end())) - 0.5*alphaSum + 1;
	double si = alphaSum + beta;

	//3. generate the mean sigma point
	double meanWeight = 1- alphaSum/(2*si);
	SigmaPts.push_back(SigmaPoint(mean, meanWeight , meanWeight));

	//4. generate the remaining 4x(S_DIMENSION) sigma points
	for(int i = 0; i < sqrtCovar.cols(); ++i){
		auto sig1 = mean + sqrt(si/(m*alphas[i]))*sqrtCovar.col(i);
		auto sig2 = mean - sqrt(si/(m*alphas[i]))*sqrtCovar.col(i);
		auto sig3 = mean + sqrt(si/((1-m)*alphas[i]))*sqrtCovar.col(i);
		auto sig4 = mean - sqrt(si/((1-m)*alphas[i]))*sqrtCovar.col(i);

		double w1 = m*alphas[i]/(4*si);
		double w2 = (1-m)*alphas[i]/(4*si);

		SigmaPts.push_back(SigmaPoint(sig1, w1, w1));
		SigmaPts.push_back(SigmaPoint(sig2, w1, w1));
		SigmaPts.push_back(SigmaPoint(sig3, w2, w2));
		SigmaPts.push_back(SigmaPoint(sig4, w2, w2));
	}

	return SigmaPts;
} 



//evolve augmented sigma points into state-only sigma points according to motion model. 
//NOTE: evolution of state subvectors is NOT an in-place operation. 
std::vector<SigmaPoint> kalman::evolveSigmaPoints(std::vector<SigmaPoint>& SPoints, const VectorXd& control, double dt){
	VectorXd state(S_DIMENSION);
	VectorXd x_bar(S_DIMENSION);
	std::vector<SigmaPoint> evolved;

	//pass the state component of each sigma point through motion model with a noisy control
	for(auto s: SPoints){
		state = s.vec.head(S_DIMENSION); //extract state vector 
		x_bar = expectedMotion(state, control, dt);
		evolved.push_back(SigmaPoint(x_bar,s.meanWeight, s.covarWeight)); 
	}
	return evolved; //return vector of evolved states-only
}


//incorporate measurement into the correction of state sigma points
//return a normal distribution of the corrected states
NormalDistribution kalman::correctSigmaPoints(NormalDistribution stateDistr,  Measurement& measure, MAP& correspondence, MatrixXd& measureNoiseCovar, double lambda){

	std::vector<SigmaPoint> readingSigmas;
	MatrixXd zeroMat = MatrixXd::Zero(M_DIMENSION, M_DIMENSION);
	VectorXd expectedRead = VectorXd(M_DIMENSION);
	VectorXd state(S_DIMENSION);
	double weight = 0;
	
	MatrixXd crossCovar = MatrixXd::Zero(S_DIMENSION, M_DIMENSION);
	MatrixXd kalmanGain(S_DIMENSION, M_DIMENSION);
	VectorXd correctedStateMean(stateDistr.mean);
	MatrixXd correctedStateCovar(stateDistr.covar);

	//1. generate sigma points from the distribution of evolved states
	auto eSigmaPoints = makeSigmaPoints(stateDistr.mean, stateDistr.covar, lambda);
	
	//2. Process multiple sensor readings contained in the input measure.
	for(auto r: measure.readings){
		
		if( correspondence.find(r.barcode) != correspondence.end() ){ 
			//3. collect expected reading associated with each sigma point
			for(auto s: eSigmaPoints){
				expectedRead = expectedReading(s.vec, r, correspondence); //pass each evolved sigma point through measurement model
				readingSigmas.push_back(SigmaPoint(expectedRead, s.meanWeight, s.covarWeight));
			}
			
			//4. compute mean and covariance of all expected readings collected above
			auto readingDistribution = distributionFromStatistics(readingSigmas, measureNoiseCovar);
			auto meanReading = readingDistribution.mean;
			auto covarReading = readingDistribution.covar;
			
			//5. compute cross-covariance (state-reading)
			for(int i= 0; i < eSigmaPoints.size(); ++i){
				weight = readingSigmas[i].covarWeight;
				state = eSigmaPoints[i].vec;
				crossCovar += weight*(state - stateDistr.mean)*((readingSigmas[i].vec - meanReading).transpose());
			}

			//6. compute kalman gain matrix
			kalmanGain = crossCovar*(covarReading.inverse());
			//7. correct mean of states using innovation 
			correctedStateMean += kalmanGain*(r.toVector() -  meanReading);
			//8. correct covariance of states
			correctedStateCovar -= kalmanGain*covarReading*(kalmanGain.transpose());
			//9. clear expected readings at Sigma points for the next set of expected readings
			readingSigmas.clear();
		}

	}


	return NormalDistribution(correctedStateMean, correctedStateCovar);
}

//calculate the mean and covariance of Sigma points with added noise covariance 
NormalDistribution kalman::distributionFromStatistics(std::vector<SigmaPoint>& SPoints, MatrixXd& noiseCovar){
	int DIM = SPoints[0].vec.rows();
	VectorXd mean = VectorXd::Zero(DIM);
	MatrixXd covariance = MatrixXd::Zero(DIM,DIM);
	
	//compute mean of distribution
	for(auto s: SPoints){
		mean += s.meanWeight * s.vec;
	}
	//compute covariance of distribution
	for(auto s: SPoints){
		covariance += s.covarWeight * (s.vec - mean) * (s.vec -mean).transpose();
	}
	
	return NormalDistribution(mean, covariance + noiseCovar);
}

//perform one iteration of the Unscented Kalman Filter Localization algorithm.
//return distribution over states of the subsequent time step
NormalDistribution kalman::UKF_known_correspondence(NormalDistribution belief, Control& ctrl, Measurement& measure, MAP& correspondence, const VectorXd& alphas, MatrixXd& measrCovar, double dt, double lambda){
	auto stateMean = belief.mean;
	auto stateCovar = belief.covar;
	MatrixXd motionNoiseCovar(S_DIMENSION, S_DIMENSION);
	
	//1.construct control-noise covariance from input parameters alphas
	MatrixXd ctrlCovar = MatrixXd::Zero(C_DIMENSION, C_DIMENSION);
	double v = abs((ctrl.lin_vel));
	double w = abs((ctrl.ang_vel == 0 ? W_MIN:ctrl.ang_vel));
	ctrlCovar(0,0) = pow(alphas[0]*v + alphas[1]*w,2);
	ctrlCovar(1,1) = pow(alphas[2]*v + alphas[3]*w,2);


	//2.generate sigma points of the distribution of augmented states
	std::vector<SigmaPoint> sigmaPoints =  makeSigmaPoints(stateMean, stateCovar, lambda);

	//3.evolve sigma points through the motion model using control input
	std::vector<SigmaPoint> evolvedSigmaPoints = evolveSigmaPoints(sigmaPoints, ctrl.toVector(), dt);
	
	//4.Incorporate motion-related noise into computing the mean and covariance of the evolved sigma points
	motionNoiseCovar = motionCovariance(ctrlCovar, motionJacobianControl(stateMean, ctrl, dt));
	NormalDistribution evolvedDistr = distributionFromStatistics(evolvedSigmaPoints, motionNoiseCovar);
	
	//5.correct evolved sigma points using readings in measurement
	auto correctedDistr = correctSigmaPoints(evolvedDistr, measure, correspondence, measrCovar, lambda);
	
	// return evolvedDistr;
	return correctedDistr;
}



