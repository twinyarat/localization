#include "kalman.h"
using MAP = std::map<int, kalman::Position>;

//construct the mean incremental vector of motion
VectorXd kalman::expectedMotion(const VectorXd& state, const Control ctrl, double dt){
	double w = (ctrl.ang_vel == 0)? W_MIN:ctrl.ang_vel;//avoid 0-division
	double v = ctrl.lin_vel; 
	double radius = v/w; 
	double x = state[0];
	double y = state[1];
	double th = state[2];
	double dth = dt*w;
	double theta_bar = th+dth;

	VectorXd mean(S_DIMENSION);
	theta_bar = atan2(sin(theta_bar), cos(theta_bar));

	mean << x + radius*(-sin(th) +sin(theta_bar)),
			y + radius*(cos(th) - cos(theta_bar)),
			theta_bar;

	return mean;
}

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

//construct the expected measurement vector from state and correspondence
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
//where M is of the form [{a(1)v+a(2)w}^2,	{a(3)v+a(4)w}^2
//						  {a(3)v+a(4)w}^2,  {a(5)v+a(6)w}^2]
MatrixXd kalman::learnControlNoiseParams(std::vector<Control> controls, std::vector<State> groundTruths, double dt){
	int timesteps = groundTruths.size() - 1; //compute expected controls using state-pairs
	
	MatrixXd sampleCovarTemp(C_DIMENSION, C_DIMENSION);
	VectorXd ctrlVec(C_DIMENSION);
	VectorXd expectedCtrlVec(C_DIMENSION);

	std::vector<double> samplecov00;
	std::vector<double> samplecov11;
	std::vector<double> samplecov01;
	std::vector<double> samplectrlv;
	std::vector<double> samplectrlw;
	
	//collect all control samples and variance samples into vectors
	for(int i = 0; i < timesteps; ++i){
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
	MatrixXd controlSamples(samplectrlv.size(),2);
	for(int i = 0; i < samplecov00.size(); ++i){
		
		sampleCovar(i,0) = sqrt(std::accumulate(samplecov00.begin(),samplecov00.end(), 0.0))/samplecov00.size();
		sampleCovar(i,1) = sqrt(std::accumulate(samplecov11.begin(), samplecov11.end(), 0.0))/samplecov11.size();
		sampleCovar(i,2) = sqrt(std::accumulate(samplecov01.begin(), samplecov01.end(), 0.0))/samplecov01.size();

		controlSamples(i,0) = samplectrlv[i];
		controlSamples(i,1) = samplectrlw[i];
	}

	//Solve A'Ax=A'b and reshape solution into row vector
	MatrixXd params = controlSamples.colPivHouseholderQr().solve(sampleCovar); 
	Map<VectorXd> alphas(params.data(), params.size());
	return alphas;
}

//learn covariance of measurement-related noise 
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
kalman::StateDistribution kalman::processObservations(VectorXd& mu_bar, MatrixXd& sigma_bar, MatrixXd& noiseCovar, Measurement& measure, MAP& correspondence ){
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
	return kalman::StateDistribution(mu_bar, sigma_bar);
}


//perform one iteration of the Extended Kalman Filter Localization algorithm (w/ known correspondence). 
//return a Gaussian distribution over states of the subsequent time step.
kalman::StateDistribution kalman::EKF_known_correspondence(const StateDistribution belief, const Control& ctrl, Measurement& measure, MAP& correspondence, VectorXd& alphas, MatrixXd measrCovar, double dt){
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
	VectorXd mu_bar = expectedMotion(mu, ctrl, dt);
	MatrixXd sigma_bar = G*sigma*(G.transpose()) + motionCovariance(M,V);

	//4. MEASUREMENT UPDATE: Incorporate all measurement readings at current timestep
	StateDistribution distr = processObservations(mu_bar, sigma_bar, measrCovar, measure, correspondence);

	return distr;
}




////////////////////////////////////////////////
//////////////// UKF Functions /////////////////
////////////////////////////////////////////////

//augment state vector to contain means of control noise distribution and of measurement noise distribution
VectorXd kalman::augmentState(VectorXd& state){
	VectorXd augmented =  VectorXd::Zero(S_DIMENSION + C_DIMENSION + M_DIMENSION);
	augmented(0) =  state[0]; 
	augmented(1) = state[1];
	return augmented;
}

//augment covariance matrix to contain covariances of control noise and measurement noise
MatrixXd kalman::augmentCovariance(MatrixXd& stateCovar, MatrixXd& ctrlNoiseCovar,MatrixXd& measureNoiseCovar){
	int DIM = S_DIMENSION + C_DIMENSION + M_DIMENSION;
	MatrixXd augmented = MatrixXd::Zero(DIM, DIM);
	int i = 0;
	augmented.block<S_DIMENSION,S_DIMENSION>(i,i) = stateCovar;
	i += S_DIMENSION;
	augmented.block<C_DIMENSION,C_DIMENSION>(i,i) = ctrlNoiseCovar;
	i += C_DIMENSION;
	augmented.block<M_DIMENSION,M_DIMENSION>(i,i) = measureNoiseCovar;
	return augmented;//augmented is block diagonal
}


//generate sigma points of the distribution parameterized by mean and covar
//parameters lambda, alpha, and beta are used to compute the weights associated with each sigma point
std::vector<kalman::SigmaPoint>  kalman::makeSigmaPoints(const VectorXd& mean, const MatrixXd& sigma, double lambda, double alpha, double beta){
	std::vector<kalman::SigmaPoint> SigmaPts;
	double gamma = 0;
	
	//compute the first statistics
	int DIM = mean.rows();
	double meanW = lambda/(DIM + lambda); //weight for recovering mean
	double covarW = meanW + (1 - pow(alpha,2)+beta); //weight for recovering covariance
	SigmaPts.push_back(SigmaPoint(mean, meanW, covarW));
	
	//Cholesky decomposition of sigma matrix
	LLT<MatrixXd> lltOfA(sigma); 
	MatrixXd sqrtSigma = lltOfA.matrixL(); 

	//collect the remaining statistics
	for(int i = 0; i < sqrtSigma.cols(); ++i){
		meanW = 1/(2*(DIM+lambda));
		covarW = meanW;
		gamma = sqrt(DIM + lambda);
		SigmaPts.push_back(SigmaPoint(mean + gamma*sqrtSigma.col(i), meanW, covarW ));
		SigmaPts.push_back(SigmaPoint(mean - gamma*sqrtSigma.col(i),meanW, covarW ));
	}
	return SigmaPts;
} 


// //evolve state and control into subsequent state according to motion model
VectorXd kalman::motionModel(const VectorXd& state, const VectorXd& control, double dt){
	double x = state[0];
	double y = state[1];
	double th = state[2];
	double v = control[0];
	double w = (control[1] == 0)? W_MIN:control[1]; //avoid -division
	double radius = v/w;

	VectorXd nextState(S_DIMENSION);
	double x_bar = x + radius*(-sin(th) +sin(th+w*dt));
	double y_bar = y + radius*(cos(th)-cos(th+w*dt));
	double th_bar = th + w*dt;
	th_bar = atan2(sin(th_bar), cos(th_bar)); //wrap within [-pi,pi]
	nextState << x_bar, y_bar, th_bar;
	
	return nextState;
}


// //evolve sigma points according to motion model
// std::vector<kalman::SigmaPoint> kalman::evolveSigmaPoints(const vector<kalman::SigmaPoint> SPoints, const VectorXd& control, double dt){


// }




