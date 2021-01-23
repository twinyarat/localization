#include<Eigen/Dense>
#include<cmath>
#include<map>
#include<vector>
#include<numeric>

using namespace Eigen;

namespace kalman{
	const int S_DIMENSION = 3; //dimension of states
	const int C_DIMENSION = 2; //dimension of controls
	const int M_DIMENSION = 2; //dimension of measurements
	const double W_MIN = 0.0001; //minimum angular velocity
	
	struct State{
		double x; //x position (m)
		double y; //y position (m)
		double theta;// heading angle (rad)
		double timeStamp;// timestamp (s)
		State(double ts, double xx, double yy, double rr): x(xx), y(yy), theta(rr), timeStamp(ts){};
		State(): timeStamp(0), x(0), y(0),theta(0) {};
		State(const State& s):x(s.x), y(s.y), theta(s.theta), timeStamp(s.timeStamp) {};		
		Eigen::VectorXd toVector(){
		 Eigen::VectorXd vec(S_DIMENSION);
		 vec << x, y, theta;
		 return vec;
		}
	};

	struct Control{
		double lin_vel; //linear velocity (m/s)
		double ang_vel; //angular velocity (rad/s)
		double timeStamp; //timestamp (s)
		Control(double tt, double lv, double av): lin_vel(lv), ang_vel(av), timeStamp(tt) {};
		Control(): timeStamp(0), lin_vel(0), ang_vel(0) {};
		Control(const Control& s):lin_vel(s.lin_vel), ang_vel(s.ang_vel), timeStamp(s.timeStamp) {};
		Eigen::VectorXd toVector(){
		 Eigen::VectorXd vec(C_DIMENSION);
		 vec << lin_vel, ang_vel;
		 return vec;
		}
	};

	struct Reading{
		double range; //distance to landmark (m)
		double bearing; //angle between landmark direction and hearing (rad)
		int barcode; //barcode number MUST be used as key in mapping to landmark ID
		double timeStamp; //timestamp (s)
		Reading(double tt, int bcode, double rr, double bb ): timeStamp(tt),barcode(bcode), range(rr), bearing(bb) {};
		Reading(): timeStamp(0), barcode(0), bearing(0), range(0) {};
		Reading(const Reading& s):range(s.range), bearing(s.bearing), barcode(s.barcode), timeStamp(s.timeStamp) {};
		Eigen::VectorXd toVector(){
		 Eigen::VectorXd vec(M_DIMENSION);
		 vec << range, bearing;
		 return vec;
		}
	};

	struct Measurement{
		std::vector<Reading> readings;
		double timeStamp;
		Measurement(double t, std::vector<Reading> r): timeStamp(t), readings(r) {};
	};

	struct Position{
		double x;
		double y;
		Position(double xx, double yy): x(xx), y(yy) {};
		Position(): x(0), y(0) {};
	};

	struct StateDistribution{
		Eigen::VectorXd mean;
		Eigen::MatrixXd covar;
		StateDistribution(Eigen::VectorXd m, Eigen::MatrixXd sigma): mean(m), covar(sigma) {};
	};

	VectorXd expectedMotion(const VectorXd& state, const Control ctrl, double dt);
	MatrixXd motionJacobianState(const VectorXd& state, const Control& ctrl, double dt);
	MatrixXd motionJacobianControl(const VectorXd state, const Control& ctrl, double dt);
	MatrixXd motionCovariance(const MatrixXd& controlCovar, const MatrixXd& motionJacobianControl);
	MatrixXd measurementJacobianState(const VectorXd& state, const Reading& m, std::map<int, Position>& correspondence);
	VectorXd expectedReading(const VectorXd& s, const Reading& m, std::map<int, Position>& correspondence);
	VectorXd expectedControl(const State& curState, const State& nextState, double dt);
	MatrixXd measurementCovariance(double rangeVar, double bearingVar);
	MatrixXd learnMeasureCovariance(std::vector<Measurement>& measurements, std::vector<State>& groundTruths, std::map<int, Position>& correspondence);
	MatrixXd learnControlNoiseParams(std::vector<Control> controls, std::vector<State> groundTruths, double dt);
	StateDistribution processObservations(VectorXd& mu_bar, MatrixXd& sigma_bar, MatrixXd& noiseCovar, Measurement& measure, std::map<int, Position>& correspondence);
	StateDistribution EKF_known_correspondence(const StateDistribution belief, const Control& ctrl, Measurement& measure, std::map<int, Position>& correspondence, VectorXd& alphas, MatrixXd Q, double dt);

}