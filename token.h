#ifndef TOKEN_H
#define TOKEN_H

#include<Eigen/Dense>
#include<vector>
namespace token{	
	const int S_DIMENSION = 3; //dimension of states
	const int C_DIMENSION = 2; //dimension of controls
	const int M_DIMENSION = 2; //dimension of measurements
	const double W_MIN = 0.000001; //minimum angular velocity
	const int NUM_PARTICLES = 500; //number of particles
	
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
		Reading(double tt, int bcode, double rr, double bb ): timeStamp(tt),barcode(bcode), range(rr), bearing(atan2(sin(bb), cos(bb))) {};
		Reading(): timeStamp(0), barcode(0), bearing(0), range(0) {};
		Reading(const Reading& s):range(s.range), bearing(atan2(sin(s.bearing), cos(s.bearing))), barcode(s.barcode), timeStamp(s.timeStamp) {};
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

	struct NormalDistribution{
		Eigen::VectorXd mean;
		Eigen::MatrixXd covar;
		NormalDistribution(Eigen::VectorXd m, Eigen::MatrixXd sigma): mean(m), covar(sigma) {};
	};

	struct SigmaPoint{
		Eigen::VectorXd vec;
		double meanWeight; //for recovering mean of distribution
		double covarWeight; //for recovering covariance of distribution
		SigmaPoint(Eigen::VectorXd pp, double mw, double cw): vec(pp), meanWeight(mw), covarWeight(cw){};
	};

	// struct Particle{
	// 	Eigen::VectorXd vec;
	// 	double weight;
	// 	Particle(Eigen::VectorXd vv, double ww): vec(vv), weight(ww){};
	// };

	struct Particle{
		Eigen::VectorXd vec;
		double weight; //importance weighting 
		double length = vec.norm(); //distance from origin
		double angle = atan2(sin(vec[1]),cos(vec[0])); //angle made by position vector and the global x-axis
		double fdiff = 0;// the first-order difference in length/angle between *this's and its right adjacent Particle
		double sdiff = 0;// the second-order difference (difference of fdiffs) in length/angle between *this's and its right adjacent Particle
		Particle(Eigen::VectorXd vv, double ww): vec(vv), weight(ww) {};
	};
}

#endif