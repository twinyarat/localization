#include<vector>
#include<map>
#include<fstream>
#include<sstream>
#include<iostream>
#include "kalman.h"

using namespace std;
using namespace kalman;


namespace preprocess{
	const double TIMEDISCOUNT = 1240000000.0;
	void readStatesFromFile(string,vector<State>&);
	void readControlsFromFile(string, vector<Control>&);
	void readMeasuresFromFile(string, vector<Measurement>&, double);
	double findMaxStartTime(vector<State>& vstate, vector<Control>& vcontrol, vector<Measurement>& vmeasure);
	double findMinEndTime(vector<State>& vstate, vector<Control>& vcontrol, vector<Measurement>& vmeasure);
	template<typename Data>
	vector<Data> resample(vector<Data>& vec, double dt);
	void trimTails(vector<State>& vstate, vector<Control>& vcontrol, vector<Measurement>& vmeasure);
	void alignTime(vector<State>& vstate, vector<Control>& vcontrol, vector<Measurement>& vmeasure, double dt = 0.2);
	map<int,int> landmarkToBarcode(string fname);
	map<int, kalman::Position> barcodeToPosition(string fname, map<int,int>);
}
