#include "preprocess.h"


void preprocess::readStatesFromFile(string fname, vector<State>& vec){ //reads States into vec
	ifstream rFile(fname);
	string line;
	if(rFile.is_open()){
		double time_stamp= 0 ;
		double x = 0;
		double y = 0;
		double rad = 0;
		while(getline(rFile, line)){
			if(line[0] != '#'){ //skip all headers 
				istringstream ss(line);
				ss >> time_stamp >> x >> y >> rad;
				State fresh(time_stamp, x , y, rad);
				vec.push_back(fresh);
			}//end if
		}//end while
	}
	else{
		cerr << "Reading States Failed\n";
	}
	rFile.close();
}

void preprocess::readControlsFromFile(string fname, vector<Control>& vec){ //reads Controls into vec
	ifstream rFile(fname);
	string line;
	if(rFile.is_open()){
		double time_stamp= 0 ;
		double vel = 0;
		double ang = 0;
		while(getline(rFile, line)){
			if(line[0] != '#'){ //skip all headers 
				istringstream ss(line);
				ss >> time_stamp >> vel >> ang ;
				vec.push_back(Control(time_stamp, vel, ang));
			}//end if
		}//end while
	}
	else{
		cerr << "Reading Controls Failed\n";
	}
	rFile.close();
}

void preprocess::readMeasuresFromFile(string fname, vector<Measurement>& vec, double dt){ //reads measurements into vec 
	ifstream rFile(fname);
		string line;
		if(rFile.is_open()){
			double time_stamp= 0 ;
			double range = 0;
			double bearing = 0;
			int barcode = 0;
			double refTime = 0;
			bool firstLineSeen = false;
			double timeThreshold = 0;
			vector<Reading> readVec;
			while( getline(rFile, line)){
				if(line[0] != '#'){ //skip all headers 
					istringstream ss(line);
					//find the first line of reading and set refTime
					if(!firstLineSeen){					
						ss >>refTime >> barcode >> range >> bearing;
						time_stamp = refTime;
						timeThreshold = refTime + dt;
						firstLineSeen = true;
					}
					else{
						ss >>time_stamp >> barcode >> range >> bearing;
					}
					
					if(time_stamp <= timeThreshold){
						readVec.push_back(Reading(time_stamp, barcode, range, bearing));
					}
					else{
						vec.push_back(Measurement(timeThreshold - dt, readVec)); //push the vector of readings over the current time step into vec of measurements
						readVec.clear(); //clear vector of readings over the current time step				
						readVec.push_back(Reading(time_stamp, barcode, range, bearing));//push the first reading over the next timestep
						timeThreshold = time_stamp + dt; //increment time threshold
					}
				}//end if
			}//end while
		}
		else{
			cerr << "Reading Measurements Failed\n";
		}
		rFile.close();
}

//find maximum starting time-stamp among the input vectors
double preprocess::findMaxStartTime(vector<State>& vstate, vector<Control>& vcontrol, vector<Measurement>& vmeasure){
	double sTime = (vstate.begin())->timeStamp ;
	double cTime = (vcontrol.begin())->timeStamp;
	double mTime = (vmeasure.begin())->timeStamp;
	return max(max(sTime, mTime), max(sTime, cTime));
}

//find minimum ending time-stamp among the input vectors
double preprocess::findMinEndTime(vector<State>& vstate, vector<Control>& vcontrol, vector<Measurement>& vmeasure){
	double sTime = (vstate.rbegin())->timeStamp;
	double cTime = (vcontrol.rbegin())->timeStamp;
	double mTime = (vmeasure.rbegin())->timeStamp;
	return min(min(sTime, mTime), min(sTime, cTime));
}

//resample data in vec in increments of dt
template<typename Data>
vector<Data> preprocess::resample(vector<Data>& vec, double dt){
	double refTime_Stamp = (vec.begin()->timeStamp);
	auto vIter = vec.begin();
	double base = vIter->timeStamp;
	vector<Data> resampled;
	while(vIter < vec.end()){
		resampled.push_back(*vIter);
		refTime_Stamp+= dt;
		vIter = lower_bound(vIter, vec.end(), refTime_Stamp, [](const Data& data, double ref){return data.timeStamp < ref;});
	
	}
	return resampled;
}

//trim all input vectors down to equal size
void preprocess::trimTails(vector<State>& vstate, vector<Control>& vcontrol, vector<Measurement>& vmeasure){
	int minIndex = min(min(vstate.size(), vcontrol.size()), min(vstate.size(), vmeasure.size()));
	vstate = vector<State>(vstate.begin(), vstate.begin()+ minIndex);
	vcontrol = vector<Control>(vcontrol.begin(), vcontrol.begin() + minIndex);
	vmeasure = vector<Measurement>(vmeasure.begin(),vmeasure.begin()+ minIndex);
}

//Make input vectors be of equal length and contain equally spaced(based on their time-stamps) data points
void preprocess::alignTime(vector<State>& vstate, vector<Control>& vcontrol, vector<Measurement>& vmeasure, double dt){
	double beginTime = findMaxStartTime(vstate, vcontrol, vmeasure);
	double endTime = findMinEndTime(vstate, vcontrol, vmeasure);

	//align left ends
	auto stateIterLeft = lower_bound(vstate.begin(), vstate.end(), beginTime, [](const State& st, double ref){return st.timeStamp <= ref;});
	auto controlIterLeft = lower_bound(vcontrol.begin(), vcontrol.end(), beginTime, [](const Control& crt, double ref){return crt.timeStamp <= ref;});
	auto measureIterLeft = lower_bound(vmeasure.begin(), vmeasure.end(), beginTime, [](const Measurement& measure, double ref){return measure.timeStamp <= ref;});
	
	//align right ends
	auto stateIterRight = upper_bound(vstate.begin(), vstate.end(), endTime, [](double ref, const State& st){return  ref <= st.timeStamp ;});
	auto controlIterRight = upper_bound(vcontrol.begin(), vcontrol.end(), endTime, [](double ref, const Control& crt){return  ref <= crt.timeStamp;});
	auto measureIterRight = upper_bound(vmeasure.begin(), vmeasure.end(), endTime, [](double ref, const Measurement& measure){return  ref <= measure.timeStamp;});
	
	//slice vectors
	vstate = vector<State>(stateIterLeft, stateIterRight);
	vcontrol = vector<Control>(controlIterLeft, controlIterRight);
	vmeasure = vector<Measurement>(measureIterLeft,measureIterRight);

	
	//resample data in vectors in increments of dt
	vstate = resample(vstate, dt);
	vcontrol = resample(vcontrol, dt);
	vmeasure = resample(vmeasure,dt);

	//trim excess tails
	trimTails(vstate, vcontrol, vmeasure);
}

//construct a map from subject# to #barcode of landmarks from a file
map<int,int> preprocess::landmarkToBarcode(string fname){
	ifstream rFile(fname);
	string line;
	map<int,int> table;
	if(rFile.is_open()){
		int key = 0;
		int val = 0;
		while(getline(rFile, line)){
			if(line[0] != '#'){ //skip all headers 
				istringstream ss(line);
				ss >> key >> val;
				table[key] = val;
			}
		}//end while		
	}//end if
	else{
		cerr << "Reading Barcode Mapping Failed\n";
	}
	rFile.close();
	return table; 
}

//construct a map from barcodes of landmark to their x-y coordinates 
map<int, Position> preprocess::barcodeToPosition(string fname, map<int, int> landmarkToBarcode ){
	ifstream rFile(fname);
	string line;
	map<int,Position> table;

	if(rFile.is_open()){
		int key = 0;
		double xpos = 0;
		double ypos = 0;

		while(getline(rFile, line)){
			if(line[0] != '#'){ //skip all headers 
				istringstream ss(line);
				ss >> key >> xpos >> ypos; //discard the 2 remaining columns
				table[landmarkToBarcode[key]] = Position(xpos, ypos);
			}
		}//end while		
	}//end if
	else{
		cerr << "Reading Landmark Mapping Failed\n";
	}
	rFile.close();
	return table; 
}



