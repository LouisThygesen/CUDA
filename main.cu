bool DEBUG = true;

// Standard libraries 
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <ctime>
#include <chrono>
#include <map>
#include <algorithm>
#include <random>
#include <cassert>
#include <memory>

// External libraries 
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <nccl.h>
#include <math.h>

// Declarations 
std::string CONFIG_PATH = "./conf/config.csv";

class Config;   
class Logger;          
class Tensor;    
class Dataset;     
class Data_loader;  
class MLP;
class Linear_layer;
class ReLU;

void load_configuration(Config& cfg);
float* allocate_array(size_t x_dim, size_t y_dim, std::string loc);

/****************************************************************************/
/******************************** Utility ***********************************/
/****************************************************************************/

struct Config {
	/* Store hyperparameters */
	size_t num_epochs = 0;
	size_t batch_size = 0;
	float lr = -1.0f;
	std::vector<size_t> hidden = {};
	std::string loc = "";
	std::string log_dir = "";
	std::string data_dir = "";
};

void load_configuration(Config& cfg) {
	/* Reads the configuration file and adds values to the configuration struct */
	std::cout << "Loading configuration...\n" << std::endl;

	// Open configuration file (CSV)
	std::ifstream cfg_file(CONFIG_PATH);

	// Read configuration file line by line.
	std::string line;
	std::string key;
	std::string value;

	while (std::getline(cfg_file, line)) {
		// Split line by comma ("key,value")
		std::istringstream ss(line);
		std::getline(ss, key, ',');
		std::getline(ss, value, ',');

		// Debugging: Print key and value (one line)
		if (DEBUG) {
			std::cout << "Key: " << key << "; Value: " << value << "\n";
		};

		// Create key-value pairs in configuration struct 
		if (key == "num_epochs") {
			cfg.num_epochs = std::stoi(value);

			// Make sure number of epochs is positive
			assert(cfg.num_epochs > 0);
		}
		else if (key == "batch_size") {
			cfg.batch_size = std::stoi(value);

			// Make sure batch size is positive
			assert(cfg.batch_size > 0);
		}
		else if (key == "lr") {
			cfg.lr = std::stof(value);

			// Make sure learning rate is positive
			assert(cfg.lr > 0.0f);
		}
		else if (key == "hidden") {
			// Value is a {x1;x2;...,xn} string which should be saved as a vector of size_t
			std::string hidden_str = value.substr(1, value.size() - 2);
			std::istringstream hidden_ss(hidden_str);
			std::string hidden_value;

			while (std::getline(hidden_ss, hidden_value, ';')) {
				cfg.hidden.push_back(std::stoi(hidden_value));

				// Make sure hidden layer sizes are positive
				assert(cfg.hidden.back() > 0);
			};
		}
		else if (key == "loc") {
			cfg.loc = value;

			// Make sure location name is either cpu, gpu or pinned
			assert(cfg.loc == "cpu" || cfg.loc == "gpu" || cfg.loc == "pinned");
		};
		else if (key == "log_dir") {
			cfg.log_dir = value;
		}
		else if (key == "data_dir") {
			cfg.data_dir = value;
		};
	};
};

/****************************************************************************/

class Logger {
	/* A logger which can save model metrics to a txt file when necessary */

public:
	// Constructor and destructor
	Logger();
	~Logger();

	// Function prototypes
	void log(int epoch, float acc);
	void log_config(Config cfg);

	// Getters
	std::string get_project_name() const { return project_name; };

private:
	// Member variables
	std::string project_name;
	std::string log_path;
	std::ofstream log_file;
};

Logger::Logger() {
	std::cout << "Initializing logger...\n" << std::endl;

	// Get current date and time in format MM-DD-HH-MM-SS (month, day, hour, min, sec) 
	auto now = std::chrono::system_clock::now();
	std::time_t now_c = std::chrono::system_clock::to_time_t(now);

	// don't use localtime because it's not thread safe
	std::tm* now_tm = std::gmtime(&now_c);

	// Convert time to string
	std::string month = std::to_string(now_tm->tm_mon + 1);
	std::string day = std::to_string(now_tm->tm_mday);
	std::string hour = std::to_string(now_tm->tm_hour);
	std::string min = std::to_string(now_tm->tm_min);
	std::string sec = std::to_string(now_tm->tm_sec);

	this->project_name = month + "-" + day + "-" + hour + "-" + min + "-" + sec;

	// Print project name
	std::cout << "Project name: " << this->project_name << "\n" << std::endl;

	// Concatenate LOG_DIR and project_name to create log path (string)
	std::string log_path = LOG_DIR + "/" + this->project_name + ".txt";

	// Open log file in append mode
	this->log_file.open(log_path, std::ios::app);
	this->log_file << "epoch,loss,accuracy,epoch_time\n";
};

Logger::~Logger() {
	/* Close log file */
	this->log_file.close();
};

void Logger::log_config(Config cfg) {
	/* Log configuration to file */
	this->log_file << "Configuration\n";
	this->log_file << "num_epochs," << cfg.num_epochs << "\n";
	this->log_file << "batch_size," << cfg.batch_size << "\n";
	this->log_file << "lr," << cfg.lr << "\n";
	this->log_file << "hidden,";

	for (int i = 0; i < cfg.hidden.size(); i++) {
		this->log_file << cfg.hidden[i] << ";";
	};

	this->log_file << "\n";
	this->log_file << "location_name," << cfg.location_name << "\n";
	this->log_file << "location_id," << cfg.location_id << "\n";
};

void Logger::log(int epoch, float acc) {
	/* Log current epoch, loss and acc */
	this->log_file << epoch << "," << acc  << "\n";
};

/****************************************************************************/

float* allocate_array(size_t x_dim, size_t y_dim, std::string loc) {
	/* Allocate either a vector or matrix as a 1D array as to favour CUDA kernel 
	   configuration (the same memory layout is used for both host and device for 
	   consistency).
	*/
	
    assert (loc == "cpu" || loc == "gpu" || loc == "pinned");

	float* array;

	// Case 1: Host (not pinned)
	if (loc == "cpu") {
		array = new float[x_dim * y_dim];
		assert (array != nullptr);
	}

	// Case 2: Host (pinned)
	else if (loc == "pinned") {
		cudaHostAlloc(&array, x_dim * y_dim * sizeof(float), cudaHostAllocDefault);
		assert (array != nullptr);
	}

	// Case 3: Device (not pinned)
	else if (loc == "gpu") {
		cudaMalloc(&array, x_dim * y_dim * sizeof(float));
		assert (array != nullptr);
	};

	return array;
};

class Tensor {
	/* A matrix class with dynamic allocation of data on host and device */

public:
	// Constructors and destructor
	Tensor(size_t x_dim = 1, size_t y_dim = 1, std::string loc = "cpu");

	// Function prototypes
	void set_data(int mode);
	void to(std::string loc);

	// Getters 
	std::vector<size_t> get_shape() { return { x_dim, y_dim }; };
	std::string get_location() { return loc; };
	float* get_data() { return data; };

private:
	// Pointers to data (allocated on host or device heap). Note that smart pointers 
	// are not an option with CUDA (so we have to make due with regular pointers).
	// This is also why I chose wrap the data pointer in a class (Tensor) - the
	// destructor will free the memory when the object goes out of scope.
	float* data_h;
	float* data_d;

	// Meta data
	size_t x_dim, y_dim;
	std::string loc;
};

Tensor::Tensor(size_t x_dim, size_t y_dim, std::string loc) {

	this->x_dim = x_dim; 
	this->y_dim = y_dim;
	this->loc= loc;

	// Even a vector should be considered as a matrix with one row or column
	assert (x_dim > 0 && y_dim > 0);

	// Allocate matrix or vector in proper location (host or gpu or both)
	if (loc == "cpu") {
		this->data_h = allocate_array(x_dim, y_dim, "cpu");
		this->data_d = nullptr;
	}
	else if (loc == "gpu") {
		this->data_h = nullptr;
		this->data_d = allocate_array(x_dim, y_dim, "gpu");
	}
	else if (loc == "pinned") {
		this->data_h = allocate_array(x_dim, y_dim, "pinned");
		this->data_d = allocate_array(x_dim, y_dim, "gpu");
	};
};

void Tensor::set_data(int mode) {
	/* Initialize matrix as either zeros or random values (depending on input) */

	// Temporary array on host of the same size as the data array in the Tensor object
	float* new_data = new float[x_dim * y_dim];

	// If input mode is 0 (zeros), set new data to zeros
	if (mode == 0) {
		for (int i = 0; i < x_dim; i++) {
			for (int j = 0; j < y_dim; j++) {
				new_data[i * y_dim + j] = 0.0f;
			};
		};
	}

	// If input mode is 1 (random), set new data to random values between 0 and 1
	else if (mode == 1) {
		for (int i = 0; i < x_dim; i++) {
			for (int j = 0; j < y_dim; j++) {
				new_data[i * y_dim + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			};
		};
	};

	// If device memory is used, copy data from new data to data_d
	// and free temp memory (and remove pointer to temp data)
	if (loc == "gpu") {
		cudaMemcpy(data_d, new_data, x_dim * y_dim * sizeof(float), cudaMemcpyHostToDevice);
		delete new_data;
	};

	// If host memory is used, set data_h to new data and remove old pointer
	if (loc == "cpu") {
		if (data_h != nullptr) {
			delete data_h;
		};

		data_h = new_data;
	};

	// If pinned memory is used, copy new data to data_d  and make data_h point to new data
	if (loc == "pinned") {
		// Copy new data to device memory
		cudaMemcpy(data_d, new_data, x_dim * y_dim * sizeof(float), cudaMemcpyHostToDevice);

		// Make data_h point to new data
		if (data_h != nullptr) {
			delete data_h;
		};

		data_h = new_data;
	};
};

void Tensor::to(std::string dest) {
	/* Copy data from pinned host to device or vice versa */

	assert (this->loc == "pinned");

	if (dest == "host") {
		cudaMemcpy(data_h, data_d, x_dim * y_dim * sizeof(float), cudaMemcpyDeviceToHost);
	}
	else if (dest == "device") {
		cudaMemcpy(data_d, data_h, x_dim * y_dim * sizeof(float), cudaMemcpyHostToDevice);
	};
};

/****************************************************************************/
/******************************* Data ***************************************/
/****************************************************************************/

class Dataset {
	/* Class to store dataset information and provide methods to access data */

public:
	// Constructor 
	Dataset(std::string partition_dir);

	// Function prototypes
	std::pair<cv::Mat, int> get_item(int idx);

	// Getters
	int get_len() { return len; };
	size_t get_num_classes() { return num_classes; };
	size_t get_img_size() { return img_size; };

	// Storage on the heap (dynamic)
	std::vector<std::string> img_paths;
	std::map<std::string, int> path2label;

private:
	// Storage on the stack (fixed size)
	int len;
	size_t num_classes;
	std::string partition_dir;
	size_t img_size;
};

Dataset::Dataset(std::string partition_dir) {
	/* Find image paths and create path2label map */

	// Temporary storages 
	std::vector<std::string> all_paths;
	std::string labels_path;

	// Glob all paths in data folder (images and label file)
	cv::glob(partition_dir, all_paths, false);

	// Extract image paths and label csv path
	for (int i = 0; i < all_paths.size(); i++) {
		if (all_paths[i].find(".png") != std::string::npos) {
			this->img_paths.push_back(all_paths[i]);
		}
		if (all_paths[i].find(".csv") != std::string::npos) {
			labels_path = all_paths[i];
		};
	};

	// Set size of dataset
	this->len = img_paths.size();

	// Read labels from csv file and create path2label map
	// (ignore first line as it is a header)
	std::ifstream labels_file(labels_path);
	std::string line;

	while (std::getline(labels_file, line)) {
		if (line.find("path") == std::string::npos) {
			std::istringstream ss(line);
			std::string path;
			std::string label;

			std::getline(ss, path, ',');
			std::getline(ss, label, ',');

			// Print path and label
			this->path2label[path] = std::stoi(label);
		};
	};

	// Get unique labels
	std::vector<int> unique_labels;
	for (auto const& x : path2label) {
		unique_labels.push_back(x.second);
	};

	// Remove duplicates
	std::sort(unique_labels.begin(), unique_labels.end());
	unique_labels.erase(std::unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());

	// Set number of classes
	this->num_classes = unique_labels.size();

	// Get image size (flattened)
	cv::Mat img = cv::imread(img_paths[0], cv::IMREAD_GRAYSCALE);
	this->img_size = img.rows * img.cols;
};

std::pair<cv::Mat, int> Dataset::get_item(int idx) {
	/* Get image and label at dataset (path list) index. Use unique pointer to 
	   ensure that the space on the heap is freed after it has been used for 
	   augmentation etc. in the dataloader
	*/

	std::string img_path = img_paths[idx];

	// Load image data on the heap (for cheap pass by value for wrapper class instance)
	cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);	 // Unsigned int 8-bit (0-255)
	int label = this->path2label[img_path];

	return std::make_pair(img, label);
};

/****************************************************************************/

class Data_loader {
	/* The main function of the data loader is to provide batches of data to the model.
	   It does this by calling get_item on the dataset object to get a single image and label,
	   and then stacking these images and labels into a batch.
	*/

public:
	// Constructor
	Data_loader(Dataset dataset, size_t batch_size, bool shuffle, std::string loc);

	// Function prototypes
	void prepare_epoch();
	auto get_batch(int batch_idx);
	size_t get_num_batches() { return this->batch_indices.size(); };

private:
	// Storage on the stack
	size_t batch_size;
	size_t num_classes;
	bool shuffle;
	Dataset dataset;

	// Storge on the heap 
	std::vector<int> obs_indices;
	std::vector<std::vector<int>> all_batch_indices;
	std::vector<int> batch_indices;
	size_t img_size;

	// Tensors for images (flattened) and labels (one-hot encoded)
	Tensor X;
	Tensor y;
};

Data_loader::Data_loader(Dataset dataset, size_t batch_size, bool shuffle, std::string loc) : dataset{ dataset }, batch_size{ batch_size }, shuffle{ shuffle } {
	std::cout << "Initializing data loader...\n" << std::endl;

	this->batch_size = batch_size;
	this->shuffle = shuffle;
	this->dataset = dataset;

	// Get image size (flattened) and number of classes from dataset
	this->img_size = dataset.get_img_size();
	this->num_classes = dataset.get_num_classes();

	// Create indices (list of integers) from 0 to len(dataset)
	this->obs_indices.reserve(dataset.get_len());

	for (int i = 0; i < dataset.get_len(); i++) {
		this->obs_indices.push_back(i);  
	};

	// Create batches of indices (list of lists)
	prepare_epoch();

	// Allocate pinned memory for images and labels on host (standard and last batch) and if 
	// main location is device, then keep same size memory allocated (so new batches
	// only update the storage when moved from cpu to device - no reallocation). 
	X = Tensor(this->img_size, this->batch_size, "pinned");
	y = Tensor(this->num_classes, this->batch_size, "pinned");
};

void Data_loader::prepare_epoch() {
	/* Shuffle indices and split into batches */

	// Shuffle indices
	std::random_device rnd_device;
	std::mt19937 mersenne_engine{ rnd_device() }; 
	std::shuffle(begin(this->obs_indices), end(this->obs_indices), mersenne_engine);

	// Split into batches of size batch_size (last batch might be smaller)
	for (int i = 0; i < this->obs_indices.size(); i += this->batch_size) {
		std::vector<int> batch(this->obs_indices.begin() + i, this->obs_indices.begin() + i + this->batch_size);
		this->all_batch_indices.push_back(batch);
	};

	assert (this->all_batch_indices[0].size() == this->batch_size);
};

auto Data_loader::get_batch(int batch_idx) {
	// Get observation indices for batch
	this->batch_indices = this->all_batch_indices[batch_idx];
 
	// Update tensors with new batch data (flat images, labels)  
	// [NUMA optimized due to where the function is first called]
	for (int obs_idx = 0; obs_idx < this->batch_size; obs_idx++) { 
		// Get image and label from dataset 
		std::pair<cv::Mat, int> obs = this->dataset.get_item(batch_indices[obs_idx]);
		cv::Mat img = obs.first;
		int label = obs.second;

		// Convert image to float [0;1] 
		img.convertTo(img, CV_32F, 1.0 / 255.0);

		// One-hot encode label and save as y tensor data 
		for (int i = 0; i < this->num_classes; i++) {
			y.get_data()[i][obs_idx] = (i == label) ? 1.0f : 0.0f;
		};
		
		// Flatten image and and save as X tensor data 
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				X.get_data()[i * img.cols + j][obs_idx] = img.at<float>(i, j);
			};
		};
	};

	// Return pair of tensors (images and labels) 
	return std::make_pair(X, y); 
};

/****************************************************************************/
/******************************** Model *************************************/
/****************************************************************************/

class ReLU {
	/* Class to store activation layer information and perform forward and backward passes */

public:
	// Constructor
	ReLU(size_t in_features, std::string loc, size_t batch_size);

	// Function prototypes
	Tensor forward(Tensor in);
	Tensor backward(Tensor grad);

private:
	// Hyperparameters and meta data
	std::string loc;
	size_t in_features;
	size_t batch_size;

	// Kernel configurations
	dim3 grid_size_fw;
	dim3 block_size_fw;

	dim3 grid_size_bw;
	dim3 block_size_bw;

	// Tensor storages
	Tensor in;
	Tensor out;
	Tensor grad_in;
};

ReLU::ReLU(size_t in_features, std::string loc, size_t batch_size) : in_features{ in_features }, loc{ loc }, batch_size{ batch_size } {
	std::cout << "Initializing ReLU layer...\n" << std::endl;

	// Allocate input, output and gradient tensors
	this->in = Tensor(in_features, batch_size, loc);
	this->out = Tensor(in_features, batch_size, loc);
	this->grad_in = Tensor(in_features, batch_size, loc);

	// Initialize grad to zeros (important)
	this->grad_in.set_data(0);

	// Initialize input and output to zeros (for debugging purposes)
	this->in.set_data(0);
	this->out.set_data(0);
};

void ReLU::reset_grads() {
	/* Reset gradients to zeros */
	grad_in.set_data(0);
};

Tensor ReLU::forward(Tensor x) {
	/* Perform forward pass on batch */

	// Save input for backward pass
	this->in = x;

	// Perform forward pass
	if (in.get_location() == "cpu") {
		_relu_host_fw(in.get_data(), out.get_data(), in_features, batch_size);
	}
	else if (in.get_location() == "gpu") {
		_relu_device_fw<<<this->grid_size_fw, this->block_size_bw>>>(in.get_data(), out.get_data(), in_features, batch_size);
	};

	return out;
}

Tensor ReLU::backward(Tensor grad_out) {
	/* Perform backward pass on batch */

	if (grad_out.get_location() == "host") {
		_relu_host_bw(in.get_data(), out.get_data(), grad_out.get_data(), grad_in.get_data(), in_features, batch_size);
	}
	else if (grad.get_location().get_name() == "device") {
		_relu_device_bw<<<this->grid_size_bw, this->block_size_bw>>>(in.get_data(), out.get_data(), grad_out.get_data(), grad_in.get_data(), in_features, batch_size);
	};

	return grad_in;
};

__host__ void _relu_host_fw(float* t_in, float* t_out, size_t in_features, size_t batch_size) {
	/* Perform forward pass on batch (ReLU) */

	for (int i = 0; i < in_features; i++) {
		for (int j = 0; j < batch_size; j++) {
			t_out[i * batch_size + j] = (t_in[i * batch_size + j] > 0) ? t_in[i * batch_size + j] : 0;
		};
	};
};

__global__ void _relu_host_bw(float* t_in, float* t_out, float* t_grad_in, float* t_grad_out, size_t in_features, size_t batch_size) {
	/* Perform backward pass on batch (ReLU) */

	for (int i = 0; i < in_features; i++) {
		for (int j = 0; j < batch_size; j++) {
			t_grad_in[i * batch_size + j] = (t_out[i * batch_size + j] > 0) ? t_grad_out[i * batch_size + j] : 0;
		};
	};
};

__host__ void _relu_device_fw(float* t_in, float* t_out, size_t in_features, size_t batch_size) {
	

};

__global__ void _relu_device_bw(float* t_in, float* t_out, float* t_grad_in, float* t_grad_out, size_t in_features, size_t batch_size) {

};

/****************************************************************************/

class Linear_layer {
	/* Class to store layer information and perform forward and backward passes */

public:
	// Constructor
	Linear_layer(size_t in_features, size_t out_features, std::string loc, size_t batch_size);

	// Function prototypes
	Tensor forward(Tensor in);
	Tensor backward(Tensor grad, Tensor grad_weights, Tensor grad_bias);

	// Getters
	std::vector<Tensor> get_weights { return weights; };
	std::vector<Tensor> get_bias { return bias; };

	std::vector<Tensor> get_grad_weights { return grad_weights; };
	std::vector<Tensor> get_grad_bias { return grad_bias; };

private:
	// Hyperparameters and meta data
	size_t in_features, out_features;
	size_t max_batch_size;
	std::string loc;
	bool pinned_output;

	// Kernel configurations
	dim3 grid_size_fw1;
	dim3 block_size_fw1;

	dim3 grid_size_fw2;
	dim3 block_size_fw2;

	dim3 grid_size_bw1;
	dim3 block_size_bw1;

	dim3 grid_size_bw2;
	dim3 block_size_bw2;

	// Tensor storages
	Tensor weights;   // W.T
	Tensor bias;      // b

	Tensor in;	      // x
	Tensor out1;      // W.T * x 
	Tensor out2;      // W.T * x + b

	Tensor grad_weight;  // dW
	Tensor grad_bias;     // db

	Tensor grad_in;      // dx
	Tensor grad_out1;      // d(W.T * x)
};

Linear_layer::Linear_layer(size_t in_features, size_t out_features, std::string loc, size_t batch_size) : loc{ loc }, in_features{ in_features }, out_features{ out_features }, max_batch_size{ batch_size } {
	std::cout << "Initializing layer...\n" << std::endl;

	assert (in_features > 0 && out_features > 0 && batch_size > 0);
	assert (loc == "cpu" || loc == "gpu");

	// Allocate weights and biases 
	this->weights = Tensor(out_features, in_features, loc);
	this->bias = Tensor(out_features, 1, loc);

	// Allocate output tensor 
	this->out1 = Tensor(out_features, batch_size, loc);

	if (loc == "cpu") {
		this->out2 = Tensor(out_features, batch_size, "cpu");
		this->pinned_output = false;
	}
	else if (loc == "gpu") {
		this->out2 = Tensor(out_features, batch_size, "pinned");
		this->pinned_output = true;
	};

	// Allocate gradient of loss w.r.t. parameters tensors 
	this->grad_weight = Tensor(out_features, in_features, loc);
	this->grad_bias = Tensor(out_features, 1, loc);

	// Allocate gradient of loss w.r.t. input tensor (step 1 and 2)
	this->grad_in = Tensor(in_features, batch_size, loc);
	this->grad_out1 = Tensor(out_features, batch_size, loc);  // TODO: Is this correct size?

	// Initialize weights, bias and gradients to zeros (important)
	this->weights.set_data(1);
	this->bias.set_data(0);

	this->grad_weight.set_data(0);
	this->grad_bias.set_data(0);

	this->grad_in.set_data(0);
	this->grad_out1.set_data(0);

	// Initialize input and step outputs as zeros (only for debugging purposes)
	this->in.set_data(0);

	this->out1.set_data(0);
	this->out2.set_data(0);

	// Compute optimal kernel configuration for forward pass (step 1)



	// Compute optimal kernel configuration for forward pass (step 2)


	// Compute optimal kernel configuration for backward pass (step 1)


	// Compute optimal kernel configuration for backward pass (step 2)


};

void Linear_layer::reset_grads() {
	/* Reset gradients to zeros */

	grad_weight.set_data(0);
	grad_bias.set_data(0);
	grad_in.set_data(0);
	grad_out1.set_data(0);
};

Tensor Linear_layer::forward(Tensor x) {
	/* Perform forward pass on batch where step 1 is out1=(W.T * x) and step 2 is out2=(out1 + expand(b)) */

	// Save x for backward pass
	this->in = x;

	// Perform forward pass 
	if (in.get_location() == "cpu") {
		_ll_host_fwd_1(in.get_data(), weights.get_data(), out1.get_data(), in_features, out_features, batch_size);
		_ll_host_fwd_2(out1.get_data(), bias.get_data(), out2.get_data(), in_features, out_features, batch_size);
	}

	else if (in.get_location() == "gpu") {
		_ll_device_fwd_1<<<this->grid_size_fw1, this->block_size_fw1>>>(in.get_data(), weights.get_data(), out1.get_data(), in_features, out_features, batch_size);
		_ll_device_fwd_2<<<this->grid_size_fw2, this->block_size_fw2>>>(out1.get_data(), bias.get_data(), out2.get_data(), in_features, out_features, batch_size);
	};

	return out2;
};

Tensor Linear_layer::backward(Tensor grad_out) {
	/* Perform backward pass */

	if (grad.get_location()== "cpu") {
		_ll_host_bwd_2(out1.get_data(), bias.get_data(), out2.get_data(), grad_bias.get_data(), grad_out2.get_data(), grad_out1.get_data(), in_features, out_features, batch_size);
		_ll_host_bwd_1(in.get_data(), weights.get_data(), out1.get_data(), grad_weights.get_data(), grad_out1.get_data(), grad_in.get_data(), in_features, out_features, batch_size);
	}

	else if (grad.get_location() == "gpu") {
		_ll_device_bwd_2<<<this->grid_size_bw2, this->block_size_bw2>>>(out1.get_data(), bias.get_data(), out2.get_data(), grad_bias.get_data(), grad_out2.get_data(), grad_out1.get_data(), in_features, out_features, batch_size);
		_ll_device_bwd_1<<<this->grid_size_bw1, this->block_size_bw1>>>(in.get_data(), weights.get_data(), out1.get_data(), grad_weights.get_data(), grad_out1.get_data(), grad_in.get_data(), in_features, out_features, batch_size);
	};

	return grad_in;
};

__host__ void _ll_host_fwd(float* t_in, float* t_weights, float* t_out1, size_t in_features, size_t out_features, size_t batch_size) {
	/* Perform forward pass on batch (all steps of linear layer) */

	// W.T * x
	for (int i = 0; i < out_features; i++) {
		for (int j = 0; j < batch_size; j++) {
			t_out1[i * batch_size + j] = 0.0f;

			for (int k = 0; k < in_features; k++) {
				t_out1[i * batch_size + j] += t_weights[i * in_features + k] * t_in[k * batch_size + j];
			};
		};
	};

	// (W.T * x) + b 
	for (int i = 0; i < out_features; i++) {
		for (int j = 0; j < batch_size; j++) {
			t_out1[i * batch_size + j] += t_bias[i];
		};
	};
};

__host__ void _ll_host_bwd(float* t_out1, float* t_bias, float* t_out2, float* t_grad_bias, float* t_grad_out1, size_t in_features, size_t out_features, size_t batch_size) {
	/* Perform backward pass on batch (all steps of linear layer) */

	// Compute gradient of loss w.r.t. bias
	for (int i = 0; i < out_features; i++) {
		t_grad_bias[i] = 0.0f;

		for (int j = 0; j < batch_size; j++) {
			t_grad_bias[i] += t_grad_out1[i * batch_size + j];
		};
	};

	// Compute gradient of loss w.r.t. weights
	for (int i = 0; i < out_features; i++) {
		for (int j = 0; j < in_features; j++) {
			t_grad_weights[i * in_features + j] = 0.0f;

			for (int k = 0; k < batch_size; k++) {
				t_grad_weights[i * in_features + j] += t_grad_out1[i * batch_size + k] * t_in[j * batch_size + k];
			};
		};
	};

	// Compute gradient of loss w.r.t. input
	for (int i = 0; i < in_features; i++) {
		for (int j = 0; j < batch_size; j++) {
			t_grad_in[i * batch_size + j] = 0.0f;

			for (int k = 0; k < out_features; k++) {
				t_grad_in[i * batch_size + j] += t_weights[k * in_features + i] * t_grad_out1[k * batch_size + j];
			};
		};
	};


};

__global__ void _ll_device_fwd_1(float* t_in, float* t_weights, float* t_out1, size_t in_features, size_t out_features, size_t batch_size) {
	
};

__global__ void _ll_device_fwd_2(float* t_out1, float* t_bias, float* t_out2, size_t in_features, size_t out_features, size_t batch_size) {
	
};

__global__ void _ll_device_bwd_1(float* t_in, float* t_weights, float* t_out1, float* t_grad_weights, float* t_grad_out1, size_t in_features, size_t out_features, size_t batch_size) {
	
};

__global__ void _ll_device_bwd_2(float* t_out1, float* t_bias, float* t_out2, float* t_grad_bias, float* t_grad_out1, size_t in_features, size_t out_features, size_t batch_size) {
	
};

/****************************************************************************/

class MLP {
	/* Class to store neural network configuration and perform forward and backward passes */

public:
	// Constructor
	MLP(size_t in_dim, std::vector<size_t> hidden_dims, size_t out_dim, std::string loc, size_t batch_size);

	// Function prototypes
	Tensor forward(Tensor x);
	Tensor backward(Tensor grad);

	// Getters
	std::vector<Tensor> get_weights() { return layer_weights; };
	std::vector<Tensor> get_biases() { return layer_bias; };

	std::vector<Tensor> get_grad_weights() { return layer_grad_weights; };
	std::vector<Tensor> get_grad_biases() { return layer_grad_bias; };

private:
	// Hyperparameters and meta data
	std::string loc;
	size_t batch_size;

	size_t in_dim;
	std::vector<size_t> hidden_dims;
	size_t out_dim;

	// Module lists 
	std::vector<Linear_layer> layers;
	std::vector<ReLU> activations;

	// Tensor storages
	std::vector<Tensor> layer_weights;
	std::vector<Tensor> layer_bias;

	std::vector<Tensor> layer_grad_weights;
	std::vector<Tensor> layer_grad_bias;

	Tensor out;
	Tensor grad_out;
};

MLP::MLP(size_t in_dim, std::vector<size_t> hidden_dims, size_t out_dim, std::string loc, size_t batch_size) : in_dim{ in_dim }, hidden_dims{ hidden_dims }, out_dim{ out_dim }, loc{ loc }, batch_size{ batch_size } {
	std::cout << "Initializing neural network...\n" << std::endl;

	// Initialize network layers and activations
	layers.push_back(Linear_layer(in_dim, hidden_dims[0], loc, batch_size));
	activations.push_back(ReLU(hidden_dims[0], loc, batch_size));

	for (int i = 0; i < hidden_dims.size() - 1; i++) {
		layers.push_back(Linear_layer(hidden_dims[i], hidden_dims[i + 1], loc, batch_size));
		activations.push_back(ReLU(hidden_dims[i+1], loc, batch_size));
	};

	layers.push_back(Linear_layer(hidden_dims.back(), out_dim, loc, batch_size));

	// Get weight tensors for each layer and collect in vector
	for (int i = 0; i < layers.size(); i++) {
		layer_weights.push_back(layers[i].get_weights());
	}

	// Get bias tensors for each layer and collect in vector
	for (int i = 0; i < layers.size(); i++) {
		layer_bias.push_back(layers[i].get_bias());
	}

	// Get gradient of weights tensors for each layer and collect in vector
	for (int i = 0; i < layers.size(); i++) {
		layer_grad_weights.push_back(layers[i].get_grad_weights());
	}

	// Get gradient of bias tensors for each layer and collect in vector
	for (int i = 0; i < layers.size(); i++) {
		layer_grad_bias.push_back(layers[i].get_grad_bias());
	}
};

void MLP::reset_grads() {
	/* Reset gradients to zeros */

	for (int i = 0; i < layers.size() - 1; i++) {
		layers[i].reset_grads();
		activations[i].reset_grads();
	};

	layers.back().reset_grads();
};

Tensor MLP::forward(Tensor x) {
	/* Perform forward pass */

	out = x;

	for (int i = 0; i < layers.size() - 1; i++) {
		out = layers[i].forward(out);
		out = activations[i].forward(out);
	};

	out = layers.back().forward(out);

	return out;
};

Tensor MLP::backward(Tensor grad_out) {
	/* Perform backward pass */

	for (int i = layers.size() - 1; i >= 1; i--) {
		this->grad_out = activations[i].backward(this->grad_out);
		this->grad_out = layers[i].backward(this->grad_out);
	};

	this->grad_out = layers[0].backward(this->grad_out);

	return this->grad_out;
};

/****************************************************************************/
/******************************* Loss ***************************************/
/****************************************************************************/

class Cross_entropy {
	/* Class to store loss function information and perform forward and backward passes */

public:
	// Constructor
	Cross_entropy(size_t in_features, std::string loc, size_t batch_size);

	// Function prototypes
	Tensor forward(Tensor x, Tensor y);
	Tensor backward(Tensor grad_out);

private:
	// Hyperparameters and meta data
	size_t num_classes;
	size_t batch_size;
	std::string loc;

	// Kernel configurations
	dim3 grid_size_fw1;
	dim3 block_size_fw1;

	dim3 grid_size_fw2;
	dim3 block_size_fw2;

	dim3 grid_size_fw3;
	dim3 block_size_fw3;

	dim3 grid_size_fw4;
	dim3 block_size_fw4;

	dim3 grid_size_fw5;
	dim3 block_size_fw5;

	// Tensor storages
	Tensor pred_logits;     // Logits from model
	Tensor true_probs;      // One-hot encoded labels

	Tensor out1;            
	Tensor out2;
	Tensor out3;
	Tensor out4;
	Tensor out5;            // Final output (loss)

	Tensor grad_out1;
	Tensor grad_out2;
	Tensor grad_out3;
	Tensor grad_out4;
	Tensor grad_out5;       // Gradient of loss w.r.t. logits (input)
};

Cross_entropy::Cross_entropy(size_t num_classes, std::string loc, size_t batch_size) : loc { loc }, num_classes{ num_classes }, batch_size{ batch_size } {
	std::cout << "Initializing Cross-entropy loss function...\n" << std::endl;

	// Allocate intermediate output tensors (not pinned - only used for computation)
	this->out1 = Tensor(num_classes, batch_size, loc);
	this->out2 = Tensor(num_classes, batch_size, loc);
	this->out3 = Tensor(num_classes, batch_size, loc);
	this->out4 = Tensor(num_classes, batch_size, loc);

	// Allocate final output (loss) tensor (pinned if device is main location)
	if (loc == "cpu") {
		this->out5 = Tensor(1, batch_size, "cpu");
	}
	else if (loc == "gpu") {
		this->out5 = Tensor(1, batch_size, "pinned");
	};

	// Allocate gradient tensors (not pinned - only used for computation)
	this->grad_out1 = Tensor(num_classes, batch_size, loc);
	this->grad_out2 = Tensor(num_classes, batch_size, loc);
	this->grad_out3 = Tensor(num_classes, batch_size, loc);
	this->grad_out4 = Tensor(num_classes, batch_size, loc);
	this->grad_out5 = Tensor(num_classes, batch_size, loc);

	// Initialize grad to zeros (important)
	this->grad_out1.set_data(0);
	this->grad_out2.set_data(0);
	this->grad_out3.set_data(0);
	this->grad_out4.set_data(0);
	this->grad_out5.set_data(0);

	// Initialize output to zeros (for debugging purposes)
	this->out1.set_data(0);
	this->out2.set_data(0);
	this->out3.set_data(0);
	this->out4.set_data(0);
	this->out5.set_data(0);

	// Compute optimal kernel configuration for forward pass (step 1)

	// Compute optimal kernel configuration for forward pass (step 2)

	// Compute optimal kernel configuration for forward pass (step 3)

	// Compute optimal kernel configuration for forward pass (step 4)

	// Compute optimal kernel configuration for forward pass (step 5)

};

void Cross_entropy::reset_grads() {
	/* Reset gradients to zeros */

	this->grad_out1.set_data(0);
	this->grad_out2.set_data(0);
	this->grad_out3.set_data(0);
	this->grad_out4.set_data(0);
	this->grad_out5.set_data(0);
};

Tensor Cross_entropy::forward(Tensor x, Tensor y) {
	/* Perform forward pass through a computationally stable version of cross 
	   entropy operating on logits instead of softmax output (using log-sum-exp 
	   trick). 

	   General cross entropy: L = -sum(y_j * log(yhat_j))

	   The output from the model is not softmaxed, but rather the logits. The
	   softmax function is numerically unstable for large values, so we use the
	   log-sum-exp trick to compute the softmax probabilities.

	   Numerically unstable softmax: log(yhat_j) = log(exp(x_j) / sum(exp(x_k)))
	   Numerically stable softmax:   log(yhat_j) = x_j - log(sum(exp(x_k)))

	   Note that here the term (log(sum(exp(x_k))) is the same for all classes
	   (used later). Insert log(yhat_j) into the cross entropy formula:

	   Stable cross entropy: L = -sum(y_j * (x_j - log(sum(exp(x_k))))

	   To compute this loss using CUDA efficiently we need to split the forward
	   pass into multiple kernels (steps) similar to how we used 2 steps in the
	   implementation of the linear layer. For each prediction (one-hot-encoding)
	   in the batch we need to compute the loss using the following steps:

	   Step 1: Compute the exp(x) vector (using element-wise kernel) 
	   Step 2: Compute the s_1 = sum(exp(x_k)) scalar (using reduction kernel)
	   Step 3: Compute the s_2 = log(s) scalar (apply log to scalar from step 2)
	   Step 4: Compute y_j * (x_j - s_2) (using element-wise/dot kernel)
	   Step 5: Compute the sum of the loss (using reduction kernel again)

	   The forward kernel configuration can be chosen so that the number of threads
	   is equal to the number of classes (num_classes) and the number of blocks is
	   equal to the number of observations in the batch (batch_size). This way we
	   can compute the loss for each observation in the batch in parallel.
	*/

	// Save logits and labels for backward pass
	this->pred_logits = x;
	this->true_probs = y;

	// Forward pass
	if (x.get_location() == "cpu") {
		_ce_host_fwd_1(pred_logits.get_data(), out1.get_data(), num_classes, batch_size);
		_ce_host_fwd_2(out1.get_data(), out2.get_data(), num_classes, batch_size);
		_ce_host_fwd_3(out2.get_data(), out3.get_data(), num_classes, batch_size);
		_ce_host_fwd_4(out3.get_data(), out4.get_data(), true_probs.get_data(), num_classes, batch_size);
		_ce_host_fwd_2(out4.get_data(), out5.get_data(), num_classes, batch_size);
	}
	else if (x.get_location() == "gpu") {
		_ce_device_fwd_1<<<grid_size_fwd_1, block_size_fwd_1>>>(pred_logits.get_data(), out1.get_data(), num_classes, batch_size);
		_ce_device_fwd_2<<<grid_size_fwd_2, block_size_fwd_2>>>(out1.get_data(), out2.get_data(), num_classes, batch_size);
		_ce_device_fwd_3<<<grid_size_fwd_3, block_size_fwd_3>>>(out2.get_data(), out3.get_data(), num_classes, batch_size);
		_ce_device_fwd_4<<<grid_size_fwd_4, block_size_fwd_4>>>(out3.get_data(), out4.get_data(), true_probs.get_data(), num_classes, batch_size);
		_ce_device_fwd_2<<<grid_size_fwd_5, block_size_fwd_5>>>(out4.get_data(), out5.get_data(), num_classes, batch_size);
	};

	return out5;
};

Tensor Cross_entropy::backward(Tensor grad_out5) {
	/* Perform backward pass through computationally stable cross entropy loss. 
	   To match the forward pass, the backward pass is split into 5 steps:

	   Step 5: Compute the gradient of the loss w.r.t. the logits (x_j - s_2)
	   Step 4: Compute the gradient of the loss w.r.t. the scalar s_2
	   Step 3: Compute the gradient of the loss w.r.t. the sum of exp(x_k)
	   Step 2: Compute the gradient of the loss w.r.t. the exp(x_k) vector
	   Step 1: Compute the gradient of the loss w.r.t. the logits (x_j)
	*/
	if (grad.get_location().get_name() == "cpu") {
		_ce_host_bwd_5(out4.get_data(), grad_out5.get_data(), num_classes, batch_size);
		_ce_host_bwd_4(out3.get_data(), out4.get_data(), grad_out4.get_data(), num_classes, batch_size);
		_ce_host_bwd_3(out2.get_data(), out3.get_data(), grad_out3.get_data(), num_classes, batch_size);
		_ce_host_bwd_2(out1.get_data(), out2.get_data(), grad_out2.get_data(), num_classes, batch_size);
		_ce_host_bwd_1(pred_logits.get_data(), out1.get_data(), grad_out1.get_data(), num_classes, batch_size);	
	}
	else if (grad.get_location().get_name() == "gpu") {
		_ce_device_bwd_5<<<grid_size_bwd_5, block_size_bwd_5>>>(out4.get_data(), grad_out5.get_data(), num_classes, batch_size);
		_ce_device_bwd_4<<<grid_size_bwd_4, block_size_bwd_4>>>(out3.get_data(), out4.get_data(), grad_out4.get_data(), num_classes, batch_size);
		_ce_device_bwd_3<<<grid_size_bwd_3, block_size_bwd_3>>>(out2.get_data(), out3.get_data(), grad_out3.get_data(), num_classes, batch_size);
		_ce_device_bwd_2<<<grid_size_bwd_2, block_size_bwd_2>>>(out1.get_data(), out2.get_data(), grad_out2.get_data(), num_classes, batch_size);
		_ce_device_bwd_1<<<grid_size_bwd_1, block_size_bwd_1>>>(pred_logits.get_data(), out1.get_data(), grad_out1.get_data(), num_classes, batch_size);
	};

	return grad_out1;
};

__host__ float* _ce_host_fwd(float* t_out4, float* t_out5, size_t num_classes, size_t batch_size) {
	/* Compute all steps of the forward pass of the stable cross entropy loss: L = -sum_j(y_j * (x_j - log(sum_k(exp(x_k))))*/

	// Compute the sum of exp(x_k) for each observation in the batch
	for (int i = 0; i < batch_size; i++) {
		float sum_exp = 0.0f;
		for (int j = 0; j < num_classes; j++) {
			sum_exp += exp(t_out4[i * num_classes + j]);
		}
		t_out5[i] = sum_exp;
	}

	// Compute the loss for each observation in the batch
	for (int i = 0; i < batch_size; i++) {
		float loss = 0.0f;
		for (int j = 0; j < num_classes; j++) {
			float y_j = t_out1[i * num_classes + j];
			float x_j = t_pred_logits[i * num_classes + j];
			float log_sum_exp = log(t_out5[i]);
			loss -= y_j * (x_j - log_sum_exp);
		}
		t_out4[i] = loss;
	}

	return t_out4;








};

__host__ float* _ce_host_bwd(float* t_pred_logits, float* t_out1, float* t_grad_out1, size_t num_classes, size_t batch_size) {
	

};

__global__ float* _ce_device_fwd_1(float* t_pred_logits, float* t_out1, size_t num_classes, size_t batch_size) {
	/* Step 1: Compute the exp(x) vector (using element-wise kernel) */

};

__global__ float* _ce_device_fwd_2(float* t_out1, float* t_out2, size_t num_classes, size_t batch_size) {
	/* Step 2: Compute the s_1 = sum(exp(x_k)) scalar (using reduction kernel) */

};

__global__ float* _ce_device_fwd_3(float* t_out2, float* t_out3, size_t num_classes, size_t batch_size) {
	/* Step 3: Compute the s_2 = log(s) scalar (apply log to scalar from step 2) */

};

__global__ float* _ce_device_fwd_4(float* t_out3, float* t_out4, float* t_true_probs, size_t num_classes, size_t batch_size) {
	/* Step 4: Compute y_j * (x_j - s_2) (using element-wise/dot kernel) */

};

__global__ float* _ce_device_bwd_1(float* t_pred_logits, float* t_out1, float* t_grad_out1, size_t num_classes, size_t batch_size) {
	/* Step 1: Compute the gradient of the loss w.r.t. the logits (x_j) */

};

__global__ float* _ce_device_bwd_2(float* t_out1, float* t_out2, float* t_grad_out2, size_t num_classes, size_t batch_size) {
	/* Step 2: Compute the gradient of the loss w.r.t. the exp(x_k) vector */

};

__global__ float* _ce_device_bwd_3(float* t_out2, float* t_out3, float* t_grad_out3, size_t num_classes, size_t batch_size) {
	/* Step 3: Compute the gradient of the loss w.r.t. the sum of exp(x_k) */

};

__global__ float* _ce_device_bwd_4(float* t_out3, float* t_out4, float* t_grad_out4, size_t num_classes, size_t batch_size) {
	/* Step 4: Compute the gradient of the loss w.r.t. the scalar s_2 */

};

/****************************************************************************/
/******************************* Optimizer **********************************/
/****************************************************************************/

class SGD {
	/* Class to store optimizer information and perform optimization steps */

public:
	// Constructor
	SGD(float lr, MLP* model, Cross_entropy* loss_fn, std::string loc);

	// Function prototypes
	void step();
	void reset_grads();

private:
	// Store  hyperparameters and meta data
	float lr;
	std::string loc;

	// Store reference to model and loss function
	MLP* model;
	Cross_entropy* loss_fn;

	// Kernel configurations

	// Tensor storages
	std::vector<Tensor> weight;
	std::vector<Tensor> bias;
	std::vector<Tensor> grad_weight;
	std::vector<Tensor> grad_bias;
};

SGD::SGD(float lr, MLP* model, Cross_entropy* loss_fn, std::string loc) : lr{ lr }, loc{ loc } model{ model }, loss_fn{ loss_fn } {
	std::cout << "Initializing optimizer...\n" << std::endl;

	// Get weights and biases from model
	weights = model->get_weights();
	bias = model->get_biases();

	// Get gradients of weights and biases from model
	grad_weights = model->get_grad_weights();
	grad_bias = model->get_grad_biases();
};

void SGD::reset_grads() {
	/* Reset gradients to zero */

	this->model->reset_grads();
	this->loss_fn->reset_grads();
};

void SGD::step() {
	/* Perform optimization step (after backward has been called) */

	if (loc.get_name() == "cpu") {
		for (int i = 0; i < weights.size(); i++) {
			_update_host(weights[i], grad_weights[i], bias[i], grad_bias[i], lr);
		};
	}
	else if (loc.get_name() == "gpu") {
		for (int i = 0; i < weights.size(); i++) {
			_weight_update_device<<<grid_size, block_size_weights>>>(weights[i], grad_weights[i], lr);
			_bias_update_device<<<grid_size, block_size_bias>>>(bias[i], grad_bias[i], lr);
		};
	};
};

__host__ void _update_host(float* t_weights, float* t_grad_weights, float* t_bias, float* t_grad_bias, float lr) {
	/* Update weights and biases on host */

};

__global__ void update_device(float* t_weights, float* t_grad_weights, float lr) {
	/* Update weights on device */

};


/****************************************************************************/
/****************************************************************************/

void process(Config cfg, Data_loader data_loader, MLP model, Cross_entropy loss_fn, \
	         SGD optimizer, Logger logger, std::string mode) {
	/* Train or test the neural network */
	std::cout << "Training neural network...\n" << std::endl;

	// Setup variables
	int epoch_size = data_loader.get_num_batches();

	// Storage 
	float epoch_acc = 0;

	for (int batch_idx = 0; batch_idx < epoch_size; batch_idx++) {
		// Get batch: Pointers to host memory locations (pinned)
		auto batch = data_loader.get_batch(batch_idx);    
		Tensor imgs = batch.first;
		Tensor labels = batch.second;

		// TODO: cudaMemcpyAsync (above) is non-blocking so we need to synchronize here
		//       but to use it we need to make sure that the data is not being used in
		//       the forward pass (which is the case here). Thus, we would need to have
		// 	     a separate buffer for the data that we can copy to and from (so that we
		//       can copy the data to the buffer and then copy from the buffer to the
		//       pinned memory). This would require a separate buffer for each batch
		//       which is not optimal. We could also use streams to overlap computation
		//       and data transfer but this is more complex and requires more memory
		//       as we need to have multiple streams for each batch.


		// Move from pinned host to device memory
		if (main_location.get_name() == "device") {
			x.copy_to_device();
			y.copy_to_device();
		};

		// Forward pass
		auto logits = model.forward(imgs);
		auto loss = loss_fn.forward(logits, labels);

		// Backward pass and optimization step 
		if (mode == "train") {
			model.backward(loss_fn.backward(1.0f));
			optimizer.step();
			optimizer.reset_grads();
		};

		// Move from device to pinned host memory
		if (main_location.get_name() == "device") {
			logits.copy_to_host();
			labels.copy_to_host();
		};

		// Batch accuracy and loss
		float batch_acc = accuracy(logits, labels);
		
		cudaDeviceSynchronize();

		epoch_acc += batch_acc.to("cpu").get_data()[0] 
	};

	// Average epoch loss and accuracy
	epoch_acc /= epoch_size;

	// Log epoch loss and accuracy
	int idx = (mode == "train") ? epoch_idx + 1 : -1;
	logger.log(idx, epoch_acc);
};

/****************************************************************************/

int main() {
	double t_begin, t_end;
	t_begin = omp_get_wtime();

	// Logger (for epoch metrics)
	Logger logger;

	// Hyperparameters 
	Config cfg;
	load_configuration(cfg);
	logger.log_config(cfg);

	// Dataset partitions (train and test)
	std::string train_dir = DATA_DIR + "/train";
	std::string test_dir = DATA_DIR + "/test";

	Dataset train_set(train_dir);
	Dataset test_set(test_dir);

	// Initialize Loaders for data partitions. Note that we already know allocate 
	// the host memory for the batches so we don't have to reallocate every new batch.
	// Note also that this does not mess with later NUMA optimization as we do not
	// have first touch before .get_item(idx) is called (where we copy the data from
	// the dataset). If main location is device, we keep the same size memory allocated
	// (so new batches only update the storage when moved from cpu to device - no reallocation)
	Data_loader train_loader(train_set, cfg.batch_size, true, cfg.loc);
	Data_loader test_loader(test_set, cfg.batch_size, false, cfg.loc);

	// Neural network (always initialize on host). CPU parallelization of weight/bias 
	// initialization. This will not be NUMA optimized as all threads needs access to
	// all weights and biases (so we do not benefit from spreading them out with CPU
	// parallelized first touch). I don't initialize on device as PyTorch also does not
	// do that.
	MLP model(train_set.get_img_size(), cfg.hidden, train_set.get_num_classes(), cfg.loc, cfg.batch_size);

	// Initialize loss function. Note that we allocate backward pass data tensors for the 
	// loss in main_location as we don't initialize this data (no learnable params in cross 
	// entropy loss) unlike the model which have weights that are initialized to specfic values.
	Cross_entropy loss_fn(cfg.hidden.back(), cfg.loc, cfg.batch_size);

	// Initialize optimizer. We initialize it with pointers to the model parameters 
	// in the main location so that we can always access them when we do updates with .step()
	SGD optimizer(cfg.lr, model.get_weights(), model.get_biases(), model.get_grad_weights(), model.get_grad_biases(), cfg.loc);

	// Print setup time
	t_end = omp_get_wtime();
	std::cout << "Setup time: " << (t_end - t_begin) << " seconds\n";

	// Training phase
	process(cfg, train_loader, model, loss_fn, optimizer, logger, "train");

	double avg_time = 0;
	for (int epoch_idx = 1; epoch_idx < cfg.num_epochs; epoch_idx++) {  
		t_begin = omp_get_wtime();
		process(cfg, train_loader, model, loss_fn, optimizer, logger, "train");
		t_end = omp_get_wtime();

		avg_time += t_end - t_begin;
	};

	avg_time /= (cfg.num_epochs - 1);

	std::cout << "Average training epoch time: " << avg_time << " seconds\n";

	// Testing phase
	t_begin = omp_get_wtime();
	process(cfg, train_loader, model, loss_fn, optimizer, logger, "test");
	t_end = omp_get_wtime();
	std::cout << "Total testing time: " << (t_end - t_begin) << " seconds\n";

	return 0;
};




// https://luniak.io/cuda-neural-network-implementation-part-1/ (fwd/bwd parallel)
// https://stackoverflow.com/questions/53498952/tensorflow-horovod-nccl-and-mpi (nccl and MPI data parallism)
// https://github.com/olcf-tutorials/MPI_ping_pong (cuda-aware MPI)


