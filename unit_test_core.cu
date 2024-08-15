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
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <math.h>

// Utility testing functions
__global__ void _assign_values(float* matrix) {
    // Assign values to element [1][1] and [2,2]

    matrix[1 * 3 + 1] = 1.0;
    matrix[1 * 3 + 2] = -1.0;
};

__global__ void _multiply_by_2(float* matrix) {
    // Multiply all elements by 2
    matrix[1 * 3 + 1] = 2.0;
    matrix[1 * 3 + 2] = -2.0;
};


/*****************************************************************************************/
/************************************** Allocation ***************************************/
/*****************************************************************************************/


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

void test_allocation(std::string loc) {
    /* Test if my allocation methods are correct */

    // Allocate array on device
    float* matrix = allocate_array(3, 3, loc);

    if (loc == "cpu") {
        // Assign values to element [1][1] and [2,2]
        matrix[1 * 3 + 1] = 1.0;
        matrix[1 * 3 + 2] = -1.0;

        // Now check diagonal elements 
        assert (matrix[1 * 3 + 1] == 1.0);
        assert (matrix[1 * 3 + 2] == -1.0);

        // Deallocate memory
        delete[] matrix;

    } else if (loc == "gpu") {
        // Assign values to element [1][1] and [2,2]
        _assign_values<<<1,1>>>(matrix);

        // Copy data back to host
        float* matrix_host = allocate_array(3, 3, "cpu");
        cudaMemcpy(matrix_host, matrix, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        // Now check diagonal elements
        assert (matrix_host[1 * 3 + 1] == 1.0);
        assert (matrix_host[1 * 3 + 2] == -1.0);

        // Deallocate memory
        delete[] matrix_host;
        cudaFree(matrix);

    } else if (loc == "pinned") {
        // Allocate pinned host matric and a device matric
        float* matrix_host = allocate_array(3, 3, "pinned");
        float* matrix_device = allocate_array(3, 3, "gpu");

        // Assign values to element [1][1] and [2,2] to pinned host matrix
        matrix[1 * 3 + 1] = 1.0;
        matrix[1 * 3 + 2] = -1.0;

        // Copy to device
        cudaMemcpy(matrix_host, matrix, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        // Simple computation on device
        _multiply_by_2<<<1,1>>>(matrix_device);

        // Copy back to host
        cudaMemcpy(matrix_host, matrix_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        // Now check diagonal elements
        assert (matrix_host[1 * 3 + 1] == 2.0);
        assert (matrix_host[1 * 3 + 2] == -2.0);
    };
}


/*****************************************************************************************/
/****************************** Forward pass (CPU) ***************************************/
/*****************************************************************************************/

__host__ void _relu_host_fwd(float* t_in, float* t_out, size_t in_features, size_t batch_size) {
    /* Forward pass of ReLU activation function
    */
    
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_features; j++) {
            t_out[i * in_features + j] = fmax(0.0, t_in[i * in_features + j]);
        };
    };
};

__host__ void _mm_host_fwd(float* t_in, float* t_weights, float* t_out, size_t in_features, size_t out_features, size_t batch_size) {
    /* Forward pass of matrix multiplication t_in * t_weights = t_out.
        :param t_in: Input tensor of shape (in_features, batch_size)
        :param t_weights: Weight tensor of shape (out_features, in_features)
    */

   for (int i = 0; i < batch_size; i++) {
       for (int j = 0; j < out_features; j++) {
           for (int k = 0; k < in_features; k++) {
               t_out[i * out_features + j] += t_in[i * in_features + k] * t_weights[j * in_features + k];
           };
       };
   };

   
};

__host__ void _ba_host_fwd(float* t_out1, float* t_bias, float* t_out2, size_t in_features, size_t out_features, size_t batch_size) {
    /* Forward pass of bias addition t_out1 + t_bias = t_out2.
        :param t_out1: Input tensor of shape (in_features, batch_size)
        :param t_bias: Bias tensor of shape (out_features, 1)

        To obtain the correct dimensions, we need to broadcast the bias tensor (aka. expand dims)
    */

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_features; j++) {
            t_out2[i * out_features + j] = t_out1[i * in_features + j] + t_bias[j];
        };
    };
};

__host__ void _ce_host_fwd(float* t_in, float* t_y, float* t_out, size_t num_classes, size_t batch_size) {
    /* Compute stable softmax and cross-entropy loss. Formula: L = -sum_j(y_j * (x_j - log(sum_k(exp(x_k)))). 
       Here j is the class index and so is k. 
    
    :param t_in: Matrix of shape (num_classes, batch_size) with predicted logits 
    :param t_y: Matrix of shape (num_classes, batch_size) with one-hot encoded labels
    :param t_out: Vector of shape (batch_size) with cross-entropy loss
    */

    // Compute stable softmax cross-entropy loss for each batch one-by-one (loop)
    for (int i = 0; i < batch_size; i++) {
        // Compute the sum of exponentials of the logits
        float sum_exp = 0.0;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += exp(t_in[j * batch_size + i]);
        };

        // Compute the cross-entropy loss
        for (int j = 0; j < num_classes; j++) {
            t_out[i] -= t_y[j * batch_size + i] * (t_in[j * batch_size + i] - log(sum_exp));
        };
    };
}


/*****************************************************************************************/
/****************************** Backward pass (CPU) **************************************/
/*****************************************************************************************/


__host__ void _ba_host_bwd(float* t_out1, float* t_bias, float* t_out2, float* t_grad_bias, float* t_grad_out1, size_t in_features, size_t out_features, size_t batch_size) {
    /* Backward pass of bias addition t_out1 + t_bias = t_out2.
        :param t_out1: Input tensor of shape (in_features, batch_size)
        :param t_bias: Bias tensor of shape (out_features, 1)
        :param t_out2: Output tensor of shape (out_features, batch_size)
        :param t_grad_bias: Gradient of loss w.r.t. bias tensor of shape (out_features, 1)
        :param t_grad_out1: Gradient of loss w.r.t. t_out1 tensor of shape (in_features, batch_size)
    */

    // Compute loss gradient w.r.t. bias. Use grad_out to compute dL/db
    for (int i = 0; i < out_features; i++) {
        for (int j = 0; j < batch_size; j++) {
            t_grad_bias[i] += t_out2[j * out_features + i];
        };
    };

    // Compute loss gradient w.r.t. input. Use grad_out to compute dL/dX
    for (int i = 0; i < in_features; i++) {
        for (int j = 0; j < batch_size; j++) {
            t_grad_out1[j * in_features + i] = t_grad_out1[j * in_features + i];
        };
    };
};

__host__ void _mm_host_bwd(float* t_in, float* t_weights, float* t_out, float* t_grad_weights, float* t_grad_in, size_t in_features, size_t out_features, size_t batch_size) {
    /* Backward pass of matrix multiplication t_in * t_weights = t_out.
        :param t_in: Input tensor X of shape (in_features, batch_size)
        :param t_weights: Weight tensor W.T of shape (out_features, in_features)
        :param t_out: Output tensor Y of shape (out_features, batch_size)
        :param t_grad_weights: Gradient of loss w.r.t. W.T tensor of shape (out_features, in_features)
        :param t_grad_in: Gradient of loss w.r.t. X of shape (in_features, batch_size)
    */

    // Compute loss gradient w.r.t. weights. Use X and grad_out to compute dL/dW
    for (int i = 0; i < out_features; i++) {
        for (int j = 0; j < in_features; j++) {
            for (int k = 0; k < batch_size; k++) {
                t_grad_weights[i * in_features + j] += t_in[k * in_features + j] * t_out[k * out_features + i];
            };
        };
    };

    // Compute loss gradient w.r.t. input. Use W.T and frad:out to compute dL/dX
    for (int i = 0; i < in_features; i++) {
        for (int j = 0; j < batch_size; j++) {
            for (int k = 0; k < out_features; k++) {
                t_grad_in[j * in_features + i] += t_weights[k * in_features + i] * t_out[j * out_features + k];
            };
        };
    };
};

__host__ void _relu_host_bwd(float* t_in, float* t_out, float* t_grad_in, float* t_grad_out, size_t in_features, size_t batch_size) {
    /* Backward pass of ReLU activation function
    */
    
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_features; j++) {
            t_grad_in[i * in_features + j] = (t_out[i * in_features + j] > 0) ? t_grad_out[i * in_features + j] : 0.0;
        };
    };
};

__host__ void _ce_host_bwd(float* t_in, float* t_y, float* t_grad_in, size_t num_classes, size_t batch_size) {
    /* Compute the gradient of the stable softmax cross-entropy loss w.r.t. the input logits (t_in) on host

    :param t_in: Matrix of shape (num_classes, batch_size) with predicted logits
    :param t_y: Matrix of shape (num_classes, batch_size) with one-hot encoded labels
    :param t_grad_in: Matrix of shape (num_classes, batch_size) with gradients of loss w.r.t. input
    :param t_grad_out: Vector of shape (batch_size) with gradients of loss w.r.t. output
    */

    // Compute the gradient of the stable softmax cross-entropy loss w.r.t. the input logits
    for (int i = 0; i < batch_size; i++) {
        // Compute the sum of exponentials of the logits
        float sum_exp = 0.0;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += exp(t_in[j * batch_size + i]);
        };

        // Compute the gradient of the loss w.r.t. the input logits
        for (int j = 0; j < num_classes; j++) {
            t_grad_in[j * batch_size + i] = (exp(t_in[j * batch_size + i]) / sum_exp) - t_y[j * batch_size + i];
        };
    };
};


/*****************************************************************************************/
/********************************** SGD step (CPU) ***************************************/
/*****************************************************************************************/


__host__ void _update_weights(float* t_weights, float* t_grad_weights, float lr, size_t in_features, size_t out_features) {
	/* Update weights and biases on host (SGD step) */

    // Update weights
    for (int i = 0; i < out_features; i++) {
        for (int j = 0; j < in_features; j++) {
            t_weights[i * in_features + j] -= lr * t_grad_weights[i * in_features + j];
        };
    };
};

__host__ void _update_bias(float* t_bias, float* t_grad_bias, float lr, size_t out_features) {
    /* Update biases on host (SGD step) */

    for (int i = 0; i < out_features; i++) {
        t_bias[i] -= lr * t_grad_bias[i];
    };
};


/*****************************************************************************************/
/********************************** Forward pass (GPU) ************************************/
/*****************************************************************************************/

__global__ void _relu_device_fwd(float* t_in, float* t_out, size_t in_features, size_t batch_size) {
    /* Forward pass of ReLU activation function (on batched data)
    */
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < batch_size && y < in_features) {
        t_out[x * in_features + y] = max(0.0, t_in[x * in_features + y]);
    };
};

__global__ void _mm_device_fwd(float* t_in, float* t_weights, float* t_out, size_t in_features, size_t out_features, size_t batch_size) {
    /* Matrix multiplication where grid_size is 1 and block_size is (out_features, in_features)
    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < batch_size && y < out_features) {
        for (int i = 0; i < in_features; i++) {
            t_out[x * out_features + y] += t_in[x * in_features + i] * t_weights[y * in_features + i];
        };
    };
};

__global__ void _ba_device_fwd(float* t_out1, float* t_bias, float* t_out2, size_t in_features, size_t out_features, size_t batch_size) {
    /* Forward pass of bias addition t_out1 + t_bias = t_out2.
        :param t_out1: Input tensor of shape (in_features, batch_size)
        :param t_bias: Bias tensor of shape (out_features, 1)
    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < batch_size && y < out_features) {
        t_out2[x * out_features + y] = t_out1[x * in_features + y] + t_bias[y];
    };
};

__global__ void _ce_device_fwd_1(float* t_pred_logits, float* t_out1, size_t num_classes, size_t batch_size) {
	/* Step 1: Compute exp(x) element-wise of entire matrix */

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            t_out1[i * num_classes + j] = exp(t_pred_logits[j * batch_size + i]);
        };
    };
};

__global__ void _ce_device_fwd_2(float* t_out1, float* t_out2, size_t num_classes, size_t batch_size) {
	/* Step 2: Compute the s_1 = sum_k(exp(x_k)) scalar (using tree reduction kernel).
    
    :param t_out1: Matrix of shape (num_classes, batch_size) with exp(x_ij) elements
    :param t_out2: Vector of shape (batch_size) with sum(exp(x_k)) elements

    In other words, we need to sum accrross the rows of the matrix t_out1 to get the vector t_out2.
    It is assumed that num_classes <= 32 (i.e. the number of threads in a block). Thus we can use
    a tree reduction kernel to sum the elements of each row in t_out1 inside each block. We have
    a block size of num_classes and a grid size of batch_size.

    for (int i = 1; i < 32; i *= 2)
        value += __shfl_xor_sync(-1, value, i);

    */

    // Tree reduction using warp shuffle
    for (int i = 1; i < num_classes; i *= 2) {
        t_out1[threadIdx.x] += __shfl_xor_sync(-1, t_out1[threadIdx.x], i);
    };

    // Store the sum of exponentials of the logits in the output vector
    if (threadIdx.x == 0) {
        t_out2[blockIdx.x] = t_out1[0];
    };
};

__global__ void _ce_device_fwd_3(float* t_out2, float* t_out3, size_t num_classes, size_t batch_size) {
	/* Step 3: Compute the s_2 = log(s) scalar on device (using element-wise kernel). A grid size of 1
       and block size of batch_size is used.
    
    :param t_out2: Vector of shape (batch_size) with sum(exp(x_k)) elements
    :param t_out3: Vector of shape (batch_size) with log(sum(exp(x_k))) elements

    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < batch_size) {
        t_out3[x] = log(t_out2[x]);
    };
};

__global__ void _ce_device_fwd_4(float* t_out3, float* t_pred_logits, float* t_out4, size_t num_classes, size_t batch_size) {
	/* Step 4: Compute (x_j - s_j) (simple element-wise subtraction) where s_j is the log(sum_k(exp(x_k)))
       scalar which should be broadcasted across the entire row of the matrix t_pred_logits. The grid size
       is batch_size and block size is num_classes.
    
    :param t_out3: Vector of shape (batch_size) with log(sum_k(exp(x_k))) elements (one for each class)
    :param t_pred_logits: Matrix of shape (num_classes, batch_size) with predicted logits
    :param t_out4: Matrix of shape (num_classes, batch_size) with (x_j - s_j) elements    

    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < batch_size && y < num_classes) {
        t_out4[y * batch_size + x] = t_pred_logits[y * batch_size + x] - t_out3[x];
    };




    

    



    




};

__global__ void _ce_device_fwd_5(float* t_out4, float* t_true_probs, float* t_out5, size_t num_classes, size_t batch_size) {
    /* Step 5 compute dot product of t_out4 and t_true_probs (element-wise multiplication and sum) */

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            t_out5[i] -= t_true_probs[j * batch_size + i] * t_out4[j * batch_size + i];
        };
    };
}


/*****************************************************************************************/
/********************************** Backward pass (GPU) ***********************************/
/*****************************************************************************************/


__global__ void _ba_device_bwd(float* t_out1, float* t_bias, float* t_out2, float* t_grad_bias, float* t_grad_out1, size_t in_features, size_t out_features, size_t batch_size) {
    /* Backward pass of bias addition t_out1 + t_bias = t_out2.
        :param t_out1: Input tensor of shape (in_features, batch_size)
        :param t_bias: Bias tensor of shape (out_features, 1)
        :param t_out2: Output tensor of shape (out_features, batch_size)
        :param t_grad_bias: Gradient of loss w.r.t. bias tensor of shape (out_features, 1)
        :param t_grad_out1: Gradient of loss w.r.t. t_out1 tensor of shape (in_features, batch_size)
    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_features) {
        // Compute loss gradient w.r.t. bias. Use grad_out to compute dL/db
        for (int i = 0; i < batch_size; i++) {
            t_grad_bias[x] += t_out2[i * out_features + x];
        };

        // Compute loss gradient w.r.t. input. Use grad_out to compute dL/dX
        for (int i = 0; i < in_features; i++) {
            t_grad_out1[y * in_features + i] = t_grad_out1[y * in_features + i];
        };
    };
};

__global__ void _mm_device_bwd(float* t_in, float* t_weights, float* t_out, float* t_grad_weights, float* t_grad_in, size_t in_features, size_t out_features, size_t batch_size) {
    /* Backward pass of matrix multiplication where grid_size is 1 and block_size is (out_features, in_features)
    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < batch_size && y < out_features) {
        // Compute loss gradient w.r.t. weights. Use X and grad_out to compute dL/dW
        for (int i = 0; i < in_features; i++) {
            t_grad_weights[y * in_features + i] += t_in[x * in_features + i] * t_out[x * out_features + y];
        };

        // Compute loss gradient w.r.t. input. Use W.T and grad:out to compute dL/dX
        for (int i = 0; i < in_features; i++) {
            t_grad_in[x * in_features + i] += t_weights[y * in_features + i] * t_out[x * out_features + y];
        };
    };
};

__global__ void _relu_device_bwd(float* t_in, float* t_out, float* t_grad_in, float* t_grad_out, size_t in_features, size_t batch_size) {
    /* Backward pass of ReLU activation function (on batched data)
    */
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < batch_size && y < in_features) {
        t_grad_in[x * in_features + y] = (t_out[x * in_features + y] > 0) ? t_grad_out[x * in_features + y] : 0.0;
    };
};






__global__ void _ce_device_bwd_1(float* t_pred_logits, float* t_out1, size_t num_classes, size_t batch_size) {
    /* Compute the gradient of the stable softmax cross-entropy loss w.r.t. the input logits (t_in) on device

    :param t_pred_logits: Matrix of shape (num_classes, batch_size) with predicted logits
    :param t_out1: Matrix of shape (num_classes, batch_size) with exp(x_ij) elements
    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < batch_size && y < num_classes) {
        t_out1[y * batch_size + x] = exp(t_pred_logits[y * batch_size + x]);
    };
};

__global__ void _ce_device_bwd_2(float* t_out1, float* t_out2, size_t num_classes, size_t batch_size) {
    /* Compute the gradient of the stable softmax cross-entropy loss w.r.t. the input logits (t_in) on device

    :param t_out1: Matrix of shape (num_classes, batch_size) with exp(x_ij) elements
    :param t_out2: Vector of shape (batch_size) with sum(exp(x_k)) elements

    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < batch_size) {
        // Tree reduction using warp shuffle
        for (int i = 1; i < num_classes; i *= 2) {
            t_out1[x] += __shfl_xor_sync(-1, t_out1[x], i);
        };

        // Store the sum of exponentials of the logits in the output vector
        if (threadIdx.x == 0) {
            t_out2[blockIdx.x] = t_out1[0];
        };
    };
};

__global__ void _ce_device_bwd_3(float* t_out2, float* t_out3, size_t num_classes, size_t batch_size) {
    /* Compute the gradient of the stable softmax cross-entropy loss w.r.t. the input logits (t_in) on device

    :param t_out2: Vector of shape (batch_size) with sum(exp(x_k)) elements
    :param t_out3: Vector of shape (batch_size) with log(sum(exp(x_k))) elements

    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < batch_size) {
        t_out3[x] = log(t_out2[x]);
    };
};

__global__ void _ce_device_bwd_4(float* t_out3, float* t_pred_logits, float* t_out4, size_t num_classes, size_t batch_size) {
    /* Compute the gradient of the stable softmax cross-entropy loss w.r.t. the input logits (t_in) on device

    :param t_out3: Vector of shape (batch_size) with log(sum_k(exp(x_k))) elements
    :param t_pred_logits: Matrix of shape (num_classes, batch_size) with predicted logits
    :param t_out4: Matrix of shape (num_classes, batch_size) with (x_j - s_j) elements

    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < batch_size && y < num_classes) {
        t_out4[y * batch_size + x] = t_pred_logits[y * batch_size + x] - t_out3[x];
    };
};

__global__ void _ce_device_bwd_5(float* t_out4, float* t_true_probs, float* t_out5, size_t num_classes, size_t batch_size) {
    /* Compute the gradient of the stable softmax cross-entropy loss w.r.t. the input logits (t_in) on device

    :param t_out4: Matrix of shape (num_classes, batch_size) with (x_j - s_j) elements
    :param t_true_probs: Matrix of shape (num_classes, batch_size) with one-hot encoded labels
    :param t_out5: Vector of shape (batch_size) with gradients of loss w.r.t. output

    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < batch_size && y < num_classes) {
        t_out5[x] -= t_true_probs[y * batch_size + x] * t_out4[y * batch_size + x];
    };
};

/*****************************************************************************************/
/********************************** SGD step (GPU) ***************************************/
/*****************************************************************************************/


__global__ void update_device(float* t_parameters, float* t_grad_parameters, float lr, size_t in_features, size_t out_features) {
	/* Update weights or biases on device 
    
    :param t_parameters: Weights tensor (shape (out_features, in_features)) or biases tensor (shape (out_features)))
    :param t_grad_parameters: Gradient of loss w.r.t. weights or biases tensor (same shape as t_parameters)
    
    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_features && y < in_features) {
        t_parameters[x * in_features + y] -= lr * t_grad_parameters[x * in_features + y];
    };

};

/*****************************************************************************************/
/*********************************** Test functions **************************************/
/*****************************************************************************************/

void test_host_relu() {
    // Allocate array on device
    float* input_host = allocate_array(3, 3, "cpu");
    float* output_host = allocate_array(3, 3, "cpu");
    float* grad_in_host = allocate_array(3, 3, "cpu");
    float* grad_out_host = allocate_array(3, 3,  "cpu");

    // Assign values to all elements
    for (int i = 0; i < 3; i++) {
        input_host[i * 3 + 0] = -1.0f;
        input_host[i * 3 + 1] = 0.0f;
        input_host[i * 3 + 2] = 1.0f;
    };

    for (int i = 0; i<3; i++) {
        for (int j = 0; j<3; j++) {
            output_host[i * 3 + j] = 0.0f;
            grad_in_host[i * 3 + j] = 0.0f;
            grad_out_host[i * 3 + j] = 1.0f;
        };
    };

    // Apply ReLU forward pass
    _relu_host_fwd(input_host, output_host, 3, 3);

    // Now check diagonal elements
    assert (output_host[0] == 0);
    assert (output_host[1 * 3 + 1] == 0);
    assert (output_host[1 * 3 + 2] == 1);

    // Apply ReLU backward pass
    _relu_host_bwd(input_host, output_host, grad_in_host, grad_out_host, 3, 3);

    // Now check diagonal elements
    assert (grad_in_host[0] == 0);
    assert (grad_in_host[1 * 3 + 1] == 0);
    assert (grad_in_host[1 * 3 + 2] == 1);

    // Deallocate memory
    delete[] input_host;
    delete[] output_host;
    delete[] grad_in_host;
    delete[] grad_out_host;
};

void test_host_mm() {
    // Allocate array on host
    float* input_host = allocate_array(3, 3, "cpu");
    float* weights_host = allocate_array(3, 3, "cpu");
    float* output_host = allocate_array(3, 3, "cpu");
    float* grad_weights_host = allocate_array(3, 3, "cpu");
    float* grad_in_host = allocate_array(3, 3, "cpu");
    float* grad_out_host = allocate_array(3, 3, "cpu");

    // Assign 1's to all input elements and 0 to output and gradient elements
    // ecept grad_out (given by earlier layer in pipeline)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            input_host[i * 3 + j] = 1.0;
            output_host[i * 3 + j] = 0.0;
            grad_weights_host[i * 3 + j] = 0.0;
            grad_in_host[i * 3 + j] = 0.0;
            grad_out_host[i * 3 + j] = 1.0;
        };
    };

    // Assign 1 to 9 to weights
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            weights_host[i * 3 + j] = i * 3 + j + 1;
        };
    };

    // Apply forward pass
    _mm_host_fwd(input_host, weights_host, output_host, 3, 3, 3);

    // Now check diagonal elements
    assert (output_host[0] == 6);
    assert (output_host[1 * 3 + 1] == 15);
    assert (output_host[1 * 3 + 2] == 24);

    // Apply backward pass
    _mm_host_bwd(input_host, weights_host, output_host, grad_weights_host, grad_in_host, 3, 3, 3);

    // Deallocate memory
    delete[] input_host;
    delete[] weights_host;
    delete[] output_host;
    delete[] grad_weights_host;
};

void test_host_ba() {
    // Allocate array on host
    float* input_host = allocate_array(3, 3, "cpu");
    float* output_host = allocate_array(3, 3, "cpu");
    float* bias_host = allocate_array(3, 3, "cpu");
    float* grad_out_host = allocate_array(3, 3, "cpu");
    float* grad_bias_host = allocate_array(3, 3, "cpu");
    float* grad_in_host = allocate_array(3, 3, "cpu");

    // Assign 1's to all input elements and 0 to output and gradient elements
    // except grad_out (given by earlier layer in pipeline - so we set it to 1s)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            input_host[i * 3 + j] = 1.0;
            output_host[i * 3 + j] = 0.0;
            bias_host[i * 3 + j] = j+1;
            grad_out_host[i * 3 + j] = 1.0;
            grad_bias_host[i * 3 + j] = 0.0;
            grad_in_host[i * 3 + j] = 0.0;
        };
    };

    // Apply forward pass
    _ba_host_fwd(input_host, bias_host, output_host, 3, 3, 3);

    // Now check diagonal elements
    assert (output_host[0] == 2);
    assert (output_host[1 * 3 + 1] == 3);
    assert (output_host[1 * 3 + 2] == 4);

    // Apply backward pass
    _ba_host_bwd(input_host, bias_host, output_host, grad_bias_host, grad_in_host, 3, 3, 3);

    // Deallocate memory
    delete[] output_host;
    delete[] bias_host;
    delete[] grad_out_host;
    delete[] grad_bias_host;
    delete[] grad_in_host;
};

void test_host_ce() {
    // Allocate array on host
    float* input_host = allocate_array(10, 3, "cpu");
    float* y_host = allocate_array(10, 3, "cpu");
    float* output_host = allocate_array(1, 3, "cpu");
    float* grad_in_host = allocate_array(10, 3, "cpu");

    // Assign 0's to output and grad_in. One-hot-encode 1, 2, 3 along the rows
    // of input and y and 3, 2, 1 along the rows of input
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 3; j++) {
            input_host[i * 3 + j] = (3 - j == i) ? 1.0 : 0.0;
            y_host[i * 3 + j] = (i == j) ? 1.0 : 0.0;
            output_host[j] = 0.0;
            grad_in_host[i * 3 + j] = 0.0;
        };
    };

    // Forward pass
    _ce_host_fwd(input_host, y_host, output_host, 10, 3);

    // Now check diagonal elements
    assert (output_host[0] == 0);
    assert (output_host[1] == 0);
    assert (output_host[2] == 0);

    // Backward pass
    _ce_host_bwd(input_host, y_host, grad_in_host, 10, 3);

    // Now check diagonal elements
    assert (grad_in_host[0] == 0);
    assert (grad_in_host[1 * 3 + 1] == 0);
    assert (grad_in_host[2 * 3 + 2] == 0);

    // Deallocate memory
    delete[] input_host;
    delete[] y_host;
    delete[] output_host;
    delete[] grad_in_host;
};

void test_host_step() {
    // Allocate array on host
    float* weights_host = allocate_array(3, 3, "cpu");
    float* grad_weights_host = allocate_array(3, 3, "cpu");
    float* bias_host = allocate_array(3, 3, "cpu");
    float* grad_bias_host = allocate_array(3, 3, "cpu");

    // Assign 1's to all input elements and 0 to output and gradient elements
    // ecept grad_out (given by earlier layer in pipeline)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            weights_host[i * 3 + j] = 1.0;
            grad_weights_host[i * 3 + j] = 0.0;
            bias_host[i * 3 + j] = 1.0;
            grad_bias_host[i * 3 + j] = 0.0;
        };
    };

    // Apply update step
    _update_host(weights_host, grad_weights_host, bias_host, grad_bias_host, 0.1);

    // Now check diagonal elements
    assert (weights_host[0] == 1.0);
    assert (weights_host[1 * 3 + 1] == 1.0);
    assert (weights_host[2 * 3 + 2] == 1.0);

    // Deallocate memory
    delete[] weights_host;
    delete[] grad_weights_host;
    delete[] bias_host;
    delete[] grad_bias_host;
};


void test_device_relu() {
    // Allocate array on device
    float* input_host = allocate_array(3, 3, "cpu");
    float* output_host = allocate_array(3, 3, "cpu");
    float* grad_in_host = allocate_array(3, 3, "cpu");
    float* grad_out_host = allocate_array(3, 3,  "cpu");

    // Assign values to all elements
    for (int i = 0; i < 3; i++) {
        input_host[i * 3 + 0] = -1.0f;
        input_host[i * 3 + 1] = 0.0f;
        input_host[i * 3 + 2] = 1.0f;
    };

    for (int i = 0; i<3; i++) {
        for (int j = 0; j<3; j++) {
            output_host[i * 3 + j] = 0.0f;
            grad_in_host[i * 3 + j] = 0.0f;
            grad_out_host[i * 3 + j] = 1.0f;
        };
    };

    // Allocate device arrays
    float* input_device = allocate_array(3, 3, "gpu");
    float* output_device = allocate_array(3, 3, "gpu");
    float* grad_in_device = allocate_array(3, 3, "gpu");
    float* grad_out_device = allocate_array(3, 3, "gpu");

    // Copy data to device
    cudaMemcpy(input_device, input_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_in_device, grad_in_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_out_device, grad_out_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Apply ReLU forward pass
    _relu_device_fwd<<<1,dim3(3,3)>>>(input_device, output_device, 3, 3);

    // Copy data back to host
    cudaMemcpy(output_host, output_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Now check diagonal elements
    assert (output_host[0] == 0);
    assert (output_host[1 * 3 + 1] == 0);
    assert (output_host[1 * 3 + 2] == 1);

    // Apply ReLU backward pass
    _relu_device_bwd<<<1,dim3(3,3)>>>(input_device, output_device, grad_in_device, grad_out_device, 3, 3);

    // Copy data back to host
    cudaMemcpy(grad_in_host, grad_in_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Now check diagonal elements
    assert (grad_in_host[0] == 0);
    assert (grad_in_host[1 * 3 + 1] == 0);
    assert (grad_in_host[1 * 3 + 2] == 1);

    // Deallocate memory
    cudaFree(input_device);
    cudaFree(output_device);
    cudaFree(grad_in_device);
    cudaFree(grad_out_device);

    delete[] input_host;
    delete[] output_host;
    delete[] grad_in_host;
    delete[] grad_out_host;
};

void test_device_mm() {
    // Allocate array on host
    float* input_host = allocate_array(3, 3, "cpu");
    float* weights_host = allocate_array(3, 3, "cpu");
    float* output_host = allocate_array(3, 3, "cpu");
    float* grad_weights_host = allocate_array(3, 3, "cpu");
    float* grad_in_host = allocate_array(3, 3, "cpu");
    float* grad_out_host = allocate_array(3, 3, "cpu");

    // Assign 1's to all input elements and 0 to output and gradient elements
    // ecept grad_out (given by earlier layer in pipeline)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            input_host[i * 3 + j] = 1.0;
            output_host[i * 3 + j] = 0.0;
            grad_weights_host[i * 3 + j] = 0.0;
            grad_in_host[i * 3 + j] = 0.0;
            grad_out_host[i * 3 + j] = 1.0;
        };
    };

    // Assign 1 to 9 to weights
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            weights_host[i * 3 + j] = i * 3 + j + 1;
        };
    };

    // Allocate device arrays
    float* input_device = allocate_array(3, 3, "gpu");
    float* weights_device = allocate_array(3, 3, "gpu");
    float* output_device = allocate_array(3, 3, "gpu");
    float* grad_weights_device = allocate_array(3, 3, "gpu");
    float* grad_in_device = allocate_array(3, 3, "gpu");
    float* grad_out_device = allocate_array(3, 3, "gpu");

    // Copy data to device
    cudaMemcpy(input_device, input_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_device, weights_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_weights_device, grad_weights_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_in_device, grad_in_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_out_device, grad_out_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Apply forward pass
    _mm_device_fwd<<<1,dim3(3,3)>>>(input_device, weights_device, output_device, 3, 3, 3);

    // Copy data back to host
    cudaMemcpy(output_host, output_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Now check diagonal elements
    assert (output_host[0] == 6);
    assert (output_host[1 * 3 + 1] == 15);
    assert (output_host[1 * 3 + 2] == 24);

    // Apply backward pass
    _mm_device_bwd<<<1,dim3(3,3)>>>(input_device, weights_device, output_device, grad_weights_device, grad_in_device, 3, 3, 3);

    // Copy data back to host
    cudaMemcpy(grad_in_host, grad_in_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_weights_host, grad_weights_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate memory
    cudaFree(input_device);
    cudaFree(weights_device);
    cudaFree(output_device);
    cudaFree(grad_weights_device);
    cudaFree(grad_in_device);
    cudaFree(grad_out_device);

    delete[] input_host;
    delete[] weights_host;
    delete[] output_host;
    delete[] grad_weights_host;
    delete[] grad_in_host;
    delete[] grad_out_host;
};

void test_device_ba() {
    // Allocate array on host
    float* input_host = allocate_array(3, 3, "cpu");
    float* output_host = allocate_array(3, 3, "cpu");
    float* bias_host = allocate_array(3, 3, "cpu");
    float* grad_out_host = allocate_array(3, 3, "cpu");
    float* grad_bias_host = allocate_array(3, 3, "cpu");
    float* grad_in_host = allocate_array(3, 3, "cpu");

    // Assign 1's to all input elements and 0 to output and gradient elements
    // except grad_out (given by earlier layer in pipeline - so we set it to 1s)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            input_host[i * 3 + j] = 1.0;
            output_host[i * 3 + j] = 0.0;
            bias_host[i * 3 + j] = j+1;
            grad_out_host[i * 3 + j] = 1.0;
            grad_bias_host[i * 3 + j] = 0.0;
            grad_in_host[i * 3 + j] = 0.0;
        };
    };

    // Allocate device arrays
    float* input_device = allocate_array(3, 3, "gpu");
    float* output_device = allocate_array(3, 3, "gpu");
    float* bias_device = allocate_array(3, 3, "gpu");
    float* grad_out_device = allocate_array(3, 3, "gpu");
    float* grad_bias_device = allocate_array(3, 3, "gpu");
    float* grad_in_device = allocate_array(3, 3, "gpu");

    // Copy data to device
    cudaMemcpy(input_device, input_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_out_device, grad_out_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_bias_device, grad_bias_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_in_device, grad_in_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Apply forward pass
    _ba_device_fwd<<<1,dim3(3,3)>>>(input_device, bias_device, output_device, 3, 3, 3);

    // Copy data back to host
    cudaMemcpy(output_host, output_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Now check diagonal elements
    assert (output_host[0] == 2);
    assert (output_host[1 * 3 + 1] == 3);
    assert (output_host[1 * 3 + 2] == 4);

    // Apply backward pass
    _ba_device_bwd<<<1,dim3(3,3)>>>(input_device, bias_device, output_device, grad_bias_device, grad_in_device, 3, 3, 3);

    // Copy data back to host
    cudaMemcpy(grad_in_host, grad_in_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_bias_host, grad_bias_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate memory
    cudaFree(input_device);
    cudaFree(output_device);
    cudaFree(bias_device);
    cudaFree(grad_out_device);
    cudaFree(grad_bias_device);
    cudaFree(grad_in_device);

    delete[] input_host;
    delete[] output_host;
    delete[] bias_host;
    delete[] grad_out_host;
    delete[] grad_bias_host;
    delete[] grad_in_host;
};

void test_device_ce() {
    // Allocate array on host
    float* input_host = allocate_array(10, 3, "cpu");
    float* y_host = allocate_array(10, 3, "cpu");
    float* output_host = allocate_array(1, 3, "cpu");
    float* grad_in_host = allocate_array(10, 3, "cpu");

    // Assign 0's to output and grad_in. One-hot-encode 1, 2, 3 along the rows
    // of input and y and 3, 2, 1 along the rows of input
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 3; j++) {
            input_host[i * 3 + j] = (3 - j == i) ? 1.0 : 0.0;
            y_host[i * 3 + j] = (i == j) ? 1.0 : 0.0;
            output_host[j] = 0.0;
            grad_in_host[i * 3 + j] = 0.0;
        };
    };

    // Allocate device arrays
    float* input_device = allocate_array(10, 3, "gpu");
    float* y_device = allocate_array(10, 3, "gpu");

    float* out1_device = allocate_array(10, 3, "gpu");
    float* out2_device = allocate_array(10, 3, "gpu");
    float* out3_device = allocate_array(10, 3, "gpu");
    float* out4_device = allocate_array(10, 3, "gpu");
    float* out5_device = allocate_array(10, 3, "gpu");

    float* grad_out1_device = allocate_array(10, 3, "gpu");
    float* grad_out2_device = allocate_array(10, 3, "gpu");
    float* grad_out3_device = allocate_array(10, 3, "gpu");
    float* grad_out4_device = allocate_array(10, 3, "gpu");
    float* grad_out5_device = allocate_array(10, 3, "gpu");

    // Copy data to device
    cudaMemcpy(input_device, input_host, 10 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, 10 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(out1_device, output_host, 10 * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out2_device, output_host, 10 * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out3_device, output_host, 10 * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out4_device, output_host, 10 * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out5_device, output_host, 10 * 3 * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaMemcpy(grad_out1_device, grad_in_host, 10 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_out2_device, grad_in_host, 10 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_out3_device, grad_in_host, 10 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_out4_device, grad_in_host, 10 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_out5_device, grad_in_host, 10 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Apply forward pass
    _ce_device_fwd_1<<<1,dim3(10,3)>>>(input_device, out1_device, 10, 3);
    _ce_device_fwd_2<<<1,dim3(10,3)>>>(out1_device, out2_device, 10, 3);
    _ce_device_fwd_3<<<1,dim3(10,3)>>>(out2_device, out3_device, 10, 3);
    _ce_device_fwd_4<<<1,dim3(10,3)>>>(out3_device, out4_device, y_device, 10, 3);
    _ce_device_fwd_2<<<1,dim3(10,3)>>>(out4_device, out5_device, 10, 3);

    // Apply backward pass
    _ce_device_bwd_1<<<1,dim3(10,3)>>>(input_device, y_device, grad_out1_device, 10, 3);
    _ce_device_bwd_2<<<1,dim3(10,3)>>>(out1_device, out2_device, grad_out2_device, 10, 3);
    _ce_device_bwd_3<<<1,dim3(10,3)>>>(out2_device, out3_device, grad_out3_device, 10, 3);
    _ce_device_bwd_4<<<1,dim3(10,3)>>>(out3_device, out4_device, grad_out4_device, 10, 3);
    _ce_device_bwd_2<<<1,dim3(10,3)>>>(out4_device, out5_device, grad_out5_device, 10, 3);

    // Copy data back to host
    cudaMemcpy(output_host, out5_device, 10 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_in_host, grad_out5_device, 10 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate memory
    cudaFree(input_device);
    cudaFree(y_device);

    cudaFree(out1_device);
    cudaFree(out2_device);
    cudaFree(out3_device);
    cudaFree(out4_device);
    cudaFree(out5_device);

    cudaFree(grad_out1_device);
    cudaFree(grad_out2_device);
    cudaFree(grad_out3_device);
    cudaFree(grad_out4_device);
    cudaFree(grad_out5_device);

    delete[] input_host;
    delete[] y_host;
    delete[] output_host;
    delete[] grad_in_host;
};


void test_device_step() {
    // Allocate array on host
    float* weights_host = allocate_array(3, 3, "cpu");
    float* grad_weights_host = allocate_array(3, 3, "cpu");
    float* bias_host = allocate_array(3, 3, "cpu");
    float* grad_bias_host = allocate_array(3, 3, "cpu");

    // Assign 1's to all input elements and 0 to output and gradient elements
    // ecept grad_out (given by earlier layer in pipeline)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            weights_host[i * 3 + j] = 1.0;
            grad_weights_host[i * 3 + j] = 0.0;
            bias_host[i * 3 + j] = 1.0;
            grad_bias_host[i * 3 + j] = 0.0;
        };
    };

    // Allocate device arrays
    float* weights_device = allocate_array(3, 3, "gpu");
    float* grad_weights_device = allocate_array(3, 3, "gpu");
    float* bias_device = allocate_array(3, 3, "gpu");
    float* grad_bias_device = allocate_array(3, 3, "gpu");
    
    // Copy data to device
    cudaMemcpy(weights_device, weights_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_weights_device, grad_weights_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_bias_device, grad_bias_host, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Apply update step
    update_device<<<1,dim3(3,3)>>>(weights_device, grad_weights_device, 0.1, 3, 3);
    update_device<<<1,dim3(3,3)>>>(bias_device, grad_bias_device, 0.1, 3, 3);

    // Copy data back to host
    cudaMemcpy(weights_host, weights_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_weights_host, grad_weights_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bias_host, bias_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_bias_host, grad_bias_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Now check diagonal elements
    assert (weights_host[0] == 1.0);
    assert (weights_host[1 * 3 + 1] == 1.0);
    assert (weights_host[2 * 3 + 2] == 1.0);

    // Deallocate memory
    cudaFree(weights_device);
    cudaFree(grad_weights_device);
    cudaFree(bias_device);
    cudaFree(grad_bias_device);

    delete[] weights_host;
    delete[] grad_weights_host;
    delete[] bias_host;
    delete[] grad_bias_host;
};

/*****************************************************************************************/
/*****************************************************************************************/


int main() {
    // Test 1: Array allocation [PASSED]
    test_allocation("cpu");
    test_allocation("gpu");
    test_allocation("pinned");

    // Test 2: Forward and backward pass CPU 
    test_host_relu();
    test_host_mm();
    test_host_ba();
    test_host_ce();

    // Test 3: Forward and backward pass GPU
    test_device_relu();
    test_device_mm();
    test_device_ba();
    test_device_ce();

    // Test 5: SGD step 
    test_host_step();
    test_device_step();

    return 0;

};