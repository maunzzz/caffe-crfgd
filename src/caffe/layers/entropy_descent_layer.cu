#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/entropy_descent_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_max_over_channels(const int num, const int channels, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num) {
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[c * num + index], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void stable_mult_div(const int n, const Dtype div_eps, const Dtype* xnow_data, const Dtype* xnext_data, Dtype* xnow_diff) {
  CUDA_KERNEL_LOOP(i, n) {
    if(xnow_data[i] < div_eps){
        if(xnext_data[i] >= div_eps){
          xnow_diff[i] *= xnext_data[i]/(xnow_data[i] + div_eps); //denominator below div_epsilon_, need to add smalle number to avoid numerical instabilities
        } //if both are below div_epsilon_, nothing needs to be done (multiply by 1)
      }else{
        xnow_diff[i] *= xnext_data[i]/xnow_data[i]; //division can be done without any numerical trouble
      }
  }
}

template <typename Dtype>
void EntropyDescentLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* xnow = bottom[0]->gpu_data();
  const Dtype* derivative = bottom[1]->gpu_data();

  Dtype* xnext = top[0]->mutable_gpu_data();
  Dtype* xnext_ptr; //helper pointer to access mid points of top blob
  Dtype* scale_data = scale_.mutable_gpu_data();

  int channels = bottom[0]->shape(prob_axis_);
  int dim = bottom[0]->count() / outer_num_; //width*height*channels

  caffe_copy(bottom[1]->count(), derivative, xnext); //xnext = f'
  caffe_gpu_scal(top[0]->count(), -1*t_k_, xnext);       //xnext = -t_k*f'

  //subtract max to avoid numerical issues
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, xnext + i * dim, scale_data);
    kernel_max_over_channels<Dtype><<<CAFFE_GET_BLOCKS(inner_num_), CAFFE_CUDA_NUM_THREADS>>>(inner_num_, channels, xnext + i * dim, scale_data);
    
    // subtraction
    for (int j = 0; j < channels; j++) {
      caffe_gpu_sub(inner_num_, xnext + i*dim + j*inner_num_, scale_data, xnext + i*dim + j*inner_num_);
    }
  }

  caffe_gpu_exp(top[0]->count(), xnext, xnext);        //xnext = exp(-t_k*f')
  caffe_gpu_mul(top[0]->count(), xnow, xnext, xnext); //xnext = xnow*exp(-t_k*f')

  // Only thing left is normalization
  for (int i = 0; i < outer_num_; ++i) {

    caffe_copy(inner_num_, xnext + i * dim, scale_data);      
    for (int j = 1; j < channels; j++) {
      xnext_ptr = xnext + i * dim + j * inner_num_;        
      caffe_gpu_add(inner_num_, scale_data, xnext_ptr, scale_data); //scale data now has \sum_l xnow_l*exp(-t_k*f'_l) 
    }
    // division
    for (int j = 0; j < channels; j++) {
      xnext_ptr = xnext + i * dim + j * inner_num_;  
      caffe_gpu_div(inner_num_, xnext_ptr, scale_data, xnext_ptr);
    }
  }
}

template <typename Dtype>
void EntropyDescentLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* xnext_diff = top[0]->gpu_diff();
  const Dtype* xnext_data = top[0]->gpu_data();
  Dtype* xnow_diff = bottom[0]->mutable_gpu_diff();
  Dtype* xnow_data = bottom[0]->mutable_gpu_data();
  Dtype* derivative_diff = bottom[1]->mutable_gpu_diff();

  int channels = top[0]->shape(prob_axis_);
  int dim = top[0]->count() / outer_num_;

  //use scale_data to store intermediate data needed
  Dtype* scale_data = scale_.mutable_gpu_data();
  Dtype* tmp_data = tmp_.mutable_gpu_data();

  // start by calculating common term for both xnow_diff and derivative_diff (dL/dxnext - \sum_j dL/dxnex * xnext_j)
  caffe_copy(bottom[0]->count(), xnext_diff, xnow_diff);
  for (int i = 0; i < outer_num_; ++i) {
    //calculate the \sum_j dL/dxnex * xnext_j and store it in scale data      
    caffe_gpu_set(inner_num_, Dtype(0), scale_data);
    for (int c = 0; c < channels; ++c) {
      caffe_copy(inner_num_, xnext_data + i*dim + c*inner_num_, tmp_data);
      caffe_gpu_mul(inner_num_, tmp_data, xnext_diff + i*dim + c*inner_num_, tmp_data); //multiply by dL/dxnext
      caffe_gpu_add(inner_num_, scale_data, tmp_data, scale_data);  
    }
    //subtract it from dL/dxnow
    for (int c = 0; c < channels; ++c) {
        caffe_gpu_sub(inner_num_, xnow_diff + i*dim + c*inner_num_, scale_data, xnow_diff + i*dim + c*inner_num_);
    }            
  }

  // copy to derivative_diff
  caffe_copy(bottom[0]->count(), xnow_diff, derivative_diff);

  //calc last part of xnow_diff
  //old
  //caffe_gpu_mul(bottom[0]->count(), xnow_diff, xnext_data, xnow_diff); 
  //caffe_gpu_div(bottom[0]->count(), xnow_diff, xnow_data, xnow_diff);
  //new 
  //need to handle division by zero, or small number 
  stable_mult_div<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(), div_epsilon_, xnow_data, 
            xnext_data, xnow_diff);

  //calc last part of derivative_diff
  caffe_gpu_mul(bottom[0]->count(), derivative_diff, xnext_data, derivative_diff); 
  caffe_gpu_scal(bottom[0]->count(), -1*t_k_, derivative_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(EntropyDescentLayer);


}  // namespace caffe
