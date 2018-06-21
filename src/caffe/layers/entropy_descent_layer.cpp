#include <algorithm>
#include <vector>

#include "caffe/layers/entropy_descent_layer.hpp"
#include "caffe/util/math_functions.hpp"

//#include "caffe/util/tvg_util.hpp"

namespace caffe {

template <typename Dtype>
void EntropyDescentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  div_epsilon_ = 1e-5;
  prob_axis_ = 1;
  top[0]->ReshapeLike(*bottom[0]);  
  outer_num_ = bottom[0]->count(0, prob_axis_); //number of instances
  inner_num_ = bottom[0]->count(prob_axis_ + 1); // width x height
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[0] = 1;
  scale_dims[prob_axis_] = 1;
  scale_.Reshape(scale_dims);
  tmp_.Reshape(scale_dims);
}

template <typename Dtype>
void EntropyDescentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* xnow = bottom[0]->cpu_data();
  const Dtype* derivative = bottom[1]->cpu_data();

  Dtype* xnext = top[0]->mutable_cpu_data();
  Dtype* xnext_ptr; //helper pointer to access mid points of top blob
  Dtype* scale_data = scale_.mutable_cpu_data();

  int channels = bottom[0]->shape(prob_axis_);
  int dim = bottom[0]->count() / outer_num_; //width*height*channels

  caffe_copy(bottom[1]->count(), derivative, xnext); //xnext = f'
  caffe_scal(top[0]->count(), -1*t_k_, xnext);       //xnext = -t_k*f'

  //subtract max to avoid numerical issues
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, xnext + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            xnext[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    for (int j = 0; j < channels; j++) {
      caffe_sub(inner_num_, xnext + i*dim + j*inner_num_, scale_data, xnext + i*dim + j*inner_num_);
    }
  }

  caffe_exp(top[0]->count(), xnext, xnext);        //xnext = exp(-t_k*f')
  caffe_mul(top[0]->count(), xnow, xnext, xnext); //xnext = xnow*exp(-t_k*f')

  // Only thing left is normalization
  for (int i = 0; i < outer_num_; ++i) {

    caffe_copy(inner_num_, xnext + i * dim, scale_data);      
    for (int j = 1; j < channels; j++) {
      xnext_ptr = xnext + i * dim + j * inner_num_;        
      caffe_add(inner_num_, scale_data, xnext_ptr, scale_data); //scale data now has \sum_l xnow_l*exp(-t_k*f'_l) 
    }
    // division
    for (int j = 0; j < channels; j++) {
      xnext_ptr = xnext + i * dim + j * inner_num_;  
      caffe_div(inner_num_, xnext_ptr, scale_data, xnext_ptr);
    }
  }
}

template <typename Dtype>
void EntropyDescentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* xnext_diff = top[0]->cpu_diff();
  const Dtype* xnext_data = top[0]->cpu_data();
  Dtype* xnow_diff = bottom[0]->mutable_cpu_diff();
  Dtype* xnow_data = bottom[0]->mutable_cpu_data();
  Dtype* derivative_diff = bottom[1]->mutable_cpu_diff();

  int channels = top[0]->shape(prob_axis_);
  int dim = top[0]->count() / outer_num_;

  //use scale_data to store intermediate data needed
  Dtype* scale_data = scale_.mutable_cpu_data();
  Dtype* tmp_data = tmp_.mutable_cpu_data();

  // start by calculating common term for both xnow_diff and derivative_diff (dL/dxnext - \sum_j dL/dxnex * xnext_j)
  caffe_copy(bottom[0]->count(), xnext_diff, xnow_diff);
  for (int i = 0; i < outer_num_; ++i) {
    //calculate the \sum_j dL/dxnex * xnext_j and store it in scale data      
    caffe_set(inner_num_, Dtype(0), scale_data);
    for (int c = 0; c < channels; ++c) {
      caffe_copy(inner_num_, xnext_data + i*dim + c*inner_num_, tmp_data);
      caffe_mul(inner_num_, tmp_data, xnext_diff + i*dim + c*inner_num_, tmp_data); //multiply by dL/dxnext
      caffe_add(inner_num_, scale_data, tmp_data, scale_data);  
    }
    //subtract it from dL/dxnow
    for (int c = 0; c < channels; ++c) {
        caffe_sub(inner_num_, xnow_diff + i*dim + c*inner_num_, scale_data, xnow_diff + i*dim + c*inner_num_);
    }            
  }

  //caffe::PrintBlob(top[0], true, "xnext diff");
  //caffe::PrintBlob(bottom[0], true, "xnow diff first");

  // copy to derivative_diff
  caffe_copy(bottom[0]->count(), xnow_diff, derivative_diff);

  //calc last part of xnow_diff

  //old version
  //caffe_mul(bottom[0]->count(), xnow_diff, xnext_data, xnow_diff); 
  //caffe_div(bottom[0]->count(), xnow_diff, xnow_data, xnow_diff);

  //new version
  //need to handle division by zero, or small number 
  for(int i = 0; i < bottom[0]->count() ; i++){
      if(xnow_data[i] < div_epsilon_){
        if(xnext_data[i] >= div_epsilon_){
          xnow_diff[i] *= xnext_data[i]/(xnow_data[i] + div_epsilon_); //denominator below div_epsilon_, need to add smalle number to avoid numerical instabilities
        } //if both are below div_epsilon_, nothing needs to be done (multiply by 1)
      }else{
        xnow_diff[i] *= xnext_data[i]/xnow_data[i]; //division can be done without any numerical trouble
      }
  }
  
  //caffe::PrintBlob(bottom[0], true, "xnow diff final");

  //calc last part of derivative_diff
  caffe_mul(bottom[0]->count(), derivative_diff, xnext_data, derivative_diff); 
  caffe_scal(bottom[0]->count(), -1*t_k_, derivative_diff);

  //caffe::PrintBlob(bottom[1], true, "derr diff final");
}


#ifdef CPU_ONLY
STUB_GPU(EntropyDescentLayer);
#endif

INSTANTIATE_CLASS(EntropyDescentLayer);
REGISTER_LAYER_CLASS(EntropyDescent);
}  // namespace caffe
