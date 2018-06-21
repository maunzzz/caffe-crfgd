#include <vector>

#include "caffe/layers/edge_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void calculate_multmask(const int n, const Dtype* in, Dtype* mask) {
  CUDA_KERNEL_LOOP(index, n) {
     mask[index] = (in[index] == 255) ? Dtype(0) : Dtype(1);
  }
}

template <typename Dtype>
void EdgeLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  calculate_multmask<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,bottom[1]->gpu_data(),multmask_.mutable_gpu_data());

  // scale
  caffe_gpu_scale(count, Dtype(1)/Dtype(256), bottom[1]->gpu_data(), scaledbottom_.mutable_gpu_data());

  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      scaledbottom_.gpu_data(),
      diff_.mutable_gpu_data());

  // multiply with mask to zero out 255 inds          
  caffe_gpu_mul(count, multmask_.gpu_data(), diff_.gpu_data(), diff_.mutable_gpu_data());

  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EdgeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EdgeLossLayer);

}  // namespace caffe
