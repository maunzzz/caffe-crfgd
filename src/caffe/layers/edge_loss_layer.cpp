#include <vector>

#include "caffe/layers/edge_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EdgeLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
 
  diff_.ReshapeLike(*bottom[0]);
  multmask_.ReshapeLike(*bottom[0]);
  scaledbottom_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EdgeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  // calc mask
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* mask_data = multmask_.mutable_cpu_data();
  for(int i = 0; i < count ; i++){
      mask_data[i] = (label_data[i] == 255) ? Dtype(0) : Dtype(1);
  }

  // scale
  caffe_cpu_scale(count, Dtype(1)/Dtype(256), bottom[1]->cpu_data(), scaledbottom_.mutable_cpu_data());

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      scaledbottom_.cpu_data(),
      diff_.mutable_cpu_data());

  // multiply with mask to zero out 255 inds          
  caffe_mul(count, multmask_.cpu_data(), diff_.cpu_data(), diff_.mutable_cpu_data());

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EdgeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EdgeLossLayer);
#endif

INSTANTIATE_CLASS(EdgeLossLayer);
REGISTER_LAYER_CLASS(EdgeLoss);

}  // namespace caffe
