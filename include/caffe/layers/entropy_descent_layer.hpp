#ifndef CAFFE_ENTROPY_DESCENT_LAYER_HPP_
#define CAFFE_ENTROPY_DESCENT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class EntropyDescentLayer : public Layer<Dtype> {
 public:
  explicit EntropyDescentLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EntropyDescent"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  inline void SetUpdateStep(Dtype newval) {t_k_ = newval;}
  inline void SetDivEpsilon(Dtype newval) {div_epsilon_ = newval;}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype t_k_; // the "step size"
  Dtype div_epsilon_; //to avoid numerical instabilities when dividing
  int outer_num_;
  int inner_num_;
  int prob_axis_;
  /// scale and tmp is an intermediate Blob to hold temporary results.
  Blob<Dtype> scale_;
  Blob<Dtype> tmp_;
};

}  // namespace caffe

#endif  // CAFFE_ENTROPY_DESCENT_LAYER_HPP_
