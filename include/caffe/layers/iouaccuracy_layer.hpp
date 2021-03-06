#ifndef CAFFE_IOUACCURACY_LAYER_HPP_
#define CAFFE_IOUACCURACY_LAYER_HPP_
#include <string>
#include <utility>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <boost/shared_array.hpp>

namespace caffe {

template <typename Dtype>
class IOUAccuracyLayer : public LossLayer<Dtype>{

 public:
  explicit IOUAccuracyLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
        softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
  virtual inline const char* type() const { return "IOUAccuracy"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int pixel_num_;
  int test_iterations_;
  int current_count_;

  boost::shared_array<int> pred_labels_;
  boost::shared_array<int> gt_counts_;
  boost::shared_array<int> pred_counts_;
  boost::shared_array<int> intersect_counts_;

  /// The internal SoftmaxLayer used to map predictions to a distribution.
  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// Whether to normalize the loss by the total number of values present
  /// (otherwise just by the batch size).
  bool normalize_;
};
}

#endif



