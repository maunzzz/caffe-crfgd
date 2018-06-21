#include <vector>
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/iouaccuracy_layer.hpp"

#include <cmath>
#include <string>
#include <stdio.h>
#include <time.h>

#include <cfloat>

namespace caffe {

template <typename Dtype>
void IOUAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  /** Softmax layer init **/
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.iou_accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.iou_accuracy_param().ignore_label();
  }
  normalize_ = this->layer_param_.iou_accuracy_param().normalize();

  /** Init related to IOU accuracy **/
  test_iterations_ = this->layer_param_.iou_accuracy_param().test_iterations();
  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  pixel_num_ = height_ * width_;

  pred_labels_.reset(new int[pixel_num_]);
  caffe_set(pixel_num_, 0, pred_labels_.get());

  gt_counts_.reset(new int[channels_]);
  pred_counts_.reset(new int[channels_]);
  intersect_counts_.reset(new int[channels_]);

  current_count_ = 0;
  caffe_set(channels_, 0, gt_counts_.get());
  caffe_set(channels_, 0, pred_counts_.get());
  caffe_set(channels_, 0, intersect_counts_.get());

  //if (num_ != 1 || channels_ != 21) {
  //  printf("Violation of assertions num_ = %d, channels = %d\n. Halting.", num_, channels_);
  //  exit(1);
 // }

  LOG(INFO) << "Configured IOU Accuracy layer. test_iterations " << test_iterations_ << "\n";
}

template <typename Dtype>
void IOUAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}


/**
 * bottom[0] - Probabilities
 * bottom[1] - Labels
 *
 * top[0] - Softmax loss
 */
template <typename Dtype>
void IOUAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Playing safe.
  if (current_count_ == 0) {
    caffe_set(channels_, 0, gt_counts_.get());
    caffe_set(channels_, 0, pred_counts_.get());
    caffe_set(channels_, 0, intersect_counts_.get());
  }

  const Dtype* likelihood_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();

  for (int i = 0; i < pixel_num_; ++i) {

    if (static_cast<int>(gt_data[i]) == ignore_label_) {
      continue;
    }

    Dtype cur_max = likelihood_data[i];
    int cur_label = 0;
    for (int c = 1; c < channels_; ++c) {
      Dtype cur_val = likelihood_data[c * pixel_num_ + i];
      if (cur_val > cur_max) {
        cur_max = cur_val;
        cur_label = c;
      }
    }
    pred_labels_[i] = cur_label;
  }

  for (int i = 0; i < pixel_num_; ++i) {

    const int gt_value = static_cast<int>(gt_data[i]);
    if (gt_value == ignore_label_) {
          continue;
    }
    const int pred_value = pred_labels_[i];

    /*//TODO: Remove this!!!
    if (gt_value >= channels_ || pred_value >= channels_) {
      CHECK_GE(1, 2);
      printf("Busted!!!!!\n");
      exit(1);
    }*/

    ++pred_counts_[pred_value];
    ++gt_counts_[gt_value];

    if (pred_value == gt_value) {
      ++intersect_counts_[pred_value];
    }
  }

  ++current_count_;

  if (current_count_ == test_iterations_) {

    double tot = 0.0;

    for (int c = 0; c < channels_; ++c) {
      double cur_acc = 100 * (static_cast<double>(intersect_counts_[c]) / static_cast<double>(gt_counts_[c] + pred_counts_[c] - intersect_counts_[c]+1E-10));
      tot += cur_acc;
      // printf("Accuracy for class = %d is %.3f\n", c, cur_acc);
    }

    LOG(INFO) << "IOU Accuracy " << (tot / channels_);

    // reset variables
    current_count_ = 0;
    caffe_set(channels_, 0, gt_counts_.get());
    caffe_set(channels_, 0, pred_counts_.get());
    caffe_set(channels_, 0, intersect_counts_.get());
  }

  // Softmax loss calculation. Copied from that layer.
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      const int label_value = static_cast<int>(label[i * spatial_dim + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.channels());
      loss -= log(std::max(prob_data[i * dim + label_value * spatial_dim + j], Dtype(FLT_MIN)));
      ++count;
    }
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / num;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template<typename Dtype>
void IOUAccuracyLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  LOG(INFO) << "Backward_cpu() on IOU Accuracy. Should NOT happen.";
  exit(1);
}

INSTANTIATE_CLASS(IOUAccuracyLayer);
REGISTER_LAYER_CLASS(IOUAccuracy);
}  // namespace caffe
