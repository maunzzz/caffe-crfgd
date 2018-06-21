#ifndef CAFFE_CRFFILTERBANKED_LAYER_HPP_
#define CAFFE_CRFFILTERBANKED_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <boost/shared_array.hpp>
#include "caffe/layers/entropy_descent_layer.hpp"

namespace caffe {

/**
 * @brief
 */

template <typename Dtype>
class CrfFilterbankEdLayer : public Layer<Dtype>  {
 public:
    explicit CrfFilterbankEdLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "CrfFilterbankEd"; }

 protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    
    // ggdr specific parameters
    Dtype step_size_;
    Dtype step_size_factor_;
    Dtype log_eps_;
    int num_iterations_;
    int kernel_size_;
    bool debug_mode_;
    Dtype unary_weight_init_;
    bool use_cudnn_;
    bool log_time_;
    bool no_redundancy_weights_; 
    int num_; 
   	int num_pixels_;
    int num_classes_;
    int num_filters_per_class_;
    bool skip_pw_term_;
    bool keep_mid_ind_zero_;
   	
    shared_ptr<Layer<Dtype> > conv_layer_;
	  shared_ptr<EntropyDescentLayer<Dtype> > ed_layer_;

    // Intermediate states need to be saved internally
    vector<Blob<Dtype>*> internal_intermediate_states_;
    vector<Blob<Dtype>*> conv_top_blobs_;
    
    vector<int> conv_top_blob_shape_;
    bool has_updated_internal_states;

};

}  // namespace caffe

#endif  // CAFFE_CRFFILTERBANKED_LAYER_HPP_
