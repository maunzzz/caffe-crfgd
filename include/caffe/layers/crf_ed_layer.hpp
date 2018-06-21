#ifndef CAFFE_CRFED_LAYER_HPP_
#define CAFFE_CRFED_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/permutohedral.hpp"
#include <boost/shared_array.hpp>
#include "caffe/layers/entropy_descent_layer.hpp"

namespace caffe {

/**
 * @brief
 */

template <typename Dtype>
class CrfEdLayer : public Layer<Dtype>  {
 public:
    explicit CrfEdLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "CrfEd"; }

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
    Dtype step_size_factor_;
    Dtype log_eps_;
    int num_iterations_;
    int num_classes_;
    int kernel_size_;
    bool debug_mode_;
    Dtype pl_filter_init_scale_;
    Dtype unary_weight_init_;
    bool use_cudnn_;
    bool calculate_energy_;
    bool log_time_;
    double start_time_, start_time_whole_, stop_time_;
    bool print_time_;
    Dtype clip_pl_gradients_;
    bool skip_pl_term_;
    bool skip_conv_term_;
    float spatial_filter_init_weight_;

    //int writecount_;

    Blob<Dtype> internal_lattice_blob_;
     
   	int num_pixels_;
   	Dtype step_size_;

	shared_ptr<Layer<Dtype> > pl_layer_;
	shared_ptr<Layer<Dtype> > conv_layer_;
    shared_ptr<EntropyDescentLayer<Dtype> > ed_layer_;
	
    // Intermediate states need to be saved internally
    vector<Blob<Dtype>*> internal_intermediate_states_;
    bool has_updated_internal_states;
    private:
        void ToMat(const char *fname, bool write_diff, Blob<Dtype>* blobtowrite);
};

}  // namespace caffe

#endif  // CAFFE_CRFED_LAYER_HPP_
