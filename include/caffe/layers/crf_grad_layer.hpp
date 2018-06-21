#ifndef CAFFE_CRFGRAD_LAYER_HPP_
#define CAFFE_CRFGRAD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/permutohedral.hpp"
#include <boost/shared_array.hpp>

namespace caffe {

/**
 * @brief
 */

template <typename Dtype>
class CrfGradLayer : public Layer<Dtype>  {
 public:
    explicit CrfGradLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "CrfGrad"; }

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
    Dtype leak_factor_;
    Dtype step_size_;
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
    
    Blob<Dtype> internal_lattice_blob_;
     
   	int num_pixels_;
   	
	shared_ptr<Layer<Dtype> > pl_layer_;
	shared_ptr<Layer<Dtype> > conv_layer_;
	
    // Intermediate states need to be saved internally
    vector<Blob<Dtype>*> internal_intermediate_states_;
    bool has_updated_internal_states;

 private:
    void ProjectBlobSimplex_(Blob<Dtype>* blob);
    void ProjectOntoSimplexBw_(const Blob<Dtype>* in,const Blob<Dtype>* outder, Blob<Dtype>* inder);
    void ProjectBlobSimplex_gpu_(Blob<Dtype>* blob, Blob<Dtype>* temp_mem);
    void ProjectOntoSimplexBw_gpu_(const Blob<Dtype>* in,const Blob<Dtype>* outder, Blob<Dtype>* inder, Blob<Dtype>* temp_mem);
};

}  // namespace caffe

#endif  // CAFFE_CRFGRAD_LAYER_HPP_
