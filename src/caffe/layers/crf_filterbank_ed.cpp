#include <vector>
#include <algorithm>
#include <math.h>

#include "caffe/layers/crf_filterbank_ed.hpp"
#include "caffe/layers/entropy_descent_layer.hpp"
#include "caffe/layer.hpp"

#include "caffe/filler.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/cudnn_conv_layer.hpp"

#include <iostream>

namespace caffe
{

/*
bottom[0]: NxCxHxW - q0 values 
bottom[1]: NxFxHxW - mapping to feature dimension for each pixel (used by feat_dim_filter)

top[0]: NxCxHxW - qT values 
*/
template <typename Dtype>
void CrfFilterbankEdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top)
{
    // Read parameters
    const caffe::CrfFilterbankEdParameter crf_param = this->layer_param_.crffilterbanked_param();

    num_iterations_ = crf_param.num_iterations(); //number of gradient descent steps
    step_size_factor_ = crf_param.step_size_factor();           //step length for gradient descent
    log_eps_ = crf_param.log_eps();               //epsilon for calculating the unary weights u = -log(x + log_eps_);
    debug_mode_ = crf_param.debug_mode();         //debug_mode initializes pw weight to non-zero
    kernel_size_ = crf_param.kernel_size();
    unary_weight_init_ = crf_param.unary_weight_init();
    use_cudnn_ = crf_param.use_cudnn();
    no_redundancy_weights_ = crf_param.no_redundancy_weights();
    skip_pw_term_ = crf_param.skip_pw_term();
    keep_mid_ind_zero_ = crf_param.keep_mid_ind_zero();

    // other settings
    num_ = bottom[0]->shape(0);
    num_classes_ = bottom[0]->shape(1); //number of classes for segmentation task (defines number of input and output channels)
    num_filters_per_class_ = bottom[1]->shape(1); //number of filter per class (defines number of output channels for conv)
    num_pixels_ = bottom[0]->shape(2)*bottom[0]->shape(3);

    //initialize blobs for all internal states (need to save x and f')
    internal_intermediate_states_.resize(num_iterations_-1);
    for(int i = 0; i < num_iterations_-1; ++i) {
        Blob<Dtype> *tmp = new Blob<Dtype>(bottom[0]->shape());
        internal_intermediate_states_[i] = tmp;
    }
    has_updated_internal_states = false;

    //initialize blobs for all conv outputs
    conv_top_blob_shape_.clear();
    conv_top_blob_shape_.push_back(num_);
    conv_top_blob_shape_.push_back(num_classes_*num_filters_per_class_);
    conv_top_blob_shape_.push_back(bottom[0]->shape(2));
    conv_top_blob_shape_.push_back(bottom[0]->shape(3));
    

    conv_top_blobs_.resize(num_iterations_);
    for (int i = 0; i < num_iterations_; ++i)
    {
        Blob<Dtype> *tmp = new Blob<Dtype>(conv_top_blob_shape_);
        conv_top_blobs_[i] = tmp;
    }

    // set parameters and create internal cudnn conv layer
    LayerParameter layer_param;
    ConvolutionParameter *conv_param = layer_param.mutable_convolution_param();
    conv_param->set_num_output(num_classes_*num_filters_per_class_);
    conv_param->set_bias_term(false);
    conv_param->add_pad((kernel_size_ - 1) / 2);
    conv_param->add_kernel_size(kernel_size_);

    conv_param->mutable_weight_filler()->set_type("xavier");

    if (use_cudnn_)
    {
        conv_layer_.reset(new CuDNNConvolutionLayer<Dtype>(layer_param));
    }
    else
    {
        conv_layer_.reset(new ConvolutionLayer<Dtype>(layer_param));
    }

    vector<Blob<Dtype> *> tmp_bottom;
    tmp_bottom.push_back(bottom[0]);

    vector<Blob<Dtype> *> tmp_top;
    tmp_top.push_back(conv_top_blobs_[0]);

    conv_layer_->SetUp(tmp_bottom, tmp_top);

    // set parameters and create internal entropy descent layer
    LayerParameter layer_param2;
    ed_layer_.reset(new EntropyDescentLayer<Dtype>(layer_param2));
    tmp_bottom.push_back(bottom[0]);
    ed_layer_->SetUp(tmp_bottom,top);
    ed_layer_->SetDivEpsilon(crf_param.div_epsilon());

    // Handle the parameters: weights
    // - blobs_[0] holds the unary term
    // - blobs_[1] holds the 2d filter weights (pairwise terms)

    // if blob already initialized
    if (this->blobs_.size() > 0)
    {
        CHECK_EQ(2, this->blobs_.size())
            << "Incorrect number of weight blobs.";

        LOG(INFO) << "Skipping parameter initialization";
    }
    else
    {
        this->blobs_.resize(2);
        // Initialize and fill the weights:
        this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
        FillerParameter filler_param;
        filler_param.set_value(unary_weight_init_);
        ConstantFiller<Dtype> weight_filler1(filler_param);
        weight_filler1.Fill(this->blobs_[0].get());

        // initialize conv filter weights
        vector<int> weight_shape(4);
        weight_shape[0] = num_classes_;
        weight_shape[1] = num_classes_*num_filters_per_class_;
        weight_shape[2] = kernel_size_;
        weight_shape[3] = kernel_size_;

        if(crf_param.init_filters()){
            this->blobs_[1].reset(new Blob<Dtype>(weight_shape));

            CHECK_EQ(num_filters_per_class_,3) << "Filter initialization only supported if num_filters_pre_class_ is 3";
            LOG(INFO) << "Initializing filters";
            // Small gaussian noise on all parameters
            filler_param.set_mean(0);
            filler_param.set_std(0.001);
            GaussianFiller<Dtype> weight_filler2(filler_param);
            weight_filler2.Fill(this->blobs_[1].get());

            Dtype *weights_pw = this->blobs_[1].get()->mutable_cpu_data();
            Dtype expdist;
            int mid_ind = kernel_size_ / 2;
            Dtype sign_factor = Dtype(1);
            for (int class_out = 0; class_out < num_classes_; ++class_out) 
            {
                for (int class_in = 0; class_in < num_classes_; ++class_in)
                {
                    if (class_in != class_out)
                    {
                        sign_factor = Dtype(1);//sign_factor = Dtype(1);
                    }else{
                        sign_factor = Dtype(-1);// sign_factor = Dtype(-1);
                    }
                    for(int x = 0; x < kernel_size_; x++)
                    {
                        for(int y = 0; y < kernel_size_; y++)
                        {
                            expdist = exp(Dtype(-1)*(pow(Dtype(x-mid_ind),2) + pow(Dtype(y-mid_ind),2))/(Dtype(kernel_size_)));
                            caffe_set(1,sign_factor*Dtype(0.1)*Dtype(expdist),weights_pw + this->blobs_[1]->offset(class_out,0*num_classes_+class_in,x,y)); //x-edge
                            caffe_set(1,sign_factor*Dtype(0.1)*Dtype(expdist),weights_pw + this->blobs_[1]->offset(class_out,1*num_classes_+class_in,x,y)); //y-edge
                            caffe_set(1,sign_factor*Dtype(1)*Dtype(expdist),weights_pw + this->blobs_[1]->offset(class_out,2*num_classes_+class_in,x,y)); //no-edge                            
                        }
                    }
                }
            }
        }else{
            this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
            filler_param.set_value(0);
            ConstantFiller<Dtype> weight_filler2(filler_param);
            weight_filler2.Fill(this->blobs_[1].get());
            if (debug_mode_)
            {
                int mid_ind = kernel_size_ / 2;
                Dtype *weights_pw = this->blobs_[1].get()->mutable_cpu_data();
                for (int i = 0; i < num_classes_; ++i)
                {
                    for (int j = 0; j < num_classes_; ++j)
                    {
                        if (i != j)
                        {
                            weights_pw[((i * num_classes_ + j) * kernel_size_ + mid_ind + 1) * kernel_size_ + mid_ind] = Dtype(1);
                            weights_pw[((i * num_classes_ + j) * kernel_size_ + mid_ind - 1) * kernel_size_ + mid_ind] = Dtype(1);
                            weights_pw[((i * num_classes_ + j) * kernel_size_ + mid_ind) * kernel_size_ + mid_ind + 1] = Dtype(1);
                            weights_pw[((i * num_classes_ + j) * kernel_size_ + mid_ind) * kernel_size_ + mid_ind - 1] = Dtype(1);
                        }
                    }
                }
            }
        }
        if(no_redundancy_weights_) //zero out all derivative for class j < class i as well as mid pixel for all filters
        { 
            int mid_ind = kernel_size_ / 2;
            Dtype *weights_pw = this->blobs_[1]->mutable_cpu_data();
            for (int i = 0; i < num_classes_; ++i)
            {
                for (int j = 0; j < num_classes_; ++j)
                {
                    if(j < i){
                        caffe_set(kernel_size_*kernel_size_*num_filters_per_class_,Dtype(0),weights_pw + this->blobs_[1]->offset(i,num_filters_per_class_*j,0,0));                            
                    }else{ //only set mid pixel to zero
                        for (int k = 0; k < num_filters_per_class_; ++k){
                            caffe_set(1,Dtype(0),weights_pw + this->blobs_[1]->offset(i,num_filters_per_class_*j+k,mid_ind,mid_ind));
                        }
                    }
                }
            }
        }

        if(keep_mid_ind_zero_){
            Dtype *weights_pw1 = this->blobs_[1]->mutable_cpu_data();
            int mid_ind1 = kernel_size_ / 2;

            for (int i = 0; i < num_classes_; ++i)
            {
                for (int j = 0; j < num_classes_; ++j)
                {
                    for (int k = 0; k < num_filters_per_class_; ++k){                        
                        caffe_set(1,Dtype(0),weights_pw1 + this->blobs_[1]->offset(i,num_filters_per_class_*j+k,mid_ind1,mid_ind1));
                    }
                }    
            }
        }

        // sync with conv layer
        caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(), conv_layer_->blobs()[0]->mutable_cpu_data());
    }

    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CrfFilterbankEdLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top)
{
    // Reshape the internal states
    for(int i = 0; i < num_iterations_-1; ++i) {
        internal_intermediate_states_[i]->Reshape(bottom[0]->shape());
    }

    num_ = bottom[0]->shape(0);
    num_pixels_ = bottom[0]->shape(2)*bottom[0]->shape(3);

    conv_top_blob_shape_.clear();
    conv_top_blob_shape_.push_back(num_);
    conv_top_blob_shape_.push_back(num_classes_*num_filters_per_class_);
    conv_top_blob_shape_.push_back(bottom[0]->shape(2));
    conv_top_blob_shape_.push_back(bottom[0]->shape(3));

    for (int i = 0; i < num_iterations_; ++i)
    {
        conv_top_blobs_[i]->Reshape(conv_top_blob_shape_);
    }

    vector<Blob<Dtype> *> tmp_bottom;
    tmp_bottom.push_back(bottom[0]);

    vector<Blob<Dtype> *> tmp_top;
    tmp_top.push_back(conv_top_blobs_[0]);

    conv_layer_->Reshape(tmp_bottom, tmp_top);

    tmp_bottom.push_back(bottom[0]);
    ed_layer_->Reshape(tmp_bottom,top);

    top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void CrfFilterbankEdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top)
{
    // sync weights with internal layers
    caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(), conv_layer_->blobs()[0]->mutable_cpu_data());

    //Get pointers to weights
    const Dtype *weights_unary = this->blobs_[0]->cpu_data();

    //initialize unary term
    Blob<Dtype> unary(bottom[0]->shape());

    //calculate unary as -1*weight_unary*log(x+log_eps), x is input data (bottom)
    // These unary terms are constant during gradient descent steps
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), unary.mutable_cpu_data()); //set to x
    Dtype *unary_vals = unary.mutable_cpu_data();
    caffe_add_scalar(unary.count(), log_eps_, unary.mutable_cpu_data()); //add log_eps
    caffe_log(unary.count(), unary.cpu_data(), unary.mutable_cpu_data());
    caffe_scal(unary.count(), Dtype(-1.0) * weights_unary[0], unary_vals);

    Blob<Dtype> current_state(bottom[0]->shape());
    current_state.CopyFrom(*bottom[0]); //initialize current state as input

    vector<Blob<Dtype> *> current_state_vec;
    current_state_vec.push_back(&current_state);

    Blob<Dtype> derivative_blob(bottom[0]->shape()); // helper blob for storing the derivative (of gradient descent steps)
    Dtype *derivative_vals = derivative_blob.mutable_cpu_data();
    vector<Blob<Dtype> *> derivative_blob_vec;
    derivative_blob_vec.push_back(&derivative_blob);

    // output for conv layer    
    vector<Blob<Dtype> *> conv_out_blob_vec;
    conv_out_blob_vec.resize(1);

    // temp blobs
    Blob<Dtype> temp_blob(conv_top_blob_shape_);
    Blob<Dtype> tmp_der_blob(bottom[0]->shape());

    // ed input output    
    vector<Blob<Dtype>*> ed_bottom_vec;
    ed_bottom_vec.push_back(&current_state);
    ed_bottom_vec.push_back(&derivative_blob);

    vector<Blob<Dtype>*> ed_top_vec;
    ed_top_vec.push_back(&tmp_der_blob);

    //LOOP OVER GRADIENT DESCENT STEPS
    for (int it = 0; it < num_iterations_; ++it)
    {
        //std::cout << "ITERATION: " << it << std::endl;

        //Calculate pairwise part with pl and spatial filtering
        conv_out_blob_vec[0] = conv_top_blobs_[it];
        conv_layer_->Forward(current_state_vec, conv_out_blob_vec);        

        // Summation over "feature values"      
        for (int n = 0; n < num_ ; n++){ //loop over batch
            for (int c = 0; c < num_classes_; c++){ //loop over class
                // multiply filter output with "feature values"
                caffe_mul(num_pixels_*num_filters_per_class_
                    ,bottom[1]->cpu_data() + bottom[1]->offset(n,0,0,0)
                    ,conv_out_blob_vec[0]->cpu_data() + conv_out_blob_vec[0]->offset(n,c*num_filters_per_class_,0,0)
                    ,temp_blob.mutable_cpu_data() + temp_blob.offset(n,c*num_filters_per_class_,0,0));

                //sum over feature dimension                        
                caffe_copy(num_pixels_
                    ,temp_blob.cpu_data() + temp_blob.offset(n,c*num_filters_per_class_,0,0)
                    ,derivative_blob.mutable_cpu_data() + derivative_blob.offset(n,c,0,0));
                for (int f = 1; f < num_filters_per_class_ ; f++){
                    caffe_add(num_pixels_
                        ,derivative_blob.cpu_data() + derivative_blob.offset(n,c,0,0)
                        ,temp_blob.cpu_data() + temp_blob.offset(n,c*num_filters_per_class_+f,0,0)
                        ,derivative_blob.mutable_cpu_data() + derivative_blob.offset(n,c,0,0));
                }                    
            }
        }

        //Add unary and pairwise gradient contribution (derivative_blob now contains the derivative before all projections)
        if(skip_pw_term_){
            caffe_copy(derivative_blob.count(), unary.cpu_data(), derivative_vals);        
        }else{
            caffe_add(derivative_blob.count(), derivative_blob.cpu_data(), unary.cpu_data(), derivative_vals);
        }

        // save intermediate steps
        // Push current state on intermedieate state vector (we need values before the last projection to be able to calculate backward derivatives)
        if(it > 0){
            caffe_copy(internal_intermediate_states_[it-1]->count(),current_state.cpu_data(),internal_intermediate_states_[it-1]->mutable_cpu_data());
        }

        //Do entropy descent update step
        step_size_ = sqrt(Dtype(2)*log(num_classes_)/(it+1))/step_size_factor_;
        ed_layer_->SetUpdateStep(step_size_);
        ed_layer_->Forward(ed_bottom_vec,ed_top_vec);
        caffe_copy(tmp_der_blob.count(),tmp_der_blob.cpu_data(),current_state.mutable_cpu_data()); //copy output to current state            
    }
    //after iteration is done, copy current state to top (output)
    
    caffe_copy(top[0]->count(), current_state.cpu_data(), top[0]->mutable_cpu_data());
    has_updated_internal_states = true;
}

template <typename Dtype>
void CrfFilterbankEdLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                            const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom)
{

    //set bottom[1] diff to zero, accumulates over iterations
    caffe_set(bottom[1]->count(),Dtype(0),bottom[1]->mutable_cpu_diff());
    Blob<Dtype> tmp_feat_bottom_blob(bottom[1]->shape());

    CHECK_EQ(has_updated_internal_states, true)
        << "No updated internal states found, forward must be done previous to backward";

    const Dtype *bottom_vals = bottom[0]->cpu_data();
    const Dtype *weight_vals_u = this->blobs_[0]->cpu_data();    

    // temporary variables that hold weight diffs during iterations
    Blob<Dtype> tmp_weight_diff_pw(this->blobs_[1]->shape());

    // dL_dtheta_u
    Blob<Dtype> dL_dtheta_u(top[0]->shape());
    Dtype *dL_dtheta_u_vals = dL_dtheta_u.mutable_cpu_data();

    //blobs for ed
    Blob<Dtype> blob_x_kplusone(top[0]->shape());
    Blob<Dtype> blob_x_k(top[0]->shape());
    Blob<Dtype> blob_fprim_k(top[0]->shape());

    // conv input/output
    vector<Blob<Dtype>*> conv_bottom;
    conv_bottom.push_back(&blob_x_k);
    vector<Blob<Dtype> *> conv_top;
    conv_top.resize(1);    

    //ed input/output
    vector<Blob<Dtype>*> ed_bottom;
    ed_bottom.push_back(&blob_x_k);
    ed_bottom.push_back(&blob_fprim_k);
		
    vector<Blob<Dtype>*> ed_top;
    ed_top.push_back(&blob_x_kplusone);

    // Start by setting dL/d(xplusone) = dL/d(out)
    caffe_copy(blob_x_kplusone.count(), top[0]->cpu_diff(), blob_x_kplusone.mutable_cpu_diff());

    //Loop backwards through gradient iterations
    for (int it = num_iterations_ - 1; it >= 0; --it)
    {
        //std::cout << "ITERATION " << it << std::endl;

        if(it > 0){
            caffe_copy(blob_x_k.count(), internal_intermediate_states_[it-1]->cpu_data(), blob_x_k.mutable_cpu_data());
        }else{
            caffe_copy(blob_x_k.count(), bottom[0]->cpu_data(), blob_x_k.mutable_cpu_data());
        }
        if(it == num_iterations_-1){
            caffe_copy(blob_x_kplusone.count(), top[0]->cpu_data(), blob_x_kplusone.mutable_cpu_data());
        }else{
            caffe_copy(blob_x_kplusone.count(), internal_intermediate_states_[it]->cpu_data(), blob_x_kplusone.mutable_cpu_data());
        }

        //get output blob from corresponding iteration
        conv_top[0] = conv_top_blobs_[it];

        //Backward through entropy descent update step
        step_size_ = sqrt(Dtype(2)*log(num_classes_)/(it+1))/step_size_factor_;
        ed_layer_->SetUpdateStep(step_size_);
        ed_layer_->Backward(ed_top,propagate_down,ed_bottom);

        //Update dL/dxkplusone to next iteration (the der for this iteration is no longer needed)                        
        caffe_copy(blob_x_kplusone.count(), blob_x_k.cpu_diff(), blob_x_kplusone.mutable_cpu_diff());

        // Backward through summing of derivatives with features
        for (int n = 0; n < num_ ; n++){
            for (int c = 0; c < num_classes_; c++){
                for (int f = 0; f < num_filters_per_class_; f++){
                    // multiply filter output with "feature values"
                    caffe_mul(num_pixels_
                        ,bottom[1]->cpu_data() + bottom[1]->offset(n,f,0,0)
                        ,blob_fprim_k.cpu_diff() + blob_fprim_k.offset(n,c,0,0)
                        ,conv_top[0]->mutable_cpu_diff() + conv_top[0]->offset(n,c*num_filters_per_class_+f,0,0));
                }
            }
        }

        //backward through spatial filtering
        caffe_set(conv_layer_->blobs()[0]->count(), Dtype(0), conv_layer_->blobs()[0]->mutable_cpu_diff());
        conv_layer_->Backward(conv_top, propagate_down, conv_bottom);
        caffe_copy(tmp_weight_diff_pw.count(), conv_layer_->blobs()[0]->cpu_diff(), tmp_weight_diff_pw.mutable_cpu_diff());
        caffe_add(blob_x_kplusone.count(), conv_bottom[0]->cpu_diff(), blob_x_kplusone.cpu_diff(), blob_x_kplusone.mutable_cpu_diff()); //add spatial dL/dxkplusone df/dy dy/dxk term

        //Backward through feature weights
        caffe_set(tmp_feat_bottom_blob.count(),Dtype(0),tmp_feat_bottom_blob.mutable_cpu_diff());       
        for (int n = 0; n < num_ ; n++){
            for (int c = 0; c < num_classes_; c++){
                for (int f = 0; f < num_filters_per_class_; f++){
                    caffe_mul(num_pixels_
                        ,conv_top[0]->cpu_data() + conv_top[0]->offset(n,c*num_filters_per_class_+f,0,0)
                        ,blob_fprim_k.cpu_diff() + blob_fprim_k.offset(n,c,0,0)
                        ,conv_top[0]->mutable_cpu_diff() + conv_top[0]->offset(n,c*num_filters_per_class_+f,0,0));
                    
                    
                    caffe_add(num_pixels_
                        ,tmp_feat_bottom_blob.cpu_diff() + tmp_feat_bottom_blob.offset(n,f,0,0)
                        ,conv_top[0]->cpu_diff() + conv_top[0]->offset(n,c*num_filters_per_class_+f,0,0)
                        ,tmp_feat_bottom_blob.mutable_cpu_diff() + tmp_feat_bottom_blob.offset(n,f,0,0));
                }
            }
        }

        // accumulate feature bottom diff vals
        caffe_add(tmp_feat_bottom_blob.count(), tmp_feat_bottom_blob.cpu_diff(), bottom[1]->cpu_diff(), bottom[1]->mutable_cpu_diff());

        // accumulate pairwise spatial weight diff vals
        caffe_add(tmp_weight_diff_pw.count(), tmp_weight_diff_pw.cpu_diff(), this->blobs_[1]->cpu_diff(), this->blobs_[1]->mutable_cpu_diff());

        // accumulate dL_dtheta_u dL_dtheta_u_vals
        caffe_add(dL_dtheta_u.count(), blob_fprim_k.cpu_diff(), dL_dtheta_u.cpu_data(), dL_dtheta_u.mutable_cpu_data());
    }

    //caffe_scal(bottom[1]->count(), Dtype(0.1), bottom[1]->mutable_cpu_diff()); // DEBUG

    // bottom diff
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* dL_dx0_vals = blob_x_kplusone.cpu_diff();
    for (int i = 0; i < bottom[0]->count(); ++i)
    {
        bottom_diff[i] = dL_dtheta_u_vals[i] * Dtype(-1.0) * weight_vals_u[0] / (bottom_vals[i] + log_eps_) + dL_dx0_vals[i];
        //bottom_diff[i] = Dtype(0);
    }

    Dtype *weight_diff_u = this->blobs_[0]->mutable_cpu_diff();
    weight_diff_u[0] = Dtype(0);
    for (int i = 0; i < bottom[0]->count(); ++i)
    {
        weight_diff_u[0] += dL_dtheta_u_vals[i] * Dtype(-1) * log(bottom_vals[i] + log_eps_);
    }

    if(no_redundancy_weights_) //zero out all derivative for class j < class i as well as mid pixel for all filters
    { 
        int mid_ind = kernel_size_ / 2;
        Dtype *diff_weights_pw = this->blobs_[1]->mutable_cpu_diff();
        for (int i = 0; i < num_classes_; ++i)
        {
            for (int j = 0; j < num_classes_; ++j)
            {
                if(j < i){
                    caffe_set(kernel_size_*kernel_size_*num_filters_per_class_,Dtype(0),diff_weights_pw + this->blobs_[1]->offset(i,num_filters_per_class_*j,0,0));                            
                }else{ //only set mid pixel to zero
                    for (int k = 0; k < num_filters_per_class_; ++k){
                        caffe_set(1,Dtype(0),diff_weights_pw + this->blobs_[1]->offset(i,num_filters_per_class_*j+k,mid_ind,mid_ind));
                    }
                }
            }
        }
    }

    if(keep_mid_ind_zero_){
        Dtype *diff_pw1 = this->blobs_[1]->mutable_cpu_diff();
        int mid_ind1 = kernel_size_ / 2;

        for (int i = 0; i < num_classes_; ++i)
        {
            for (int j = 0; j < num_classes_; ++j)
            {
                for (int k = 0; k < num_filters_per_class_; ++k){                        
                    caffe_set(1,Dtype(0),diff_pw1 + this->blobs_[1]->offset(i,num_filters_per_class_*j+k,mid_ind1,mid_ind1));
                }
            }    
        }
    }

}


#ifdef CPU_ONLY
STUB_GPU(CrfFilterbankEdLayer);
#endif

INSTANTIATE_CLASS(CrfFilterbankEdLayer);
REGISTER_LAYER_CLASS(CrfFilterbankEd);
} // namespace caffe