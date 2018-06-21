#include <vector>
#include "caffe/layers/crf_filterbank_ed.hpp"

namespace caffe
{

// KERNEL FUNCTIONS
template <typename Dtype>
__global__ void scale_unary_weight_kernel(const int n, const Dtype *alpha, Dtype *y)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        y[index] *= alpha[0];
    }
}

template <typename Dtype>
__global__ void calc_bottom_diff(const int N, Dtype *bottom_diff, const Dtype *dL_dtheta_u, const Dtype *unary_weight, const Dtype *bottom_data, const Dtype log_eps, const Dtype *dL_dx_it)
{
    CUDA_KERNEL_LOOP(i, N)
    {
        bottom_diff[i] = dL_dtheta_u[i] * Dtype(-1.0) * unary_weight[0] / (bottom_data[i] + log_eps) + dL_dx_it[i];
    }
}

template <typename Dtype>
void CrfFilterbankEdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top)
{
    // SETUP 
    // sync weights with internal layer
    caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->gpu_data(), conv_layer_->blobs()[0]->mutable_gpu_data());

    // Initialize blobs needed through iterations
    Blob<Dtype> unary(bottom[0]->shape());

    Blob<Dtype> current_state(bottom[0]->shape());
    current_state.CopyFrom(*bottom[0]); //initialize current state as input
    vector<Blob<Dtype> *> current_state_vec;
    current_state_vec.push_back(&current_state);

    Blob<Dtype> derivative_blob(bottom[0]->shape()); // helper blob for storing the derivative (of gradient descent steps)
   
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

    // calculate unary weights
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), unary.mutable_gpu_data()); //set to x
    caffe_gpu_add_scalar(unary.count(), log_eps_, unary.mutable_gpu_data());         //add log_eps
    caffe_gpu_log(unary.count(), unary.gpu_data(), unary.mutable_gpu_data());
    scale_unary_weight_kernel<<<CAFFE_GET_BLOCKS(unary.count()), CAFFE_CUDA_NUM_THREADS>>>(unary.count(), this->blobs_[0]->gpu_data(), unary.mutable_gpu_data());
    caffe_gpu_scal(unary.count(), Dtype(-1), unary.mutable_gpu_data());

    //LOOP OVER GRADIENT DESCENT STEPS
    for (int it = 0; it < num_iterations_; ++it)
    {
        conv_out_blob_vec[0] = conv_top_blobs_[it];
        conv_layer_->Forward(current_state_vec, conv_out_blob_vec);

        // Summation over "feature values"      
        for (int n = 0; n < num_ ; n++){
            for (int c = 0; c < num_classes_; c++){
                //multiply filter output with "feature values"
                caffe_gpu_mul(num_pixels_*num_filters_per_class_
                    ,bottom[1]->gpu_data() + bottom[1]->offset(n,0,0,0)
                    ,conv_out_blob_vec[0]->gpu_data() + conv_out_blob_vec[0]->offset(n,c*num_filters_per_class_,0,0)
                    ,temp_blob.mutable_gpu_data() + temp_blob.offset(n,c*num_filters_per_class_,0,0));

                //sum over feature dimension                        
                caffe_copy(num_pixels_
                    ,temp_blob.gpu_data() + temp_blob.offset(n,c*num_filters_per_class_,0,0)
                    ,derivative_blob.mutable_gpu_data() + derivative_blob.offset(n,c,0,0));
                for (int f = 1; f < num_filters_per_class_ ; f++){
                    caffe_gpu_add(num_pixels_
                        ,derivative_blob.gpu_data() + derivative_blob.offset(n,c,0,0)
                        ,temp_blob.gpu_data() + temp_blob.offset(n,c*num_filters_per_class_+f,0,0)
                        ,derivative_blob.mutable_gpu_data() + derivative_blob.offset(n,c,0,0));
                }                    
            }
        }            

        //Add unary and pairwise gradient contribution
        if(skip_pw_term_){
            caffe_copy(derivative_blob.count(), unary.gpu_data(), derivative_blob.mutable_gpu_data());        
        }else{
            caffe_gpu_add(derivative_blob.count(), derivative_blob.gpu_data(), unary.gpu_data(), derivative_blob.mutable_gpu_data());
        }

        // save intermediate steps
        // Push current state on intermedieate state vector (we need values before the last projection to be able to calculate backward derivatives)
        if(it > 0){
            caffe_copy(internal_intermediate_states_[it-1]->count(),current_state.gpu_data(),internal_intermediate_states_[it-1]->mutable_gpu_data());
        }

        //Do entropy descent update step
        step_size_ = sqrt(Dtype(2)*log(num_classes_)/(it+1))/step_size_factor_;
        ed_layer_->SetUpdateStep(step_size_);
        ed_layer_->Forward(ed_bottom_vec,ed_top_vec);
        caffe_copy(tmp_der_blob.count(),tmp_der_blob.gpu_data(),current_state.mutable_gpu_data()); //copy output to current state            
    }

    //after iteration is done, copy current state to top (output)
    caffe_copy(top[0]->count(), current_state.gpu_data(), top[0]->mutable_gpu_data());
    has_updated_internal_states = true;
}

template <typename Dtype>
void CrfFilterbankEdLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                            const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom)
{
    CHECK_EQ(has_updated_internal_states, true)
        << "No updated internal states found, forward must be done previous to backward";

    //set bottom[1] diff to zero, accumulates over iterations
    caffe_gpu_set(bottom[1]->count(),Dtype(0),bottom[1]->mutable_gpu_diff());        

    const Dtype *bottom_vals = bottom[0]->gpu_data();
    const Dtype *weight_vals_u = this->blobs_[0]->gpu_data();    

    // temporary variables that hold weight diffs during iterations
    Blob<Dtype> tmp_weight_diff_pw(this->blobs_[1]->shape());
    Blob<Dtype> tmp_feat_bottom_blob(bottom[1]->shape());

    // dL_dtheta_u
    Blob<Dtype> dL_dtheta_u(top[0]->shape());
    Dtype *dL_dtheta_u_vals = dL_dtheta_u.mutable_gpu_data();

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
    caffe_copy(blob_x_kplusone.count(), top[0]->gpu_diff(), blob_x_kplusone.mutable_gpu_diff());

    //Loop backwards through gradient iterations
    for (int it = num_iterations_ - 1; it >= 0; --it)
    {
        if(it > 0){
            caffe_copy(blob_x_k.count(), internal_intermediate_states_[it-1]->gpu_data(), blob_x_k.mutable_gpu_data());
        }else{
            caffe_copy(blob_x_k.count(), bottom[0]->gpu_data(), blob_x_k.mutable_gpu_data());
        }
        if(it == num_iterations_-1){
            caffe_copy(blob_x_kplusone.count(), top[0]->gpu_data(), blob_x_kplusone.mutable_gpu_data());
        }else{
            caffe_copy(blob_x_kplusone.count(), internal_intermediate_states_[it]->gpu_data(), blob_x_kplusone.mutable_gpu_data());
        }

        //get output blob from corresponding iteration
        conv_top[0] = conv_top_blobs_[it];

        //Backward through entropy descent update step
        step_size_ = sqrt(Dtype(2)*log(num_classes_)/(it+1))/step_size_factor_;
        ed_layer_->SetUpdateStep(step_size_);
        ed_layer_->Backward(ed_top,propagate_down,ed_bottom);

        //Update dL/dxkplusone to next iteration (the der for this iteration is no longer needed)                        
        caffe_copy(blob_x_kplusone.count(), blob_x_k.gpu_diff(), blob_x_kplusone.mutable_gpu_diff());

        // Backward through summing of derivatives with features
        for (int n = 0; n < num_ ; n++){
            for (int c = 0; c < num_classes_; c++){
                for (int f = 0; f < num_filters_per_class_; f++){
                    // multiply filter output with "feature values"
                    caffe_gpu_mul(num_pixels_
                        ,bottom[1]->gpu_data() + bottom[1]->offset(n,f,0,0)
                        ,blob_fprim_k.gpu_diff() + blob_fprim_k.offset(n,c,0,0)
                        ,conv_top[0]->mutable_gpu_diff() + conv_top[0]->offset(n,c*num_filters_per_class_+f,0,0));
                }
            }
        }

        caffe_gpu_set(conv_layer_->blobs()[0]->count(), Dtype(0), conv_layer_->blobs()[0]->mutable_gpu_diff());
        conv_layer_->Backward(conv_top, propagate_down, conv_bottom);
        caffe_copy(tmp_weight_diff_pw.count(), conv_layer_->blobs()[0]->gpu_diff(), tmp_weight_diff_pw.mutable_gpu_diff());
        caffe_gpu_add(blob_x_kplusone.count(), conv_bottom[0]->gpu_diff(), blob_x_kplusone.gpu_diff(), blob_x_kplusone.mutable_gpu_diff()); //add spatial dL/dxkplusone df/dy dy/dxk term

        //Backward through feature weights
        caffe_gpu_set(tmp_feat_bottom_blob.count(),Dtype(0),tmp_feat_bottom_blob.mutable_gpu_diff());       
        for (int n = 0; n < num_ ; n++){
            for (int c = 0; c < num_classes_; c++){
                for (int f = 0; f < num_filters_per_class_; f++){
                    caffe_gpu_mul(num_pixels_
                        ,conv_top[0]->gpu_data() + conv_top[0]->offset(n,c*num_filters_per_class_+f,0,0)
                        ,blob_fprim_k.gpu_diff() + blob_fprim_k.offset(n,c,0,0)
                        ,conv_top[0]->mutable_gpu_diff() + conv_top[0]->offset(n,c*num_filters_per_class_+f,0,0));
                    
                    
                    caffe_gpu_add(num_pixels_
                        ,tmp_feat_bottom_blob.gpu_diff() + tmp_feat_bottom_blob.offset(n,f,0,0)
                        ,conv_top[0]->gpu_diff() + conv_top[0]->offset(n,c*num_filters_per_class_+f,0,0)
                        ,tmp_feat_bottom_blob.mutable_gpu_diff() + tmp_feat_bottom_blob.offset(n,f,0,0));
                }
            }
        }            

        // accumulate feature bottom diff vals
        caffe_gpu_add(tmp_feat_bottom_blob.count(), tmp_feat_bottom_blob.gpu_diff(), bottom[1]->gpu_diff(), bottom[1]->mutable_gpu_diff());

        // accumulate pairwise spatial weight diff vals
        caffe_gpu_add(tmp_weight_diff_pw.count(), tmp_weight_diff_pw.gpu_diff(), this->blobs_[1]->gpu_diff(), this->blobs_[1]->mutable_gpu_diff());

        // accumulate dL_dtheta_u dL_dtheta_u_vals
        caffe_gpu_add(dL_dtheta_u.count(), blob_fprim_k.gpu_diff(), dL_dtheta_u.gpu_data(), dL_dtheta_u.mutable_gpu_data());
    }
    // bottom diff
    calc_bottom_diff<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(), bottom[0]->mutable_gpu_diff(), dL_dtheta_u.gpu_data(), weight_vals_u, bottom[0]->gpu_data(), log_eps_, blob_x_kplusone.gpu_diff());

    //Use inp_prev_it as temporary storage
    caffe_copy(blob_x_k.count(), bottom[0]->gpu_data(), blob_x_k.mutable_gpu_data());
    caffe_gpu_add_scalar(blob_x_k.count(), log_eps_, blob_x_k.mutable_gpu_data()); //add log_eps
    caffe_gpu_log(blob_x_k.count(), blob_x_k.gpu_data(), blob_x_k.mutable_gpu_data());
    caffe_gpu_scal(blob_x_k.count(), Dtype(-1), blob_x_k.mutable_gpu_data());
    caffe_gpu_mul(blob_x_k.count(), blob_x_k.gpu_data(), dL_dtheta_u.gpu_data(), blob_x_k.mutable_gpu_data());

    //parallell way of summing elements in matrix (dot product with ones)
    caffe_gpu_set(blob_x_kplusone.count(), Dtype(1), blob_x_kplusone.mutable_gpu_data());
    Dtype sum = 0;
    caffe_gpu_dot(blob_x_kplusone.count(), blob_x_kplusone.gpu_data(), blob_x_k.gpu_data(), &sum);
    this->blobs_[0]->mutable_cpu_diff()[0] = sum;

    if(no_redundancy_weights_) //zero out all derivative for class j < class i as well as mid pixel for all filters
    { 
        int mid_ind = kernel_size_ / 2;
        Dtype *diff_weights_pw = this->blobs_[1]->mutable_gpu_diff();
        for (int i = 0; i < num_classes_; ++i)
        {
            for (int j = 0; j < num_classes_; ++j)
            {
                if(j < i){
                    caffe_gpu_set(kernel_size_*kernel_size_*num_filters_per_class_,Dtype(0),diff_weights_pw + this->blobs_[1]->offset(i,num_filters_per_class_*j,0,0));                            
                }else{ //only set mid pixel to zero
                    for (int k = 0; k < num_filters_per_class_; ++k){
                        caffe_gpu_set(1,Dtype(0),diff_weights_pw + this->blobs_[1]->offset(i,num_filters_per_class_*j+k,mid_ind,mid_ind));
                    }
                }
            }
        }
    }

    if(keep_mid_ind_zero_){
        Dtype *diff_pw1 = this->blobs_[1]->mutable_gpu_diff();
        int mid_ind1 = kernel_size_ / 2;

        for (int i = 0; i < num_classes_; ++i)
        {
            for (int j = 0; j < num_classes_; ++j)
            {
                for (int k = 0; k < num_filters_per_class_; ++k){                        
                    caffe_gpu_set(1,Dtype(0),diff_pw1 + this->blobs_[1]->offset(i,num_filters_per_class_*j+k,mid_ind1,mid_ind1));
                }
            }    
        }
    }

}

INSTANTIATE_LAYER_GPU_FUNCS(CrfFilterbankEdLayer);
} // namespace caffe
