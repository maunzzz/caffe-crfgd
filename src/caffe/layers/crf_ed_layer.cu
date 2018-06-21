#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/crf_ed_layer.hpp"
#include "caffe/layers/entropy_descent_layer.hpp"
#include "caffe/layers/permutohedral_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <sstream>

namespace caffe {

template <typename Dtype>
__global__ void scale_unary_weight_kernel(const int n, const Dtype* alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] *= alpha[0];
  }
}

template <typename Dtype>
__global__ void clip_gradients_cu(const int n, Dtype thresh, Dtype* x) {
  CUDA_KERNEL_LOOP(i, n) {
    x[i] = max(min(x[i],thresh),Dtype(-1)*thresh);
  }
}

template <typename Dtype>
__global__ void calc_bottom_diff(const int N, Dtype* bottom_diff, const Dtype* dL_dtheta_u,const Dtype* unary_weight, const Dtype* bottom_data,const Dtype log_eps,const Dtype* dL_dx_it){
    CUDA_KERNEL_LOOP(i, N) {
        bottom_diff[i] = dL_dtheta_u[i]*Dtype(-1.0)*unary_weight[0]/(bottom_data[i]+log_eps) + dL_dx_it[i];
    }
}

template <typename Dtype>
void CrfEdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // sync weights with internal layers
    caffe_copy(this->blobs_[1]->count(),this->blobs_[1]->gpu_data(),conv_layer_->blobs()[0]->mutable_gpu_data());
    caffe_copy(this->blobs_[2]->count(),this->blobs_[2]->gpu_data(),pl_layer_->blobs()[0]->mutable_gpu_data());

    //Get pointers to weights
    const Dtype* weights_unary = this->blobs_[0]->gpu_data();

    //initialize unary term
    Blob<Dtype> unary(bottom[0]->shape());

    //calculate unary as -1*weight_unary*log(x+log_eps), x is input data (bottom)
    // These unary terms are constant during gradient descent steps
    caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),unary.mutable_gpu_data()); //set to x
    Dtype* unary_vals = unary.mutable_gpu_data();
    caffe_gpu_add_scalar(unary.count(), log_eps_, unary.mutable_gpu_data()); //add log_eps
    caffe_gpu_log(unary.count(), unary.gpu_data(), unary.mutable_gpu_data());    
    scale_unary_weight_kernel<<<CAFFE_GET_BLOCKS(unary.count()), CAFFE_CUDA_NUM_THREADS>>>(unary.count(),weights_unary,unary.mutable_gpu_data());
    caffe_gpu_scal(unary.count(), Dtype(-1), unary.mutable_gpu_data());

    //Dtype tmp_sum = Dtype(0.0);
    //Dtype tmp_tosubtract = Dtype(0.0);

    //Blob current state keeps x for current iteration
    Blob<Dtype> current_state(bottom[0]->shape());
    current_state.CopyFrom(*bottom[0]); 			//initialize current state as input    
    vector<Blob<Dtype>*> current_state_vec;
    current_state_vec.push_back(&current_state);

    // helper blob for storing the derivative (of gradient descent steps)
    Blob<Dtype> derivative_blob(bottom[0]->shape()); 
    
    Blob<Dtype> tmp_der_blob(bottom[0]->shape());
    vector<Blob<Dtype>*> tmp_der_blob_vec;
    tmp_der_blob_vec.push_back(&tmp_der_blob);

    //create vec for pl-filtering input/output
    vector<Blob<Dtype>*> pl_bottom;
	pl_bottom.push_back(&current_state);
	pl_bottom.push_back(bottom[1]);
	pl_bottom.push_back(bottom[1]);

	vector<Blob<Dtype>*> pl_top;
	pl_top.push_back(&derivative_blob);
	pl_top.push_back(&internal_lattice_blob_);

    // ed input output    
    vector<Blob<Dtype>*> ed_bottom_vec;
    ed_bottom_vec.push_back(&current_state);
    ed_bottom_vec.push_back(&derivative_blob);
    //ed_bottom_vec.push_back(&asd);

    vector<Blob<Dtype>*> ed_top_vec;
    ed_top_vec.push_back(&tmp_der_blob);

    //LOOP OVER GRADIENT DESCENT STEPS
    for(int it = 0; it < num_iterations_; ++it) {
        //Calculate pairwise part with pl and spatial filtering
        
        if(!skip_conv_term_){
            conv_layer_->Forward(current_state_vec,tmp_der_blob_vec);
        }
        if(!skip_pl_term_){
            pl_layer_->Forward(pl_bottom,pl_top);
        }
        
        if(calculate_energy_){
            Blob<Dtype> tmp_energy_blob_1(bottom[0]->shape()); //store total energy
            Blob<Dtype> tmp_energy_blob_2(bottom[0]->shape()); //store each intermediate energy term

            // -- Calculate pixel,classwise energy
            // unary energy
            caffe_gpu_mul(tmp_energy_blob_1.count(),current_state.gpu_data(),unary.gpu_data(),tmp_energy_blob_1.mutable_gpu_data());

            // spatial pw energy
            caffe_gpu_mul(tmp_energy_blob_2.count(),current_state.gpu_data(),tmp_der_blob.gpu_data(),tmp_energy_blob_2.mutable_gpu_data());
            caffe_gpu_add(tmp_energy_blob_2.count(),tmp_energy_blob_2.gpu_data(),tmp_energy_blob_1.gpu_data(),tmp_energy_blob_1.mutable_gpu_data());

            // bilateral pw energy
            caffe_gpu_mul(tmp_energy_blob_2.count(),current_state.gpu_data(),derivative_blob.gpu_data(),tmp_energy_blob_2.mutable_gpu_data());
            caffe_gpu_add(tmp_energy_blob_2.count(),tmp_energy_blob_2.gpu_data(),tmp_energy_blob_1.gpu_data(),tmp_energy_blob_1.mutable_gpu_data());

            // -- Sum over all pixels
            //parallell way of summing elements in matrix (dot product with ones)
            caffe_gpu_set(tmp_energy_blob_2.count(),Dtype(1),tmp_energy_blob_2.mutable_gpu_data());
            Dtype energy = 0;
            caffe_gpu_dot(tmp_energy_blob_2.count(), tmp_energy_blob_1.gpu_data(), tmp_energy_blob_2.gpu_data(),&energy);
            Dtype* energy_top_data = top[1]->mutable_cpu_data();
            energy_top_data[it] = energy;            
        }
        
        if((it == 0) & (!skip_pl_term_)){ // if first iteration save lattice blob
			caffe_copy(internal_lattice_blob_.count(),pl_top[1]->gpu_data(),internal_lattice_blob_.mutable_gpu_data());			
			pl_bottom.push_back(&internal_lattice_blob_);
			pl_top.pop_back();
		}
		
        if(!skip_pl_term_){
            if(!skip_conv_term_){
		        //add spatial and sparse pl contribution
		        caffe_gpu_add(current_state.count(), tmp_der_blob.gpu_data(), derivative_blob.gpu_data(), derivative_blob.mutable_gpu_data());
            }
        }else{
            if(!skip_conv_term_){
                caffe_copy(current_state.count(), tmp_der_blob.gpu_data(), derivative_blob.mutable_gpu_data()); //DEBUG
            }else{
                caffe_gpu_set(derivative_blob.count(),Dtype(0) , derivative_blob.mutable_gpu_data()); //debug
            }
        }

        //Add unary and pairwise gradient contribution (derivative_blob now contains the derivative before all projections)
        caffe_gpu_add(derivative_blob.count(), derivative_blob.gpu_data(), unary.gpu_data(), derivative_blob.mutable_gpu_data());
		
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
    caffe_copy(top[0]->count(),current_state.gpu_data(),top[0]->mutable_gpu_data());
    has_updated_internal_states = true;	
}

template <typename Dtype>
void CrfEdLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {    

    if(propagate_down[0]){
        CHECK_EQ(has_updated_internal_states, true)
        << "No updated internal states found, forward must be done previous to backward";

        // dL_dtheta_u
        Blob<Dtype> dL_dtheta_u(top[0]->shape());
        caffe_gpu_set(dL_dtheta_u.count(),Dtype(0),dL_dtheta_u.mutable_gpu_data());

        Blob<Dtype> blob_x_kplusone(top[0]->shape());
        Blob<Dtype> blob_x_k(top[0]->shape());
        Blob<Dtype> blob_fprim_k(top[0]->shape());

        //pl input/output
		vector<Blob<Dtype>*> pl_bottom;
		pl_bottom.push_back(&blob_x_k);
		pl_bottom.push_back(bottom[1]);
		pl_bottom.push_back(bottom[1]);
		pl_bottom.push_back(&internal_lattice_blob_);
		
		vector<Blob<Dtype>*> pl_top;
		pl_top.push_back(&blob_fprim_k);

        //conv input/output
		vector<Blob<Dtype>*> conv_bottom;
		conv_bottom.push_back(&blob_x_k);
		
		vector<Blob<Dtype>*> conv_top;
		conv_top.push_back(&blob_fprim_k);

        //ed input/output
        vector<Blob<Dtype>*> ed_bottom;
		ed_bottom.push_back(&blob_x_k);
        ed_bottom.push_back(&blob_fprim_k);
		
		vector<Blob<Dtype>*> ed_top;
		ed_top.push_back(&blob_x_kplusone);

        // Start by setting dL/d(xplusone) = dL/d(out)
        caffe_copy(blob_x_kplusone.count(), top[0]->gpu_diff(), blob_x_kplusone.mutable_gpu_diff());

        //Loop backwards through gradient iterations
        for(int it = num_iterations_-1 ; it >= 0 ; --it){

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

            //Backward through entropy descent update step
            step_size_ = sqrt(Dtype(2)*log(num_classes_)/(it+1))/step_size_factor_;
            ed_layer_->SetUpdateStep(step_size_);
            ed_layer_->Backward(ed_top,propagate_down,ed_bottom);

            //Update dL/dxkplusone to next iteration (the der for this iteration is no longer needed)                        
            caffe_copy(blob_x_kplusone.count(), blob_x_k.gpu_diff(), blob_x_kplusone.mutable_gpu_diff());

            if(!skip_pl_term_){
                //backward through pl filtering
                caffe_gpu_set(pl_layer_->blobs()[0]->count(),Dtype(0),pl_layer_->blobs()[0]->mutable_gpu_diff()); // zero internal layer weight diffs
			    pl_layer_->Backward(pl_top,propagate_down,pl_bottom); 
                caffe_gpu_add(blob_x_kplusone.count(), pl_bottom[0]->gpu_diff(), blob_x_kplusone.gpu_diff() , blob_x_kplusone.mutable_gpu_diff()); //add bilateral dL/dxkplusone df/dy dy/dxk term
            }

            if(!skip_conv_term_){
                //backward through spatial filtering
                caffe_gpu_set(conv_layer_->blobs()[0]->count(),Dtype(0),conv_layer_->blobs()[0]->mutable_gpu_diff()); // zero internal layer weight diffs
			    conv_layer_->Backward(conv_top,propagate_down,conv_bottom);
                caffe_gpu_add(blob_x_kplusone.count(), conv_bottom[0]->gpu_diff(), blob_x_kplusone.gpu_diff(), blob_x_kplusone.mutable_gpu_diff()); //add spatial dL/dxkplusone df/dy dy/dxk term
            }

            if(!skip_pl_term_){
                // accumulate pairwise pl weight diff vals
                caffe_gpu_add(pl_layer_->blobs()[0]->count(),pl_layer_->blobs()[0]->gpu_diff(),this->blobs_[2]->gpu_diff(),this->blobs_[2]->mutable_gpu_diff());
            }
            
            if(!skip_conv_term_){
                // accumulate pairwise spatial weight diff vals
                caffe_gpu_add(conv_layer_->blobs()[0]->count(),conv_layer_->blobs()[0]->gpu_diff(),this->blobs_[1]->gpu_diff(),this->blobs_[1]->mutable_gpu_diff());
            }
            
            // accumulate dL_dtheta_u dL_dtheta_u_vals
            caffe_gpu_add(dL_dtheta_u.count(),blob_fprim_k.gpu_diff(),dL_dtheta_u.gpu_data(),dL_dtheta_u.mutable_gpu_data());
        }
           
        // variables for accessing data
        const Dtype* weight_vals_u = this->blobs_[0]->gpu_data();
        Dtype* weight_diff_u = this->blobs_[0]->mutable_gpu_diff();
        Dtype* dL_dtheta_u_vals = dL_dtheta_u.mutable_gpu_data();

        const Dtype* bottom_vals = bottom[0]->gpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const Dtype* dL_dx0_vals = blob_x_kplusone.mutable_gpu_diff();

        // bottom diff
        calc_bottom_diff<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(),bottom[0]->mutable_gpu_diff(),
            dL_dtheta_u.gpu_data(),weight_vals_u,bottom[0]->gpu_data(),log_eps_,blob_x_kplusone.gpu_diff());
        
        // unary weight diff            
        //Use blob_x_k and blob_x_kplusone as temporary storage
        caffe_copy(blob_x_k.count(),bottom[0]->gpu_data(),blob_x_k.mutable_gpu_data());
        caffe_gpu_add_scalar(blob_x_k.count(), log_eps_, blob_x_k.mutable_gpu_data()); //add log_eps
        caffe_gpu_log(blob_x_k.count(), blob_x_k.gpu_data(), blob_x_k.mutable_gpu_data());
        caffe_gpu_scal(blob_x_k.count(), Dtype(-1), blob_x_k.mutable_gpu_data());
        caffe_gpu_mul(blob_x_k.count(),blob_x_k.gpu_data(),dL_dtheta_u.gpu_data(),blob_x_k.mutable_gpu_data());

        //parallell way of summing elements in matrix (dot product with ones)
        caffe_gpu_set(blob_x_kplusone.count(),Dtype(1),blob_x_kplusone.mutable_gpu_data());
        Dtype sum = 0;
        caffe_gpu_dot(blob_x_kplusone.count(), blob_x_kplusone.gpu_data(), blob_x_k.gpu_data(),&sum);
        this->blobs_[0]->mutable_cpu_diff()[0] = sum;

        if (clip_pl_gradients_ > 0){
            clip_gradients_cu<<<CAFFE_GET_BLOCKS(this->blobs_[2]->count()), CAFFE_CUDA_NUM_THREADS>>>(this->blobs_[2]->count(),clip_pl_gradients_,this->blobs_[2]->mutable_gpu_diff());
        }
    } 
}

INSTANTIATE_LAYER_GPU_FUNCS(CrfEdLayer);


}  // namespace caffe
