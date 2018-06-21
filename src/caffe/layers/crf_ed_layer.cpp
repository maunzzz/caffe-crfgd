#include <vector>
#include <algorithm>
#include <math.h>

#include "caffe/layers/crf_ed_layer.hpp"
#include "caffe/layers/entropy_descent_layer.hpp"
#include "caffe/layers/permutohedral_layer.hpp"
#include "caffe/layer.hpp"

#include "caffe/filler.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/cudnn_conv_layer.hpp"

#include <iostream>

namespace caffe {

template <typename Dtype>
void CrfEdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // Read parameters
    const caffe::CrfEdParameter crfed_param = this->layer_param_.crfed_param();

    num_iterations_ = crfed_param.num_iterations(); //number of gradient descent steps
    step_size_factor_ = crfed_param.step_size_factor(); //step size factor for gradient descent (L_f in article)
    num_classes_ = crfed_param.num_classes(); //number of classes for segmentation task (defines number of input and output channels)
    log_eps_ = crfed_param.log_eps(); //epsilon for calculating the unary weights u = -log(x + log_eps_);
    debug_mode_ = crfed_param.debug_mode(); //debug_mode initializes pw weight to non-zero
    kernel_size_ = crfed_param.kernel_size();
	pl_filter_init_scale_ = crfed_param.pl_filter_init_scale();
	unary_weight_init_ = crfed_param.unary_weight_init();
	use_cudnn_ = crfed_param.use_cudnn();
    calculate_energy_ = crfed_param.calculate_energy();
    clip_pl_gradients_ = crfed_param.clip_pl_gradients();
    skip_pl_term_ = crfed_param.skip_pl_term();
    skip_conv_term_ = crfed_param.skip_conv_term();
    spatial_filter_init_weight_ = crfed_param.spatial_filter_init_weight();
    //writecount_ = 0;

    //initialize blobs for all internal states (need to save x and f')
    internal_intermediate_states_.resize(num_iterations_-1);
    for(int i = 0; i < num_iterations_-1; ++i) {
        Blob<Dtype> *tmp = new Blob<Dtype>(bottom[0]->shape());
        internal_intermediate_states_[i] = tmp;
    }
    has_updated_internal_states = false;
    
    // set parameters and create internal cudnn conv layer
    LayerParameter layer_param;
    ConvolutionParameter* conv_param =	layer_param.mutable_convolution_param();
    conv_param->set_num_output(num_classes_);
    conv_param->set_bias_term(false);
    conv_param->add_pad((kernel_size_-1)/2);
    conv_param->add_kernel_size(kernel_size_);

    conv_param->mutable_weight_filler()->set_type("constant");
    conv_param->mutable_weight_filler()->set_value(0);

	if(use_cudnn_){
    	conv_layer_.reset(new CuDNNConvolutionLayer<Dtype>(layer_param));
	}else{
		conv_layer_.reset(new ConvolutionLayer<Dtype>(layer_param));
	}
    
    vector<Blob<Dtype>*> tmp_bottom;
	tmp_bottom.push_back(bottom[0]);
	
    if (calculate_energy_){
        vector<Blob<Dtype>*> tmp_top;
	    tmp_top.push_back(top[0]);
        conv_layer_->SetUp(tmp_bottom,tmp_top);
    }else{
        conv_layer_->SetUp(tmp_bottom,top);
    }
    
    // set parameters and create internal entropy descent layer
    LayerParameter layer_param3;

    ed_layer_.reset(new EntropyDescentLayer<Dtype>(layer_param3));

    tmp_bottom.push_back(bottom[0]);
    
    if (calculate_energy_){
        vector<Blob<Dtype>*> tmp_top;
	    tmp_top.push_back(top[0]);
        ed_layer_->SetUp(tmp_bottom,tmp_top);
    }else{
        ed_layer_->SetUp(tmp_bottom,top);
    }
    ed_layer_->SetDivEpsilon(crfed_param.div_epsilon());

    tmp_bottom.pop_back();

    // set parameters and create internal pl layer
    LayerParameter layer_param2;
    PermutohedralParameter* pl_param =	layer_param2.mutable_permutohedral_param();

	pl_param->set_offset_type(PermutohedralParameter_OffsetType_NONE); //set weights "manually" instead
	pl_param->set_num_output(num_classes_);
	pl_param->set_neighborhood_size(1);
	
	
	pl_layer_.reset(new PermutohedralLayerTemplate<Dtype, permutohedral::Permutohedral>(layer_param2));
    
	tmp_bottom.push_back(bottom[1]);
	tmp_bottom.push_back(bottom[1]);

    if (calculate_energy_){
        vector<Blob<Dtype>*> tmp_top;
	    tmp_top.push_back(top[0]);
        pl_layer_->SetUp(tmp_bottom,tmp_top);
    }else{
        pl_layer_->SetUp(tmp_bottom,top);
    }	

    // Handle the parameters: weights
    // - blobs_[0] holds the unary term
    // - blobs_[1] holds the spatial filter weights (pairwise terms)
    // - blobs_[2] holds the sparse filter weights

    // if blob already initialized
    if (this->blobs_.size() > 0) {
        CHECK_EQ(3, this->blobs_.size())
            << "Incorrect number of weight blobs.";
        caffe_copy(this->blobs_[1]->count(),this->blobs_[1]->cpu_data(),conv_layer_->blobs()[0]->mutable_cpu_data()); //copy spatial filter weights
        caffe_copy(this->blobs_[2]->count(),this->blobs_[2]->cpu_data(),pl_layer_->blobs()[0]->mutable_cpu_data());  //copy sparse filter weights
        
        LOG(INFO) << "Skipping parameter initialization";
    } else {
        this->blobs_.resize(3);
        // Initialize and fill the weights:
        this->blobs_[0].reset(new Blob<Dtype>(1,1,1,1));
        FillerParameter filler_param;
        filler_param.set_value(unary_weight_init_);
        ConstantFiller<Dtype> weight_filler1(filler_param);
        weight_filler1.Fill(this->blobs_[0].get());
		
		// initialize spatial filter weights
		vector<int> weight_shape(4);
        weight_shape[0] = num_classes_;
        weight_shape[1] = num_classes_;
        weight_shape[2] = kernel_size_;
        weight_shape[3] = kernel_size_;

        this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
        filler_param.set_value(0);
        ConstantFiller<Dtype> weight_filler2(filler_param);
        weight_filler2.Fill(this->blobs_[1].get());

        if(spatial_filter_init_weight_ > 0){
            this->blobs_[1].reset(new Blob<Dtype>(weight_shape));

            LOG(INFO) << "Initializing spatial filters";

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
                        sign_factor = Dtype(1);
                    }else{
                        sign_factor = Dtype(-1);
                    }
                    for(int x = 0; x < kernel_size_; x++)
                    {
                        for(int y = 0; y < kernel_size_; y++)
                        {
                            expdist = exp(Dtype(-1)*(pow(Dtype(x-mid_ind),2) + pow(Dtype(y-mid_ind),2))/(Dtype(kernel_size_)));
                            caffe_set(1,sign_factor*Dtype(spatial_filter_init_weight_)*Dtype(expdist),weights_pw + this->blobs_[1]->offset(class_out,class_in,x,y)); 
                        }
                    }
                }
            }
        }

        if(debug_mode_){
            int mid_ind = kernel_size_/2;
            Dtype* weights_pw = this->blobs_[1].get()->mutable_cpu_data();
            for(int i = 0; i < num_classes_; ++i) {
                for(int j = 0; j < num_classes_; ++j) {
                    if (i != j) {
                        weights_pw[((i*num_classes_+j)*kernel_size_ + mid_ind+1)*kernel_size_ + mid_ind ] = Dtype(1);
                        weights_pw[((i*num_classes_+j)*kernel_size_ + mid_ind-1)*kernel_size_ + mid_ind ] = Dtype(1);
                        weights_pw[((i*num_classes_+j)*kernel_size_ + mid_ind)*kernel_size_ + mid_ind+1] = Dtype(1);
                        weights_pw[((i*num_classes_+j)*kernel_size_ + mid_ind)*kernel_size_ + mid_ind-1] = Dtype(1);
                    }
                }
            }

        }

        // sync with conv layer
        caffe_copy(this->blobs_[1]->count(),this->blobs_[1]->cpu_data(),conv_layer_->blobs()[0]->mutable_cpu_data());
        
		// Set sparse filter weights
		this->blobs_[2].reset(new Blob<Dtype>(pl_layer_->blobs()[0]->shape()));

		// Get the current incarnation of a Gaussian filter.
		typedef permutohedral::Permutohedral<Dtype> permutohedral_type;
		const int filter_size = permutohedral_type::get_filter_size(1, 5);
  		typename permutohedral_type::gauss_type gauss(1, 5);
  		const Dtype* gauss_filter = gauss.filter();
  		
  		// copy gaussian weights to non-diagonal entries (between class interactions)
  		Dtype* w_ptr = this->blobs_[2]->mutable_cpu_data();
		for(int c1 = 0; c1 < num_classes_ ; ++c1){
			for (int i = 0; i < filter_size; ++i) {
				w_ptr[this->blobs_[2]->offset(c1, c1, 0, i)] = Dtype(-1)*pl_filter_init_scale_*gauss_filter[i];
			}
		}
        
		// sync with pl layer		
        caffe_copy(this->blobs_[2]->count(),this->blobs_[2]->cpu_data(),pl_layer_->blobs()[0]->mutable_cpu_data());
    }

    // Propagate gradients to the parameters (as directed by backward pass).
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CrfEdLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // Reshape the internal states
    for(int i = 0; i < num_iterations_-1; ++i) {
        internal_intermediate_states_[i]->Reshape(bottom[0]->shape());
    }

    num_pixels_ = bottom[0]->shape(2)*bottom[0]->shape(3);

	//
	
	vector<Blob<Dtype>*> tmp_bottom;
	tmp_bottom.push_back(bottom[0]);
	
	conv_layer_->Reshape(tmp_bottom,top);

    tmp_bottom.push_back(bottom[0]);
    ed_layer_->Reshape(tmp_bottom,top);
    tmp_bottom.pop_back();
	
	tmp_bottom.push_back(bottom[1]);
	tmp_bottom.push_back(bottom[1]);
	
    pl_layer_->Reshape(tmp_bottom,top);	

    top[0]->Reshape(bottom[0]->shape());
    if(calculate_energy_){
        top[1]->Reshape(bottom[0]->shape(0),1,1,num_iterations_);
    }

    internal_lattice_blob_.Reshape(bottom[0]->shape(0), 3, 1, sizeof(void*)); //needed for pl filtering
}

template <typename Dtype>
void CrfEdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // sync weights with internal layers
    caffe_copy(this->blobs_[1]->count(),this->blobs_[1]->cpu_data(),conv_layer_->blobs()[0]->mutable_cpu_data());
    caffe_copy(this->blobs_[2]->count(),this->blobs_[2]->cpu_data(),pl_layer_->blobs()[0]->mutable_cpu_data());

    //Get pointers to weights
    const Dtype* weights_unary = this->blobs_[0]->cpu_data();

    //initialize unary term
    Blob<Dtype> unary(bottom[0]->shape());

    //calculate unary as -1*weight_unary*log(x+log_eps), x is input data (bottom)
    // These unary terms are constant during gradient descent steps
    caffe_copy(bottom[0]->count(),bottom[0]->cpu_data(),unary.mutable_cpu_data()); //set to x
    Dtype* unary_vals = unary.mutable_cpu_data();
    caffe_add_scalar(unary.count(), log_eps_, unary.mutable_cpu_data()); //add log_eps
    caffe_log(unary.count(), unary.cpu_data(), unary.mutable_cpu_data());
    caffe_scal(unary.count(), Dtype(-1.0)*weights_unary[0], unary_vals);

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
            caffe_mul(tmp_energy_blob_1.count(),current_state.cpu_data(),unary.cpu_data(),tmp_energy_blob_1.mutable_cpu_data());

            // spatial pw energy
            caffe_mul(tmp_energy_blob_2.count(),current_state.cpu_data(),tmp_der_blob.cpu_data(),tmp_energy_blob_2.mutable_cpu_data());
            caffe_add(tmp_energy_blob_2.count(),tmp_energy_blob_2.cpu_data(),tmp_energy_blob_1.cpu_data(),tmp_energy_blob_1.mutable_cpu_data());

            // bilateral pw energy
            caffe_mul(tmp_energy_blob_2.count(),current_state.cpu_data(),derivative_blob.cpu_data(),tmp_energy_blob_2.mutable_cpu_data());
            caffe_add(tmp_energy_blob_2.count(),tmp_energy_blob_2.cpu_data(),tmp_energy_blob_1.cpu_data(),tmp_energy_blob_1.mutable_cpu_data());

            // -- Sum over all pixels
            Dtype* energy_top_data = top[1]->mutable_cpu_data();
            energy_top_data[it] = Dtype(0);
            Dtype* tmp_energy_blob_1_mdata = tmp_energy_blob_1.mutable_cpu_data();
            for(int i = 0; i < tmp_energy_blob_1.count() ; ++i){
               energy_top_data[it] += tmp_energy_blob_1_mdata[i];
            }
            
        }
        
        if((it == 0) & (!skip_pl_term_)){ // if first iteration save lattice blob
			caffe_copy(internal_lattice_blob_.count(),pl_top[1]->cpu_data(),internal_lattice_blob_.mutable_cpu_data());			
			pl_bottom.push_back(&internal_lattice_blob_);
			pl_top.pop_back();
		}
		
		//add spatial and sparse pl contribution
        if(!skip_pl_term_){
            if(!skip_conv_term_){
		        caffe_add(current_state.count(), tmp_der_blob.cpu_data(), derivative_blob.cpu_data(), derivative_blob.mutable_cpu_data());
            }
        }else{
            if(!skip_conv_term_){
                caffe_copy(current_state.count(), tmp_der_blob.cpu_data() , derivative_blob.mutable_cpu_data()); //debug    
            }else{
                caffe_set(derivative_blob.count(),Dtype(0) , derivative_blob.mutable_cpu_data()); //debug
            }
        }

        //Add unary and pairwise gradient contribution (derivative_blob now contains the derivative before all projections)
        caffe_add(derivative_blob.count(), derivative_blob.cpu_data(), unary.cpu_data(), derivative_blob.mutable_cpu_data());
		
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
    caffe_copy(top[0]->count(),current_state.cpu_data(),top[0]->mutable_cpu_data());
    has_updated_internal_states = true;	
}

template <typename Dtype>
void CrfEdLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {    

    if(propagate_down[0]){
        CHECK_EQ(has_updated_internal_states, true)
        << "No updated internal states found, forward must be done previous to backward";

        // dL_dtheta_u
        Blob<Dtype> dL_dtheta_u(top[0]->shape());
        caffe_set(dL_dtheta_u.count(),Dtype(0),dL_dtheta_u.mutable_cpu_data());

        Blob<Dtype> blob_x_kplusone(top[0]->shape());
        Blob<Dtype> blob_x_k_forpl(top[0]->shape());
        Blob<Dtype> blob_x_k(top[0]->shape());
        Blob<Dtype> blob_fprim_k(top[0]->shape());

        //pl input/output
		vector<Blob<Dtype>*> pl_bottom;
		pl_bottom.push_back(&blob_x_k_forpl);
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
        caffe_copy(blob_x_kplusone.count(), top[0]->cpu_diff(), blob_x_kplusone.mutable_cpu_diff());

        //Loop backwards through gradient iterations
        for(int it = num_iterations_-1 ; it >= 0 ; --it){
            //std::cout << "ITERATION " << it << std::endl;

            if(it > 0){
                caffe_copy(blob_x_k.count(), internal_intermediate_states_[it-1]->cpu_data(), blob_x_k.mutable_cpu_data());
                caffe_copy(blob_x_k_forpl.count(), internal_intermediate_states_[it-1]->cpu_data(), blob_x_k_forpl.mutable_cpu_data());
            }else{
                caffe_copy(blob_x_k.count(), bottom[0]->cpu_data(), blob_x_k.mutable_cpu_data());
                caffe_copy(blob_x_k_forpl.count(), bottom[0]->cpu_data(), blob_x_k_forpl.mutable_cpu_data());
            }
            if(it == num_iterations_-1){
                caffe_copy(blob_x_kplusone.count(), top[0]->cpu_data(), blob_x_kplusone.mutable_cpu_data());
            }else{
                caffe_copy(blob_x_kplusone.count(), internal_intermediate_states_[it]->cpu_data(), blob_x_kplusone.mutable_cpu_data());
            }

            //Backward through entropy descent update step
            step_size_ = sqrt(Dtype(2)*log(num_classes_)/(it+1))/step_size_factor_;
            ed_layer_->SetUpdateStep(step_size_);
            ed_layer_->Backward(ed_top,propagate_down,ed_bottom);

            //copy to pl bot
            caffe_copy(blob_x_k.count(), blob_x_k.cpu_diff(), blob_x_k_forpl.mutable_cpu_diff());

            //Update dL/dxkplusone to next iteration (the der for this iteration is no longer needed)                        
            caffe_copy(blob_x_kplusone.count(), blob_x_k.cpu_diff(), blob_x_kplusone.mutable_cpu_diff());
            
            if(!skip_pl_term_){
                //backward through pl filtering
                caffe_set(pl_layer_->blobs()[0]->count(),Dtype(0),pl_layer_->blobs()[0]->mutable_cpu_diff()); // zero internal layer weight diffs
                //caffe_set(pl_bottom[0]->count(),Dtype(0),pl_bottom[0]->mutable_cpu_diff()); // zero bottom diff
			    pl_layer_->Backward(pl_top,propagate_down,pl_bottom); 
                caffe_add(blob_x_kplusone.count(), pl_bottom[0]->cpu_diff(), blob_x_kplusone.cpu_diff() , blob_x_kplusone.mutable_cpu_diff()); //add bilateral dL/dxkplusone df/dy dy/dxk term
            }

            if(!skip_conv_term_){
                //backward through spatial filtering
                caffe_set(conv_layer_->blobs()[0]->count(),Dtype(0),conv_layer_->blobs()[0]->mutable_cpu_diff()); // zero internal layer weight diffs
			    conv_layer_->Backward(conv_top,propagate_down,conv_bottom);
                caffe_add(blob_x_kplusone.count(), conv_bottom[0]->cpu_diff(), blob_x_kplusone.cpu_diff(), blob_x_kplusone.mutable_cpu_diff()); //add spatial dL/dxkplusone df/dy dy/dxk term
            }

            if(!skip_pl_term_){
                // accumulate pairwise pl weight diff vals
                caffe_add(pl_layer_->blobs()[0]->count(),pl_layer_->blobs()[0]->cpu_diff(),this->blobs_[2]->cpu_diff(),this->blobs_[2]->mutable_cpu_diff());
            }

            if(!skip_conv_term_){
                // accumulate pairwise spatial weight diff vals
                caffe_add(conv_layer_->blobs()[0]->count(),conv_layer_->blobs()[0]->cpu_diff(),this->blobs_[1]->cpu_diff(),this->blobs_[1]->mutable_cpu_diff());
            }

            // accumulate dL_dtheta_u dL_dtheta_u_vals
            caffe_add(dL_dtheta_u.count(),blob_fprim_k.cpu_diff(),dL_dtheta_u.cpu_data(),dL_dtheta_u.mutable_cpu_data());
        }

        // variables for accessing data
        const Dtype* weight_vals_u = this->blobs_[0]->cpu_data();
        Dtype* weight_diff_u = this->blobs_[0]->mutable_cpu_diff();
        Dtype* dL_dtheta_u_vals = dL_dtheta_u.mutable_cpu_data();

        const Dtype* bottom_vals = bottom[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* dL_dx0_vals = blob_x_kplusone.mutable_cpu_diff();

        // bottom diff
        for(int i = 0; i < bottom[0]->count() ; ++i){
            bottom_diff[i] = dL_dtheta_u_vals[i]*Dtype(-1.0)*weight_vals_u[0]/(bottom_vals[i]+log_eps_) + dL_dx0_vals[i];
            //bottom_diff[i] = dL_dx0_vals[i];
        }

        // unary weight diff
        weight_diff_u[0] = Dtype(0);
        for(int i = 0; i < bottom[0]->count() ; ++i){
            weight_diff_u[0] += dL_dtheta_u_vals[i]*Dtype(-1)*log(bottom_vals[i]+log_eps_);
        }

        if (clip_pl_gradients_ > 0){
            Dtype* pl_diff = this->blobs_[2]->mutable_cpu_diff();
            for(int i = 0; i < this->blobs_[2]->count(); i++){
                pl_diff[i] = std::max(std::min(pl_diff[i],clip_pl_gradients_),Dtype(-1)*clip_pl_gradients_);
            }
        }
    } 
}

#ifdef CPU_ONLY
STUB_GPU(CrfEdLayer);
#endif

INSTANTIATE_CLASS(CrfEdLayer);
REGISTER_LAYER_CLASS(CrfEd);

}  // namespace caffe
