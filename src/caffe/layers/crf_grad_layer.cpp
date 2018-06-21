#include <vector>
#include <algorithm>
#include <math.h>

#include "caffe/layers/crf_grad_layer.hpp"
#include "caffe/layers/permutohedral_layer.hpp"
#include "caffe/layer.hpp"

#include "caffe/filler.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/cudnn_conv_layer.hpp"

#include <iostream>

namespace caffe {

template <typename Dtype>
void CrfGradLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // Read parameters
    const caffe::CrfGradParameter crfgrad_param = this->layer_param_.crfgrad_param();

    num_iterations_ = crfgrad_param.num_iterations(); //number of gradient descent steps
    step_size_ = crfgrad_param.step_size(); //step length for gradient descent
    leak_factor_ = crfgrad_param.leak_factor(); //leak factor for projection onto [0 1]
    num_classes_ = crfgrad_param.num_classes(); //number of classes for segmentation task (defines number of input and output channels)
    log_eps_ = crfgrad_param.log_eps(); //epsilon for calculating the unary weights u = -log(x + log_eps_);
    debug_mode_ = crfgrad_param.debug_mode(); //debug_mode initializes pw weight to non-zero
    kernel_size_ = crfgrad_param.kernel_size();
	pl_filter_init_scale_ = crfgrad_param.pl_filter_init_scale();
	unary_weight_init_ = crfgrad_param.unary_weight_init();
	use_cudnn_ = crfgrad_param.use_cudnn();
    calculate_energy_ = crfgrad_param.calculate_energy();

    //initialize blobs for all internal states
    internal_intermediate_states_.resize(num_iterations_);
    for(int i = 0; i < num_iterations_; ++i) {
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
    
    // set parameters and create internal pl  layer
    LayerParameter layer_param2;
    PermutohedralParameter* pl_param =	layer_param2.mutable_permutohedral_param();

	pl_param->set_offset_type(PermutohedralParameter_OffsetType_NONE); //set weights "manually" instead
	pl_param->set_num_output(num_classes_);
	pl_param->set_neighborhood_size(1);
	
	
	pl_layer_.reset(new PermutohedralLayerTemplate<Dtype, permutohedral::Permutohedral>(layer_param2));
    
	tmp_bottom.push_back(bottom[1]);
	tmp_bottom.push_back(bottom[1]);

    pl_layer_->SetUp(tmp_bottom,top);	

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
void CrfGradLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // Reshape the internal states
    for(int i = 0; i < num_iterations_; ++i) {
        internal_intermediate_states_[i]->Reshape(bottom[0]->shape());
    }

    num_pixels_ = bottom[0]->shape(2)*bottom[0]->shape(3);

	//
	
	vector<Blob<Dtype>*> tmp_bottom;
	tmp_bottom.push_back(bottom[0]);
	
	conv_layer_->Reshape(tmp_bottom,top);
	
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
void CrfGradLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

    Blob<Dtype> current_state(bottom[0]->shape());
    current_state.CopyFrom(*bottom[0]); 			//initialize current state as input
    //Dtype* current_state_vals = current_state.mutable_cpu_data();
    
    vector<Blob<Dtype>*> current_state_vec;
    current_state_vec.push_back(&current_state);
    
    Blob<Dtype> derivative_blob(bottom[0]->shape()); // helper blob for storing the derivative (of gradient descent steps)
    Dtype* derivative_vals = derivative_blob.mutable_cpu_data();
    
    Blob<Dtype> tmp_der_blob(bottom[0]->shape());
    vector<Blob<Dtype>*> tmp_der_blob_vec;
    tmp_der_blob_vec.push_back(&tmp_der_blob);
    
    //create vec for pl-filtering input
    vector<Blob<Dtype>*> pl_bottom;
	pl_bottom.push_back(&current_state);
	pl_bottom.push_back(bottom[1]);
	pl_bottom.push_back(bottom[1]);
	
	vector<Blob<Dtype>*> pl_top;
	pl_top.push_back(&derivative_blob);
	pl_top.push_back(&internal_lattice_blob_);
    
    //LOOP OVER GRADIENT DESCENT STEPS
    for(int it = 0; it < num_iterations_; ++it) {
		//std::cout << "ITERATION: " << it << std::endl;

        //Calculate pairwise part with pl and spatial filtering
        conv_layer_->Forward(current_state_vec,tmp_der_blob_vec);
        pl_layer_->Forward(pl_bottom,pl_top);

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
		if(it == 0){ // if first iteration save lattice blob
			caffe_copy(internal_lattice_blob_.count(),pl_top[1]->cpu_data(),internal_lattice_blob_.mutable_cpu_data());			
			pl_bottom.push_back(&internal_lattice_blob_);
			pl_top.pop_back();
		}
		
		//add spatial and sparse pl contribution
		caffe_add(current_state.count(), tmp_der_blob.cpu_data(), derivative_blob.cpu_data(),derivative_vals);

        //Add unary and pairwise gradient contribution (derivative_blob now contains the derivative before all projections)
        caffe_add(derivative_blob.count(), derivative_blob.cpu_data(), unary.cpu_data(), derivative_vals);
		
        //Take step in descent direction
        caffe_scal(derivative_blob.count(),Dtype(-1.0)*step_size_,derivative_blob.mutable_cpu_data());
        caffe_add(current_state.count(),derivative_blob.cpu_data(),current_state.cpu_data(),current_state.mutable_cpu_data());

        // Push current state on intermedieate state vector (we need values before the last projection to be able to calculate backward derivatives)
        caffe_copy(internal_intermediate_states_[it]->count(),current_state.cpu_data(),internal_intermediate_states_[it]->mutable_cpu_data());

		// do the projection onto simplex
		this->ProjectBlobSimplex_(&current_state);
    }
    //after iteration is done, copy current state to top (output)
    caffe_copy(top[0]->count(),current_state.cpu_data(),top[0]->mutable_cpu_data());
    //caffe_copy(top[0]->count(),projected_gradient.cpu_data(),top[0]->mutable_cpu_data());
    has_updated_internal_states = true;	
}

template <typename Dtype>
void CrfGradLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    if(propagate_down[0]){
        CHECK_EQ(has_updated_internal_states, true)
        << "No updated internal states found, forward must be done previous to backward";

        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        Dtype* weight_diff_u = this->blobs_[0]->mutable_cpu_diff();

        const Dtype* bottom_vals = bottom[0]->cpu_data();
        const Dtype* weight_vals_u = this->blobs_[0]->cpu_data();
        //const Dtype* weight_vals_pw = this->blobs_[1]->cpu_data();

        // temporary variables that hold weight diffs during iterations
        Blob<Dtype> tmp_weight_diff_pw_spatial(this->blobs_[1]->shape());
        Blob<Dtype> tmp_weight_diff_pw_pl(this->blobs_[2]->shape());

        // dL_dtheta_u
        Blob<Dtype> dL_dtheta_u(top[0]->shape());
        Dtype* dL_dtheta_u_vals = dL_dtheta_u.mutable_cpu_data();

        // dL_dx_it holds the error derivatives with respect to the input of iteration it of the rnn
        vector<Blob<Dtype>*> dL_dx_it_vec;
        Blob<Dtype> dL_dx_it(top[0]->shape());
        dL_dx_it.CopyFrom(*top[0],true,false); 			//initialize as top diff
        dL_dx_it_vec.push_back(&dL_dx_it);
        Dtype* dL_dx_it_vals = dL_dx_it.mutable_cpu_diff();

        Blob<Dtype> temp_blob(top[0]->shape());
        //Dtype* temp_vals = temp_blob.mutable_cpu_diff();
        
        Blob<Dtype> inp_prev_it_conv_blob(bottom[0]->shape());
        vector<Blob<Dtype>*> inp_prev_it_conv_blob_vec;
    	inp_prev_it_conv_blob_vec.push_back(&inp_prev_it_conv_blob);


        Blob<Dtype> inp_prev_it(top[0]->shape());
        Dtype* inp_prev_it_vals = inp_prev_it.mutable_cpu_data();

		vector<Blob<Dtype>*> pl_bottom;
		pl_bottom.push_back(&inp_prev_it);
		pl_bottom.push_back(bottom[1]);
		pl_bottom.push_back(bottom[1]);
		pl_bottom.push_back(&internal_lattice_blob_);
		
		vector<Blob<Dtype>*> pl_top;
		pl_top.push_back(&dL_dx_it);
	
        //Loop backwards through gradient iterations
        for(int it = num_iterations_-1 ; it >= 0 ; --it){
            //std::cout << "ITERATION " << it << std::endl;

            // inp_prev_it is the input to conv-backward, during forward process this has been projected before the convolution, 
            // if its not the first iteration. Hence we need to de the same here
            if (it > 0){
                caffe_copy(internal_intermediate_states_[it-1]->count(),internal_intermediate_states_[it-1]->cpu_data(),inp_prev_it_vals); //set prev it vals to previous internal state
                this->ProjectBlobSimplex_(&inp_prev_it);
            }else{ //For the first iteration we do not need to do this
                caffe_copy(bottom[0]->count(),bottom[0]->cpu_data(),inp_prev_it_vals);
            }
            // previous iteration data is also needed by conv layer    
            caffe_copy(bottom[0]->count(),inp_prev_it.cpu_data(),inp_prev_it_conv_blob.mutable_cpu_data());

            //Backward through simplex projection            
            caffe_set(temp_blob.count(),Dtype(0),temp_blob.mutable_cpu_diff());
            this->ProjectOntoSimplexBw_(internal_intermediate_states_[it],&dL_dx_it,&temp_blob);
            caffe_copy(temp_blob.count(),temp_blob.cpu_diff(),dL_dx_it.mutable_cpu_diff());
            
            //backward through pl filtering
            caffe_set(pl_layer_->blobs()[0]->count(),Dtype(0),pl_layer_->blobs()[0]->mutable_cpu_diff());
			pl_layer_->Backward(pl_top,propagate_down,pl_bottom); //pl_bottom[0] is inp_prev_it
			caffe_copy(tmp_weight_diff_pw_pl.count(),pl_layer_->blobs()[0]->cpu_diff(),tmp_weight_diff_pw_pl.mutable_cpu_diff());
			
			//backward through spatial filtering
            caffe_set(conv_layer_->blobs()[0]->count(),Dtype(0),conv_layer_->blobs()[0]->mutable_cpu_diff());
			conv_layer_->Backward(pl_top,propagate_down,inp_prev_it_conv_blob_vec);
			caffe_copy(tmp_weight_diff_pw_spatial.count(),conv_layer_->blobs()[0]->cpu_diff(),tmp_weight_diff_pw_spatial.mutable_cpu_diff());

            // accumulate pairwise pl weight diff vals
            caffe_scal(tmp_weight_diff_pw_pl.count(),Dtype(-1)*step_size_,tmp_weight_diff_pw_pl.mutable_cpu_diff());
            caffe_add(tmp_weight_diff_pw_pl.count(),tmp_weight_diff_pw_pl.cpu_diff(),this->blobs_[2]->cpu_diff(),this->blobs_[2]->mutable_cpu_diff());
            
            // accumulate pairwise spatial weight diff vals
            caffe_scal(tmp_weight_diff_pw_spatial.count(),Dtype(-1)*step_size_,tmp_weight_diff_pw_spatial.mutable_cpu_diff());
            caffe_add(tmp_weight_diff_pw_spatial.count(),tmp_weight_diff_pw_spatial.cpu_diff(),this->blobs_[1]->cpu_diff(),this->blobs_[1]->mutable_cpu_diff());

            // accumulate dL_dtheta_u dL_dtheta_u_vals
            caffe_scal(dL_dtheta_u.count(),Dtype(-1.0)*step_size_,temp_blob.mutable_cpu_diff());
            caffe_add(dL_dtheta_u.count(),temp_blob.cpu_diff(),dL_dtheta_u.cpu_data(),dL_dtheta_u.mutable_cpu_data());

            // Change dl_dx_it to next iteration
            // Copy inp_prev_it_vec_it: diff field to dL_dx_it_vals for next iteration, we get two parts, one from spatial and one from pl filtering
            //pl
            caffe_scal(inp_prev_it.count(),Dtype(-1)*step_size_,inp_prev_it.mutable_cpu_diff());
            caffe_add(dL_dx_it.count(),dL_dx_it.cpu_diff(),inp_prev_it.cpu_diff(),dL_dx_it.mutable_cpu_diff());
            //conv
            caffe_scal(inp_prev_it_conv_blob.count(),Dtype(-1)*step_size_,inp_prev_it_conv_blob.mutable_cpu_diff());
            caffe_add(dL_dx_it.count(),dL_dx_it.cpu_diff(),inp_prev_it_conv_blob.cpu_diff(),dL_dx_it.mutable_cpu_diff());
        }

        // bottom diff
        for(int i = 0; i < bottom[0]->count() ; ++i){
            bottom_diff[i] = dL_dtheta_u_vals[i]*Dtype(-1.0)*weight_vals_u[0]/(bottom_vals[i]+log_eps_) + dL_dx_it_vals[i];
            //bottom_diff[i] = Dtype(0);
        }

        weight_diff_u[0] = Dtype(0);
        for(int i = 0; i < bottom[0]->count() ; ++i){
            weight_diff_u[0] += dL_dtheta_u_vals[i]*Dtype(-1)*log(bottom_vals[i]+log_eps_);
        }
    } 
}


template <typename Dtype>
void CrfGradLayer<Dtype>::ProjectBlobSimplex_(Blob<Dtype>* blobinout) {

	Dtype* blobvals = blobinout->mutable_cpu_data();
	
	const vector<int>* blob_shape;
    blob_shape = &blobinout->shape();
    const int n_batches = (*blob_shape)[0];
    const int n_channels = (*blob_shape)[1];
    const int im_height = (*blob_shape)[2];
    const int im_width = (*blob_shape)[3];
    
	Dtype x[n_channels];
	Dtype tmpsum,tmax;
	bool bget = false;
	for (int n = 0; n < n_batches; ++n) { //loop over batches
		for (int h = 0; h < im_height; ++h) { //loop over image height
			for (int w = 0; w < im_width; ++w) { //loop over image width
	        	for (int k = 0; k < n_channels; ++k) { //loop over channels
		        	x[k] = blobvals[((n*n_channels+k)*im_height+h)*im_width+w];
		        }
	            std::sort(x,x+n_channels);
				tmpsum = Dtype(0);
                bget = false;
				for (int ii = 1; ii < n_channels ; ++ii) { //calculate t	
	            	tmpsum += x[n_channels-ii];
	               	tmax = (tmpsum - Dtype(1))/Dtype(ii);
	               	if(tmax >= x[n_channels-ii-1]) {
	               	 	bget = true;
	               		break;
	               	}
	            }
	            if(!bget){
	               	tmax = (tmpsum + x[0] - Dtype(1))/Dtype(n_channels);
	            }
	            for (int k = 0; k < n_channels; ++k) { //do the actual projection
	               	if(blobvals[((n*n_channels+k)*im_height+h)*im_width+w]-tmax > 0){
	               		blobvals[((n*n_channels+k)*im_height+h)*im_width+w] = blobvals[((n*n_channels+k)*im_height+h)*im_width+w]-tmax;
	               	}else{
	               		blobvals[((n*n_channels+k)*im_height+h)*im_width+w] = leak_factor_*(blobvals[((n*n_channels+k)*im_height+h)*im_width+w]-tmax);
	               	}
	            }
            }
		}
	}
}

template <typename Dtype>
void CrfGradLayer<Dtype>::ProjectOntoSimplexBw_(const Blob<Dtype>* in,const Blob<Dtype>* outder, Blob<Dtype>* inder) {

	const Dtype* invals = in->cpu_data();
	const Dtype* outdervals = outder->cpu_diff();
	Dtype* indervals = inder->mutable_cpu_diff();

	const vector<int>* blob_shape;
    blob_shape = &in->shape();
    const int n_batches = (*blob_shape)[0];
    const int n_channels = (*blob_shape)[1];
    const int im_height = (*blob_shape)[2];
    const int im_width = (*blob_shape)[3];
    
	Dtype x[n_channels];
	Dtype dt_dx[n_channels];
	Dtype tmpsum,tmax;
    int ii;
	bool bget = false;
	for (int n = 0; n < n_batches; ++n) { //loop over batches
		for (int h = 0; h < im_height; ++h) { //loop over image height
			for (int w = 0; w < im_width; ++w) { //loop over image width
			
				// This performs the backward step
	        	for (int k = 0; k < n_channels; ++k) { //loop over channels
		        	x[k] = invals[((n*n_channels+k)*im_height+h)*im_width+w];
		        }
	            std::sort(x,x+n_channels);
				tmpsum = Dtype(0);
                bget = false;
				for (ii = 1; ii < n_channels ; ++ii) { //calculate t	
	            	tmpsum += x[n_channels-ii];
	               	tmax = (tmpsum - Dtype(1))/Dtype(ii);
	               	if(tmax >= x[n_channels-ii-1]) {
	               	 	bget = true;
	               		break;
	               	}
	            }

	            if(!bget){
	               	tmax = (tmpsum + x[0] - Dtype(1))/Dtype(n_channels);
	               	for (int kk = 0; kk < n_channels; ++kk){
	               		dt_dx[kk] = Dtype(1)/Dtype(n_channels);
	               	}
	            }else{
	            	for (int kk = 0; kk < n_channels; ++kk){
	            		if(invals[((n*n_channels+kk)*im_height+h)*im_width+w] > tmax){
	               			dt_dx[kk] = Dtype(1)/Dtype(ii);
	               		}else{
	               			dt_dx[kk] = Dtype(0);
	               		}
	               	}
	            }
	            
	            for (int k1 = 0; k1 < n_channels; ++k1) { 
	            	for (int k2 = 0; k2 < n_channels; ++k2) {
	            		if(k1 == k2){
	            			if(invals[((n*n_channels+k2)*im_height+h)*im_width+w] > tmax){
	            				indervals[((n*n_channels+k1)*im_height+h)*im_width+w] += outdervals[((n*n_channels+k1)*im_height+h)*im_width+w] - dt_dx[k1] * outdervals[((n*n_channels+k2)*im_height+h)*im_width+w];
	            			}else{
	            				indervals[((n*n_channels+k1)*im_height+h)*im_width+w] += leak_factor_*(outdervals[((n*n_channels+k1)*im_height+h)*im_width+w] - dt_dx[k1] * outdervals[((n*n_channels+k2)*im_height+h)*im_width+w]);	            			
	            			}
	            		}else{
	            			if(invals[((n*n_channels+k2)*im_height+h)*im_width+w] > tmax){
	            				indervals[((n*n_channels+k1)*im_height+h)*im_width+w] -= dt_dx[k1] * outdervals[((n*n_channels+k2)*im_height+h)*im_width+w];
	            			}else{
	            				indervals[((n*n_channels+k1)*im_height+h)*im_width+w] -= leak_factor_*(dt_dx[k1] * outdervals[((n*n_channels+k2)*im_height+h)*im_width+w]);	           
	            			}
	            		}
					}	
	            }
            }
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(CrfGradLayer);
#endif

INSTANTIATE_CLASS(CrfGradLayer);
REGISTER_LAYER_CLASS(CrfGrad);

}  // namespace caffe
