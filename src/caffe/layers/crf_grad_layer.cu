#include <vector>
#include "caffe/layers/crf_grad_layer.hpp"
namespace caffe {

// KERNEL FUNCTIONS
/*
Input: vals - array of values
       n_vals - size of vals

Output: outval - next highest value after vals[prev_highest_ind]       
        

Explanation: Returns the highest value in vals, also sets this value to -10001
*/
template <typename Dtype>
__device__ void getHighestValue(Dtype *outval, Dtype *vals, int n_vals)
{
    *outval = Dtype(-1000);
    int ind = -2;
    for (int i = 0 ; i < n_vals ; i++)
    {
        if(vals[i] > *outval)
        {
            *outval = vals[i];
            ind = i;
        }
    }
    if (ind > -1){
        vals[ind] = Dtype(-1001);
    }
}


template <typename Dtype>
__global__ void simplex_proj_kernel_fast(Dtype *blobvals, const int n_batches, const int n_channels, const int im_size, const Dtype this_leak_factor, Dtype *blob_for_sorting)
{

    Dtype tmpsum, tmax, tmpvalthis, tmpvalnext;
    bool bget;
    int n, ii; // variables to keep track of what index we are at, n is batchindex, ii is image index

    CUDA_KERNEL_LOOP(index, n_batches * im_size)
    {
        ii = index % im_size;
        n = (index - ii) / im_size;

        Dtype *x = &blob_for_sorting[(n * im_size + ii) * n_channels]; //blob_for_sorting is temp blob needed to do the sorting, x points to where this thread can use storage

        if (ii < im_size && n < n_batches)
        {
            for (int k = 0; k < n_channels; ++k)
            { //loop over channels
                x[k] = blobvals[(n * n_channels + k) * im_size + ii];
            }           

            tmpsum = Dtype(0);
            bget = false;

            getHighestValue(&tmpvalthis , x, n_channels); //get highest value
            for (int kk = 1; kk < n_channels; ++kk)
            { //calculate t
                getHighestValue(&tmpvalnext , x, n_channels); //get next highest value

                tmpsum += tmpvalthis;
                tmax = (tmpsum - Dtype(1)) / Dtype(kk);
                if (tmax >= tmpvalnext)
                {
                    bget = true;
                    break;
                }else
                {
                    tmpvalthis = tmpvalnext;
                }
                
            }

            if (!bget)
            {
                tmax = (tmpsum + tmpvalthis - Dtype(1)) / Dtype(n_channels);
            }
            for (int k = 0; k < n_channels; ++k)
            { //do the actual projection
                if (blobvals[(n * n_channels + k) * im_size + ii] - tmax > 0)
                {
                    blobvals[(n * n_channels + k) * im_size + ii] = blobvals[(n * n_channels + k) * im_size + ii] - tmax;
                }
                else
                {
                    blobvals[(n * n_channels + k) * im_size + ii] = this_leak_factor * (blobvals[(n * n_channels + k) * im_size + ii] - tmax);
                }
            }
        }
    }
}

template <typename Dtype>
__global__ void simplex_proj_kernel_bw_fast(const Dtype *invals, const Dtype *outdervals, Dtype *indervals, const int n_batches, const int n_channels, const int im_size, const Dtype this_leak_factor, Dtype *blob_for_sorting)
{

    Dtype tmpsum, tmax, tmpvalthis, tmpvalnext;
    bool bget;
    int n, ii, kk; // variables to keep track of what index we are at, n is batchindex, ii is image index, kk is the equivalent of ii in cpu and matlab veriosn
    CUDA_KERNEL_LOOP(index, n_batches * im_size)
    {
        ii = index % im_size;
        n = (index - ii) / im_size;

        Dtype *x = &blob_for_sorting[(n * im_size + ii) * n_channels]; //blob_for_sorting is temp blob needed to do the sorting, x points to where this thread can use storage

        if (ii < im_size && n < n_batches)
        {
            // This performs the backward step
            for (int k = 0; k < n_channels; ++k)
            { //loop over channels
                x[k] = invals[(n * n_channels + k) * im_size + ii];
            }

            tmpsum = Dtype(0);
            bget = false;
            getHighestValue(&tmpvalthis , x, n_channels); //get highest value
            for (kk = 1; kk < n_channels; ++kk)
            { //calculate t
                getHighestValue(&tmpvalnext , x, n_channels); //get next highest value

                tmpsum += tmpvalthis;
                tmax = (tmpsum - Dtype(1)) / Dtype(kk);
                if (tmax >= tmpvalnext)
                {
                    bget = true;
                    break;
                }else
                {
                    tmpvalthis = tmpvalnext;
                }
                
            }

            Dtype *dt_dx = &blob_for_sorting[(n * im_size + ii) * n_channels]; //blob_for_sorting is temp blob needed to do the sorting, x points to where this thread can use storage

            if (!bget)
            {
                tmax = (tmpsum + tmpvalthis - Dtype(1)) / Dtype(n_channels);
                for (int k = 0; k < n_channels; ++k)
                {
                    dt_dx[k] = Dtype(1) / Dtype(n_channels);
                }
            }
            else
            {
                for (int k = 0; k < n_channels; ++k)
                {
                    if (invals[(n * n_channels + k) * im_size + ii] > tmax)
                    {
                        dt_dx[k] = Dtype(1) / Dtype(kk);
                    }
                    else
                    {
                        dt_dx[k] = Dtype(0);
                    }
                }
            }

            for (int k1 = 0; k1 < n_channels; ++k1)
            {
                for (int k2 = 0; k2 < n_channels; ++k2)
                {
                    if (k1 == k2)
                    {
                        if (invals[(n * n_channels + k2) * im_size + ii] > tmax)
                        {
                            indervals[(n * n_channels + k1) * im_size + ii] += outdervals[(n * n_channels + k1) * im_size + ii] - dt_dx[k1] * outdervals[(n * n_channels + k2) * im_size + ii];
                        }
                        else
                        {
                            indervals[(n * n_channels + k1) * im_size + ii] += this_leak_factor * (outdervals[(n * n_channels + k1) * im_size + ii] - dt_dx[k1] * outdervals[(n * n_channels + k2) * im_size + ii]);
                        }
                    }
                    else
                    {
                        if (invals[(n * n_channels + k2) * im_size + ii] > tmax)
                        {
                            indervals[(n * n_channels + k1) * im_size + ii] -= dt_dx[k1] * outdervals[(n * n_channels + k2) * im_size + ii];
                        }
                        else
                        {
                            indervals[(n * n_channels + k1) * im_size + ii] -= this_leak_factor * (dt_dx[k1] * outdervals[(n * n_channels + k2) * im_size + ii]);
                        }
                    }
                }
            }
        }
    }
}


template <typename Dtype>
__global__ void scale_unary_weight_kernel(const int n, const Dtype* alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] *= alpha[0];
  }
}

template <typename Dtype>
__global__ void calc_sum_over_classes(const Dtype* in, Dtype* out, const int N, const int C, const int IS, const int this_c) {

    int this_n,this_ii;
    CUDA_KERNEL_LOOP(i, N*IS) {
        this_ii = i % IS;
        this_n = (i-this_ii)/IS;
        out[i] += in[(this_n*C+this_c)*IS+this_ii];
    }
}

template <typename Dtype>
__global__ void subtract_class_sum(const Dtype* insum, const Dtype* in, Dtype* out, const int N, const int C, const int IS) {

    int this_n,this_c,this_ii;
    CUDA_KERNEL_LOOP(i, N*C*IS) {
        this_ii = i % IS;
        this_c = (i-this_ii)/IS % C;
        this_n = ((i-this_ii)/IS-this_c)/C;
        out[i] = in[i] - insum[this_n*IS + this_ii];
    }
}


template <typename Dtype>
__global__ void leaky_01_projection(const int N, Dtype* inout, const Dtype leak_factor) {

    CUDA_KERNEL_LOOP(i, N) {
        if(inout[i] < 0) {
            inout[i] = leak_factor*inout[i];
        }else if(inout[i] > 1){
            inout[i] = Dtype(1) + leak_factor*(inout[i]-Dtype(1));
        } //else do nothing
    }
}

template <typename Dtype>
__global__ void backward_leaky_01_projection(const int N, const Dtype* x, Dtype* der, const Dtype leak_factor) {

    CUDA_KERNEL_LOOP(i, N) {
        if(x[i] < 0 || x[i] > 1) {
            der[i] *= leak_factor;
        } //else do nothing
    }
}

template <typename Dtype>
__global__ void part_of_backward_sum_projection( Dtype* out, const Dtype* in, Dtype one_over_nclasses, const int N, const int C, const int IS, const int that_c){
    int this_ii,this_c,this_n;
    //Dtype DEBUGVARIABLE;
    CUDA_KERNEL_LOOP(i, N*IS*C) {
        this_ii = i % IS;
        this_c = (i-this_ii)/IS % C;
        this_n = ((i-this_ii)/IS-this_c)/C;
        out[i] -= one_over_nclasses*in[(this_n*C+that_c)*IS+this_ii];
    }
}

template <typename Dtype>
__global__ void calc_bottom_diff(const int N, Dtype* bottom_diff, const Dtype* dL_dtheta_u,const Dtype* unary_weight, const Dtype* bottom_data,const Dtype log_eps,const Dtype* dL_dx_it){
    CUDA_KERNEL_LOOP(i, N) {
        bottom_diff[i] = dL_dtheta_u[i]*Dtype(-1.0)*unary_weight[0]/(bottom_data[i]+log_eps) + dL_dx_it[i];
    }
}


template <typename Dtype>
void CrfGradLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	
	// sync weights with internal layer  
    caffe_copy(this->blobs_[1]->count(),this->blobs_[1]->gpu_data(),conv_layer_->blobs()[0]->mutable_gpu_data());
    caffe_copy(this->blobs_[2]->count(),this->blobs_[2]->gpu_data(),pl_layer_->blobs()[0]->mutable_gpu_data());  
	
    const Dtype* weights_unary = this->blobs_[0]->gpu_data();
    const Dtype* weights_pw = this->blobs_[1]->gpu_data();

    Blob<Dtype> unary(bottom[0]->shape());

    caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),unary.mutable_gpu_data()); //set to x
    caffe_gpu_add_scalar(unary.count(), log_eps_, unary.mutable_gpu_data()); //add log_eps
    caffe_gpu_log(unary.count(), unary.gpu_data(), unary.mutable_gpu_data());
    scale_unary_weight_kernel<<<CAFFE_GET_BLOCKS(unary.count()), CAFFE_CUDA_NUM_THREADS>>>(unary.count(),weights_unary,unary.mutable_gpu_data());
    caffe_gpu_scal(unary.count(), Dtype(-1), unary.mutable_gpu_data());

    //SETUP BEFORE LOOP
    const vector<int>* blob_shape;
    blob_shape = &bottom[0]->shape();
    const int n_batches = (*blob_shape)[0];
    const int n_channels = (*blob_shape)[1];
    const int im_height = (*blob_shape)[2];
    const int im_width = (*blob_shape)[3];

    Blob<Dtype> current_state(bottom[0]->shape());
    current_state.CopyFrom(*bottom[0]); 			//initialize current state as input
    
    vector<Blob<Dtype>*> current_state_vec;
    current_state_vec.push_back(&current_state);

    Blob<Dtype> derivative_blob(bottom[0]->shape()); // helper blob for storing the derivative (of gradient descent steps)

    Blob<Dtype> projected_gradient(derivative_blob.shape());

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

        conv_layer_->Forward(current_state_vec,tmp_der_blob_vec);
        pl_layer_->Forward(pl_bottom,pl_top);
        
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
        
        if(it == 0){ // if first iteration save lattice blob
			caffe_copy(internal_lattice_blob_.count(),pl_top[1]->gpu_data(),internal_lattice_blob_.mutable_gpu_data());
			pl_bottom.push_back(&internal_lattice_blob_);
			pl_top.pop_back();
		}
		
		//add spatial and sparse pl contribution
		caffe_gpu_add(current_state.count(), tmp_der_blob.gpu_data(), derivative_blob.gpu_data(),derivative_blob.mutable_gpu_data());
		
        //Add unary and pairwise gradient contribution (derivative_blob now contains the derivative before all projections)
        caffe_gpu_add(derivative_blob.count(), derivative_blob.gpu_data(), unary.gpu_data(), projected_gradient.mutable_gpu_data());
		
        //Take step in descent direction
        caffe_gpu_scal(projected_gradient.count(),Dtype(-1.0)*step_size_,projected_gradient.mutable_gpu_data());
        caffe_gpu_add(current_state.count(),projected_gradient.gpu_data(),current_state.gpu_data(),current_state.mutable_gpu_data());

        // Push current state on intermedieate state vector (we need values before the last projection to be able to calculate backward derivatives)
        caffe_copy(internal_intermediate_states_[it]->count(),current_state.gpu_data(),internal_intermediate_states_[it]->mutable_gpu_data());
        
        // Project down on simplex (projected_gradient is only used for temporary storage)
        this->ProjectBlobSimplex_gpu_(&current_state,&projected_gradient);
    }

    //after iteration is done, copy current state to top (output)
    caffe_copy(top[0]->count(),current_state.gpu_data(),top[0]->mutable_gpu_data());
    has_updated_internal_states = true;
}


template <typename Dtype>
void CrfGradLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {   
    
    if(propagate_down[0]){
        CHECK_EQ(has_updated_internal_states, true)
        << "No updated internal states found, forward must be done previous to backward";

        Dtype* weight_diff_pw = this->blobs_[1]->mutable_gpu_diff();

        //const Dtype* bottom_vals = bottom[0]->cpu_data();
        const Dtype* weight_vals_u = this->blobs_[0]->gpu_data();
        const Dtype* weight_vals_pw = this->blobs_[1]->gpu_data();

        // temporary variables that hold weight diffs during iterations
        Blob<Dtype> tmp_weight_diff_pw_spatial(this->blobs_[1]->shape());
        Blob<Dtype> tmp_weight_diff_pw_pl(this->blobs_[2]->shape());

        // dL_dtheta_u
        Blob<Dtype> dL_dtheta_u(top[0]->shape());
        
        // blob for when gpu needs temporary memory
        Blob<Dtype> temp_mem(top[0]->shape());

        // dL_dx_it holds the error derivatives with respect to the input of iteration it of the rnn
        vector<Blob<Dtype>*> dL_dx_it_vec;
        Blob<Dtype> dL_dx_it(top[0]->shape());
        dL_dx_it.CopyFrom(*top[0],true,false); 			//initialize as top diff
        dL_dx_it_vec.push_back(&dL_dx_it);
        Dtype* dL_dx_it_vals = dL_dx_it.mutable_gpu_diff();

        Blob<Dtype> temp_blob(top[0]->shape());
        Dtype* temp_vals = temp_blob.mutable_gpu_diff();

        const vector<int>* blob_shape;
        blob_shape = &top[0]->shape();
        const int n_batches = (*blob_shape)[0];
        const int n_channels = (*blob_shape)[1];
        const int im_height = (*blob_shape)[2];
        const int im_width = (*blob_shape)[3];

        Blob<Dtype> inp_prev_it(top[0]->shape());

		Blob<Dtype> inp_prev_it_conv_blob(bottom[0]->shape());
        vector<Blob<Dtype>*> inp_prev_it_conv_blob_vec;
    	inp_prev_it_conv_blob_vec.push_back(&inp_prev_it_conv_blob);

		vector<Blob<Dtype>*> pl_bottom;
		pl_bottom.push_back(&inp_prev_it);
		pl_bottom.push_back(bottom[1]);
		pl_bottom.push_back(bottom[1]);
		pl_bottom.push_back(&internal_lattice_blob_);
		
		vector<Blob<Dtype>*> pl_top;
		pl_top.push_back(&temp_blob);
		
        //Loop backwards through gradient iterations
        for(int it = num_iterations_-1 ; it >= 0 ; --it){
            // inp_prev_it is the input to conv-backward, during forward process this has been projected before the convolution, 
            // if its not the first iteration. Hence we need to de the same here
            if (it > 0){
                caffe_copy(internal_intermediate_states_[it-1]->count(),internal_intermediate_states_[it-1]->gpu_data(),inp_prev_it.mutable_gpu_data()); //set prev it vals to previous internal state
                this->ProjectBlobSimplex_gpu_(&inp_prev_it,&temp_mem); 
            }else{ //For the first iteration we do not need to do this
                caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),inp_prev_it.mutable_gpu_data());
            }

            // previous iteration data is also needed by conv layer    
            caffe_copy(bottom[0]->count(),inp_prev_it.gpu_data(),inp_prev_it_conv_blob.mutable_gpu_data());
            
            //Backward through simplex projection
			caffe_gpu_set(temp_blob.count(),Dtype(0),temp_blob.mutable_gpu_diff()); //need to zero out data before calculating projection
          	this->ProjectOntoSimplexBw_gpu_(internal_intermediate_states_[it],&dL_dx_it,&temp_blob,&temp_mem);
            caffe_copy(temp_blob.count(),temp_blob.gpu_diff(),dL_dx_it.mutable_gpu_diff());           

            caffe_gpu_set(pl_layer_->blobs()[0]->count(),Dtype(0),pl_layer_->blobs()[0]->mutable_gpu_diff());
			pl_layer_->Backward(pl_top,propagate_down,pl_bottom); //pl_bottom[0] is inp_prev_it
			caffe_copy(tmp_weight_diff_pw_pl.count(),pl_layer_->blobs()[0]->gpu_diff(),tmp_weight_diff_pw_pl.mutable_gpu_diff());
			
            caffe_gpu_set(conv_layer_->blobs()[0]->count(),Dtype(0),conv_layer_->blobs()[0]->mutable_gpu_diff());
			conv_layer_->Backward(pl_top,propagate_down,inp_prev_it_conv_blob_vec);
			caffe_copy(tmp_weight_diff_pw_spatial.count(),conv_layer_->blobs()[0]->gpu_diff(),tmp_weight_diff_pw_spatial.mutable_gpu_diff());
            
            // Copy inp_prev_it_vec_it: diff field to dL_dx_it_vals for next iteration, we get two parts, one from spatial and one from pl filtering
            //pl
            caffe_gpu_scal(dL_dx_it.count(),Dtype(-1)*step_size_,inp_prev_it.mutable_gpu_diff());
            caffe_gpu_add(dL_dx_it.count(),dL_dx_it.gpu_diff(),inp_prev_it.gpu_diff(),dL_dx_it.mutable_gpu_diff());
            //conv
            caffe_gpu_scal(inp_prev_it_conv_blob.count(),Dtype(-1)*step_size_,inp_prev_it_conv_blob.mutable_gpu_diff());
            caffe_gpu_add(dL_dx_it.count(),dL_dx_it.gpu_diff(),inp_prev_it_conv_blob.gpu_diff(),dL_dx_it.mutable_gpu_diff());
            
            // accumulate pairwise pl weight diff vals
            caffe_gpu_scal(tmp_weight_diff_pw_pl.count(),Dtype(-1)*step_size_,tmp_weight_diff_pw_pl.mutable_gpu_diff());
            caffe_gpu_add(tmp_weight_diff_pw_pl.count(),tmp_weight_diff_pw_pl.gpu_diff(),this->blobs_[2]->gpu_diff(),this->blobs_[2]->mutable_gpu_diff());
            
            // accumulate pairwise spatial weight diff vals
            caffe_gpu_scal(tmp_weight_diff_pw_spatial.count(),Dtype(-1)*step_size_,tmp_weight_diff_pw_spatial.mutable_gpu_diff());
            caffe_gpu_add(tmp_weight_diff_pw_spatial.count(),tmp_weight_diff_pw_spatial.gpu_diff(),this->blobs_[1]->gpu_diff(),this->blobs_[1]->mutable_gpu_diff());

            // accumulate dL_dtheta_u dL_dtheta_u_vals
            caffe_gpu_scal(dL_dtheta_u.count(),Dtype(-1)*step_size_,temp_blob.mutable_gpu_diff());
            caffe_gpu_add(dL_dtheta_u.count(),temp_blob.gpu_diff(),dL_dtheta_u.gpu_data(),dL_dtheta_u.mutable_gpu_data());
        }

        //start_time = getUnixTime2();
        // bottom diff
        calc_bottom_diff<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(),bottom[0]->mutable_gpu_diff(),dL_dtheta_u.gpu_data(),weight_vals_u,bottom[0]->gpu_data(),log_eps_,dL_dx_it.gpu_diff());

        //Use inp_prev_it as temporary storage
        caffe_copy(inp_prev_it.count(),bottom[0]->gpu_data(),inp_prev_it.mutable_gpu_data());
        caffe_gpu_add_scalar(inp_prev_it.count(), log_eps_, inp_prev_it.mutable_gpu_data()); //add log_eps
        caffe_gpu_log(inp_prev_it.count(), inp_prev_it.gpu_data(), inp_prev_it.mutable_gpu_data());
        caffe_gpu_scal(inp_prev_it.count(), Dtype(-1), inp_prev_it.mutable_gpu_data());
        caffe_gpu_mul(inp_prev_it.count(),inp_prev_it.gpu_data(),dL_dtheta_u.gpu_data(),inp_prev_it.mutable_gpu_data());

        //parallell way of summing elements in matrix (dot product with ones)
        caffe_gpu_set(temp_blob.count(),Dtype(1),temp_blob.mutable_gpu_data());
        Dtype sum = 0;
        caffe_gpu_dot(temp_blob.count(), temp_blob.gpu_data(), inp_prev_it.gpu_data(),&sum);
        this->blobs_[0]->mutable_cpu_diff()[0] = sum;
    }
}
    
template <typename Dtype>
void CrfGradLayer<Dtype>::ProjectBlobSimplex_gpu_(Blob<Dtype>* blobinout, Blob<Dtype>* temp_mem) {
	const vector<int>* blob_shape;
    blob_shape = &blobinout->shape();
    const int n_batches = (*blob_shape)[0];
    const int n_channels = (*blob_shape)[1];
    const int im_height = (*blob_shape)[2];
    const int im_width = (*blob_shape)[3];
	
	simplex_proj_kernel_fast<<<CAFFE_GET_BLOCKS(n_batches*im_height*im_width), CAFFE_CUDA_NUM_THREADS>>>(blobinout->mutable_gpu_data(), n_batches, n_channels, im_height*im_width, leak_factor_,temp_mem->mutable_gpu_data());
}

template <typename Dtype>
void CrfGradLayer<Dtype>::ProjectOntoSimplexBw_gpu_(const Blob<Dtype>* in,const Blob<Dtype>* outder, Blob<Dtype>* inder, Blob<Dtype>* temp_mem) {
	const vector<int>* blob_shape;
    blob_shape = &in->shape();
    const int n_batches = (*blob_shape)[0];
    const int n_channels = (*blob_shape)[1];
    const int im_height = (*blob_shape)[2];
    const int im_width = (*blob_shape)[3];
	
	simplex_proj_kernel_bw_fast<<<CAFFE_GET_BLOCKS(n_batches*im_height*im_width), CAFFE_CUDA_NUM_THREADS>>>(in->gpu_data(), outder->gpu_diff(), inder->mutable_gpu_diff(), n_batches, n_channels, im_height*im_width, leak_factor_, temp_mem->mutable_gpu_data());
}


INSTANTIATE_LAYER_GPU_FUNCS(CrfGradLayer);

}  // namespace caffe
