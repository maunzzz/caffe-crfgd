function w_matlab = weights_caffe_to_matlab(w_caffe)
    w_size = size(w_caffe);
    n_classes = w_size(4);
    n_features = w_size(3)/n_classes;
    
    w_matlab = zeros(w_size);
    
    for c_in = 1:n_classes
        for c_out = 1:n_classes
            for ff = 1:n_features
                w_matlab(:,:,3*(c_out-1)+ff,c_in)=w_caffe(:,:,2*(ff-1)+c_in,c_out);
            end
        end
    end
end