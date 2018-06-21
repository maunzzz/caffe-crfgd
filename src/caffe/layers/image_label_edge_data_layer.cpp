#include <opencv2/core/core.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/image_label_edge_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/imgproc/imgproc.hpp>

namespace {

cv::Mat PadImage(cv::Mat &image, int min_size, double value = -1, bool pad_centre = true) {
  if (image.rows >= min_size && image.cols >= min_size) {
    return image;
  }
  int top, bottom, left, right;
  top = bottom = left = right = 0;
  if (image.rows < min_size) {
    top = (min_size - image.rows) / 2;
    bottom = min_size - image.rows - top;

    if (!pad_centre){
      top = 0;
      bottom = min_size - image.rows;
    }
  }

  if (image.cols < min_size) {
    left = (min_size - image.cols) / 2;
    right = min_size - image.cols - left;

    if (!pad_centre){
      //left = 0;
      //right = min_size - image.cols;
      // Looks like I swapped left and right around
      right = 0;
      left = min_size - image.cols;
    }
  }
  cv::Mat big_image;
  if (value < 0) {
    cv::copyMakeBorder(image, big_image, top, bottom, right, left,
                       cv::BORDER_REFLECT_101);
  } else {
    cv::copyMakeBorder(image, big_image, top, bottom, right, left,
                       cv::BORDER_CONSTANT, cv::Scalar(value));
  }
  return big_image;
}

cv::Mat ExtendLabelMargin(cv::Mat &image, int margin_w, int margin_h,
                          double value = -1) {
  cv::Mat big_image;
  if (value < 0) {
    cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w,
                       cv::BORDER_REFLECT_101);
  } else {
    cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w,
                       cv::BORDER_CONSTANT, cv::Scalar(value));
  }
  return big_image;
}

void ApplyHSVNoise (cv::Mat &image, const int h_noise, const int s_noise, const int v_noise, std::mt19937 * rng){

  cv::cvtColor(image, image, CV_BGR2HSV);

  // Cannot take modulus with 0
  // unsigned int h_delta = 0; if (h_noise > 0) { h_delta = caffe::caffe_rng_rand() % h_noise; std::cout << "H " << h_delta << std::endl; }
  // unsigned int s_delta = 0; if (s_noise > 0) { s_delta = caffe::caffe_rng_rand() % s_noise; std::cout << "S " << s_delta << std::endl;}
  // unsigned int v_delta = 0; if (v_noise > 0) { v_delta = caffe::caffe_rng_rand() % v_noise; std::cout << "V " << v_delta << std::endl;}

  int h_delta = std::uniform_int_distribution<int>( -h_noise, h_noise)(*rng);
  int s_delta = std::uniform_int_distribution<int>( -s_noise, s_noise)(*rng);
  int v_delta = std::uniform_int_distribution<int>( -v_noise, v_noise)(*rng);
  // std::cout << "H: " << h_delta << " S: " << s_delta << " V: " << v_delta << std::endl; 

  for (int y = 0; y < image.rows; ++y){
    for (int x = 0; x < image.cols; ++x){

      int cur1 = image.at<cv::Vec3b>(cv::Point(x,y))[0];
      int cur2 = image.at<cv::Vec3b>(cv::Point(x,y))[1];
      int cur3 = image.at<cv::Vec3b>(cv::Point(x,y))[2];
      cur1 += h_delta;
      cur2 += s_delta;
      cur3 += v_delta;
      if(cur1 < 0) cur1= 0; else if(cur1 > 255) cur1 = 255;
      if(cur2 < 0) cur2= 0; else if(cur2 > 255) cur2 = 255;
      if(cur3 < 0) cur3= 0; else if(cur3 > 255) cur3 = 255;

      image.at<cv::Vec3b>(cv::Point(x,y))[0] = cur1;
      image.at<cv::Vec3b>(cv::Point(x,y))[1] = cur2;
      image.at<cv::Vec3b>(cv::Point(x,y))[2] = cur3;

    }
  }

  cv::cvtColor(image, image, CV_HSV2BGR);  
}

template <typename Dtype>
void GetLabelSlice(const Dtype *labels, int rows, int cols,
                   const caffe::Slice &label_slice, Dtype *slice_data) {
  // for (int c = 0; c < channels; ++c) {
  labels += label_slice.offset(0) * cols;
  for (int h = 0; h < label_slice.dim(0); ++h) {
    labels += label_slice.offset(1);
    for (int w = 0; w < label_slice.dim(1); ++w) {
      slice_data[w] = labels[w * label_slice.stride(1)];
    }
    labels += cols * label_slice.stride(0) - label_slice.offset(1);
    slice_data += label_slice.dim(1);
  }
  //t_label_data += this->label_margin_h_ * (label_width + this->label_margin_w_ * 2);
  // }
}

}

namespace caffe {

template <typename Dtype>
ImageLabelEdgeDataLayer<Dtype>::ImageLabelEdgeDataLayer(
    const LayerParameter &param) : BasePrefetchingData4Layer<Dtype>(param) {
  std::random_device rand_dev;
  rng_ = new std::mt19937(rand_dev());
}

template <typename Dtype>
ImageLabelEdgeDataLayer<Dtype>::~ImageLabelEdgeDataLayer() {
  this->StopInternalThread();
  delete rng_;
}

template <typename Dtype>
void ImageLabelEdgeDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  auto &data_param = this->layer_param_.image_label_edge_data_param();
  string data_dir = data_param.data_dir();
  string image_dir = data_param.image_dir();
  string label1_dir = data_param.label1_dir();
  string label2_dir = data_param.label2_dir();
  string label3_dir = data_param.label3_dir();
  string label4_dir = data_param.label4_dir();

  if (image_dir == "" && label1_dir == "" && label2_dir == "" && data_dir != "") {
    image_dir = data_dir;
    label1_dir = data_dir;
    label2_dir = data_dir;
    label3_dir = data_dir;
    label4_dir = data_dir;
  }

  // Read the file with filenames and labels
  const string& image_list_path =
      this->layer_param_.image_label_edge_data_param().image_list_path();
  LOG(INFO) << "Opening image list " << image_list_path;
  std::ifstream infile(image_list_path.c_str());
  string filename;
  while (infile >> filename) {
    image_lines_.push_back(filename);
  }

  const string& label1_list_path =
      this->layer_param_.image_label_edge_data_param().label1_list_path();
  LOG(INFO) << "Opening label list " << label1_list_path;
  std::ifstream in_label1(label1_list_path.c_str());
  while (in_label1 >> filename) {
    label1_lines_.push_back(filename);
  }

  const string& label2_list_path =
      this->layer_param_.image_label_edge_data_param().label2_list_path();
  LOG(INFO) << "Opening label list " << label2_list_path;
  std::ifstream in_label2(label2_list_path.c_str());
  while (in_label2 >> filename) {
    label2_lines_.push_back(filename);
  }

  const string& label3_list_path =
      this->layer_param_.image_label_edge_data_param().label3_list_path();
  LOG(INFO) << "Opening label list " << label3_list_path;
  std::ifstream in_label3(label3_list_path.c_str());
  while (in_label3 >> filename) {
    label3_lines_.push_back(filename);
  }

  const string& label4_list_path =
      this->layer_param_.image_label_edge_data_param().label4_list_path();
  LOG(INFO) << "Opening label list " << label4_list_path;
  std::ifstream in_label4(label4_list_path.c_str());
  while (in_label4 >> filename) {
    label4_lines_.push_back(filename);
  }


  if (this->layer_param_.image_label_edge_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << image_lines_.size() << " images.";
  LOG(INFO) << "A total of " << label1_lines_.size() << " label 1.";
  LOG(INFO) << "A total of " << label2_lines_.size() << " label 2.";
  LOG(INFO) << "A total of " << label3_lines_.size() << " label 3.";
  LOG(INFO) << "A total of " << label4_lines_.size() << " label 4.";
  CHECK_EQ(image_lines_.size(), label1_lines_.size());
  CHECK_EQ(image_lines_.size(), label2_lines_.size());
  CHECK_EQ(image_lines_.size(), label3_lines_.size());
  CHECK_EQ(image_lines_.size(), label4_lines_.size());

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_label_edge_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.image_label_edge_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(image_lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[lines_id_]);
  CHECK(cv_img.data) << "Could not load " << image_lines_[lines_id_];
  int crop_size = -1;
  auto transform_param = this->layer_param_.transform_param();
  if (transform_param.has_crop_size()) {
    crop_size = transform_param.crop_size();
  }
  cv_img = PadImage(cv_img, crop_size);

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> data_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(data_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_label_edge_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  data_shape[0] = batch_size;
  top[0]->Reshape(data_shape);

  /*
  * HSV noise
  */
  hsv_noise_ = this->layer_param_.image_label_edge_data_param().hsv_noise();
  h_noise_ = this->layer_param_.image_label_edge_data_param().h_noise();
  s_noise_ = this->layer_param_.image_label_edge_data_param().s_noise();
  v_noise_ = this->layer_param_.image_label_edge_data_param().v_noise();

  /*
  *pad centre or not
  */
  pad_centre_ = this->layer_param_.image_label_edge_data_param().pad_centre();

  /*
   * label
   */
  auto &label_slice = this->layer_param_.image_label_edge_data_param().label_slice();
  label_margin_h_ = label_slice.offset(0);
  label_margin_w_ = label_slice.offset(1);
  LOG(INFO) << "Assuming image and label map sizes are the same";
  vector<int> label_shape(4);
  label_shape[0] = batch_size;
  label_shape[1] = 1;
  label_shape[2] = label_slice.dim(0);
  label_shape[3] = label_slice.dim(1);
  top[1]->Reshape(label_shape);
  top[2]->Reshape(label_shape);
  top[3]->Reshape(label_shape);
  top[4]->Reshape(label_shape);

  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(data_shape);
    this->prefetch_[i]->label1_.Reshape(label_shape);
    this->prefetch_[i]->label2_.Reshape(label_shape);
    this->prefetch_[i]->label3_.Reshape(label_shape);
    this->prefetch_[i]->label4_.Reshape(label_shape);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
  << top[0]->channels() << "," << top[0]->height() << ","
  << top[0]->width();

  LOG(INFO) << "output label size: " << top[1]->num() << ","
  << top[1]->channels() << "," << top[1]->height() << ","
  << top[1]->width();
}

template <typename Dtype>
void ImageLabelEdgeDataLayer<Dtype>::ShuffleImages() {
//  caffe::rng_t* prefetch_rng =
//      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
//  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
//  LOG(FATAL) <<
//      "ImageLabelDataLayer<Dtype>::ShuffleImages() is not implemented";
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  vector<int> order(image_lines_.size());
  for (int i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  shuffle(order.begin(), order.end(), prefetch_rng);
  vector<std::string> new_image_lines(image_lines_.size());
  vector<std::string> new_label1_lines(label1_lines_.size());
  vector<std::string> new_label2_lines(label2_lines_.size());
  vector<std::string> new_label3_lines(label3_lines_.size());
  vector<std::string> new_label4_lines(label4_lines_.size());
  for (int i = 0; i < order.size(); ++i) {
    new_image_lines[i] = image_lines_[order[i]];
    new_label1_lines[i] = label1_lines_[order[i]];
    new_label2_lines[i] = label2_lines_[order[i]];
    new_label3_lines[i] = label3_lines_[order[i]];
    new_label4_lines[i] = label4_lines_[order[i]];
  }
  swap(image_lines_, new_image_lines);
  swap(label1_lines_, new_label1_lines);
  swap(label2_lines_, new_label2_lines);
  swap(label3_lines_, new_label3_lines);
  swap(label4_lines_, new_label4_lines);
}

template <typename Dtype>
void ImageLabelEdgeDataLayer<Dtype>::SampleScale(cv::Mat *image, cv::Mat *label1, cv::Mat *label2, cv::Mat *label3, cv::Mat *label4) {
  ImageLabelEdgeDataParameter data_param =
      this->layer_param_.image_label_edge_data_param();
  if (!data_param.rand_scale()) return;
  double scale = std::uniform_real_distribution<double>(
      data_param.min_scale(), data_param.max_scale())(*rng_);
  cv::Size zero_size(0, 0);

  cv::resize(*label1, *label1, cv::Size(0, 0),
             scale, scale, cv::INTER_NEAREST);
  cv::resize(*label2, *label2, cv::Size(0, 0),
             scale, scale, cv::INTER_NEAREST);                     
  cv::resize(*label3, *label3, cv::Size(0, 0),
             scale, scale, cv::INTER_NEAREST);
  cv::resize(*label4, *label4, cv::Size(0, 0),
             scale, scale, cv::INTER_NEAREST);             

  
  if (scale > 1) {
    cv::resize(*image, *image, zero_size, scale, scale, cv::INTER_CUBIC);
  } else {
    cv::resize(*image, *image, zero_size, scale, scale, cv::INTER_AREA);
  }
}

template <typename Dtype>
void AssignEvenLabelWeight(const Dtype *labels, int num, Dtype *weights) {
  Dtype max_label = labels[0];
  for (int i = 0; i < num; ++i) {
    if (labels[i] != 255) {
      max_label = std::max(labels[i], max_label);
    }
  }
  int num_labels = static_cast<int>(max_label) + 1;
  vector<int> counts(num_labels, 0);
  vector<double> label_weight(num_labels);
  for (int i = 0; i < num; ++i) {
    if (labels[i] != 255) {
      counts[static_cast<int>(labels[i])] += 1;
    }
  }
  for (int i = 0; i < num_labels; ++i) {
    if (counts[i] == 0) {
      label_weight[i] = 0;
    } else {
      label_weight[i] = 1.0 / counts[i];
    }
  }
  for (int i = 0; i < num; ++i) {
    weights[i] = label_weight[static_cast<int>(labels[i])];
  }
}

template <typename Dtype>
void ImageLabelEdgeDataLayer<Dtype>::load_batch(Batch4<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->transformed_data_.count());
  ImageLabelEdgeDataParameter data_param =
      this->layer_param_.image_label_edge_data_param();
  const int batch_size = data_param.batch_size();

  string data_dir = data_param.data_dir();
  string image_dir = data_param.image_dir();
  string label1_dir = data_param.label1_dir();
  string label2_dir = data_param.label2_dir();
  string label3_dir = data_param.label3_dir();
  string label4_dir = data_param.label4_dir();

  if (image_dir == "" && label1_dir == "" && label2_dir == "" && label3_dir == "" && label4_dir == "" && data_dir != "") {
    image_dir = data_dir;
    label1_dir = data_dir;
    label2_dir = data_dir;
    label3_dir = data_dir;
    label4_dir = data_dir;
  }

  int crop_size = -1;
  auto transform_param = this->layer_param_.transform_param();
  if (transform_param.has_crop_size()) {
    crop_size = transform_param.crop_size();
  }

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[lines_id_], true);

  cv_img = PadImage(cv_img, crop_size);

  CHECK(cv_img.data) << "Could not load " << image_lines_[lines_id_];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  // read label 1
  cv::Mat cv_label1 = ReadImageToCVMat(label1_dir + label1_lines_[lines_id_],
                                      false);
  CHECK(cv_label1.data) << "Could not load " << label1_lines_[lines_id_];
  cv_label1 = PadImage(cv_label1, crop_size);
 
  vector<int> label1_shape = this->data_transformer_->InferBlobShape(cv_label1);

  this->transformed_label1_.Reshape(label1_shape);

  auto &label1_slice = this->layer_param_.image_label_edge_data_param().label_slice();

  label1_shape[0] = batch_size;
  label1_shape[2] = label1_slice.dim(0);
  label1_shape[3] = label1_slice.dim(1);
  batch->label1_.Reshape(label1_shape);

  // read label 2
  cv::Mat cv_label2 = ReadImageToCVMat(label2_dir + label2_lines_[lines_id_],
                                      false);
  CHECK(cv_label2.data) << "Could not load " << label2_lines_[lines_id_];
  cv_label2 = PadImage(cv_label2, crop_size);
 
  vector<int> label2_shape = this->data_transformer_->InferBlobShape(cv_label2);

  this->transformed_label2_.Reshape(label2_shape);

  auto &label2_slice = this->layer_param_.image_label_edge_data_param().label_slice();

  label2_shape[0] = batch_size;
  label2_shape[2] = label2_slice.dim(0);
  label2_shape[3] = label2_slice.dim(1);
  batch->label2_.Reshape(label2_shape);

  // read label 3
  cv::Mat cv_label3 = ReadImageToCVMat(label3_dir + label3_lines_[lines_id_],
                                      false);
  CHECK(cv_label3.data) << "Could not load " << label3_lines_[lines_id_];
  cv_label3 = PadImage(cv_label3, crop_size);
 
  vector<int> label3_shape = this->data_transformer_->InferBlobShape(cv_label3);

  this->transformed_label3_.Reshape(label3_shape);

  auto &label3_slice = this->layer_param_.image_label_edge_data_param().label_slice();

  label3_shape[0] = batch_size;
  label3_shape[2] = label3_slice.dim(0);
  label3_shape[3] = label3_slice.dim(1);
  batch->label3_.Reshape(label3_shape);

  // read label 4
  cv::Mat cv_label4 = ReadImageToCVMat(label4_dir + label4_lines_[lines_id_],
                                      false);
  CHECK(cv_label4.data) << "Could not load " << label4_lines_[lines_id_];
  cv_label4 = PadImage(cv_label4, crop_size);
 
  vector<int> label4_shape = this->data_transformer_->InferBlobShape(cv_label4);

  this->transformed_label4_.Reshape(label4_shape);

  auto &label4_slice = this->layer_param_.image_label_edge_data_param().label_slice();

  label4_shape[0] = batch_size;
  label4_shape[2] = label4_slice.dim(0);
  label4_shape[3] = label4_slice.dim(1);
  batch->label4_.Reshape(label4_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label1 = batch->label1_.mutable_cpu_data();
  Dtype* prefetch_label2 = batch->label2_.mutable_cpu_data();
  Dtype* prefetch_label3 = batch->label3_.mutable_cpu_data();
  Dtype* prefetch_label4 = batch->label4_.mutable_cpu_data();

  // datum scales
  auto lines_size = image_lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[lines_id_]);
    cv::Mat cv_label1 = ReadImageToCVMat(label1_dir + label1_lines_[lines_id_],
                                        false);
    cv::Mat cv_label2 = ReadImageToCVMat(label2_dir + label2_lines_[lines_id_],
                                        false);                                        
    cv::Mat cv_label3 = ReadImageToCVMat(label3_dir + label3_lines_[lines_id_],
                                        false);
    cv::Mat cv_label4 = ReadImageToCVMat(label4_dir + label4_lines_[lines_id_],
                                        false);                                                                                

    // do HSV noise here
    if (hsv_noise_){
      ApplyHSVNoise(cv_img, h_noise_, s_noise_, v_noise_, rng_);
    }

    SampleScale(&cv_img, &cv_label1, &cv_label2, &cv_label3, &cv_label4);
    switch (data_param.padding()) {
      case ImageLabelDataParameter_Padding_ZERO:
        cv_img = ExtendLabelMargin(cv_img, label_margin_w_, label_margin_h_, 0);
        cv_img = PadImage(cv_img, crop_size, 0, pad_centre_);
        break;
      case ImageLabelDataParameter_Padding_REFLECT:
        cv_img = ExtendLabelMargin(cv_img, label_margin_w_, label_margin_h_, -1);
        cv_img = PadImage(cv_img, crop_size, -1, pad_centre_);
        break;
      default:
        LOG(FATAL) << "Unknown Padding";
    }

    cv_label1 = ExtendLabelMargin(cv_label1, label_margin_w_, label_margin_h_, 255);
    cv_label1 = PadImage(cv_label1, crop_size, 255, pad_centre_);

    cv_label2 = ExtendLabelMargin(cv_label2, label_margin_w_, label_margin_h_, 255);
    cv_label2 = PadImage(cv_label2, crop_size, 255, pad_centre_);

    cv_label3 = ExtendLabelMargin(cv_label3, label_margin_w_, label_margin_h_, 255);
    cv_label3 = PadImage(cv_label3, crop_size, 255, pad_centre_);

    cv_label4 = ExtendLabelMargin(cv_label4, label_margin_w_, label_margin_h_, 255);
    cv_label4 = PadImage(cv_label4, crop_size, 255, pad_centre_);

    CHECK(cv_img.data) << "Could not load " << image_lines_[lines_id_];
    CHECK(cv_label1.data) << "Could not load " << label1_lines_[lines_id_];
    CHECK(cv_label2.data) << "Could not load " << label2_lines_[lines_id_];
    CHECK(cv_label3.data) << "Could not load " << label2_lines_[lines_id_];
    CHECK(cv_label4.data) << "Could not load " << label2_lines_[lines_id_];

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image

    int image_offset = batch->data_.offset(item_id);
    int label1_offset = batch->label1_.offset(item_id);
    int label2_offset = batch->label2_.offset(item_id);
    int label3_offset = batch->label3_.offset(item_id);
    int label4_offset = batch->label4_.offset(item_id);

    this->transformed_data_.set_cpu_data(prefetch_data + image_offset);
    // this->transformed_label_.set_cpu_data(prefetch_label + label_offset);
    this->data_transformer_->Transform(cv_img, cv_label1, cv_label2, cv_label3, cv_label4,
                                       &(this->transformed_data_), &(this->transformed_label1_), 
                                       &(this->transformed_label2_), &(this->transformed_label3_), &(this->transformed_label4_));

    Dtype *label1_data = prefetch_label1 + label1_offset;
    Dtype *label2_data = prefetch_label2 + label2_offset;
    Dtype *label3_data = prefetch_label3 + label3_offset;
    Dtype *label4_data = prefetch_label4 + label4_offset;
    
    const Dtype *t_label1_data = this->transformed_label1_.cpu_data();
    const Dtype *t_label2_data = this->transformed_label2_.cpu_data();
    const Dtype *t_label3_data = this->transformed_label3_.cpu_data();
    const Dtype *t_label4_data = this->transformed_label4_.cpu_data();
//    for (int c = 0; c < label_channel; ++c) {
//      t_label_data += this->label_margin_h_ * (label_width + this->label_margin_w_ * 2);
//      for (int h = 0; h < label_height; ++h) {
//        t_label_data += this->label_margin_w_;
//        for (int w = 0; w < label_width; ++w) {
//          label_data[w] = t_label_data[w];
//        }
//        t_label_data += this->label_margin_w_ + label_width;
//        label_data += label_width;
//      }
//      t_label_data += this->label_margin_h_ * (label_width + this->label_margin_w_ * 2);
//    }
    GetLabelSlice(t_label1_data, crop_size, crop_size, label1_slice, label1_data);
    GetLabelSlice(t_label2_data, crop_size, crop_size, label2_slice, label2_data);
    GetLabelSlice(t_label3_data, crop_size, crop_size, label3_slice, label3_data);
    GetLabelSlice(t_label4_data, crop_size, crop_size, label4_slice, label4_data);
//    CHECK_EQ(t_label_data - this->transformed_label_.cpu_data(),
//             this->transformed_label_.count());
//    cv::Mat_<Dtype> cropped_label(label_height, label_width,
//                                  prefetch_label + label_offset);
//    cropped_label = transformed_label(
//        cv::Range(label_margin_h_, label_margin_h_ + label_height),
//        cv::Range(label_margin_w_, label_margin_w_ + label_width));
    trans_time += timer.MicroSeconds();

    // prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_label_edge_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }

  batch_timer.Stop();
  /*DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";*/
}

INSTANTIATE_CLASS(ImageLabelEdgeDataLayer);
REGISTER_LAYER_CLASS(ImageLabelEdgeData);

}  // namespace caffe
