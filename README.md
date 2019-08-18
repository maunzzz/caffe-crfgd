# Caffe-crfgd
Public repository containing implementations of the work done in "A Projected Gradient Descent Method for CRF Inference Allowing End-to-End Training of Arbitrary Pairwise Potentials" and "Revisiting Deep Structured Models in Semantic Segmentation with Gradient-Based Inference". The latter is not yet published but can be found as a part of "End-to-End Learning of Deep Structured Models for Semantic Segmentation".

This repository also uses code from [this repo](https://github.com/MPI-IS/bilateralNN) that implements "Learning sparse high dimensional filters : Image Filtering, Dense CRFs and Bilateral Neural Networks".

## Usage
The different types of CRF models presented in the paper are implemented as caffe layers. For example usage see the crfgd_tools folder. This folder contains code for data handling, training and result visualization.

## Citation
Please consider citing the following publications if it helps your research:

	@PhdThesis{crfe2e2018,
	  author =      {Larsson, M{\aa}ns},
	  title =      {End-to-End Learning of Deep Structured Models for Semantic Segmentation},
	  school =      {Chalmers University of Technology (CTH), Gothenburg, Sweden},
	  year =      {2018},
	  type =      {Licentiate Thesis},
	  month =      {Mar.}
	}
	
	@article{larsson2018revisiting,
  		title={Revisiting Deep Structured Models for Pixel-Level Labeling with Gradient-Based Inference},
  		author={Larsson, M{\aa}ns and Arnab, Anurag and Zheng, Shuai and Torr, Philip and Kahl, Fredrik},
  		journal={SIAM Journal on Imaging Sciences},
  		volume={11},
  		number={4},
  		pages={2610--2628},
  		year={2018},
  		publisher={SIAM}
	}

	@InProceedings{crfgd2018,
	  author="Larsson, M{\aa}ns and Arnab, Anurag and Kahl, Fredrik and Zheng, Shuai and Torr, Philip", editor="Pelillo, Marcello and Hancock, Edwin",
	  title="A Projected Gradient Descent Method for CRF Inference Allowing End-to-End Training of Arbitrary Pairwise Potentials",
	  booktitle="Energy Minimization Methods in Computer Vision and Pattern Recognition",
	  year="2018",
	  publisher="Springer International Publishing",
	  address="Cham",
	  pages="564--579"
	}

## Caffe License and Citation
Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.


Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
