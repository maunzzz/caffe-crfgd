#!/bin/bash
TOOLS=../../../build/tools
WEIGHTS=/home/cvia/Documents/Code/arbitrary_crfs_for_segmentation/genericcrf/voc/edgenet/deeplab-fov/fromdeeplab/train2_iter_8000.caffemodel
SOLVER=solver.prototxt
LOG=log.log
GPU=1

$TOOLS/caffe train -solver $SOLVER -weights $WEIGHTS -gpu $GPU 2>&1 | tee $LOG
