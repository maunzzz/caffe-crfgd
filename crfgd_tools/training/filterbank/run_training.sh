#!/bin/bash
TOOLS=../../../build/tools
WEIGHTS=../snapshots/filterbank_pretrain_iter_92232.caffemodel
SOLVER=solver_train.prototxt
LOG=log_train.log
GPU=1

$TOOLS/caffe train -solver $SOLVER -weights $WEIGHTS -gpu $GPU 2>&1 | tee $LOG
