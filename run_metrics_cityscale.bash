#!/bin/bash

dir="$1"

cd ./metrics_cityscale

./cal_pixelscore.bash $dir 
./cal_apls.bash $dir 
./cal_topo.bash $dir 


