#!/bin/bash

dir="$1"

cd ./metrics_spacenet

./cal_pixelscore.bash $dir 
./cal_apls.bash $dir
./cal_topo.bash $dir 


