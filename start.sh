#!/usr/bin/bash
mypid=$(netstat -apn|grep 5052|awk '{print $7}'|cut -d/ -f1);
echo $mypid
kill -9 $mypid
source activate
conda deactivate
conda activate cmgan
python /home/zxcl/workspace/zsm/project/Speaker-Verification/verification.py
