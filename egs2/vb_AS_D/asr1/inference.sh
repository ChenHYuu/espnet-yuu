#!/usr/bin/env bash
spks="644"
# spks="547 644"
len="8 4 2 1"

# tag="adapter"
# tags=""
tag='wo_sp_rand'

random_mode=false

echo $random_mode > local/mode
for spk in $spks; do
    echo $spk > local/spk
    ./run.sh --stage 1 --stop_stage 10 \
    --skip_train true 
    for num in $len; do
        echo $num > local/num
        # for tag in $tags; do
            
        # done
        ./run.sh --stage 12 --stop_stage 13 \
        --skip_train true \
        --asr_tag A_SD_${tag}_${num}c_${spk}
        sleep 20
    done
done