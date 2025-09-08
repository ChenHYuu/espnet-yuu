#!/usr/bin/env bash
spks="439"
# spks="547 644"
len="16"

# tag="adapter"
# tags=""
tag='cifv2_impr_seq_mode_real'

random_mode=true

echo $random_mode > local/mode
for spk in $spks; do
    echo $spk > local/spk
    #./run.sh --stage 1 --stop_stage 10 \
    #--skip_train true 
    for num in $len; do
        echo $num > local/num
        # for tag in $tags; do
            
        # done
        # mkdir exp/asr_U_SI_${spk}
        # cp /mnt/slave1/modelzoo/asr/NER/NER_pro_pretrain_24e_mhsa/train.acc.best.pth exp/asr_U_SI_${spk}/valid.acc.ave.pth

        ./run.sh --stage 12 --stop_stage 13 \
        --skip_train true \
        --asr_tag A_SD_${tag}_${num}c_${spk} 
        sleep 20
    done
done
