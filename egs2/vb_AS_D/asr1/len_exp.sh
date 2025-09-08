#!/usr/bin/env bash
spks="573"
# spks="644"
len="4"
model="/mnt/slave1/modelzoo/asr/NER/NER_pro_pretrain_24e_mhsa/train.acc.best.pth"
# model="../../vb_AS_I/asr1/exp/asr_A_SI_rm_phase2_fd_wotm/valid.acc.ave.pth"
# model="../../vb_AS_I/asr1/exp/asr_A_SI_rm_phase2_fd_wotm_wo_phrase/valid.acc.ave.pth"

random_mode=false

tag="from_ner"

# tag='from_ner'
echo $random_mode > local/mode
for spk in $spks; do
    echo $spk > local/spk
    for num in $len; do
        echo $num > local/num
        ./run.sh --stage 1 --stop_stage 11 \
        --asr_tag A_SD_${tag}_${num}c_${spk} \
        --pretrained_model ${model}
        sleep 10
    done
done