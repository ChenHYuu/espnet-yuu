#!/usr/bin/env bash
spks="573"
# spks=""
len="8 4 2 1"
model="/mnt/slave1/modelzoo/asr/NER/NER_pro_pretrain_24e_mhsa/train.acc.best.pth"
# model="../../vb_AS_I/asr1/exp/asr_A_SI_rm_phase2_fd_wotm/valid.acc.ave.pth"
# model="../../vb_AS_I/asr1/exp/asr_A_SI_rm_phase2_fd_wotm_wo_phrase/valid.acc.ave.pth"

random_mode=true

tag="from_ner_rand"

# tag='from_ner'
echo $random_mode > local/mode
for spk in $spks; do
    echo $spk > local/spk
    for num in $len; do
        echo $num > local/num
        
        ./run.sh --stage 1 --stop_stage 1 \
        --asr_tag A_SD_${tag}_${num}c_${spk} \
        --pretrained_model ${model}

        rm -r data/train
        cp -r rand_corpus/rand_${num}c_${spk}/train data/train

        ./run.sh --stage 2 --stop_stage 11 \
        --asr_tag A_SD_${tag}_${num}c_${spk} \
        --pretrained_model ${model}
        
        # for s in train;do
        #     rm -r rand_corpus/${tag}_${num}c_${spk}/$s
        #     mkdir -p rand_corpus/${tag}_${num}c_${spk}/$s
        #     cp -r data/$s/* rand_corpus/${tag}_${num}c_${spk}/$s
        # done
        sleep 10
    done
done