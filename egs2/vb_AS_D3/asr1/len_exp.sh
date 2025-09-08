#!/usr/bin/env bash
# spks="573 644"
spks="439"
len="16"
# model="/mnt/slave1/modelzoo/asr/NER/NER_pro_pretrain_24e_mhsa/train.acc.best.pth"
# model="../../vb_AS_I/asr1/exp/asr_A_SI_rm_phase2_fd_wotm/valid.acc.ave.pth"
model="../../vb_AS_I/asr1/exp/asr_A_SI_rm_phase2_fd_wotm_wo_phrase/valid.acc.ave.pth"
# model="../../vb_AS_I/asr1/exp/asr_cifexp_cifv2_sp_fd2/valid.acc.ave.pth"

random_mode=false

# tag="cifv2_impr_from_ner"
tag="cifv2_seq_mode_fd_fc_no_ouputlayer"

# tag='from_ner'
echo $random_mode > local/mode
for spk in $spks; do
    echo $spk > local/spk
    for num in $len; do
        echo $num > local/num
        ./run.sh --stage 11 --stop_stage 12 \
        --asr_tag A_SD_${tag}_${num}c_${spk} 
        # --pretrained_model ${model}
        sleep 20
        # python ~/scriptzoo/corpus_related/stat_text.py data/train/text
    done
done
