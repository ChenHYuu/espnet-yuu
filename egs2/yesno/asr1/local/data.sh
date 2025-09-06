#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/} 
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=1

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./db.sh
. ./path.sh
. ./cmd.sh

# === 設定資料路徑 ===
train_wav_dir=downloads/train_wavs
train_txt_dir=downloads/train_text
dev_wav_dir=downloads/dev_wavs
dev_txt_dir=downloads/dev_text
test_wav_dir=downloads/test_wavs
test_txt_dir=downloads/test_text

train_set="train_nodev"
train_dev="train_dev"
eval_set="test_yesno"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Prepare data directories from existing wav/text splits"

    mkdir -p data/local

    # === 取得檔名 list（無副檔名）===
    find ${train_wav_dir} -name '*.wav' | sed 's:.*/::;s:.wav$::' | sort > data/local/train.list
    find ${train_txt_dir} -name '*.txt' | sed 's:.*/::;s:.txt$::' | sort > data/local/train_txt.list
    find ${dev_wav_dir}   -name '*.wav' | sed 's:.*/::;s:.wav$::' | sort > data/local/dev.list
    find ${test_wav_dir}  -name '*.wav' | sed 's:.*/::;s:.wav$::' | sort > data/local/test.list

    # === 建立 scp 與 text ===
    for set in train dev test; do
        wav_dir=downloads/${set}_wavs
        txt_dir=downloads/${set}_text
        list_file=data/local/${set}.list

        # wav.scp
        awk -v d=${wav_dir} '{printf("%s %s/%s.wav\n", $1, d, $1)}' ${list_file} > data/local/${set}_wav.scp
        # text
        awk -v d=${txt_dir} '{printf("%s ", $1); system("cat " d "/" $1 ".txt")}' ${list_file} > data/local/${set}.txt
    done

    # === 建立 data/{train_yesno,train_dev,test_yesno} ===
    cp -r data/local/train_wav.scp data/local/train_yesno_wav.scp
    cp -r data/local/train.txt      data/local/train_yesno.txt
    cp -r data/local/dev_wav.scp    data/local/train_dev_wav.scp
    cp -r data/local/dev.txt        data/local/train_dev.txt
    cp -r data/local/test_wav.scp   data/local/${eval_set}_wav.scp
    cp -r data/local/test.txt       data/local/${eval_set}.txt

    for x in train_yesno train_dev ${eval_set}; do
        mkdir -p data/$x
        cp data/local/${x}_wav.scp data/$x/wav.scp
        cp data/local/$x.txt       data/$x/text

        # 建立 utt2spk，speaker = utt_id 自己
        awk '{ spk=$1; if (match(spk, /^sp[0-9.]+-/)) sub(/^sp[0-9.]+-/, "", spk); print $1, spk }' data/$x/text > data/$x/utt2spk
        utils/utt2spk_to_spk2utt.pl < data/$x/utt2spk > data/$x/spk2utt
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Splitting train_yesno into train_dev and train_nodev"

    # 檢查 train_yesno 是否存在
    [ ! -d data/train_yesno ] && echo "Missing data/train_yesno" && exit 1

    # 切出 dev 部分
    ndev=$(wc -l < data/train_dev/text)
    utils/subset_data_dir.sh --first data/train_yesno ${ndev} data/${train_dev}

    # 剩下的給 train_nodev
    ntrain=$(($(wc -l < data/train_yesno/text) - ndev))
    utils/subset_data_dir.sh --last data/train_yesno ${ntrain} data/${train_set}

    # 確保每個資料夾格式正確
    for x in ${train_set} ${train_dev} ${eval_set}; do
        sort -k1,1 data/$x/wav.scp -o data/$x/wav.scp
        sort -k1,1 data/$x/text    -o data/$x/text
        sort -k1,1 data/$x/utt2spk -o data/$x/utt2spk
        utils/utt2spk_to_spk2utt.pl data/$x/utt2spk > data/$x/spk2utt
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
