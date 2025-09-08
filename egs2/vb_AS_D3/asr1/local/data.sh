#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# This script is use to make vb2023 corpus as test set
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}



help_message=$(cat << EOF
Usage: $0

Options:
    --remove_archive (bool): true or false
      With remove_archive=True, the archives will be removed after being successfully downloaded and un-tarred.
EOF
)
SECONDS=0

log "$0 $*"


. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if [ -z "${VB2023}" ]; then
  log "Error: \$VB2023 is not set in db.sh."
  exit 2
fi

# Decide which phase to use
vb_sub_set="phase-1 phase-2 phase-3 phase-4 phase-5"
# vb_as_set=train'
# train_num=4 #1~16
train_num=$(cat local/num)
max_num=16
# spk="435"
spk=$(cat local/spk)
train_corpus=
dev_corpus="2"
test_corpus="3 4 5 6 7 8"
sen_corpus="2"
sp_corpus="3 4 5 6 7 8"

# random_mode=false
random_mode=$(cat local/mode)

train_corpus=''
# If random mode then set to max and shuffle
if [ $random_mode = true ];then
  train_num=$max_num
  echo 'Use random mode!!'
fi
for ((i=1; i<=10#$train_num; i++)); do
  a=$(printf "%02d" $i)
  train_corpus="$train_corpus $a"
done

echo $train_corpus

use_eng=false

log "Data Preparation"
train_dir=data/local/train
dev_dir=data/local/dev
test_dir=data/local/test
tmp_dir=data/local/tmp
sen_dir=data/local/sen
sp_dir=data/local/sp


mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir
mkdir -p $tmp_dir
mkdir -p $sen_dir
mkdir -p $sp_dir

if [ -f $tmp_dir/wav.flist ]; then
  rm $tmp_dir/wav.flist
fi

if [ -f $tmp_dir/txt.flist ]; then
  rm $tmp_dir/txt.flist
fi

# Get all wav file in all sub sets
for p in $vb_sub_set; do
  echo Collecting set $p corpus 
  find $VB2023/$p -iname "*.wav" >> $tmp_dir/wav.flist # Remove 16k if you want to use another sample rate like 48k or 44.1k
  find $VB2023/$p -iname "*.txt" >> $tmp_dir/txt.flist
done

vb_transcript=downloads/transcript_vb.txt

# Make transcript, if it does not exist, make one
echo "Making vb2023 transcript"
# cat $tmp_dir/vb_txt.flist | awk -F/ '{printf "%s ", $NF; system("cat "$0"")}' | awk -F.txt '{print $1 $2}' > $tmp_dir/transcript_vb.txt
python3 local/vb.py
cp $tmp_dir/transcript_vb.txt $vb_transcript

echo vb2023 transcript file at "$vb_transcript"

# remove old wav file list
for _dir in $train_dir $dev_dir $test_dir $sen_dir $sp_dir; do
  if [ -f $_dir/wav.flist ]; then
    echo "remove $_dir"
    rm $_dir/wav.flist
  fi
done

# Divid into train and dev
echo "Preparing dev set"
# dev set is consist with previous half of corpus s2 of specific spk 
for dspk in $spk; do
  in_series="01 02 03 04 05 06"
  for s in $in_series; do
    grep ${dspk}_s2${s} $tmp_dir/wav.flist >> $dev_dir/wav.flist || echo "Dev: $dspk have no $s";
  done
done

# Make test set
echo "Preparing test set"
for tspk in $spk; do
  # corpus 2 requir special process
  # testset contains after half of corpus s2
  in_series="07 08 09 10 11 12"
  for s in $in_series; do
    grep ${tspk}_s2${s} $tmp_dir/wav.flist >> $test_dir/wav.flist || echo "Test: $tspk have no $s in s2";
    grep ${tspk}_s2${s} $tmp_dir/wav.flist >> $sen_dir/wav.flist || echo "Test: $tspk have no $s in s2";
  done
  for c in $test_corpus; do
    grep ${tspk}_s${c} $tmp_dir/wav.flist >> $test_dir/wav.flist || echo "Test: $tspk have no $c"
    grep ${tspk}_s${c} $tmp_dir/wav.flist >> $sp_dir/wav.flist || echo "Test: $tspk have no $c"
  done
done

echo "Preparing train set"
# trainset is consist of corpus s1
for tspk in $spk; do
  for c in $train_corpus; do
    grep ${tspk}_s1$c $tmp_dir/wav.flist >> $train_dir/wav.flist || echo $c not in $spk;
  done
  if [ $random_mode = true ]; then
    train_num=$(cat local/num)
    mv $train_dir/wav.flist $tmp_dir/wav.flist.train
    all_num=$(cat $tmp_dir/wav.flist.train | wc -l)
    ratio=$(python local/div.py $train_num $max_num)
    line_num=$(python local/mul.py $all_num $ratio)
    echo "Total: $all_num, ratio: $ratio, Target: $line_num"
    shuf $tmp_dir/wav.flist.train | head -n $line_num > $train_dir/wav.flist
  fi
done
# cp $tmp_dir/wav.flist $train_dir/wav.flist

for dir in $train_dir $dev_dir $test_dir $sen_dir $sp_dir; do
  log Preparing $dir transcriptions
  # Remove redundent string
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' | sed -e 's/^rvtw//' | sed -e 's/^vb_phase//' |sed -e 's/^-1_//' | sed -e 's/^-2_//' | sed -e 's/^-3_//' | sed -e 's/^-4_//' | sed -e 's/^-5_//'  > $dir/utt.list
  # Spk format defined here
  awk -F '_' '{print $0,$1}' $dir/utt.list > $dir/utt2spk_all
  paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
  utils/filter_scp.pl -f 1 $dir/utt.list $vb_transcript > $dir/transcripts.txt
  awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/utt2spk_all | sort -u > $dir/utt2spk
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
  sort -u $dir/transcripts.txt > $dir/text
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

log "Successfully finished making test data [elapsed=${SECONDS}s]"

rm -r $tmp_dir

mkdir -p data/train data/dev data/test data/sen data/sp

for f in spk2utt utt2spk wav.scp text; do
  cp $train_dir/$f data/train/$f || exit 1;
  cp $dev_dir/$f data/dev/$f || exit 1;
  cp $test_dir/$f data/test/$f || exit 1;
  cp $sen_dir/$f data/sen/$f || exit 1;
  cp $sp_dir/$f data/sp/$f || exit 1;
done

# remove space in text
for x in train dev test sen sp; do
  cp data/${x}/text data/${x}/text.org
  paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " " |../../../utils/remove_punctuation.pl) \
      > data/${x}/text
#  cp data/${x}/text data/${x}/text.org
#  ../../../utils/remove_punctuation.pl < data/${x}/text.org > data/${x}/text
  rm data/${x}/text.org
done

log "Successfully finished. [elapsed=${SECONDS}s]"
