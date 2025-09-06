#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train_nodev"
valid_set="train_dev"
test_sets="train_dev test_yesno"

asr_config=conf/train_asr.yaml
inference_config=conf/decode.yaml

speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                        \
    --nj 32                                     \
    --inference_nj 32                           \
    --ngpu 2                                    \
    --lang zh                                   \
    --audio_format wav                          \
    --feats_type raw                            \
    --token_type word                           \
    --use_lm false                              \
    --asr_config "${asr_config}"                \
    --inference_config "${inference_config}"    \
    --train_set "${train_set}"                  \
    --valid_set "${valid_set}"                  \
    --test_sets "${test_sets}"                  \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" "$@"
