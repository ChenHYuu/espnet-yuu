import os
import re

def contains_english(text):
    return bool(re.search(r'[a-zA-Z]', text))

# This file is used to make transcript
tmp_dir = 'data/local/tmp'
txt_list = f'{tmp_dir}/txt.flist'
trans = f'{tmp_dir}/transcript_vb.txt'

with open(trans, 'w') as trf:
    with open(txt_list, 'r') as f:
        x = f.readline()
        while x:
            x = x.strip()
            sp = x.split("/")
            fn = sp[-1]
            id = fn.replace('.txt', '') # extension modify here
            if 'rvtw' in id:
                id = id.replace('rvtw', "")
            else:
                id = id.split('_')[-1]
                id = f"{sp[-3]}_{id}"
            with open(x, 'r') as tf:
                try:
                    text = tf.read()
                except:
                    print(id)
            text = text.strip()
            if contains_english(text):
                x = f.readline()
                continue
            trf.write(f'{id} {text}\n')
            # spk = sp[-3]
            # qua = sp[-4]
            x = f.readline()