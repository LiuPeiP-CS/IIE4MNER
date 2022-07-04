#!/usr/bin/env bash
for i in 'twitter2015' # 'conll2003' 'twitter2015' 'twitter2017'--bertlayer
do
    for k in 'MSCMT'
    do
      echo 'run_mtmner_crf.py'
      echo ${i}
      echo ${k}
      PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=2 python run_mtmner_crf.py --data_dir=data/${i} \
      --bert_model=bert-base-cased --task_name=${i} --output_dir=out/${i}_${k}/ \
      --max_seq_length=64 --do_train --do_eval --train_batch_size=24 --mm_model ${k} \
      --layer_num1=1 --layer_num2=1 --layer_num3=1 --learning_rate=3e-5
      # --do_eval --mm_model ${k} --layer_num1=1 --layer_num2=1 --layer_num3=1
    done
done
