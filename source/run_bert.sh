#%%

!python ./run_bert_2562.py \
--model_type bert \
--model_name_or_path ../model/chinese_roberta_wwm_large_ext_pytorch  \
--do_train \
--do_eval \
--do_test \
--data_dir ../data/data_StratifiedKFold_42/data_origin_0 \
--output_dir ../model/roberta_wwm_large_2562_gru1_42/roberta_wwm_large_2562_gru1_0 \
--max_seq_length 256 \
--split_num 2 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 20000 \
--freeze 0

#%%

# base:  首尾
!python ./run_bert_2562.py \
--model_type bert \
--model_name_or_path ../model/chinese_roberta_wwm_large_ext_pytorch  \
--do_train \
--do_eval \
--do_test \
--data_dir ../data/data_StratifiedKFold_42/data_origin_1 \
--output_dir ../model/roberta_wwm_large_2562_gru1_42/roberta_wwm_large_2562_gru1_1 \
--max_seq_length 256 \
--split_num 2 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 20000 \
--freeze 0

#%%

# base:  首尾
!python ./run_bert_2562.py \
--model_type bert \
--model_name_or_path ../model/chinese_roberta_wwm_large_ext_pytorch  \
--do_train \
--do_eval \
--do_test \
--data_dir ../data/data_StratifiedKFold_42/data_origin_2 \
--output_dir ../model/roberta_wwm_large_2562_gru1_42/roberta_wwm_large_2562_gru1_2 \
--max_seq_length 256 \
--split_num 2 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 20000 \
--freeze 0

#%%

# base:  首尾
!python ./run_bert_2562.py \
--model_type bert \
--model_name_or_path ../model/chinese_roberta_wwm_large_ext_pytorch  \
--do_train \
--do_eval \
--do_test \
--data_dir ../data/data_StratifiedKFold_42/data_origin_3 \
--output_dir ../model/roberta_wwm_large_2562_gru1_42/roberta_wwm_large_2562_gru1_3 \
--max_seq_length 256 \
--split_num 2 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 20000 \
--freeze 0

#%%

# base:  首尾
!python ./run_bert_2562.py \
--model_type bert \
--model_name_or_path ../model/chinese_roberta_wwm_large_ext_pytorch  \
--do_train \
--do_eval \
--do_test \
--data_dir ../data/data_StratifiedKFold_42/data_origin_4 \
--output_dir ../model/roberta_wwm_large_2562_gru1_42/roberta_wwm_large_2562_gru1_4 \
--max_seq_length 256 \
--split_num 2 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 20000 \
--freeze 0