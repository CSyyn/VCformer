export CUDA_VISIBLE_DEVICES = 0

model_name = VCformer
seq_len = 96

for pred_len in 96 192 336 720 
do
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_${seq_len}_${pred_len} \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --d_model 512 \
    --d_ff 1024 \
    --batch_size 16 \
    --dropout 0.1 \
    --train_epochs 20 \
    --patience 5 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done