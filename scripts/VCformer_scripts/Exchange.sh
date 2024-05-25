export CUDA_VISIBLE_DEVICES=1

model_name=VCformer
seq_len=96

for pred_len in 96 192 336 720 
do
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --d_model 128 \
    --d_ff 256 \
    --batch_size 16 \
    --dropout 0.1 \
    --train_epochs 10 \
    --patience 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --itr 1
done