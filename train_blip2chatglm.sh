python train_blip2chatglm.py \
    --img_data_path data/MEPAVE/product_images \
    --train_data_path data/MEPAVE/jdair.jave.train.uie.jsonl \
    --dev_data_path data/MEPAVE/jdair.jave.valid.uie.jsonl \
    --question "以{类型：属性}的形式列出图中商品的所有属性" \
    --blip2_path models/blip2zh-chatglm-6b \
    --lora_rank 16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_steps 50000 \
    --save_steps 10000 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --prediction_loss_only \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --dataloader_num_workers 8 \
    --output_dir output