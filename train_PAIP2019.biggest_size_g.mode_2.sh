export CUDA_VISIBLE_DEVICES=0

echo "Start train mode: 2."
python -u train_deep_globe.py \
--n_class 2 \
--data_path  "/data/huaxin/projects/ai/datasets/PAIP-2019-Decompressed/" \
--model_path "$(pwd)/generetedFiles/saved_models/" \
--log_path   "$(pwd)/generetedFiles/runs/" \
--task_name "PAIP2019.yaohuaxin.ibm" \
--mode 2 \
--batch_size 1 \
--sub_batch_size 4 \
--size_g 6000 \
--size_p 500 \
--path_g   "PAIP2019-global.pth" \
--path_g2l "PAIP2019-global2local.pth" \
--path_l2g "PAIP2019-local2global.pth" \
--data_loader_worker 10 \
--image_level 1 \

