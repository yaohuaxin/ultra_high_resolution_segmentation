echo "Start train mode: 1."

python -m torch.distributed.launch --nproc_per_node 4 --nnodes 1 train_deep_globe_DDP.py \
--n_class 3 \
--data_path  "/home/shhxyao/huaxin/projects/ai/contest/DatasetPAIP2019/" \
--model_path "$(pwd)/generetedFiles/saved_models/" \
--log_path   "$(pwd)/generetedFiles/runs/" \
--task_name "PAIP2019.yaohuaxin.ibm" \
--mode 1 \
--batch_size 1 \
--sub_batch_size 1 \
--size_g 2000 \
--size_p 1000 \
--path_g   "PAIP2019-global.pth" \
--path_g2l "PAIP2019-global2local.pth" \
--path_l2g "PAIP2019-local2global.pth" \
--data_loader_worker 10 \
--image_level 2 \
