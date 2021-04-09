# Graph-based-Recommendation-System

## Prerequiste of packages for our model:
```bash

python==3.7
pandas==0.24.2
matplotlib==2.2.2
argparse==1.1
tqdm==4.31.1
tensorboardX==1.7
numpy==1.17.3
torch==1.3.1
networkx==2.4
folium==0.10.1
```

## Running the model
```bash
python3 scripts/train.py --rate_num 5 \
			--lr 0.01 \
			--weight_decay 0.00001 \
			--num_epochs 1000 \
			--hidden_dim 5 \
			--side_hidden_dim 5 \
			--out_dim 5 \
			--drop_out 0.0 \
			--split_ratio 0.8 \
			--save_steps 100 \
			--log_dir './log' \
			--saved_model_folder './weights' \
			--dataset_path './ml-100k' \
			--save_processed_data_path './data' \
			--use_side_feature 1 \
			--use_data_whitening 1 \
			--use_laplacian_loss 1 \
			--laplacian_loss_weight 0.05

```

You can observe the loss curve through the training by runing:
```bash
tensorboard --logdir=log/name_of_your_saved_file
```

## Gridsearch for best hyperparameters by running shell script:
The reult(RMSE) will be saved in a **gridsearch.txt** under the text folder.
```bash
bash scripts/run.sh -lr 0.01 0.02 -epochs 1000 2000 -hidden_dim 3 5 -side_hidden_dim 3 5 -dropout 0 0.1 0.2 -use_side_feature 0 1 -use_data_whitening 0 1 -use_laplacian_loss 0 1 -laplacian_loss_weight 0.05 0.1 | tee -a text/gridsearch.txt
```
