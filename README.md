# My Project

## 概要
このプロジェクトは、SIRENにて学習されたモデルを作成するために作られました。この際に、各層ごとの画像も保存されます。

## 使用方法

<details><summary>事前確認コマンド</summary>

Please check below commnd.
```
nvidia-smi
docker
```

</details>

<details><summary>Docker環境の構築</summary>

### Build
`image_name` is free as docker image name．<br>
```
cd docker_for_build
sudo docker build -t image_name .
cd ..
```

### Run Container
`$pwd` is mount current dir．<br>
```
sudo docker run -it --shm-size 2g --gpus all -v $(pwd):/workspace image_name 
```

</details>

<details><summary>1.画像の学習</summary>
   
SIRENにて画像を学習する。この際に、各層ごとの画像も保存される。以下のコマンドを実行する： 
```
python experiment_scripts/train_img.py \
--model_type=sine \
--input_path=mip_nd2_yasuda/230627_Crest_LysMskin_LE_day13_CD3PE_EB_3D_MIP/240617_1203_EGFP_resized_512x512_no_opencv.tiff \
--num_hidden_layers=1 \
--experiment_name=experiment_oo
```
- input_path：入力ファイル（学習したい画像）
- num_hidden_layers：隠れ層の数
- experiment_name：出力ディレクトリ（学習済みモデルや各層ごとの画像を保存するフォルダ名）

</details>

<details><summary>2.推論</summary>

1.にて作成された学習済みモデルに、適当な座標数を入力する。以下のコマンドを実行する：
```
python explore_siren_ipynb_eval6.py \
--omega=30 \
--model_path=logs/experiment_oo/checkpoints/model_final.pth \
--output_dir=logs/experiment_oo/w30_hl1_1024 \
--hidden_layers=1 \
--sidelen=1024
```
- omega：基本30でいい。train_img.pyでは30で固定している。
- model_path：入力ファイル（学習済みモデル）
- output_dir：出力ディレクトリ（学習済みモデルにて作成される出力画像を保存するフォルダ）
- hidden_layers：隠れ層の数
- sidelen：座標数
  
</details>

  
## 補足
tmuxを用いて、コマンドを連続的に実行することが出来る。以下のコマンドのように実行できる。
```
python experiment_scripts/train_img.py --model_type=sine --input_path=/mnt/siren/mip_nd2_yasuda/230627_Crest_LysMskin_Sham_day13_CD3PE_EB_3D_MIP/240617_1203_EGFP_resized_512x512_no_opencv.tiff --num_hidden_layers=1 --experiment_name=experiment_oo_hl1_eachLayer && \
python experiment_scripts/train_img.py --model_type=sine --input_path=/mnt/siren/mip_nd2_yasuda/230627_Crest_LysMskin_Sham_day13_CD3PE_EB_3D_MIP/240617_1203_EGFP_resized_512x512_no_opencv.tiff --num_hidden_layers=2 --experiment_name=experiment_oo_hl2_eachLayer && \
python experiment_scripts/train_img.py --model_type=sine --input_path=/mnt/siren/mip_nd2_yasuda/230627_Crest_LysMskin_Sham_day13_CD3PE_EB_3D_MIP/240617_1203_EGFP_resized_512x512_no_opencv.tiff --num_hidden_layers=3 --experiment_name=experiment_oo_hl3_eachLayer
```
