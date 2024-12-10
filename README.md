# My Project

## 概要
このプロジェクトは、SIRENにて学習されたモデルを作成するために作られました。この際に、各層ごとの画像も保存されます。

## 使用方法
1. SIRENにて画像を学習する。この際に、各層ごとの画像も保存される。以下のコマンドを実行する：  
   python experiment_scripts/train_img.py --model_type=sine --input_path=mip_nd2_yasuda/230627_Crest_LysMskin_LE_day13_CD3PE_EB_3D_MIP/240617_1203_EGFP_resized_512x512_no_opencv.tiff --num_hidden_layers=1 --experiment_name=experiment_oo
   - input_path：入力ファイル（学習したい画像）
   - num_hidden_layers：隠れ層の数
   - experiment_name：出力ディレクトリ（学習済みモデルや各層ごとの画像を保存するフォルダ名）
