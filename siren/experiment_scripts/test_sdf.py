'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys


#特定のディレクトリをPythonのモジュール検索パスに追加するためのもの
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=16384)#default=16384 240701_2344
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
# --resolution オプションは、解像度を指定します。解像度は、3D空間をサンプリングするための格子の解像度を表します。デフォルトは 1600 です。
p.add_argument('--resolution', type=int, default=1600)#512にすることも

opt = p.parse_args()
"""
optは以下を表す。
Namespace(config_filepath=None, logging_root='./logs', experiment_name='experiment_1_rec1', 
batch_size=16384, checkpoint_path='/mnt/siren/logs/experiment_1/checkpoints/model_final.pth', model_type='sine', mode='mlp', resolution=1600)
"""

class SDFDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the model.
        if opt.mode == 'mlp':
            self.model = modules.SingleBVPNet(type=opt.model_type, final_layer_factor=1, in_features=3)
        elif opt.mode == 'nerf':
            self.model = modules.SingleBVPNet(type='relu', mode='nerf', final_layer_factor=1, in_features=3)
        
        """ 5層の全結合ニューラルネットワーク
        SingleBVPNet(
        (image_downsampling): ImageDownsampling()
        (net): FCBlock(
            (net): MetaSequential(
            (0): MetaSequential(
                (0): BatchLinear(in_features=3, out_features=256, bias=True)
                (1): Sine()
            )
            (1): MetaSequential(
                (0): BatchLinear(in_features=256, out_features=256, bias=True)
                (1): Sine()
            )
            (2): MetaSequential(
                (0): BatchLinear(in_features=256, out_features=256, bias=True)
                (1): Sine()
            )
            (3): MetaSequential(
                (0): BatchLinear(in_features=256, out_features=256, bias=True)
                (1): Sine()
            )
            (4): MetaSequential(
                (0): BatchLinear(in_features=256, out_features=1, bias=True)
            )
            )
        )
        )
        """
        self.model.load_state_dict(torch.load(opt.checkpoint_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)['model_out']


sdf_decoder = SDFDecoder()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

"""
sdf_meshing.pyのcreate_mesh関数は、3Dシーンの表現を生成し、その表現を「点群⇒メッシュ」に変換

引数：
    ・decoder:  (SDFDecoder)                        シーンの表現を生成するデコーダーモデル。
    ・filename: ('./logs/experiment_1_rec1/test')   出力のPLYファイル名。
    ・N:        (1600)                              サンプリンググリッドの解像度。
"""
sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, 'test'), N=opt.resolution)
