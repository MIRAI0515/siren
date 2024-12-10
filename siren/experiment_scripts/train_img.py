# Enable import from parent package
import sys
import os
#import torch #yasuda
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial

# メモリの使用状況を表示
#print(torch.cuda.memory_summary(device=None, abbreviated=False)) #yasuda
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='/mnt/siren/logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True, # default="experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl2",# 変更箇所 
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')#default=10000

p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')#default=1000

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

# 追加箇所 
p.add_argument('--num_hidden_layers', type=int, # default=2, 
               help='Number of hidden layers in the model. Default is 3.') 

# 追加箇所
p.add_argument('--input_path', type=str, required=True, 
               help='Path to the input image file. Example: /path/to/image.tiff')

opt = p.parse_args()

# num_hidden_layers をクラス属性として設定
if opt.num_hidden_layers:
    modules.SingleBVPNet.set_hidden_layer(opt.num_hidden_layers)

img_dataset = dataio.Collagen(input_path=opt.input_path)
#img_dataset = dataio.Camera()
# img_dataset = dataio.Collagen_1024()



#ここでのImplicit2DWrapperは、ImageFittingに該当するのかな。
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=512, compute_diff='all')# 変更箇所
# coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=64, compute_diff='all') #全体画像ではなく一部画像を学習済みモデルにいれる際に使用
image_resolution = (512, 512)# 変更箇所 # 解像度 1024x1024は隠れ層3でもメモリ不足となる
# image_resolution = (64, 64) #全体画像ではなく一部画像を学習済みモデルにいれる際に使用

#データセットからデータを取得するために、データローダーを使用し、データローダーが __getitem__ メソッドを適切に呼び出してデータを取得・変換します。つまり、これにより、transformされる
# coord_dataset: <dataio.Implicit2DWrapper object at 0x7f89b4a62d70>
# opt: Namespace(config_filepath=None, logging_root='./logs', experiment_name='my_experiment', batch_size=1, lr=0.0001, num_epochs=10000, epochs_til_ckpt=25, steps_til_summary=1000, model_type='sine', checkpoint_path=None)
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)#0

"""
# Model definition # 追加箇所
model = modules.SingleBVPNet(
    type=opt.model_type, 
    mode='mlp', 
    sidelength=image_resolution, 
    num_hidden_layers=opt.num_hidden_layers
)
"""

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh' or opt.model_type == 'selu' or opt.model_type == 'elu'\
        or opt.model_type == 'softplus':
    # image_resolution: (512, 512)
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', sidelength=image_resolution)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, sidelength=image_resolution)
else:
    raise NotImplementedError
model.cuda()

# root_path: './logs/my_experiment'
# opt.logging_root: './logs'
# opt.experiment_name: 'my_experiment'
root_path = os.path.join(opt.logging_root, opt.experiment_name)
modules.SingleBVPNet.process_root_path(root_path)

# Define the loss
# loss_functions: <module 'loss_functions' from '/mnt/siren/loss_functions.py'>
# loss_functions.image_mse: <function image_mse at 0x7f55278903a0>
# loss_fn: functools.partial(<function image_mse at 0x7f55278903a0>, None)
loss_fn = partial(loss_functions.image_mse, None)
# utils: <module 'utils' from '/mnt/siren/utils.py'>
# utils.write_image_summary: <function write_image_summary at 0x7f553229f9a0>
# image_resolution: (512, 512)
# summary_fn:functools.partial(<function write_image_summary at 0x7f553229f9a0>, (512, 512))
summary_fn = partial(utils.write_image_summary, image_resolution)
#import pdb; pdb.set_trace()

# training: <module 'training' from '/mnt/siren/training.py'>
# training.train: <function train at 0x7f89f0c1c160>
# model:SingleBVPNet~
# dataloader:<torch.utils.data.dataloader.DataLoader object at 0x7f7920e683d0>
# opt.num_epochs:10000
# opt.lr:0.0001
# opt.steps_til_summary:1000
# opt.epochs_til_ckpt:25
# root_path:'./logs/experiment_06101419'
# loss_fn:functools.partial(<function image_mse at 0x7f795ccfcca0>, None)
# summary_fn:functools.partial(<function write_image_summary at 0x7f795d390310>, (512, 512))
training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)
