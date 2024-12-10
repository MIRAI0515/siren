'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
#!/usr/bin/env python3
"""
このコード（sdf_meshing.py: 点群⇒メッシュ）は、3Dシーンの表現を生成し、それを点群からメッシュに変換して、
最終的にPLYファイルとして保存するためのスクリプトです。
"""

import logging
import numpy as np
import plyfile
#from skimage import measure 要りそう？
#import skimage.measure　元々
#from skimage.measure import marching_cubes_lewiner
import time
import skimage
import torch

"""
この関数は、3Dシーンの表現を生成し、その表現を点群からメッシュに変換する役割を担います。

引数：
    ・decoder: シーンの表現を生成するデコーダーモデル。
    ・filename: 出力のPLYファイル名。
    ・N: サンプリンググリッドの解像度。（resolution）
    ・max_batch: サンプリングのバッチサイズの制限。
    ・offset: メッシュのオフセット（オプション）。
    ・scale: メッシュのスケール（オプション）。
    
関数内での主な処理:
    ・サンプリンググリッドを作成し、そのグリッド上の点をサンプリングしてSDFの値を計算します。
    ・SDFの値はメッシュ生成のために使用され、最終的にPLYファイルとして保存されます。
"""
"""
サンプリンググリッドの特徴は、格子点が一様に配置されていることです。
格子の解像度は、格子点の間隔を調整することで変更でき、解像度が高い場合は細かいデータをキャプチャできますが、
計算コストが高くなります。逆に、解像度が低い場合は計算コストが低減しますが、細かいデータの詳細が失われる可能性があります。
"""
def create_mesh(
    decoder, filename, N=256, max_batch=64 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
   

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    """ ~overall_index~
    tensor([         0,          1,          2,  ..., 4095999997, 4095999998,
        4095999999])
    """
    
    # PyTorchのテンソル（Tensor）を使用して、形状が (N ** 3, 4) のゼロで初期化された2次元のテンソル（行列）を生成する操作
    samples = torch.zeros(N ** 3, 4)
    """
    (N ** 3) 行、各行に4つのゼロを持つ2次元のテンソル（行列）を生成し、samples 変数に代入します。
    このようなテンソルは、通常、データの格納や処理に使用されます。
    この場合、samples は N ** 3 個のデータポイントを格納できるメモリを確保し、各データポイントは4つの値を持つ
    ことが期待されています。その後、このテンソルは値が計算されて埋められることになります。
    """
    """ ~samples~  (N ** 3 は N の3乗を計算した値で、通常は立方体のサイズや3Dグリッドのサイズを表します。)
    tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        ...,
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
    """

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    """
    tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 2.0000e+00, 0.0000e+00],
        ...,
        [0.0000e+00, 0.0000e+00, 1.5970e+03, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 1.5980e+03, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 1.5990e+03, 0.0000e+00]])
    """
    samples[:, 1] = (overall_index.long() / N) % N
    """
    tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 6.2500e-04, 1.0000e+00, 0.0000e+00],
        [0.0000e+00, 1.2500e-03, 2.0000e+00, 0.0000e+00],
        ...,
        [0.0000e+00, 0.0000e+00, 1.5970e+03, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 1.5980e+03, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 1.5990e+03, 0.0000e+00]])
    """
    samples[:, 0] = ((overall_index.long() / N) / N) % N
    """
    tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [3.9062e-07, 6.2500e-04, 1.0000e+00, 0.0000e+00],
        [7.8125e-07, 1.2500e-03, 2.0000e+00, 0.0000e+00],
        ...,
        [0.0000e+00, 0.0000e+00, 1.5970e+03, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 1.5980e+03, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 1.5990e+03, 0.0000e+00]])
    """

    # transform first 3 columns
    # to be the x, y, z coordinate
    # voxel_origin[2] はこのベクトルの3番目の要素を指し、通常は z 軸方向に対応します。この値は 3Dボクセルデータ内の z 座標軸における原点の位置を示します。
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    """ ~samples~
    tensor([[-1.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
        [-1.0000e+00,  6.2500e-04,  1.0000e+00,  0.0000e+00],
        [-1.0000e+00,  1.2500e-03,  2.0000e+00,  0.0000e+00],
        ...,
        [-1.0000e+00,  0.0000e+00,  1.5970e+03,  0.0000e+00],
        [-1.0000e+00,  0.0000e+00,  1.5980e+03,  0.0000e+00],
        [-1.0000e+00,  0.0000e+00,  1.5990e+03,  0.0000e+00]])
    """
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    """ ~samples~
    tensor([[-1.0000e+00, -1.0000e+00,  0.0000e+00,  0.0000e+00],
        [-1.0000e+00, -1.0000e+00,  1.0000e+00,  0.0000e+00],
        [-1.0000e+00, -1.0000e+00,  2.0000e+00,  0.0000e+00],
        ...,
        [-1.0000e+00, -1.0000e+00,  1.5970e+03,  0.0000e+00],
        [-1.0000e+00, -1.0000e+00,  1.5980e+03,  0.0000e+00],
        [-1.0000e+00, -1.0000e+00,  1.5990e+03,  0.0000e+00]])
    """
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    """ ~samples~
    tensor([[-1.0000, -1.0000, -1.0000,  0.0000],
        [-1.0000, -1.0000, -0.9987,  0.0000],
        [-1.0000, -1.0000, -0.9975,  0.0000],
        ...,
        [-1.0000, -1.0000,  0.9975,  0.0000],
        [-1.0000, -1.0000,  0.9987,  0.0000],
        [-1.0000, -1.0000,  1.0000,  0.0000]])
    """

    num_samples = N ** 3
    """ ~num_samples~
    4096000000   (1600の3乗)
    """
    
    samples.requires_grad = False

    head = 0

    while head < num_samples:
        print(head)
        
        # 0:3 とは、0番目から2(=3-1)番目までを取り出す。
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        # head = 0 の時
        # sample_subset = samples[0:min(0+262144, 4096000000), 0:3].cuda()
        # sample_subset = samples[0:262144, 0:3].cuda()
        """ ~sample_subset~  (head=0の時)
        tensor([[-1.0000, -1.0000, -1.0000],
        [-1.0000, -1.0000, -0.9987],
        [-1.0000, -1.0000, -0.9975],
        ...,
        [-0.9999, -0.7951,  0.6773],
        [-0.9999, -0.7951,  0.6785],
        [-0.9999, -0.7951,  0.6798]], device='cuda:0')
        """
        
        """ ~sample_subset~  (head=4095737856の時)
        tensor([[ 1.0011,  0.7964, -0.6798],
        [ 1.0011,  0.7964, -0.6785],
        [ 1.0011,  0.7964, -0.6773],
        ...,
        [-1.0000, -1.0000,  0.9975],
        [-1.0000, -1.0000,  0.9987],
        [-1.0000, -1.0000,  1.0000]], device='cuda:0')
        """

        samples[head : min(head + max_batch, num_samples), 3] = (
            # head = 0 の時
            # samples[0 : min(0 + 262144, 4096000000), 3]
            decoder(sample_subset)
            .squeeze()#.squeeze(1) # テンソルから次元が1の次元を削除する操作です。例えば、2Dテンソルを1Dテンソルに変換します。
            .detach()# テンソルを計算グラフから切り離し、勾配情報を持たないテンソルに変換します。
            .cpu()
        )
        """ ~decoder~
        SDFDecoder(
        (model): SingleBVPNet(
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
        )
        """
        head += max_batch
        
    
    """ ~samples~  torch.Size([4096000000, 4])
    tensor([[-1.0000, -1.0000, -1.0000,  0.0227],
        [-1.0000, -1.0000, -0.9987,  0.0231],
        [-1.0000, -1.0000, -0.9975,  0.0236],
        ...,
        [-1.0000, -1.0000,  0.9975,  0.0252],
        [-1.0000, -1.0000,  0.9987,  0.0254],
        [-1.0000, -1.0000,  1.0000,  0.0256]])
    """
    
    sdf_values = samples[:, 3]
    """ ~sdf_values~  torch.Size([4096000000])
    tensor([0.0227, 0.0231, 0.0236,  ..., 0.0252, 0.0254, 0.0256])
    """
    
    sdf_values = sdf_values.reshape(N, N, N)
    """ ~sdf_values~ (N=1600)  1600*1600*1600 = 4096000000
    tensor([[[0.0227, 0.0231, 0.0236,  ..., 0.0253, 0.0255, 0.0257],
         [0.0230, 0.0234, 0.0239,  ..., 0.0254, 0.0256, 0.0258],
         [0.0233, 0.0237, 0.0242,  ..., 0.0255, 0.0257, 0.0260],
         ...,
         [0.0273, 0.0280, 0.0286,  ..., 0.0194, 0.0197, 0.0200],
         [0.0269, 0.0275, 0.0282,  ..., 0.0189, 0.0191, 0.0194],
         [0.0264, 0.0270, 0.0277,  ..., 0.0183, 0.0186, 0.0189]],

        [[0.0225, 0.0230, 0.0234,  ..., 0.0254, 0.0255, 0.0257],
         [0.0228, 0.0233, 0.0237,  ..., 0.0255, 0.0257, 0.0258],
         [0.0231, 0.0236, 0.0240,  ..., 0.0256, 0.0258, 0.0260],
         ...,
         [0.0276, 0.0283, 0.0289,  ..., 0.0199, 0.0201, 0.0204],
         [0.0271, 0.0278, 0.0284,  ..., 0.0193, 0.0196, 0.0199],
         [0.0267, 0.0273, 0.0279,  ..., 0.0187, 0.0190, 0.0193]],

        [[0.0224, 0.0228, 0.0233,  ..., 0.0254, 0.0256, 0.0257],
         [0.0226, 0.0231, 0.0236,  ..., 0.0255, 0.0257, 0.0259],
         [0.0229, 0.0234, 0.0239,  ..., 0.0257, 0.0258, 0.0260],
         ...,
         [0.0279, 0.0286, 0.0292,  ..., 0.0203, 0.0206, 0.0209],
         [0.0274, 0.0281, 0.0287,  ..., 0.0197, 0.0200, 0.0203],
         [0.0269, 0.0276, 0.0282,  ..., 0.0191, 0.0194, 0.0197]],

        ...,

        [[0.0111, 0.0120, 0.0129,  ..., 0.0308, 0.0307, 0.0307],
         [0.0110, 0.0119, 0.0128,  ..., 0.0312, 0.0312, 0.0312],
         [0.0109, 0.0118, 0.0128,  ..., 0.0317, 0.0317, 0.0316],
         ...,
         [0.0168, 0.0172, 0.0176,  ..., 0.1002, 0.0997, 0.0993],
         [0.0164, 0.0168, 0.0172,  ..., 0.1009, 0.1004, 0.1000],
         [0.0160, 0.0164, 0.0168,  ..., 0.0296, 0.0295, 0.0295]],

        [[0.0107, 0.0116, 0.0125,  ..., 0.0300, 0.0300, 0.0300],
         [0.0106, 0.0115, 0.0124,  ..., 0.0305, 0.0304, 0.0304],
         [0.0105, 0.0114, 0.0123,  ..., 0.0310, 0.0309, 0.0308],
         ...,
         [0.0162, 0.0166, 0.0169,  ..., 0.0995, 0.0990, 0.0986],
         [0.0158, 0.0162, 0.0166,  ..., 0.1002, 0.0997, 0.0993],
         [0.0154, 0.0158, 0.0162,  ..., 0.0288, 0.0288, 0.0288]],

        [[0.0103, 0.0111, 0.0120,  ..., 0.0293, 0.0292, 0.0292],
         [0.0102, 0.0110, 0.0119,  ..., 0.0297, 0.0297, 0.0297],
         [0.0101, 0.0109, 0.0118,  ..., 0.0302, 0.0301, 0.0301],
         ...,
         [0.0156, 0.0160, 0.0163,  ..., 0.0988, 0.0983, 0.0979],
         [0.0152, 0.0156, 0.0160,  ..., 0.0995, 0.0991, 0.0986],
         [0.0149, 0.0152, 0.0156,  ..., 0.0252, 0.0254, 0.0256]]])
    """
    end = time.time()
    print("sampling takes: %f" % (end - start))
    
    # SDFサンプルをPLYファイルに変換する
    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(), # torch.Size([1600, 1600, 1600])
        voxel_origin, # [-1, -1, -1]
        voxel_size, # 0.0012507817385866166
        ply_filename + ".ply", # './logs/experiment_1_rec1/test.ply'
        offset,
        scale,
    )

"""
この関数は、SDFのサンプルを3Dメッシュに変換し、それをPLYファイルとして保存する役割を担います。

引数:
・pytorch_3d_sdf_tensor: PyTorchテンソル形式のSDFサンプル。
・voxel_grid_origin: ボクセルグリッドの原点座標。
・voxel_size: ボクセルのサイズ。
・ply_filename_out: 出力のPLYファイル名。
・offset: メッシュのオフセット（オプション）。
・scale: メッシュのスケール（オプション）。

関数内での主な処理:
・PyTorchテンソルからNumPyテンソルに変換します。
・Marching Cubesアルゴリズムを使用して、SDFの等値面を抽出し、メッシュを生成します。
・メッシュをワールド座標に変換し、必要に応じてオフセットとスケールを適用します。
・メッシュ情報をPLYフォーマットに変換し、指定されたPLYファイルに保存します。
"""

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    """ ~numpy_3d_sdf_tensor~ (1600, 1600, 1600)
    array([[[0.02272509, 0.02314886, 0.02358348, ..., 0.0253127 ,
         0.02551375, 0.02572256],
        [0.02300811, 0.02343732, 0.02387654, ..., 0.02541402,
         0.02561804, 0.02582985],
        [0.02330326, 0.02373724, 0.02418061, ..., 0.02553471,
         0.02574085, 0.02595458],
        ...,
        [0.02733593, 0.0279814 , 0.02863313, ..., 0.01943092,
         0.01971232, 0.02000971],
        [0.0268771 , 0.02751195, 0.02815298, ..., 0.01886351,
         0.01914769, 0.01944813],
        [0.02641355, 0.02703854, 0.02766971, ..., 0.01831673,
         0.01860185, 0.01890365]],

       [[0.02253091, 0.02296745, 0.02341468, ..., 0.02535885,
         0.02553774, 0.02572487],
        [0.02281727, 0.0232593 , 0.02371127, ..., 0.02547065,
         0.025653  , 0.02584332],
        [0.02311693, 0.02356385, 0.02402002, ..., 0.0256016 ,
         0.02578637, 0.02597905],
        ...,
        [0.02760384, 0.02825835, 0.02891929, ..., 0.01985172,
         0.02013755, 0.02043823],
        [0.02713898, 0.02778249, 0.02843246, ..., 0.01927349,
         0.01956293, 0.01986737],
        [0.02666998, 0.02730319, 0.02794311, ..., 0.01871425,
         0.01900566, 0.01931247]],

       [[0.02235601, 0.02280425, 0.02326323, ..., 0.02540058,
         0.02555693, 0.02572174],
        [0.02264462, 0.02309872, 0.02356251, ..., 0.02552315,
         0.02568316, 0.02585145],
        [0.02294747, 0.02340653, 0.0238748 , ..., 0.0256646 ,
         0.02582737, 0.02599807],
        ...,
        [0.02788973, 0.02855304, 0.02922255, ..., 0.0202789 ,
         0.02056833, 0.02087091],
        [0.02741811, 0.02807004, 0.02872827, ..., 0.01969138,
         0.01998528, 0.02029249],
        [0.02694304, 0.02758441, 0.02823215, ..., 0.01912132,
         0.01941811, 0.0197283 ]],

       ...,

       [[0.01114341, 0.01203905, 0.01294944, ..., 0.03077675,
         0.03074495, 0.03073619],
        [0.01103615, 0.01193238, 0.01284322, ..., 0.03124546,
         0.03119864, 0.03117464],
        [0.01094317, 0.01184012, 0.01275188, ..., 0.03172046,
         0.03165836, 0.03161906],
        ...,
        [0.0167761 , 0.01715871, 0.01755705, ..., 0.10018122,
         0.099732  , 0.09928454],
        [0.01638273, 0.01676264, 0.01715731, ..., 0.10086682,
         0.10042517, 0.09998489],
        [0.01600125, 0.01637775, 0.01676841, ..., 0.02956175,
         0.0295392 , 0.02953944]],

       [[0.01069599, 0.01157834, 0.01247527, ..., 0.03002117,
         0.02998381, 0.02996919],
        [0.01058615, 0.01146891, 0.01236614, ..., 0.03048855,
         0.03043626, 0.03040652],
        [0.01049053, 0.01137397, 0.01227199, ..., 0.03096127,
         0.03089383, 0.03084894],
        ...,
        [0.01619038, 0.01656042, 0.01694701, ..., 0.09947672,
         0.09902723, 0.09857937],
        [0.01580524, 0.01617315, 0.01655679, ..., 0.10017819,
         0.09973627, 0.09929565],
        [0.01543255, 0.01579778, 0.01617796, ..., 0.02882368,
         0.02879567, 0.02879027]],

       [[0.01026974, 0.01113778, 0.01202053, ..., 0.02928059,
         0.02923793, 0.0292178 ],
        [0.01015805, 0.01102633, 0.01190928, ..., 0.02974465,
         0.02968724, 0.02965209],
        [0.01006006, 0.01092911, 0.01181284, ..., 0.03021314,
         0.03014079, 0.03009064],
        ...,
        [0.0156105 , 0.01596757, 0.01634202, ..., 0.09877931,
         0.09832996, 0.09788209],
        [0.01523351, 0.01558908, 0.01596121, ..., 0.09949698,
         0.09905493, 0.09861407],
        [0.01486911, 0.01522283, 0.01559228, ..., 0.02523202,
         0.02542896, 0.02563393]]], dtype=float32)
    """

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    # verts, faces, normals: array([], shape=(0, 3), dtype=float64)
    # values: array([], dtype=float64)
    try:
        # verts, faces, normals: array([], shape=(0, 3), dtype=float64)
        # values: array([], dtype=float64)
        # 3D形状から等値面（指定した閾値と一致する値の表面）を抽出する
        # numpy_3d_sdf_tensor: 3D形状
        """
        numpy_3d_sdf_tensor内の特定の等値面を抽出します。この等値面は、level=0.0として指定された閾値によって定義されています。
        具体的には、SDFテンソル内の値が0.0に等しい場所で形成される等値面が抽出されます。
        
        等値面が抽出され、それを構成するための頂点（verts）、三角形ポリゴンの面（faces）、法線ベクトル（normals）、および値（values）が取得されます。
        これらのデータは、等値面を視覚化するために使用できます。
        """
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
        print("aaaa")
    except Exception as e:
        print("エラーが発生しました:", str(e))
        pass
    """
        try:
        # verts, faces, normals: array([], shape=(0, 3), dtype=float64)
        # values: array([], dtype=float64)
        # 3D形状から等値面（指定した閾値と一致する値の表面）を抽出する
        # numpy_3d_sdf_tensor: 3D形状
        
        numpy_3d_sdf_tensor内の特定の等値面を抽出します。この等値面は、level=0.0として指定された閾値によって定義されています。
        具体的には、SDFテンソル内の値が0.0に等しい場所で形成される等値面が抽出されます。
        
        等値面が抽出され、それを構成するための頂点（verts）、三角形ポリゴンの面（faces）、法線ベクトル（normals）、および値（values）が取得されます。
        これらのデータは、等値面を視覚化するために使用できます。
        
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
        print("aaaa")
    except Exception as e:
        print("エラーが発生しました:", str(e))
        pass
    """
        

    # transform from voxel coordinates to camera coordinates（ボクセル座標からカメラ座標への変換）
    # note x and y are flipped in the output of marching_cubes（Marching_cubes の出力では x と y が反転していることに注意。）
    
    mesh_points = np.zeros_like(verts)
    # vertsは、(n, 3)のような形状をもつ。ここでn は、頂点の数を表す。
    """
    np.zeros_like(verts)は、vertsと同じ形状とデータ型を持つ新しい配列を生成します。
    すなわち、mesh_points の形状とデータ型は verts と同じであるため、mesh_points は同じ数の頂点を持つことになります。
    """
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0] # num_verts: 0
    num_faces = faces.shape[0] # num_faces: 0

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    # array([], dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    # array([], dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])
    # array([], dtype=[('vertex_indices', '<i4', (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    # PlyElement('vertex', (PlyProperty('x', 'float'), PlyProperty('y', 'float'), PlyProperty('z', 'float')), count=0, comments=[])
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")
    # PlyElement('face', (PlyListProperty('vertex_indices', 'uchar', 'int'),), count=0, comments=[])

    ply_data = plyfile.PlyData([el_verts, el_faces])
    """ ~ply_data~
    PlyData((PlyElement('vertex', (PlyProperty('x', 'float'), PlyProperty('y', 'float'), PlyProperty('z', 'float')), count=0, comments=[]), PlyElement('face', (PlyListProperty('vertex_indices', 'uchar', 'int'),), count=0, comments=[])), text=False, byte_order='<', comments=[], obj_info=[])
    """
    logging.debug("saving mesh to %s" % (ply_filename_out)) # ply_filename_out: './logs/experiment_1_rec1/test.ply'
    
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
