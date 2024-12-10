from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import cv2
import cmapy
import skimage
import pathlib
from tensorboard.backend.event_processing import event_accumulator
from moviepy.editor import VideoFileClip, clips_array, vfx, ImageSequenceClip, CompositeVideoClip, concatenate_videoclips, VideoClip, TextClip
from matplotlib import animation

"""
このスクリプトは主に機械学習モデルのトレーニングの進捗や結果の可視化に使用され、TensorBoardのデータを処理し、
視覚的な情報を生成します。データの抽出、可視化、比較に役立つユーティリティ関数が含まれています。
また、アニメーションや静的なプロットの作成もサポートされています。
"""

"""
TensorBoardのサマリーデータから画像を抽出し、指定のディレクトリに保存します。カラーマップを適用することもできます。
"""
def extract_images_from_summary(events_path, tag_names_to_look_for, suffix='', img_outdir=None, colormap=None):
    count=0
    print("Extracting data from tensorboard summary...")
    # テンソルボードのイベントファイルを読み込む
    #import pdb; pdb.set_trace()
    
    #指定されたパスにあるイベントファイルを読み込み、画像データについてはすべてのイベントを読み込むことを指定しています。
    event_acc = event_accumulator.EventAccumulator(events_path, size_guidance={'images': 0})
    event_acc.Reload()

    # 出力ディレクトリに保存する場合に名前に付加するサフィックス
    strsuffix = suffix

    # すべての画像タグを見る
    if img_outdir is not None:
        outdir = pathlib.Path(img_outdir)# 出力ディレクトリのパスを作成
        outdir.mkdir(exist_ok=True, parents=True)# ディレクトリが存在しない場合は作成

    # We are looking at all the images ...
    image_dict = defaultdict(list)
    for tag in event_acc.Tags()['images']:# すべての画像タグを反復
        # tag: 'train_gt_vs_pred'
        # tag: 'train_pred_img'
        print("processing tag %s"%tag)
        # events: ~~~~~x00\x00IEND\xaeB`\x82', width=1024, height=512
        events = event_acc.Images(tag)
        # tag_name: 'train_gt_vs_pred'
        tag_name = tag.replace('/', '_')
        # タグ名が指定されたタグリストにあるか確認
        if tag_name in tag_names_to_look_for:
            tag_name = tag_name + strsuffix

            if img_outdir is not None:
                dirpath = outdir / tag_name# 各タグのディレクトリパスを作成
                dirpath.mkdir(exist_ok=True, parents=True)# ディレクトリが存在しない場合は作成

            for index, event in enumerate(events):# すべてのイベントを反復
                print(f"index:{index}")
                #print(f"event:{event}")
                
                s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)# 画像をデコード
                image = cv2.imdecode(s, cv2.IMREAD_COLOR)# 画像に変換

                if colormap is not None:
                    image = cv2.applyColorMap(image[..., 0], cmapy.cmap(colormap))# カラーマップが指定されている場合は適用

                if img_outdir is not None:
                    outpath = dirpath / '{:04}.png'.format(index)# 画像の出力パスを作成
                    count=count+1
                    cv2.imwrite(outpath.as_posix(), image)# 画像を保存

                image_dict[tag].append(image)# 画像を辞書に追加
    print(count)
    return image_dict# 画像の辞書を返す


"""
TensorBoardのサマリーデータから特定のタグに関連する値と時間情報を抽出します。
"""
def extract_from_summary(path, value_tag):
    if os.path.isdir(path):# パスがディレクトリか確認
        path = glob.glob(os.path.join(path, "*"))[0]# ディレクトリ内の最初のファイルを取得

    origin_wall_time = None
    wall_times = []
    values = []

    for event in tf.compat.v1.train.summary_iterator(path):# サマリーイベントを反復
        if not origin_wall_time:
            origin_wall_time = event.wall_time# 元のウォールタイムを設定
        for value in event.summary.value:# イベントの値を反復
            if value.tag == value_tag:
                wall_times.append(event.wall_time - origin_wall_time)# ウォールタイムを追加
                values.append(value.simple_value)# 値を追加
                
    # Debugging: Print the extracted values(yasuda)
    print(f"Extracted values for {value_tag} from {path}: {values}")
    
    return wall_times, values# ウォールタイムと値を返す


"""
動画クリップを指定されたファイルパスに保存します。
"""
def save_video(video_clip, filepath):
    video_clip.resize(width=1080)

    height, width = video_clip.h, video_clip.w

    if height % 2:
        height += 1

    video_clip.resize(width=1080, height=height).write_videofile(filepath, fps=25,
                                                                 audio_codec='libfdk_aac', audio=False)

"""
複数のビデオクリップをグリッドに配置して新しいビデオを生成します。
"""
def make_video_grid_from_filepaths(num_rows, num_cols, video_list, trgt_name,
                                   margin_color=(255,255,255), margin_width=0,
                                   column_wise=True):

    clip_array = [[] for _ in range(num_rows)]
    #import pdb; pdb.set_trace() 
    # num_cols: 6
    for col in range(num_cols):
        # num_rows: 3
        for row in range(num_rows):
            if column_wise:
                idx = col * num_rows + row
            else:
                idx = row * num_cols + col

            video_clip = VideoFileClip(video_list[idx]).margin(margin_width, color=margin_color)
            if margin_width > 0:
                video_clip = video_clip.margin(margin_width, color=margin_color)

            clip_array[row].append(video_clip)

    final_clip = clips_array(clip_array)
    save_video(final_clip, trgt_name)


"""
グラフデータを使用してアニメーション付きの折れ線グラフを作成します。
"""
def animated_line_plot(x_axis, data, trgt_path, legend_loc='lower right', plot_type=None):
    fig, ax = plt.subplots()
    fontdict = {'size': 16}
    ax.tick_params(axis='both', which='major', direction='in', labelsize=11)
    ax.set_ylabel("PSNR", fontdict=fontdict)
    ax.set_xlabel("Iterations", fontdict=fontdict)
    if plot_type == 'image':
        ax.set_xticks([5000, 10000, 15000])
        ax.set_xticklabels(['5,000', '10,000', '15,000'])
        ax.set_xlim(0, 15000)
        ax.set_yticks([10, 20, 30, 40, 50, 60])
        ax.set_ylim(0, 60)
    elif plot_type == 'poisson':
        ax.set_xticks([1000, 2000, 3000, 4000])
        ax.set_xticklabels(['1,000', '2,000', '3,000', '4,000'])
        ax.set_xlim(0, 4000)
        ax.set_yticks([5, 10, 15, 20, 25, 30, 35])
        ax.set_ylim(0, 35)
    ax.grid()

    lines = []
    #import pdb; pdb.set_trace()
    # data.items(): dict_items([('ReLU', []), ('Tanh', []), ('ReLU P.E.', []), ('RBF-ReLU', []), ('SIREN', [])])
    for key, y in data.items():
        # x_axis: array([    0,    10,    20, ..., 14980, 14990, 15000])
        # y: []
        # key: ReLU
        lobj = ax.plot(x_axis, y, label=key)[0]
        lines.append(lobj)

    ax.legend(loc=legend_loc, bbox_to_anchor=(0.95, 0.035))

    def update(num, x, data, lines):
        for idx, (_, y) in enumerate(data.items()):
            lines[idx].set_data(x[:num], y[:num])
        return lines

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=26, bitrate=1800)
    anim = animation.FuncAnimation(fig, update, fargs=[x_axis, data, lines],
                                   frames=len(x_axis), interval=1, blit=True)
    anim.save(trgt_path, writer=writer)

"""
TensorBoardサマリーデータから画像を抽出し、異なるモデルの予測を比較するためのビデオを作成します。
"""
def make_video_from_tensorboard_summaries(summary_paths, trgt_path, image_extraction_dir, pred_tag_list,
                                          num_rows, num_cols, gt_tag_list=None, overwrite=False, colormap=None):
    #summary_paths:
    # {'ReLU': '/mnt/siren/logs/experiment_06171152_3dvoxel/relu/summaries', 'Tanh': '/mnt/siren/logs/experiment_06171152_3dvoxel/tanh/summaries', 'ReLU P.E.': '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries', 'RBF-ReLU': '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries', 'SIREN': '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries'}
    #import pdb; pdb.set_trace()
    # summary_paths.keys(): dict_keys(['ReLU', 'Tanh', 'ReLU P.E.', 'RBF-ReLU', 'SIREN'])
    for key in summary_paths.keys():
        if os.path.isdir(summary_paths[key]):
            summary_paths[key] = glob.glob(os.path.join(summary_paths[key], "*"))[0]

    video_filepaths = []

    def extract_images_make_videoclips(summary_path, root_dir, tag_list):
        if not os.path.exists(root_dir) or overwrite:
            #import pdb; pdb.set_trace()
            # root_dir: 'experiment_06171152_3dvoxel_all_06172341/gt'
            extract_images_from_summary(summary_path,
                                        tag_names_to_look_for=tag_list,
                                        img_outdir=root_dir,
                                        colormap=colormap)

        # tag_list: ['train_gt_img', 'train_gt_grad', 'train_gt_lapl']
        for tag in tag_list:
            dir = os.path.join(root_dir, tag)
            # dir: 'experiment_06171152_3dvoxel_all_06172341/gt/train_gt_img'
            video_path = os.path.join(dir, 'video.mp4')
            if not os.path.exists(video_path):
                print("Making video for %s" % dir)
                # ImageSequenceClip: 一連の画像を用いてビデオクリップを作成する
                img_clip = ImageSequenceClip(dir, 26) #元々26 #fps=26が指定されているため、1秒あたり26フレームの速度で画像が再生されるように設定されます。
                save_video(img_clip, video_path)
            video_filepaths.append(video_path)

    # Extract ground truth
    # image_extraction_dir: 'experiment_06171152_3dvoxel_all'
    gt_dir = os.path.join(image_extraction_dir, 'gt')
    summary_path = summary_paths[next(iter(summary_paths))]
    extract_images_make_videoclips(summary_path, gt_dir, gt_tag_list)

    # Extract all model predictions
    # summary_paths.items(): dict_items([('ReLU', '/mnt/siren/logs/experiment_06171152_3dvoxel/relu/summaries/events.out.tfevents.1718599641.e483456e9b62.1744200.0'), ('Tanh', '/mnt/siren/logs/experiment_06171152_3dvoxel/tanh/summaries/events.out.tfevents.1718604291.e483456e9b62.1770454.0'), ('ReLU P.E.', '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries/events.out.tfevents.1718592826.e483456e9b62.1710066.0'), ('RBF-ReLU', '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries/events.out.tfevents.1718592826.e483456e9b62.1710066.0'), ('SIREN', '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries/events.out.tfevents.1718592826.e483456e9b62.1710066.0')])
    for key, summary_path in summary_paths.items():
        # subdir: 'experiment_06171152_3dvoxel_all_06172341/ReLU'
        subdir = os.path.join(image_extraction_dir, key)
        summary_path = summary_path
        extract_images_make_videoclips(summary_path, subdir, pred_tag_list)

    # Now make joint video...
    if os.path.exists(trgt_path):
        val = input("The video %s exists. Overwrite? (y/n)" % trgt_path)
        if val == 'y':
            os.remove(trgt_path)
    
    # image_convergence.mp4の作成
    make_video_grid_from_filepaths(num_rows, num_cols, video_list=video_filepaths,
                                   trgt_name=trgt_path, margin_width=0)


"""
指定されたディレクトリ内の画像ファイルのリストを取得します。
"""
def glob_all_imgs(trgt_dir):
    '''Returns list of all images in trgt_dir
    '''
    all_imgs = []
    for ending in ['*.png', '*.tiff', '*.tif', '*.jpeg', '*.JPEG', '*.jpg', '*.bmp']:
        all_imgs.extend(glob.glob(os.path.join(trgt_dir, ending)))

    return all_imgs

"""
PSNR（ピーク信号対雑音比）の収束プロットを生成し、アニメーションまたは静的な画像として保存します。
"""
def make_convergence_plot(gt_dir, img_dirs, trgt_path, animate=False, iters_info=None, plot_type=None):
    '''
    Args:
        img_dirs: dictionary with method name as key and path to the directory with the respective images as item
    '''
    # 変数の初期化(yasuda)
    gt_images = []
    pred_images = []
    
    if gt_dir is not None:
        gt_images = sorted(glob_all_imgs(gt_dir))

    psnrs = defaultdict(list)
    #import pdb; pdb.set_trace()    
    # img_dirs.items(): dict_items([('ReLU', '/mnt/siren/logs/experiment_06171152_3dvoxel/relu/summaries'), ('Tanh', '/mnt/siren/logs/experiment_06171152_3dvoxel/tanh/summaries'), ('ReLU P.E.', '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries'), ('RBF-ReLU', '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries'), ('SIREN', '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries')])
    for key, path in tqdm(img_dirs.items()):
        psnrs_path = os.path.join(path, 'psnrs.npy')
        if os.path.exists(psnrs_path):
            #import pdb; pdb.set_trace() 
            # np.load(psnrs_path).tolist(): []
            # psnrs_path: /mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries/psnrs.npy'
            psnrs[key] = np.load(psnrs_path).tolist()
            continue

        pred_images = sorted(glob_all_imgs(path))
        for gt_path, pred_path in tqdm(zip(gt_images, pred_images)):
            #import pdb; pdb.set_trace() #ここには入っていない。
            gt_img = imageio.imread(gt_path)
            pred_img = imageio.imread(pred_path)

            psnr = skimage.measure.compare_psnr(gt_img, pred_img)
            psnrs[key].append(psnr)

        np.save(psnrs_path, np.array(psnrs[key]))

    # Debugging: Print psnrs dictionary(yasuda)
    print("PSNRs Dictionary:", psnrs)
    
    # Now make a line plot
    if gt_dir is not None:
        iterations = np.arange(len(gt_images))
    else:
        iterations = np.arange(0, iters_info['num_iters'], iters_info['step'])
    
    # Debugging: Print iterations length(yasuda)
    print("Iterations Length:", len(iterations))

    if animate:
        assert 'mp4' in trgt_path, "Filepath needs to be mp4"
        animated_line_plot(iterations, psnrs, trgt_path, plot_type=plot_type)
    else:
        fig, ax = plt.subplots()
        ax.tick_params(axis='both', which='major', direction='in', labelsize=8)
        ax.set_ylabel("PSNR")
        ax.set_xlabel("Iterations")
        ax.grid()

        ax.plot(iterations, psnrs[next(iter(psnrs))])#yasudaがコメントアウトした(240617)     
        fig.savefig(trgt_path, bbox='tight', bbox_inches='tight', pad_inches=0.)


"""
 TensorBoardサマリーデータから画像を抽出し、モデルの予測の収束を比較するビデオを作成するための関数です。
 """
def image_convergence_video():
    # この辞書は、各モデルのTensorBoardサマリーデータが保存されているディレクトリへのパスを指定します。
    summary_paths = {"ReLU": "/mnt/siren/logs/experiment_line3_black/summaries",
                    "Tanh": "/mnt/siren/logs/experiment_line3_black/summaries",
                    "ReLU P.E.": "/mnt/siren/logs/experiment_line3_black/summaries",
                    "RBF-ReLU": "/mnt/siren/logs/experiment_line3_black/summaries",
                    "SIREN": "/mnt/siren/logs/experiment_line3_black/summaries"}

    
    """
    summary_paths = {"ReLU": "/mnt/siren/logs/experiment_06131017_camera/relu/summaries",
                     "Tanh": "/mnt/siren/logs/experiment_06131017_camera/tanh/summaries",
                     "ReLU P.E.": "/mnt/siren/logs/experiment_06131017_camera/sine/summaries",
                     "RBF-ReLU": "/mnt/siren/logs/experiment_06131017_camera/sine/summaries",
                     "SIREN": "/mnt/siren/logs/experiment_06131017_camera/sine/summaries"}
    """

    # 画像を抽出して保存するディレクトリを指定します。
    image_extraction_dir = '/mnt/siren/logs/experiment_line3_black/summaries'#/mnt/siren/logs/experiment_06131017_camera_all_06180121
    os.makedirs(image_extraction_dir, exist_ok=True)

    # サマリーデータ内の画像のタグリストと予測値のタグリストを指定します。
    gt_tag_list = ['train_gt_img', 'train_gt_grad', 'train_gt_lapl']
    pred_tag_list = ['train_pred_img', 'train_pred_grad', 'train_pred_lapl']
    # ビデオのパスとフレームのレイアウトを指定して、ビデオを作成します。
    trgt_path = '/mnt/siren/logs/experiment_experiment_line3_black/summaries/image_convergence.mp4'
    #import pdb; pdb.set_trace() 
    # summary_paths: {'ReLU': '/mnt/siren/logs/experiment_06171152_3dvoxel/relu/summaries', 'Tanh': '/mnt/siren/logs/experiment_06171152_3dvoxel/tanh/summaries', 'ReLU P.E.': '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries', 'RBF-ReLU': '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries', 'SIREN': '/mnt/siren/logs/experiment_06171152_3dvoxel/sine/summaries'}
    make_video_from_tensorboard_summaries(summary_paths, trgt_path, image_extraction_dir=image_extraction_dir,
                                          pred_tag_list=pred_tag_list, num_rows=3, num_cols=len(summary_paths)+1,
                                          gt_tag_list=gt_tag_list, overwrite=True)

"""
 PSNRの収束プロットを生成し、アニメーションまたは静的な画像として保存するための関数です。
 """
def image_convergence_plot():
    # アニメーションを作成するかどうかを指定します。
    animated = True

    # この辞書は、各モデルのTensorBoardサマリーデータが保存されているディレクトリへのパスを指定します。
    summary_paths = {"ReLU": "/mnt/siren/logs/experiment_experiment_line3_black/summaries",
                     "Tanh": "/mnt/siren/logs/experiment_experiment_line3_black/summaries",
                     "ReLU P.E.": "/mnt/siren/logs/experiment_experiment_line3_black/summaries",
                     "RBF-ReLU": "/mnt/siren/logs/experiment_experiment_line3_black/summaries",
                     "SIREN": "/mnt/siren/logs/experiment_experiment_line3_black/summaries"}
    """
    summary_paths = {"ReLU": "/mnt/siren/logs/experiment_06131017_camera/relu/summaries",
                     "Tanh": "/mnt/siren/logs/experiment_06131017_camera/tanh/summaries",
                     "ReLU P.E.": "/mnt/siren/logs/experiment_06131017_camera/sine/summaries",
                     "RBF-ReLU": "/mnt/siren/logs/experiment_06131017_camera/sine/summaries",
                     "SIREN": "/mnt/siren/logs/experiment_06131017_camera/sine/summaries"}
    """

    # PSNRの収束プロットを保存するファイル名を指定します。
    #import pdb; pdb.set_trace()
    #filename = '/mnt/siren/logs/experiment_06171126_all_psnr/image_psnr_convergence' + '.mp4' if animated else '.pdf'
    filename = '/mnt/siren/logs/experiment_line3_black/summaries/image_psnr_convergence' + '.mp4' if animated else '.pdf'

    # PSNRの収束プロットを生成し、アニメーションまたは静的な画像として保存します。
    make_convergence_plot(None, img_dirs=summary_paths, animate=animated,
                          trgt_path=filename, iters_info={'num_iters':15001, 'step':5}, plot_type='image')


"""
サマリーデータからPSNRを抽出し、保存します。
"""
def extract_image_psnrs(summary_paths):
    # 各モデルのサマリーデータからPSNRを抽出して保存します。
    #import pdb; pdb.set_trace()#ここに入っていない。
    for key, item in summary_paths.items():
        # サマリーデータファイルのパスを取得します。
        summary_file = os.listdir(item)[0]
        summary_file = os.path.join(item, summary_file)

        # サマリーデータからPSNRを抽出します。
        wall_times, values = extract_from_summary(summary_file, 'train_img_psnr')
        psnrs = [values for _, values in sorted(zip(wall_times, values))]
        #import pdb; pdb.set_trace()

        # PSNRをnumpy形式で保存します。
        np.save(os.path.join(item, 'psnrs.npy'), psnrs)

#import pdb; pdb.set_trace()
if __name__ == '__main__':
    image_convergence_video()
    image_convergence_plot()

