import sys
import os
import configargparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# コマンドライン引数の設定
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--hidden_layers', type=int, default=3, help='Number of hidden layers in the model. Default is 7.')
p.add_argument('--omega', type=float, default=30, help='Omega value for the sine activation function. Default is 30.')
p.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model.')
p.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
p.add_argument('--sidelen', type=int, default=1024, help='Side length of the output grid. Default is 1024.')

# コマンドライン引数の解析
opt = p.parse_args()

# Sine アクティベーション層
class Sine(torch.nn.Module):
    def __init__(self, omega):
        super().__init__()
        self.omega = omega

    def forward(self, input):
        return torch.sin(self.omega * input)

# Fully connected block
class FCBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, omega, outermost_linear=True):
        super().__init__()
        layers = [torch.nn.Sequential(torch.nn.Linear(in_features, hidden_features), Sine(omega))]
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Sequential(torch.nn.Linear(hidden_features, hidden_features), Sine(omega)))
        if outermost_linear:
            layers.append(torch.nn.Sequential(torch.nn.Linear(hidden_features, out_features)))
        else:
            layers.append(torch.nn.Sequential(torch.nn.Linear(hidden_features, out_features), Sine(omega)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Single BVP model
class SingleBVPNet(torch.nn.Module):
    def __init__(self, in_features=2, out_features=1, hidden_features=256, num_hidden_layers=7, omega=30):
        super().__init__()
        self.net = FCBlock(in_features, out_features, num_hidden_layers, hidden_features, omega, outermost_linear=True)
        # print(num_hidden_layers)

    def forward(self, coords):
        return self.net(coords)

# 座標のグリッドを生成
def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return mgrid.reshape(-1, dim)

# 輝度値統計情報の計算
def calculate_statistics(data):
    flat_data = data.flatten()
    min_value = np.min(flat_data)
    max_value = np.max(flat_data)
    avg_value = np.mean(flat_data)
    median_value = np.median(flat_data)
    return {'min': min_value, 'max': max_value, 'average': avg_value, 'median': median_value}

# 輝度値修正の処理
def process_tensor(data_tensor):
    rows, cols = data_tensor.shape
    processed_tensor = data_tensor.clone()
    min_value = torch.min(data_tensor).item()
    correction_value = min_value - 0.001
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (data_tensor[i, j] < data_tensor[i - 1, j] and
                data_tensor[i, j] < data_tensor[i + 1, j] and
                data_tensor[i, j] < data_tensor[i, j - 1] and
                data_tensor[i, j] < data_tensor[i, j + 1]):
                processed_tensor[i, j] = correction_value
    return processed_tensor, correction_value

# モデルのロード
model = SingleBVPNet(num_hidden_layers=opt.hidden_layers, omega=opt.omega)
model.load_state_dict(torch.load(opt.model_path, map_location=torch.device('cpu')), strict=False)
model.eval()

# 出力ディレクトリの作成
os.makedirs(opt.output_dir, exist_ok=True)

# モデルの出力を生成
model_input = get_mgrid(opt.sidelen).float()
model_output = model(model_input).detach().cpu().numpy().reshape(opt.sidelen, opt.sidelen)

# 統計情報の計算
stats = calculate_statistics(model_output)
print("Model Output")
print(f"Min: {stats['min']}")
print(f"Max: {stats['max']}")
print(f"Average: {stats['average']}")
print(f"Median: {stats['median']}")

# -1より小さいものは-1に、1より大きいものは1に設定（torch.clampを使用）
model_output_clamp = torch.clamp(torch.tensor(model_output), min=-1, max=1)
# -1~1を0~255に変換
model_output_clamp = ((model_output_clamp + 1) * 127.5).to(torch.uint8) # 整数でないと、model_output_processed_rgb.pngが出力されなかった。

# 出力をCPUに移してNumPy配列に変換
model_output_clamp_np = model_output_clamp.cpu().view(opt.sidelen, opt.sidelen).detach().numpy()

# （SIRENを通した後の）画像を保存（輝度値0~255を考慮）
output_image_path = os.path.join(opt.output_dir, f'{opt.sidelen}_model_output.png')
plt.imshow(model_output_clamp_np, cmap='gray')
plt.title(f'Model Output ({opt.sidelen}x{opt.sidelen})')
plt.colorbar()
plt.savefig(output_image_path)
plt.close()
print(f"Model output saved to {output_image_path}")


# NumPy配列をTIFF形式で保存する
output_tiff_path = os.path.join(opt.output_dir, f'{opt.sidelen}_model_output.tiff')
# PILを使用してTIFF画像を保存
image = Image.fromarray(model_output_clamp_np)
image.save(output_tiff_path, format="TIFF")
print(f"Model output saved to {output_tiff_path}")


# 修正処理
if opt.sidelen >= 3:
    processed_data, modified_value = process_tensor(torch.tensor(model_output).view(opt.sidelen, opt.sidelen))
    processed_data_np = processed_data.cpu().detach().numpy()
    modified_mask = (processed_data_np == modified_value)
    processed_data_clamp_np = np.clip(processed_data_np, -1, 1)
    processed_data_clamp_np = ((processed_data_clamp_np + 1) * 127.5).astype(np.uint8)

    # RGB画像
    rgb_image = np.stack([processed_data_clamp_np] * 3, axis=-1)
    rgb_image[modified_mask, 0] = 255
    rgb_image[modified_mask, 1] = 255
    rgb_image[modified_mask, 2] = 0

    # 保存
    image_file3 = os.path.join(opt.output_dir, f'{opt.sidelen}_model_output_processed_rgb.png')
    plt.imshow(rgb_image)
    plt.title('Model Output with Highlighted Modifications')
    plt.savefig(image_file3)
    plt.close()
    print(f"画像が保存されました: {image_file3}")

    image_file5 = os.path.join(opt.output_dir, f'{opt.sidelen}_model_output_processed.png')
    plt.imshow(processed_data_clamp_np, cmap='gray')
    plt.title(f'Model Output Brightness({opt.sidelen}x{opt.sidelen})')
    plt.colorbar()
    plt.savefig(image_file5)
    plt.close()
    print(f"モデルの出力を画像として '{image_file5}' に保存しました。")

    image_file5_tiff = os.path.join(opt.output_dir, f'{opt.sidelen}_model_output_processed.tiff')
    Image.fromarray(processed_data_clamp_np).save(image_file5_tiff, format="TIFF")
    print(f"モデルの出力をTIFF画像として '{image_file5_tiff}' に保存しました。")

    model_output_processed_txt = os.path.join(opt.output_dir, 'model_output_processed.txt')
    np.savetxt(model_output_processed_txt, processed_data_clamp_np, fmt='%d')
    print(f"モデルの出力をテキストとして '{model_output_processed_txt}' に保存しました。")

    stats2 = calculate_statistics(processed_data_clamp_np)
    print("Model Output（輝度値修正後）")
    print(f"Min: {stats2['min']}")
    print(f"Max: {stats2['max']}")
    print(f"Average: {stats2['average']}")
    print(f"Median: {stats2['median']}")
else:
    print("データが小さすぎて処理できません（行列サイズが2×2以下です）。")
