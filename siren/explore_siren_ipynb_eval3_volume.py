import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
import re

def get_mgrid(sidelen, dim=3):
    tensors = [torch.linspace(-1, 1, steps=s) for s in sidelen]
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1
        return activations

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# 輝度値修正の関数（テンソル版）
def process_tensor(data_tensor):
    depth, rows, cols = data_tensor.shape
    processed_tensor = data_tensor.clone()

    # テンソルの最小値を取得
    min_value = torch.min(data_tensor).item()
    correction_value = min_value - 0.001  # 修正値を最小値から-0.001とする

    # マトリックス内の全要素に対して、周囲と比較（境界は無視）
    for z in range(1, depth - 1):
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # 周囲の要素よりも小さい場合、その値を条件に応じて変更
                if (data_tensor[z, i, j] < data_tensor[z - 1, i, j] and  # 前
                    data_tensor[z, i, j] < data_tensor[z + 1, i, j] and  # 後
                    data_tensor[z, i, j] < data_tensor[z, i - 1, j] and  # 上
                    data_tensor[z, i, j] < data_tensor[z, i + 1, j] and  # 下
                    data_tensor[z, i, j] < data_tensor[z, i, j - 1] and  # 左
                    data_tensor[z, i, j] < data_tensor[z, i, j + 1]):    # 右
                    processed_tensor[z, i, j] = correction_value  # 修正された値を設定
                
    return processed_tensor, correction_value


# 輝度値統計情報の計算関数
def calculate_statistics(data):
    flat_data = data.flatten()  # 配列を1次元に変換
    min_value = np.min(flat_data)
    max_value = np.max(flat_data)
    avg_value = np.mean(flat_data)
    median_value = np.median(flat_data)

    return {
        'min': min_value,
        'max': max_value,
        'average': avg_value,
        'median': median_value
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained SIREN model.')
    parser.add_argument('--sidelen', type=str, default='64,64,64', help='Comma-separated dimensions (e.g., 64,64,64).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs.')

    args = parser.parse_args()

    # sidelen をパース
    sidelen = tuple(map(int, args.sidelen.split(',')))
    if len(sidelen) != 3:
        raise ValueError("sidelen must have exactly three dimensions, e.g., 64,64,64")

    # 学習済みモデルの読み込み
    img_siren = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True)
    img_siren.load_state_dict(torch.load(args.model_path))

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_siren.to(device)
    img_siren.eval()

    # 座標を生成
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_input = get_mgrid(sidelen, dim=3).to(device)
    model_output, coords = img_siren(model_input)

    # 出力をCPUに移してNumPy配列に変換
    model_output_np = model_output.cpu().view(*sidelen).detach().numpy()

    # 出力サイズを保存
    output_size = sidelen

    # 輝度値の統計情報を計算
    stats = calculate_statistics(model_output_np)
    print("Model Output")
    print(f"Min: {stats['min']}")
    print(f"Max: {stats['max']}")
    print(f"Average: {stats['average']}")
    print(f"Median: {stats['median']}")

    # -1より小さいものは-1に、1より大きいものは1に設定
    model_output_clamp = torch.clamp(torch.tensor(model_output_np), min=-1, max=1)
    model_output_clamp = ((model_output_clamp + 1) * 127.5).to(torch.uint8)
    model_output_clamp_np = model_output_clamp.view(*sidelen).numpy()

    # NumPy配列を保存する
    output_npy_path = os.path.join(output_dir, 'model_output.npy')
    np.save(output_npy_path, model_output_clamp_np)
    print(f"モデルの出力をNumPyファイルとして '{output_npy_path}' に保存しました。")

    # 輝度値修正
    if all(s >= 3 for s in sidelen):
        processed_data, modified_value = process_tensor(torch.tensor(model_output_np))
        processed_data_np = processed_data.numpy()

        # 修正後の出力を保存
        processed_output_npy_path = os.path.join(output_dir, 'processed_model_output.npy')
        np.save(processed_output_npy_path, processed_data_np)
        print(f"修正後の出力をNumPyファイルとして '{processed_output_npy_path}' に保存しました。")

        stats2 = calculate_statistics(processed_data_np)
        print("Model Output（輝度値修正後）")
        print(f"Min: {stats2['min']}")
        print(f"Max: {stats2['max']}")
        print(f"Average: {stats2['average']}")
        print(f"Median: {stats2['median']}")
    else:
        print("データが小さすぎて処理できません（行列サイズが3×3×3以下です）。")
