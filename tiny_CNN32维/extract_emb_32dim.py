# extract_emb_32dim.py
# 从训练好的 TinyCNN 模型中提取 32 维特征，保存为 .npy 文件，供后续训练或分析使用。
import torch, librosa, numpy as np, json, os
from tqdm import tqdm
#明天一旦有新语音进来，还得靠它 + .pth 把新文件变成 32 维向量。
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TinyCNN(torch.nn.Module):
    def __init__(self, emb=32, n_class=56):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(40, 64, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.fc = torch.nn.Linear(128, emb)

    def forward(self, x):
        h = self.conv(x).squeeze(-1)
        return self.fc(h)

# 1. 载入训练好的权重（先训练或下载）
net = TinyCNN(emb=32).to(device)
net.load_state_dict(torch.load('tiny_cnn_32dim.pth', map_location=device))
net.eval()

# 2. 提取函数
@torch.no_grad()
def extract_32dim(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    if len(y) > 3*16000: y = y[:3*16000]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, n_mels=40)
    logmel = librosa.power_to_db(mel, ref=np.max)
    x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).to(device)
    emb = net(x).cpu().numpy().squeeze()
    return emb

# 3. 批量导出
with open(r"D:\专用轻量分类器\jsonl_dataset\train_smart_home_none_200乘5.jsonl", 'r', encoding='utf-8') as f:
    items = [json.loads(l) for l in f]

X, y = [], []
for it in tqdm(items, desc="提取 32 维"):
    e = extract_32dim(it["audio"]["path"])
    if e is not None:
        X.append(e)
        y.append(it["sentence"])

np.save("features_32.npy", np.array(X))
np.save("labels_32.npy",   np.array(y))
print("完成 32 维特征提取，形状:", np.array(X).shape)