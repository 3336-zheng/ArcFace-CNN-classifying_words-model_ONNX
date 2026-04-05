# export_onnx_rpi.py  （纯导出版）
# 直接从训练好的模型中加载 backbone 部分，导出为 ONNX 格式，供树莓派使用。

import torch, json, numpy as np, os
from tiny_CNN32维.train_tiny_cnn_32dim import TinyCNN   # 只拿模型类，不跑训练

model_path = r"D:\专用轻量分类器\tiny_CNN32维\tiny_cnn_32dim_short_arcface_best.pth"
onnx_path  = r"tiny_cnn_rpi.onnx"

# 只取 embedding 部分
class EmbedOnly(torch.nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.backbone = TinyCNN(emb=32)
        self.backbone.load_state_dict(torch.load(ckpt, map_location='cpu'))
    def forward(self, x):
        return self.backbone(x)          # 32 维向量

model = EmbedOnly(model_path).eval()
dummy = torch.randn(1, 40, 300)        # 1 句 3 秒 Fbank
torch.onnx.export(model, dummy, onnx_path,
                  input_names=['fbank'],
                  output_names=['embedding'],
                  opset_version=11,
                  dynamic_axes=None)
print("✅ ONNX 导出完成：", onnx_path)