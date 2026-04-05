# inference_pc.py
# 这个脚本实现了在PC上使用ONNX模型进行音频命令识别的功能。它包括音频预处理、特征提取、单个音频预测和批量预测等功能。还提供了从特征文件创建分类器的函数，方便用户在PC上进行命令识别。
import onnxruntime as ort
import numpy as np
import librosa
import json
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os


class TinyCNNInferencePC:
    def __init__(self, onnx_path, command_mapping_path):
        # 加载ONNX模型
        self.session = ort.InferenceSession(onnx_path)

        # 加载命令映射
        with open(command_mapping_path, 'r', encoding='utf-8') as f:
            self.command2id = json.load(f)
        self.id2command = {v: k for k, v in self.command2id.items()}

        # 音频参数（与训练时一致）
        self.sr = 16000
        self.sec = 3  # 3秒音频
        self.frame = 300  # 300帧
        self.n_mels = 40

        print(f"✅ 模型加载成功，支持 {len(self.command2id)} 个命令")
        print("支持的命令列表:")
        for cmd, idx in self.command2id.items():
            print(f"  {idx:2d}: {cmd}")

    def preprocess_audio(self, audio_path):
        """预处理音频，与训练时完全一致"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.sr)

            # 裁剪或填充到3秒
            target_length = self.sec * sr
            if len(y) > target_length:
                # 取中间3秒
                start = (len(y) - target_length) // 2
                y = y[start:start + target_length]
            elif len(y) < target_length:
                # 填充到3秒
                y = np.pad(y, (0, target_length - len(y)), 'constant')

            # 提取Mel频谱
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr,
                n_fft=512,
                hop_length=160,
                n_mels=self.n_mels
            )
            logmel = librosa.power_to_db(mel, ref=np.max)

            # 调整到固定长度300帧
            if logmel.shape[1] < self.frame:
                logmel = np.pad(logmel, ((0, 0), (0, self.frame - logmel.shape[1])), 'constant')
            else:
                logmel = logmel[:, :self.frame]

            return logmel.astype(np.float32)

        except Exception as e:
            print(f"❌ 音频预处理失败: {e}")
            return None

    def extract_embedding(self, audio_path):
        """提取32维特征向量"""
        logmel = self.preprocess_audio(audio_path)
        if logmel is None:
            return None

        # 添加batch维度
        input_data = np.expand_dims(logmel, axis=0)  # (1, 40, 300)

        # ONNX推理
        try:
            inputs = {'fbank': input_data}
            outputs = self.session.run(['embedding'], inputs)
            embedding = outputs[0][0]  # 提取32维向量
            return embedding
        except Exception as e:
            print(f"❌ 模型推理失败: {e}")
            return None

    def predict_single(self, audio_path, classifier=None, scaler=None):
        """预测单个音频"""
        embedding = self.extract_embedding(audio_path)
        if embedding is None:
            return None, None

        if classifier is not None and scaler is not None:
            # 使用分类器预测
            embedding_scaled = scaler.transform([embedding])
            pred_id = classifier.predict(embedding_scaled)[0]
            confidence = np.max(classifier.predict_proba(embedding_scaled))
            command = self.id2command.get(pred_id, "未知命令")
            return command, confidence
        else:
            # 只返回特征向量
            return embedding, None

    def batch_predict(self, audio_dir, classifier=None, scaler=None):
        """批量预测目录中的音频文件"""
        results = []
        audio_files = [f for f in os.listdir(audio_dir)
                       if f.endswith(('.wav', '.mp3', '.m4a', '.flac'))]

        for audio_file in audio_files:
            audio_path = os.path.join(audio_dir, audio_file)
            command, confidence = self.predict_single(audio_path, classifier, scaler)

            if command is not None:
                if confidence is not None:
                    results.append({
                        'file': audio_file,
                        'command': command,
                        'confidence': f"{confidence:.4f}",
                        'status': '✅ 成功'
                    })
                else:
                    results.append({
                        'file': audio_file,
                        'embedding': embedding.tolist(),
                        'status': '✅ 特征提取成功'
                    })
            else:
                results.append({
                    'file': audio_file,
                    'status': '❌ 失败'
                })

        return results


def create_classifier_from_features(feature_file, label_file):
    """从特征文件创建分类器"""
    try:
        X_train = np.load(feature_file)
        y_train = np.load(label_file)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(X_train_scaled, y_train)

        print(f"✅ 分类器创建成功，训练样本: {len(X_train)}")
        return classifier, scaler
    except Exception as e:
        print(f"❌ 分类器创建失败: {e}")
        return None, None


if __name__ == "__main__":
    # 初始化推理器
    model = TinyCNNInferencePC(
        onnx_path="tiny_cnn_rpi.onnx",
        command_mapping_path=r"D:\专用轻量分类器\tiny_CNN32维\command_mapping.json"
    )

    # 尝试加载分类器（如果有特征文件）
    classifier, scaler = None, None
    if os.path.exists(r"D:\专用轻量分类器\tiny_CNN32维\features_32_short_arcface_best.npy") and os.path.exists(r"D:\专用轻量分类器\tiny_CNN32维\labels_32_short_arcface_best.npy"):
        classifier, scaler = create_classifier_from_features(
            r"D:\专用轻量分类器\tiny_CNN32维\features_32_short_arcface_best.npy",
            r"D:\专用轻量分类器\tiny_CNN32维\labels_32_short_arcface_best.npy"
        )

    # 测试单个音频
    test_audio = r"C:\语音识别大模型\Whisper-Finetune\智能家居\智能家居适老化语料-原声\原声\空调-1.wav"  # 替换为你的测试音频路径
    if os.path.exists(test_audio):
        if classifier is not None:
            command, confidence = model.predict_single(test_audio, classifier, scaler)
            print(f"\n🎯 预测结果:")
            print(f"   文件: {test_audio}")
            print(f"   命令: {command}")
            print(f"   置信度: {confidence:.4f}")
        else:
            embedding, _ = model.predict_single(test_audio)
            print(f"\n🔍 特征提取结果:")
            print(f"   文件: {test_audio}")
            print(f"   32维特征: {embedding}")
            print(f"   特征形状: {embedding.shape}")

    # 批量测试（如果有测试目录）
    test_dir = "test_audios"
    if os.path.exists(test_dir):
        results = model.batch_predict(test_dir, classifier, scaler)
        print(f"\n📊 批量测试结果 ({len(results)} 个文件):")
        for result in results:
            print(f"   {result['file']}: {result['status']}")
            if 'command' in result:
                print(f"       命令: {result['command']}, 置信度: {result['confidence']}")