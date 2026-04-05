# smart_home_feature_extractor.py
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
from collections import Counter


class DataPreprocessor:
    def __init__(self):
        self.commands_counter = Counter()
        self.valid_data = []

    def load_and_analyze_data(self, jsonl_path):
        """加载JSONL数据并分析分布"""
        print("正在加载和分析数据...")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    audio_path = item["audio"]["path"]
                    sentence = item["sentence"]

                    # 检查文件是否存在
                    if os.path.exists(audio_path):
                        self.valid_data.append({
                            "audio_path": audio_path,
                            "sentence": sentence,
                            "duration": item.get("duration", 0)
                        })
                        self.commands_counter[sentence] += 1
                    else:
                        print(f"警告: 文件不存在 - {audio_path}")

                except json.JSONDecodeError as e:
                    print(f"JSON解析错误 (行 {line_num}): {e}")
                except KeyError as e:
                    print(f"键错误 (行 {line_num}): {e}")

        print(f"\n数据统计:")
        print(f"总有效样本: {len(self.valid_data)}")
        print(f"指令类别数: {len(self.commands_counter)}")
        print("\n指令分布:")
        for cmd, count in self.commands_counter.most_common():
            print(f"  {cmd}: {count} 样本")

        return self.valid_data

    def create_command_mapping(self, min_samples=10):
        """创建指令映射，过滤样本太少的类别"""
        self.command_mapping = {}
        valid_commands = []

        for cmd, count in self.commands_counter.items():
            if count >= min_samples:
                valid_commands.append(cmd)

        # 按字母顺序排序以确保一致性
        valid_commands.sort()
        self.command_mapping = {cmd: idx for idx, cmd in enumerate(valid_commands)}
        self.reverse_mapping = {idx: cmd for cmd, idx in self.command_mapping.items()}

        print(f"\n使用的指令类别 ({len(valid_commands)} 个):")
        for cmd in valid_commands:
            print(f"  {cmd} -> {self.command_mapping[cmd]}")

        return self.command_mapping

    def save_command_mapping_json(self, filename="command_mapping.json"):
        """保存指令映射为JSON格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.command_mapping, f, ensure_ascii=False, indent=2)
        print(f"✓ 指令映射已保存为 {filename}")


class FeatureExtractor:
    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate

    def safe_load_audio(self, audio_path):
        """安全加载音频文件，处理各种异常情况"""
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                print(f"文件不存在: {audio_path}")
                return None, None

            # 检查文件大小
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                print(f"空文件: {audio_path}")
                return None, None

            # 尝试加载音频
            y, sr = librosa.load(audio_path, sr=self.sampling_rate)

            # 检查音频长度
            if len(y) < self.sampling_rate * 0.5:  # 少于0.5秒
                print(f"音频过短: {audio_path} (长度: {len(y) / sr:.2f}秒)")
                return None, None

            return y, sr

        except Exception as e:
            print(f"加载音频失败 {audio_path}: {e}")
            return None, None

    def extract_pitch_features(self, y, sr):
        """提取基频特征，增加错误处理"""
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []

            # 只选择幅度足够大的帧
            magnitude_threshold = np.median(magnitudes) * 0.1

            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                magnitude = magnitudes[index, t]

                # 只保留有效的基频值
                if pitch > 50 and pitch < 500 and magnitude > magnitude_threshold:
                    pitch_values.append(pitch)

            if len(pitch_values) > 5:  # 至少有5个有效基频点
                return [np.mean(pitch_values), np.std(pitch_values)]
            else:
                return [0, 0]

        except Exception as e:
            print(f"基频提取失败: {e}")
            return [0, 0]

    def extract_features(self, audio_path):
        """从音频文件提取特征，增强错误处理"""
        try:
            # 安全加载音频
            y, sr = self.safe_load_audio(audio_path)
            if y is None:
                return None

            # 如果音频太长，截取中间部分
            if len(y) > sr * 3:  # 超过3秒
                start = (len(y) - sr * 3) // 2
                y = y[start:start + sr * 3]

            # 确保音频不为空
            if len(y) == 0:
                print(f"音频为空: {audio_path}")
                return None

            features = []

            # 1. 时域特征
            try:
                rms = librosa.feature.rms(y=y)
                features.extend([np.mean(rms), np.std(rms), np.max(rms)])
            except Exception as e:
                print(f"能量特征提取失败 {audio_path}: {e}")
                features.extend([0, 0, 0])

            try:
                zcr = librosa.feature.zero_crossing_rate(y)
                features.append(np.mean(zcr))
            except Exception as e:
                print(f"过零率提取失败 {audio_path}: {e}")
                features.append(0)

            # 2. 频域特征
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)
                features.extend(mfcc_mean[:8])  # 取前8个系数
            except Exception as e:
                print(f"MFCC提取失败 {audio_path}: {e}")
                features.extend([0] * 8)

            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                features.append(np.mean(spectral_centroids))
            except Exception as e:
                print(f"频谱质心提取失败 {audio_path}: {e}")
                features.append(0)

            try:
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                features.append(np.mean(spectral_bandwidth))
            except Exception as e:
                print(f"频谱带宽提取失败 {audio_path}: {e}")
                features.append(0)

            # 3. 基频特征 (音调)
            pitch_features = self.extract_pitch_features(y, sr)
            features.extend(pitch_features)

            # 检查特征是否有效
            feature_array = np.array(features)
            if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
                print(f"无效特征值: {audio_path}")
                return None

            return feature_array

        except Exception as e:
            print(f"处理文件 {audio_path} 时出错: {e}")
            return None

    def extract_features_batch(self, data_list, command_mapping, max_files=None):
        """批量提取特征，增加进度控制和内存优化"""
        print("开始批量提取特征...")

        # 可选：限制处理文件数量用于测试
        if max_files and max_files < len(data_list):
            data_list = data_list[:max_files]
            print(f"测试模式: 只处理前 {max_files} 个文件")

        features_list = []
        labels_list = []
        failed_files = []
        processed_count = 0

        for item in tqdm(data_list, desc="提取特征"):
            audio_path = item["audio_path"]
            sentence = item["sentence"]

            # 只处理在映射中的指令
            if sentence in command_mapping:
                features = self.extract_features(audio_path)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(command_mapping[sentence])
                    processed_count += 1
                else:
                    failed_files.append(audio_path)
            else:
                failed_files.append(audio_path)

            # 定期输出进度
            if processed_count % 100 == 0 and processed_count > 0:
                print(f"已处理 {processed_count} 个文件...")

        # 输出统计信息
        print(f"\n特征提取完成:")
        print(f"  - 成功提取: {len(features_list)} 个文件")
        print(f"  - 失败文件: {len(failed_files)} 个")

        if failed_files:
            print(f"  - 失败文件示例: {failed_files[:5]}")  # 只显示前5个失败文件

        if len(features_list) == 0:
            print("错误: 没有成功提取任何特征!")
            return None, None

        return np.array(features_list), np.array(labels_list)

    def save_features(self, features, labels, features_path="features.npy", labels_path="labels.npy"):
        """保存特征和标签，增加数据验证"""
        try:
            # 验证数据
            if features is None or len(features) == 0:
                print("错误: 没有有效的特征数据可保存")
                return False

            if labels is None or len(labels) == 0:
                print("错误: 没有有效的标签数据可保存")
                return False

            if len(features) != len(labels):
                print(f"警告: 特征数量({len(features)})和标签数量({len(labels)})不匹配")
                # 取较小值
                min_len = min(len(features), len(labels))
                features = features[:min_len]
                labels = labels[:min_len]

            # 保存数据
            np.save(features_path, features)
            np.save(labels_path, labels)

            # 验证保存的文件
            if os.path.exists(features_path) and os.path.exists(labels_path):
                print(f"✓ 特征已保存到 {features_path} (形状: {features.shape})")
                print(f"✓ 标签已保存到 {labels_path} (数量: {len(labels)})")
                return True
            else:
                print("错误: 保存文件失败")
                return False

        except Exception as e:
            print(f"保存特征时出错: {e}")
            return False

    def analyze_features(self, features):
        """分析特征数据质量"""
        if features is None or len(features) == 0:
            print("没有特征数据可分析")
            return

        print(f"\n特征数据分析:")
        print(f"  - 样本数量: {len(features)}")
        print(f"  - 特征维度: {features.shape[1]}")
        print(f"  - 特征范围: [{np.min(features):.4f}, {np.max(features):.4f}]")
        print(f"  - 特征均值: {np.mean(features):.4f}")
        print(f"  - 特征标准差: {np.std(features):.4f}")

        # 检查NaN和Inf值
        nan_count = np.sum(np.isnan(features))
        inf_count = np.sum(np.isinf(features))

        if nan_count > 0:
            print(f"  - 警告: 发现 {nan_count} 个NaN值")
        if inf_count > 0:
            print(f"  - 警告: 发现 {inf_count} 个Inf值")


def main():
    """主函数 - 完整的特征提取流程"""
    print("=" * 50)
    print("智能家居语音特征提取")
    print("=" * 50)

    # 步骤1: 数据预处理
    print("\n步骤1: 数据预处理")
    preprocessor = DataPreprocessor()
    data = preprocessor.load_and_analyze_data(r"D:\专用轻量分类器\jsonl_dataset\train_smart_home_none_200乘5.jsonl")
    command_mapping = preprocessor.create_command_mapping(min_samples=2)

    # 保存指令映射
    preprocessor.save_command_mapping_json("command_mapping.json")

    # 步骤2: 特征提取
    print("\n步骤2: 特征提取")
    extractor = FeatureExtractor()

    # 可选：使用 max_files 参数进行小规模测试
    # X, y = extractor.extract_features_batch(data, command_mapping, max_files=100)
    X, y = extractor.extract_features_batch(data, command_mapping, max_files=None)

    if X is not None and y is not None:
        print(f"\n特征矩阵形状: {X.shape}")
        print(f"标签数量: {len(y)}")

        # 分析特征质量
        extractor.analyze_features(X)

        # 保存特征和标签
        success = extractor.save_features(X, y)

        if success:
            print("\n🎉 特征提取完成!")
            print("下一步: 运行模型训练")
        else:
            print("\n❌ 特征提取失败")
    else:
        print("\n❌ 特征提取失败，请检查数据和错误信息")


if __name__ == "__main__":
    main()

# 我们提取特征时，每一条音频（即使是同一个指令的多个增强版本）都会独立提取特征。
# 所以，五个“开灯”指令的音频会提取出五个特征向量，每个特征向量对应一个“开灯”的标签。
# 在训练时，我们会将这五个特征向量都用于训练，并且它们都对应同一个类别（开灯）。