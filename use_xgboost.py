import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import pickle
import xgboost as xgb

AUDIO_SAMPLE_RATE = 44100
MAX_FEATURE_LENGTH = 1000


def extract_features(audio_file):

    audio, sr = librosa.load(audio_file, sr=AUDIO_SAMPLE_RATE)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    f0, voiced_flag = librosa.piptrack(y=audio, sr=sr)
    f0_mean = np.mean(f0)
    f0_std = np.std(f0)

    features = np.concatenate((mfccs_mean, mfccs_std, [zcr_mean], [zcr_std], [
                              f0_mean], [f0_std]))
    return features


def kmeans_clustering(features, num_clusters):

    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans.fit(features)
    return kmeans.labels_


# def train_xgboost_model(features, labels):
#     dtrain = xgb.DMatrix(features, label=labels)
#     params = {
#         'objective': 'multi:softmax',  # 多分类问题
#         'num_class': len(np.unique(labels)),  # 类别数量
#         'eval_metric': 'merror'  # 评估指标：错误率
#     }
#     model = xgb.train(params, dtrain)
#     return model


def train_xgboost_model(features, labels):
    dtrain = xgb.DMatrix(features, label=labels)
    params = {
        'objective': 'multi:softmax',  # 多分类问题
        'num_class': len(np.unique(labels)),  # 类别数量
        'eval_metric': 'merror'  # 评估指标：错误率
    }
    evals = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, evals=evals, verbose_eval=True)
    return model


def extract_features_and_save(audio_files, save_path):
    features = []
    for audio_file in tqdm(audio_files, desc="Extracting features"):
        feature = extract_features(audio_file)
        features.append(feature)

    features = np.array(features)

    with open(save_path, 'wb') as f:
        pickle.dump(features, f)


def load_features(file_path):
    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    return features


def identify_speakers(audio_files, output_dir, num_clusters):
    features_file = os.path.join("logs", "features.pkl")
    if os.path.exists(features_file):
        print("Loading features from file...")
        features = load_features(features_file)
    else:
        os.makedirs("logs", exist_ok=True)
        extract_features_and_save(audio_files, features_file)
        features = load_features(features_file)

    kmeans_labels = kmeans_clustering(features, num_clusters)
    xgboost_model = train_xgboost_model(features, kmeans_labels)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_clusters):
        speaker_dir = os.path.join(output_dir, f'speaker_{i}')
        os.makedirs(speaker_dir, exist_ok=True)
    for i, audio_file in tqdm(enumerate(audio_files), desc="Copying files", total=len(audio_files)):
        speaker_id = int(xgboost_model.predict(
            xgb.DMatrix([features[i]]))[0])  # 将speaker_id转换为整数
        output_speaker_dir = os.path.join(output_dir, f'speaker_{speaker_id}')
        output_path = os.path.join(
            output_speaker_dir, os.path.basename(audio_file))
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        shutil.copy(audio_file, output_path)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    for i in range(num_clusters):
        indices = np.where(kmeans_labels == i)
        plt.scatter(reduced_features[indices, 0], reduced_features[indices,
                    1], color=colors[i], label=f'Speaker {i}')

    plt.title('Audio Speakers Distribution')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('./logs/', 'speakers_distribution_xgboost.png'))


def main(input_dir, output_dir, num_clusters):
    if not os.path.exists(input_dir):
        print("Error: Input directory doesn't exist!")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Starting audio speaker identification...")

    audio_files = [os.path.join(input_dir, f)
                   for f in os.listdir(input_dir) if f.endswith('.wav')]

    if len(audio_files) == 0:
        print("Error: No audio files found in the input directory!")
        return

    identify_speakers(audio_files, output_dir, num_clusters)

    print("Audio speaker identification completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Audio speaker identification.')
    parser.add_argument('-i', '--input_dir', type=str, default='dataset',
                        help='Input directory containing audio files.')
    parser.add_argument('-o', '--output_dir', type=str, default='output',
                        help='Output directory for segmented audio files.')
    parser.add_argument('-n', '--num_clusters', type=int,
                        default=5, help='Number of speakers to identify.')

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.num_clusters)
