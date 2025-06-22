import os
import numpy as np
from data.wav_folder import read_wav
from resemblyzer import preprocess_wav, VoiceEncoder 

attr_dir = '/path/to/attr_utt_path/'
target_dir = '/path/to/tgt_utt_path/'

sampling_rate = 24000
encoder = VoiceEncoder()

attr_list = []
for attr_path in os.listdir(attr_dir):
    attr_path = os.path.join(attr_dir, attr_path)

    attr_wav, _ = read_wav(attr_path, sr=sampling_rate)

    attr_list.append(attr_wav)

target_list = []
for target_path in os.listdir(target_dir):
    target_path = os.path.join(target_dir, target_path)
    target_wav, _ = read_wav(target_path, sr=sampling_rate)
    target_list.append(target_wav)

spk_embeds_attr = np.array([encoder.embed_speaker(attr_list)])
spk_embeds_target = np.array([encoder.embed_speaker(target_list)])

spk_sim_attr = np.inner(spk_embeds_attr, spk_embeds_target)

print('CVC:{}'.format(spk_sim_attr))