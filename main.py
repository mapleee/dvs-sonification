import numpy as np
import glob
import cv2
import os
import tqdm
from utils import event2music
from utils.config import *


event_dir = 'data/event'
event_txts = glob.glob(f'{event_dir}/*.txt')
sf2_path = 'data/sf2/GeneralUser GS 1.442 MuseScore/GeneralUser GS MuseScore v1.442.sf2'

pic_dir = 'data/image'
os.makedirs(pic_dir, exist_ok=True)

video_dir = 'data/video'
os.makedirs(video_dir, exist_ok=True)

music_dir = 'data/music'
os.makedirs(music_dir, exist_ok=True)

if __name__ == '__main__':
    for event_txt in tqdm.tqdm(event_txts):
        event_type = os.path.splitext(os.path.basename(event_txt))[0]
        events_coo_list, coo_accu_list = event2music.process_events(event_txt)

        # timestamps = np.unique(events[:, 0])

        # # 写图片
        # os.makedirs(f'{pic_dir}/{event_type}', exist_ok=True)
        # for i, time in enumerate(timestamps):
        #     pic_fn = f'{pic_dir}/{event_type}/pic_{i}.png'
        #     event2music.write_event_pic(time, events, pic_fn)
        # break

        # # 写midi
        # midi_path = f'{music_dir}/{event_type}.mid'
        # event2music.pitch2midi(timestamps, events, midi_path)
        #
        # # 写audio
        # audio_path = f'{music_dir}/{event_type}.wav'
        # event2music.midi2audio(sf2_path, midi_path, audio_path)
        #
        # 写视频
        video_path = f'{video_dir}/{event_type}.avi'
        video_sound_path = f'{video_dir}/{event_type}_with_sound.avi'
        event2music.write_event_video(coo_accu_list, video_path)

        # event2music.write_event_video(events, timestamps, video_path, audio_path, video_sound_path)

