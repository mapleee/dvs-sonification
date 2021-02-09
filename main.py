import glob
import os
import tqdm
import time

from utils import label2music, event_process, visualization
from config.load_yaml import configs


event_txts = sorted(glob.glob(f'{configs["event_dir"]}/*.txt'))

os.makedirs(configs['video_dir'], exist_ok=True)
os.makedirs(configs['label_dir'], exist_ok=True)
os.makedirs(configs['music_dir'], exist_ok=True)
os.makedirs(configs['image_dir'], exist_ok=True)
os.makedirs(configs['cmap_dir'], exist_ok=True)


if __name__ == '__main__':

    for event_txt in tqdm.tqdm(event_txts):
        # event_txt = configs['event_path']
        time1 = time.time()
        event_type = os.path.splitext(os.path.basename(event_txt))[0]

        # label是一个长度为evt_num的数组，且每个元素是三维（x, y, label），label_list是"视频帧数"个label连接起来的列表
        # 聚类
        pic_list, label_list = event_process.gmm_cluster(event_txt)

        # 写midi
        midi_path = f'{configs["music_dir"]}/{event_type}.mid'
        label2music.label_list_to_midi(label_list, midi_path)

        # 写audio
        audio_path = f'{configs["music_dir"]}/{event_type}.wav'
        label2music.save_audio(configs["sf2_path"], midi_path, audio_path)

        # 写视频
        video_path = f'{configs["video_dir"]}/{event_type}.avi'
        label_path = f'{configs["label_dir"]}/{event_type}.avi'
        video_audio_path = f'{configs["label_dir"]}/{event_type}_with_sound.avi'
        visualization.write_event_video(pic_list, video_path)

        visualization.write_label_video(label_list, label_path, output_image=False,
                                        audio_path=audio_path, video_audio_path=video_audio_path)
