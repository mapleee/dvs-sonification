import glob
import os
import tqdm
from utils import event2music


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
        events_velocity = event2music.get_event_velocity(events_coo_list)
        note_place = event2music.get_note_place(events_velocity)

        # 写midi
        midi_path = f'{music_dir}/{event_type}.mid'
        event2music.pitch2midi(events_coo_list, note_place, midi_path)

        # 写audio
        audio_path = f'{music_dir}/{event_type}.wav'
        event2music.midi2audio(sf2_path, midi_path, audio_path)

        # 写视频
        video_path = f'{video_dir}/{event_type}.avi'
        video_audio_path = f'{video_dir}/{event_type}_with_sound.avi'
        event2music.write_event_video(coo_accu_list, video_path, audio_path, video_audio_path)

