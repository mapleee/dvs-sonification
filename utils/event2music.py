import cv2
import os
import numpy as np
import pretty_midi
import subprocess
from midi2audio import FluidSynth
from collections import Counter
from utils.config import *


def write_event_video(coo_accu_list, video_path, audio_path, video_audio_path):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(video_path, fourcc, FPS, (WIDTH, HEIGHT))
    for coo_accu in coo_accu_list:
        img = np.zeros((HEIGHT, WIDTH), np.uint8)
        for coo in coo_accu:
            if int(coo[2]) >= ACCUMULATE_NUM:
                img[int(coo[1]), int(coo[0])] = 255
            else:
                img[int(coo[1]), int(coo[0])] = int(255 // ACCUMULATE_NUM * coo[2])
        img_3c = cv2.merge([img, img, img])
        out.write(img_3c)
    out.release()
    os.system(f'ffmpeg -i {video_path} -i {audio_path} -c copy {video_audio_path}')


def process_events(event_txt):
    '''

    :param event_txt:
    :return: events_coo_list: 每帧图像的事件列表，列表中每个元素为一个二维（dim:[num_event, 2]）数组
             coo_accu_list: 每帧图像的事件列表，附加一个事件出现的次数，列表中每个元素为一个二维（dim:[num_event, 3]）数组
    '''
    # 获取每帧图像中事件的坐标
    events = np.genfromtxt(event_txt)
    events[:, 0] = (events[:, 0] - events[0, 0]) // PACKET_TIME
    events_coo_list = []
    coo_accu_list = []

    for i in range(int(events[-1, 0]) + 1):
        idx = np.where(events[:, 0] == i)
        events_coo = events[idx, 1:3].reshape(-1, 2)
        events_coo = numpy_multi_sort(events_coo)
        events_coo_list.append(events_coo)

        # # 对每帧图像的事件进行积分，不论极性，坐标只要出现一次，亮度增加一个常数（255//ACCUMULATE_NUM）
        coo_accu = np.array([[events_coo[0,0], events_coo[0,1], 1]])
        for j, coo in enumerate(events_coo, 1):
            if (coo[0] == coo_accu[-1, 0]) and (coo[1] == coo_accu[-1, 1]):
                coo_accu[-1, 2] += 1
            else:
                coo = np.append(coo, 1).reshape(1, 3)
                coo_accu = np.concatenate((coo_accu, coo))
        coo_accu_list.append(coo_accu)

    return events_coo_list, coo_accu_list


def pitch_played_for_frame(start_time, events_coo):
    '''
    对于一帧图像，其中事件个数在指定高度范围内的数量大于设定阈值，则该区域存在音符
    :param start_time:
    :param events_coo:
    :return:
    '''

    e_pitch = events_coo[:, 1] // (HEIGHT // (OCTAVE_NUM*12))  # 表示一个事件高度对应哪一个音高
    pitch_counter = Counter(e_pitch)

    # 去噪，如果总事件数低于设定值，则证明该帧图像为噪声
    event_num = sum(pitch_counter.values())
    if event_num < TOTAL_EVENT_THRESHOLD:
        return None, None

    # 得到所有符合条件（事件数大于阈值）的音高序列
    pitch_played_index = []
    for item in Counter(e_pitch).items():
        if item[1] > EVENT_NUM_THRESHOLD:
            pitch_played_index.append(item[0])
    # 注意：高度与音高是正好相反的
    pitch_played_index = np.sort((OCTAVE_NUM*12) - np.array(pitch_played_index))

    if len(pitch_played_index) == 0:
        return None, None

    # 求出最高音与最低音的差，查表获得使用音符
    pitch_diff = pitch_played_index[-1] - pitch_played_index[0]
    if SCALE_TYPE == 'Major':
        pitch_played = np.array(PITCH_MAJOR[int(pitch_diff)]) + ROOT_PITCH + int(pitch_played_index[0])
        notes_list = []
        for pitch in pitch_played:
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time/1e6, end=(start_time+PACKET_TIME)/1e6)
            notes_list.append(note)
        # 加上control change事件
        cc_value = int(np.mean(events_coo[:, 0]) * 127 / WIDTH)
        cc = pretty_midi.ControlChange(number=10, value=cc_value, time=start_time/1e6)
    else:
        raise NotImplementedError

    return notes_list, cc


def pitch2midi(events_coo_list, midi_path):
    midi_data = pretty_midi.PrettyMIDI()
    piano_all = pretty_midi.Instrument(program=0)
    # piano_concat = pretty_midi.Instrument(program=0)

    for i, events_coo in enumerate(events_coo_list):
        notes_list, cc = pitch_played_for_frame(i*PACKET_TIME, events_coo)
        if (notes_list is None) and (cc is None):
            continue
        piano_all.notes.extend(notes_list)
        piano_all.control_changes.append(cc)

    # midi_data.instruments.append(piano_concat)
    midi_data.instruments.append(piano_all)
    midi_data.write(midi_path)

    return midi_data


def midi2audio(sf2_path, midi_path, audio_path):
    sf2 = FluidSynth(sf2_path)
    sf2.midi_to_audio(midi_path, audio_path)


def numpy_multi_sort(array):
    value = [tuple(a) for a in array]
    dtyp = [('x', int), ('y', int)]
    array = np.array(value, dtype=dtyp)
    sorted_arr = np.sort(array, order=['x', 'y'])
    sorted_arr = np.array([list(a) for a in sorted_arr])
    return sorted_arr


def pitch_concat(midi_data):
    # 音高拼接，把所有破碎的音符拼接起来
    new_midi_data = pretty_midi.PrettyMIDI()
    piano_concat = pretty_midi.Instrument(program=0)

    piano_all = midi_data.instruments[0]
    piano_all.notes = sorted(piano_all.notes, key=lambda x: (x.pitch, x.start))  # 对所有音符按照音高和开始时间排序
    flag = 0
    for i, note in enumerate(piano_all.notes):
        if flag == 0:
            start = note.start
            end = note.end
            pitch = note.pitch
            flag = 1
        else:
            if piano_all.notes[i].pitch != piano_all.notes[i-1].pitch:
                start = note.start
                end = note.end
                pitch = note.pitch
                flag = 1
                continue
            if piano_all.notes[i].start == piano_all.notes[i-1].end:
                end = note.end
            else:
                new_note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
                piano_concat.notes.append(new_note)
                flag = 0
    new_midi_data.instruments.append(piano_concat)
    return new_midi_data


if __name__ == '__main__':
    pass