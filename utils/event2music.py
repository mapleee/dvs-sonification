import cv2
import os
import numpy as np
import pretty_midi
import subprocess
from midi2audio import FluidSynth
from collections import Counter
from config.load_yaml import configs


def frame_to_music(label):
    pass


def pic_list_to_music(label_list):
    init = True
    cluster_idx = []
    for i, label in enumerate(label_list):
        if label is not None:
            if init:
                frame_to_music(label)
                init = False
                first_idx = i
            else:

                cur_label_num = np.max(label_list[i][:, 2])
                pre_label_num = np.max(label_list[i-1][:, 2])
                if cur_label_num <= pre_label_num:
                    pass
                # 计算与上一张图的对应事件簇


def get_event_velocity(events_coo_list):
    '''
    得到每个PACKET事件的速度，计算方法就是用位移除以时间，目前只支持单一物体
    :param events_coo_list:
    :return:
    '''
    events_velocity = []
    for i, events_coo in enumerate(events_coo_list):
        if i == 0:
            events_velocity.append(0)
        else:
            if (events_coo_list[i].shape[0] < configs['total_event_thres']) or \
                    (events_coo_list[i-1].shape[0] < configs['total_event_thres']):
                events_velocity.append(0)
            else:
                x1, y1 = np.mean(events_coo_list[i-1][:, 0]), np.mean(events_coo_list[i-1][:, 1])
                x2, y2 = np.mean(events_coo_list[i][:, 0]), np.mean(events_coo_list[i][:, 1])
                # 原本是除以packet_size
                vel = np.sqrt(np.square(x2-x1)+np.square(y2-y1)) / (configs['packet_time']/1000)  # 单位：像素/毫秒
                events_velocity.append(vel)
    return events_velocity


def get_note_place(events_velocity):
    '''
    得到所有note应当出现的位置，也就是开始时间，用于控制音符密度
    使用最小的时间间隔是速度120，4/4拍的16分音符，其时值为125ms，因为一帧图像是25ms，也就是每5帧图像检查一次
    configs['vel_thres'] = [0.5, 1, 2]  # 速度分为4个档，分别对应用半音符，四分音符，八分音符和十六分音符的时间长度来演奏，注意：4个档是写死的

    :param events_velocity:
    :return:
    '''
    note_place = []
    for i, vel in enumerate(events_velocity):
        if i == 0:
            continue
        if i % configs['note_per_frame'] == 0:
            # 对连续configs['note_per_frame']帧图像取平均值
            vel_now = np.mean(events_velocity[i-configs['note_per_frame']:i])
            # 得到该速度在configs['vel_thres']的索引
            vel_index = np.where(np.sort(np.append(configs['vel_thres'], vel_now))==vel_now)[0][0]
            if vel_index == 3:
                note_place.append(1)
            elif vel_index == 2:
                if len(note_place) == 0:
                    note_place.append(1)
                elif note_place[-1] == 1:
                    note_place.append(0)
                else:
                    note_place.append(1)
            elif vel_index == 1:
                if len(note_place) == 0:
                    note_place.append(1)
                elif len(note_place) < 3:
                    if 1 not in note_place:
                        note_place.append(1)
                    else:
                        note_place.append(0)
                else:
                    if 1 not in note_place[-3:]:
                        note_place.append(1)
                    else:
                        note_place.append(0)
            else:
                note_place.append(0)

    return note_place


def note_played_for_frame(start_time, events_coo):
    '''
    对于一帧图像，其中事件个数在指定高度范围内的数量大于设定阈值，则该区域存在音符
    :param start_time:
    :param events_coo:
    :return:
    '''

    e_pitch = events_coo[:, 1] // (configs['height'] // (configs['oct_num']*12))  # 表示一个事件高度对应哪一个音高
    # e_std = np.std(events_coo[:, 0], ddof=1)
    # print(f'e_std: {e_std}')
    pitch_counter = Counter(e_pitch)

    # 去噪，如果总事件数低于设定值，则证明该帧图像为噪声
    event_num = sum(pitch_counter.values())
    if event_num < configs['total_event_thres']:
        return None, None

    # 得到所有符合条件（事件数大于阈值）的音高序列
    pitch_played_index = []
    for item in Counter(e_pitch).items():
        if item[1] > configs['part_event_thres']:
            pitch_played_index.append(item[0])
    # 注意：高度与音高是正好相反的
    pitch_played_index = np.sort((configs['oct_num']*12) - np.array(pitch_played_index))

    if len(pitch_played_index) == 0:
        return None, None

    # 求出最高音与最低音的差，查表获得使用音符
    pitch_diff = pitch_played_index[-1] - pitch_played_index[0]
    if configs['scale_type'] == 'Major':
        pitch_played = np.array(configs['pitch_major'][int(pitch_diff)]) + configs['root_pitch'] + int(pitch_played_index[0])
        notes_list = []

        for pitch in pitch_played:
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time/1e6,
                                    end=(start_time+configs['frame_time']*(len(configs['vel_thres'])+1))/1e6)
            notes_list.append(note)
        # 加上control change事件
        cc_value = int(np.mean(events_coo[:, 0]) * 127 / configs['width'])
        cc = pretty_midi.ControlChange(number=10, value=cc_value, time=start_time/1e6)
    else:
        raise NotImplementedError

    return notes_list, cc


def pitch2midi(events_coo_list, note_place, midi_path):
    midi_data = pretty_midi.PrettyMIDI()
    piano_all = pretty_midi.Instrument(program=0)
    # piano_concat = pretty_midi.Instrument(program=0)

    for i, events_coo in enumerate(events_coo_list):
        if i == 0:
            continue
        if i % configs['note_per_frame'] == 0:
            if note_place[i // configs['note_per_frame'] - 1] == 1:
                notes_list, cc = note_played_for_frame(i*configs['frame_time'], events_coo)
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