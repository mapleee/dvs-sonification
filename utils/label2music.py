import math
import numpy as np
import pretty_midi
from midi2audio import FluidSynth
from config.load_yaml import configs

# 从轻快到沉重
instruments = ['Music Box', 'Cello', 'Acoustic Grand Piano']
# instruments = ['Pan Flute', 'Cello', 'Contrabass', 'Viola']


def cluster_to_notes_and_cc(c_label, start):
    '''

    :param c_label:
    :param start:
    :return:
    '''
    # Control Change
    cc_value = get_cc_value(c_label, func='logistic')
    cc = pretty_midi.ControlChange(number=10, value=cc_value, time=start * configs['packet_time'] / 1e6)

    # Note
    velocity = get_note_velocity(c_label)
    note_off_param = get_note_off_param(c_label)
    pitch_max = int(np.max(c_label[:, 1]) / (configs['height'] / (configs['oct_num']*12)))
    pitch_min = int(np.min(c_label[:, 1]) / (configs['height'] / (configs['oct_num']*12)))
    pitch_mean = int((np.mean(c_label[:, 1])) / (configs['height'] / (configs['oct_num']*12)))
    pitch_played = np.array(configs['pitch_major'][pitch_max-pitch_min]) + configs['root_pitch'] + pitch_mean
    note_list = []
    for pitch in pitch_played:
        note = pretty_midi.Note(velocity=velocity, pitch=pitch,
                                start=start * configs['packet_time'] / 1e6,
                                end=(start + note_off_param) * configs['packet_time'] / 1e6)
        note_list.append(note)

    return note_list, cc


def get_cc_value(label, func='logistic'):
    '''
    x轴坐标和spanning的映射关系
    :param label:
    :param func:
    :return:
    '''
    mean_label_x = np.mean(label[:, 0])
    if func == 'linear':
        cc_value = int(mean_label_x * 127 / configs['width'])
    elif func == 'logistic':
        cc_value = int(128 / (1 + np.exp(-0.03 * (mean_label_x - configs['width']/2))))
    else:
        raise AttributeError(f'func only has linear and logistic, but got {func}.')
    return cc_value


def get_note_velocity(label, mapping='constant'):
    '''
    听觉频响曲线, 事件数量越多，velocity越大
    :param label:
    :param mapping: 'constant': 常数
                 'area': 根据聚类面积大小决定velocity大小
    :return:
    '''
    if mapping == 'area':
        obj_area = len(label)   # 利用事件数来估计物体的面积
        total_area = configs['width'] * configs['height']
        obj_ratio = 1.0 * obj_area / total_area
        # 以下是一个同时满足 x = 0.001, y = 60以及x = 1, y = 120的对数函数
        if obj_ratio < 0.001:
            velocity = 0
        else:
            velocity = int(20 * math.log(obj_ratio, 10) + 120)

    elif mapping == 'constant':
        velocity = 100

    else:
        raise AttributeError(f'get wrong mapping method {mapping}')

    return velocity


def get_note_off_param(label):
    '''
    返回音符应当持续的帧数
    :param label:
    :return:
    '''
    c_width = np.max(label[:, 0]) - np.min(label[:, 0])
    noff_param = 10 * c_width // configs['width'] + 1
    return noff_param


def label_list_to_midi(label_list, midi_path):
    '''
    设计该函数先以可交互方式设计，即仅进行当前帧分析，而不分析后续帧
    整体设计思想：先设计好每帧图像的声音映射方案。具体流程为首先判断该帧图像是否需要发声，若不需要直接进入下一帧，若需要则给出声音映射。
    :param label_list: 长度为evt_num的数组，且每个元素是三维（x, y, label）
    :param midi_path:
    :return:
    '''
    init = True
    midi_data = pretty_midi.PrettyMIDI()
    cluster_idx_list = []
    for i, label in enumerate(label_list):
        if label is not None:
            cluster_idx = np.unique(label[:, 2])
            if init:
                for j in cluster_idx:
                    instr_program = pretty_midi.instrument_name_to_program(instruments[j])
                    instr = pretty_midi.Instrument(program=instr_program)
                    cluster_label = label[np.where(label[:, 2] == j)]
                    x_mean = np.mean(cluster_label[:, 0])
                    y_mean = np.mean(cluster_label[:, 1])
                    note_list, control_change = cluster_to_notes_and_cc(cluster_label, i)
                    instr.notes.extend(note_list)
                    instr.control_changes.append(control_change)
                    midi_data.instruments.append(instr)
                cluster_idx_list.append(cluster_idx)
                frame_count = np.zeros([np.max(cluster_idx) + 1])   # 帧数计数
                d_sum = np.zeros([np.max(cluster_idx) + 1])         # 移动距离累加
                init = False
            else:
                # 当有新增聚类时
                if len(cluster_idx) > len(cluster_idx_list[-1]):
                    diff_element = sorted(list(set(cluster_idx) - set(cluster_idx_list[-1])))

                    if diff_element[0] == np.max(cluster_idx):
                        frame_count = np.append(frame_count, np.zeros([len(diff_element)]))
                        d_sum = np.append(d_sum, np.zeros([len(diff_element)]))
                        for j in diff_element:
                            instr_program = pretty_midi.instrument_name_to_program(instruments[j])
                            instr = pretty_midi.Instrument(program=instr_program)
                            midi_data.instruments.append(instr)
                    else:
                        for j in diff_element:
                            frame_count[j] = 0
                            d_sum[j] = 0

                for j in cluster_idx:
                    cluster_label = label[np.where(label[:, 2] == j)]
                    if frame_count[j] == configs['frame_interval']:
                        if d_sum[j] >= 10:
                            note_list, control_change = cluster_to_notes_and_cc(cluster_label, i)
                            midi_data.instruments[j].notes.extend(note_list)
                            midi_data.instruments[j].control_changes.append(control_change)
                            d_sum[j] = 0
                        frame_count[j] = 0

                    frame_count[j] += 1
                    x_mean_new = np.mean(cluster_label[:, 0])
                    y_mean_new = np.mean(cluster_label[:, 1])
                    d_sum[j] += math.sqrt((x_mean_new - x_mean)*(x_mean_new - x_mean) +
                                          (y_mean_new - y_mean)*(y_mean_new - y_mean))
                    x_mean = x_mean_new
                    y_mean = y_mean_new

                cluster_idx_list.append(cluster_idx)

    midi_data.write(midi_path)


def save_audio(sf2_path, midi_path, audio_path):
    sf2 = FluidSynth(sf2_path)
    sf2.midi_to_audio(midi_path, audio_path)
