import os
import time
import numpy as np
from scipy import ndimage, misc
import numba as nb
from config.load_yaml import configs, configs_name
from sklearn.mixture import GaussianMixture
from utils import visualization


def get_packet_events(event_txt, mode='TimeFixed'):
    '''

    :param event_txt: The input events .txt file
    :param mode: 'TimeFixed', equal time interval in each packet
                 'EventNumFixed', equal event numbers in each packet
                 Default is 'TimeFixed'
    :return: packet_events: [packet_1, packet_2, ..., packet_i]
            packet_i: [[timestamp_1, x_1, y_1], [timestamp_2, x_2, y_2], ..., [timestamp_j, x_j, y_j]]
    '''
    if configs_name in ['iros', 'insight']:
        events = np.genfromtxt(event_txt)[:, :3]   # dim:(events_num, 3), 3dim: [timestamp, x, y]
        events[:, 0] = configs['time_transfer'] * (events[:, 0] - events[0, 0])  # transfer timestamp unit to 'us'

    elif configs_name == 'matlab':
        events = np.loadtxt(event_txt, delimiter=',')
        events[:, [0, 1, 2]] = events[:, [2, 1, 0]]
        # events[:, 0] = events[:, 0] - events[0, 0] + 1
        return [events]

    else:
        raise OSError

    if mode == 'TimeFixed':
        pkt_events = []
        win_start = 0
        win_end = configs['packet_time']
        while True:
            if win_end >= events[-1, 0]:
                current_pkt = events[np.where(events[:, 0] >= win_start)]
                pkt_events.append(current_pkt)
                break
            else:
                current_pkt = events[np.where((events[:, 0] >= win_start) & (events[:, 0] < win_end))]
                pkt_events.append(current_pkt)
                win_start += configs['packet_time'] * configs['sliding_time']
                win_end += configs['packet_time'] * configs['sliding_time']

    elif mode == 'EventNumFixed':
        raise NotImplementedError
        # pkt_events = [[] for _ in range(int(len(events)//configs['c_packet_event_num'])+1)]
        # for i, item in enumerate(events):
        #     packet_idx = int(i//configs['c_packet_event_num'])
        #     pkt_events[packet_idx].append(item)

    else:
        pkt_events = None

    return pkt_events


def old_get_packet_events(event_txt):
    '''

    :param event_txt:
    :return: events_coo_list: 每帧图像的事件列表，列表中每个元素为一个二维（dim:[num_event, 2]）数组
             coo_accu_list: 每帧图像的事件列表，附加一个事件出现的次数，列表中每个元素为一个二维（dim:[num_event, 3]）数组
    '''
    # 获取每帧图像中事件的坐标
    events = np.genfromtxt(event_txt)
    events[:, 0] = configs['time_transfer'] * (events[:, 0] - events[0, 0]) // configs['packet_time']
    events_coo_list = []
    coo_accu_list = []

    for i in range(int(events[-1, 0]) + 1):
        idx = np.where(events[:, 0] == i)
        events_coo = events[idx, 1:3].reshape(-1, 2)
        events_coo = numpy_multi_sort(events_coo)
        events_coo_list.append(events_coo)

        # # 对每帧图像的事件进行积分，不论极性，坐标只要出现一次，亮度增加一个常数（255//configs['accu_num']）
        coo_accu = np.array([[events_coo[0,0], events_coo[0,1], 1]])
        for j, coo in enumerate(events_coo, 1):
            if (coo[0] == coo_accu[-1, 0]) and (coo[1] == coo_accu[-1, 1]):
                coo_accu[-1, 2] += 1
            else:
                coo = np.append(coo, 1).reshape(1, 3)
                coo_accu = np.concatenate((coo_accu, coo))
        coo_accu_list.append(coo_accu)

    return events_coo_list, coo_accu_list


def numpy_multi_sort(array):
    value = [tuple(a) for a in array]
    dtyp = [('x', int), ('y', int)]
    array = np.array(value, dtype=dtyp)
    sorted_arr = np.sort(array, order=['x', 'y'])
    sorted_arr = np.array([list(a) for a in sorted_arr])
    return sorted_arr


def denoise(evt_vec, medfilt, pic_type):
    '''

    :param evt_vec:
    :param medfilt:
    :param pic_type: sae, evt_intensity, latest_evt
                     sae: 每个像素记录最新的时间步
                     intensity: 每个像素记录事件累加的个数
                     bool: 每个像素记录是否有事件
    :return:
    '''
    height, width = configs['height'], configs['width']
    pic = np.zeros([height, width])
    for item in evt_vec:
        if pic_type == 'sae':
            pic[int(item[2])][int(item[1])] = item[0]
        elif pic_type == 'intensity':
            pic[int(item[2])][int(item[1])] += 1
        elif pic_type == 'bool':
            pic[int(item[2])][int(item[1])] = 1
        else:
            return AttributeError

    if medfilt:
        pic = ndimage.median_filter(pic, size=3)   # 4ms
        # 闭运算
        pic = ndimage.binary_dilation(pic, iterations=5).astype(int)   # 2.6ms
        evt_vec = get_denoised_evt_vec(pic, height, width)   # 1.5ms

    return evt_vec, pic


@nb.njit
def get_denoised_evt_vec(pic, height, width):
    evt_vec_out = []
    for i in range(height):
        for j in range(width):
            if pic[i][j] != 0:
                evt_vec_out.append([pic[i][j], j, i])
    return evt_vec_out


def cal_gmm_components(evt_xy, n_components=1):
    score_list = []
    for i in range(n_components, n_components+5):
        gmm = GaussianMixture(n_components=i, warm_start=True).fit(evt_xy)
        score_list.append(gmm.bic(evt_xy))

    print(score_list)


def gmm_cluster(evt_path):
    '''
    本方法仅考虑如下情况：
    1. 前后帧类别数不变的
    2. 后帧相比前帧多进入一个物体
    3. 后帧相比前帧退出一个或多个物体
    未考虑的情况：
    1. 有多个物体进入
    2. 同时有物体进入和有物体退出
    :param evt_path:
    :return:
    '''
    evt_fn = os.path.splitext(os.path.basename(evt_path))[0]
    try:
        configs[evt_fn]
    except KeyError:
        print('Not found the provided file')
        raise

    pkt_evts = get_packet_events(event_txt=evt_path)
    pic_list = []
    evt_xy_list = []
    gmm_list = []
    weights_list = []
    means_list = []
    n_components_list = [configs[evt_fn]['init_n_components']]
    nll_list = []
    used_label_list = []   # 为了保持video的时间连续性，实际使用的label
    used_label_unique_list = []
    label_list = []
    label_means_list = []
    init = True

    for i, event_vector in enumerate(pkt_evts):
        if configs[evt_fn]['start_time'] <= i < configs[evt_fn]['end_time']:
            pkt_evt, pic = denoise(event_vector, medfilt=configs['medfilt'], pic_type='bool')
            pkt_evts[i] = pkt_evt
            pic_list.append(pic)
            # misc.imsave(f'{configs["image_dir"]}/{image_fn}_{i}.jpg', pic)

            if len(pkt_evt) >= 80:
                # print('-----------')
                # print(i)
                evt_xy = np.array(pkt_evt)[:, 1:]
                evt_xy[:, 1] = configs['height'] - evt_xy[:, 1]
                evt_xy_list.append(evt_xy)
                current_n_components = n_components_list[-1]

                # 初始化gmm，送入视频的第一帧
                if init:
                    # n_components = event_process.cal_gmm_components()
                    gmm = GaussianMixture(n_components=current_n_components, random_state=0).fit(evt_xy)
                    used_label = gmm.predict(evt_xy)
                    label = gmm.predict(evt_xy)
                    init = False

                # 第二帧及往后
                else:
                    # # 1. 异常值检测, 检测用上一帧图像训练的gmm来predict这一帧图像，若有新事件簇出现，会有较多低概率点
                    # # 实验证明，低概率点并不是新出现点，但有新类出现时，确实会增加许多低概率点
                    label = gmm_list[-1].predict(evt_xy)

                    # entrance 异常值指标
                    nll = -gmm_list[-1].score(evt_xy)

                    # exit 异常值指标
                    each_label_num = np.array([np.sum(label == j) for j in range(n_components_list[-1])])
                    label_zero_num = np.sum(each_label_num == 0)

                    # 如果无异常值, 记录当前的weight和means, 并重新用现有数据fit一个新的gmm
                    if abs(nll - nll_list[-1]) / nll_list[-1] < 0.2 and label_zero_num == 0:
                        gmm = GaussianMixture(n_components=current_n_components,
                                              weights_init=weights_list[-1],
                                              means_init=means_list[-1],
                                              random_state=42).fit(evt_xy)
                        # 调整label，使得同一个物体的当前label与上一帧中的label相同
                        label = gmm.predict(evt_xy)
                        used_label = np.zeros_like(label)

                        for j, c_mean in enumerate(gmm.means_):
                            used_label_idx = np.argmin(
                                [np.linalg.norm(c_mean - p_mean) for p_mean in label_means_list[-1]])
                            used_label[np.where(label == j)] = used_label_unique_list[-1][used_label_idx]

                    else:
                        # 有异常值的情况，即已有事件退出的情况
                        if label_zero_num != 0:
                            current_n_components -= label_zero_num
                            gmm = GaussianMixture(n_components=current_n_components, random_state=42).fit(evt_xy) # 10ms
                            # 调整label，使得同一个物体的当前label与上一帧中的label相同
                            label = gmm.predict(evt_xy)
                            used_label = np.zeros_like(label)
                            for j, c_mean in enumerate(gmm.means_):
                                used_label_idx = np.argmin(
                                    [np.linalg.norm(c_mean - p_mean) for p_mean in label_means_list[-1]])
                                used_label[np.where(label == j)] = used_label_unique_list[-1][used_label_idx]

                        # 有异常值的情况，即新事件entrance的情况, 目前仅支持新进来1个物体，超过1个的物体还没有解决方案
                        if abs(nll - nll_list[-1]) / nll_list[-1] >= 0.1:
                            current_n_components += 1
                            gmm = GaussianMixture(n_components=current_n_components, random_state=42).fit(evt_xy) # 10ms

                            # 调整label，使得同一个物体的当前label与上一帧中的label相同
                            label = gmm.predict(evt_xy)
                            used_label = np.zeros_like(label)

                            # 排除某些小物体突然在某帧画面中突然消失的情况（理论上不用在此处理）
                            def repair_small_obj_lost(lost_frame_num):
                                gmm_ = GaussianMixture(n_components=current_n_components,
                                                      weights_init=weights_list[-(lost_frame_num+1)],
                                                      means_init=means_list[-(lost_frame_num+1)],
                                                      random_state=0).fit(evt_xy)
                                label_ = gmm_.predict(evt_xy)
                                used_label_ = np.zeros_like(label_)
                                for j, c_mean_ in enumerate(gmm_.means_):
                                    used_label_idx_ = np.argmin(
                                        [np.linalg.norm(c_mean_ - p_mean_) for p_mean_ in label_means_list[-(lost_frame_num+1)]])
                                    used_label_[np.where(label_ == j)] = used_label_unique_list[-(lost_frame_num+1)][used_label_idx_]

                                return label_, used_label_

                            if current_n_components == n_components_list[-2]:
                                label, used_label = repair_small_obj_lost(1)
                            elif current_n_components == n_components_list[-3]:
                                label, used_label = repair_small_obj_lost(2)
                            elif current_n_components == n_components_list[-4]:
                                label, used_label = repair_small_obj_lost(3)

                            else:
                                dist_min = []
                                for j, c_mean in enumerate(gmm.means_):
                                    dist = [np.linalg.norm(c_mean - p_mean) for p_mean in label_means_list[-1]]
                                    dist_min.append(np.min(dist))

                                existed_label_idx_list = np.where(dist_min != np.max(dist_min))[0]

                                new_idx_list = []
                                temp_idx = 0
                                for j in range(current_n_components):
                                    if j in existed_label_idx_list:
                                        new_idx = used_label_unique_list[-1][temp_idx]
                                        used_label[np.where(label == j)] = new_idx
                                        temp_idx += 1
                                    else:
                                        new_idx = np.max(used_label_unique_list[-1]) + 1
                                        used_label[np.where(label == j)] = new_idx
                                    new_idx_list.append(new_idx)

                                gmm.weights_[list(range(current_n_components))] = gmm.weights_[new_idx_list]
                                gmm.means_[list(range(current_n_components)), :] = gmm.means_[new_idx_list, :]
                                gmm = GaussianMixture(n_components=current_n_components,
                                                      weights_init=gmm.weights_,
                                                      means_init=gmm.means_,
                                                      random_state=0).fit(evt_xy)
                                label = gmm.predict(evt_xy)
                                used_label = gmm.predict(evt_xy)

                label_means = [np.mean(evt_xy[np.where(used_label == j)], axis=0) for j in np.unique(used_label)]

                gmm_list.append(gmm)
                n_components_list.append(current_n_components)
                weights_list.append(gmm.weights_)
                means_list.append(gmm.means_)
                nll_list.append(-gmm.score(evt_xy))

                used_label_unique_list.append(np.unique(used_label))
                used_label_with_evt = np.concatenate((evt_xy, used_label.reshape(-1, 1)), axis=1)
                used_label_list.append(used_label_with_evt)
                label_with_evt = np.concatenate((evt_xy, label.reshape(-1, 1)), axis=1)
                label_list.append(label_with_evt)
                label_means_list.append(label_means)

                # visualization.plot_gmm(evt_xy, used_label, f'{configs["cmap_dir"]}/{evt_fn}_{i}', means=gmm.means_)

            else:
                used_label_list.append(None)
                label_list.append(None)

    return pic_list, used_label_list
