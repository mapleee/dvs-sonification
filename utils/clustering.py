import matplotlib.pyplot as plt
from scipy import misc
import numba as nb
import numpy as np
import tqdm

import time

from sklearn.cluster import KMeans
from config.load_yaml import configs


def kmeans_cluster_optical_flow(evt_vec, kmeans_dir, idx):
    sae = np.zeros([configs['height'], configs['width']])  # sae is short for Surface of Active Events

    # generating SAE, find optical flow
    for item in evt_vec:
        if sae[int(item[2])][int(item[1])] < item[0]:
            sae[int(item[2])][int(item[1])] = item[0]

    diff_row_sae = np.diff(sae, axis=0)
    diff_col_sae = np.diff(sae, axis=1)

    # @nb.jit()
    def calc_opt_flow():
        optical_flow_vec = []
        for i in range(configs['height']-1):
            for j in range(configs['width']-1):
                if (sae[i][j] == 0) or (diff_col_sae[i][j] == 0) or (diff_row_sae[i][j] == 0) \
                        or (diff_row_sae[i][j] == diff_col_sae[i][j]) \
                        or (sae[i+1][j] == 0) or (sae[i][j+1] == 0) \
                        or (abs(diff_col_sae[i][j]) > configs['c_max_delta_t']) \
                        or (abs(diff_row_sae[i][j]) > configs['c_max_delta_t']):
                    continue
                optical_flow_vec.append([i, j, diff_row_sae[i][j], diff_col_sae[i][j]])
        return np.array(optical_flow_vec)

    optical_flow_vec = calc_opt_flow()
    # use kmeans to cluster optical flow vector
    kmeans = KMeans(n_clusters=configs['c_num']).fit(optical_flow_vec[:, -2:])
    label, cluster_center = kmeans.labels_, kmeans.cluster_centers_.T
    # theta:
    # [[y1, y2],
    #  [x1, x2]]
    valid_idx = np.where(cluster_center != 0)
    theta = np.zeros_like(cluster_center)
    theta[valid_idx] = 1/cluster_center[valid_idx]

    # discard outlier
    theta[np.where(abs(theta) > configs['c_max_px_per_unit'])] = 0

    # #################                               ################# #
    # plot Kmeans result use cluster_center, label and optical_flow_vec #
    # #################                               ################# #
    if configs['c_num'] == 2:
        label_0 = np.where(label == 0)
        label_1 = np.where(label == 1)
        plt.scatter(optical_flow_vec[label_0, 2], optical_flow_vec[label_0, 3], s=1, c='r', marker='.')
        plt.scatter(optical_flow_vec[label_1, 2], optical_flow_vec[label_1, 3], s=1, c='b', marker='.')
        plt.scatter(cluster_center[0, :], cluster_center[1, :], s=49, c='black', marker='x')
        # plt.show()
        plt.savefig(f'{kmeans_dir}/kmeans_{idx}.png')
        plt.close('all')

    return theta


def initialize(evt_vec, mode, kmeans_dir, idx):
    '''

    :param evt_vec: [[timestamp_1, x_1, y_1], [timestamp_2, x_2, y_2], ..., [timestamp_j, x_j, y_j]]
    :param plot_fig: bool
    :param mode: origin: initialize theta with zero
                 kmeans: initialize theta with kmeans optical flow
                 defalut is kmeans
           dof: describe 'degree of freedom' of movement
    :return:
    '''

    p = np.ones([len(evt_vec), configs['c_num']])/configs['c_num']
    if mode == 'kmeans':
        theta = kmeans_cluster_optical_flow(evt_vec, kmeans_dir, idx)
    elif mode == 'random':
        theta = np.zeros([configs['c_dof'], configs['c_num']])
    else:
        raise AttributeError(f'mode has only to option: kmeans and random, but got {mode}')

    return theta, p


def generate_iwe(evt_arr, theta, p, ref_time):
    '''

    :param evt_arr: shape: [event_num, 3]   3dim: [timestamp, x, y]
    :param theta: [[y1, y2],
                   [x1, x2]]
    :param p:
    :param ref_time:
    :return:
    '''

    evt_num, evt_dim = evt_arr.shape
    weighted_iwe = np.zeros([configs['height'], configs['width'], configs['c_num']])
    evt_vec_warped = np.zeros([evt_num, evt_dim, configs['c_num']])

    for l in range(configs['c_num']):
        # new_evt_vec: shape:(evt_num, 2)  2dim:[x, y]
        new_evt_vec = warp(ref_time, evt_arr, theta[:, l])

        for k, item in enumerate(new_evt_vec):
            if item[0] <= 0 or item[0] > (configs['width']-2) or item[1] <= 0 or item[1] > (configs['height']-2):
                continue
            evt_vec_warped[k, :, l] = np.append(evt_arr[k][0], item)

            weighted_iwe[int(np.around(item[1])), int(np.around(item[0])), l] += p[k][l]

    return weighted_iwe, evt_vec_warped


def save_iwe(weighted_iwe, iwe_dir, idx, iter_num):
    for j in range(configs['c_num']):
        misc.imsave(f'{iwe_dir}/iwe_{idx}_{iter_num}_{j}.jpg', weighted_iwe[:, :, j])


def save_prob_map(evt_arr, p, prob_map_dir, idx, iter_num):
    for i in range(p.shape[-1]):
        pm = plt.scatter(evt_arr[:, 1], -evt_arr[:, 2], c=p[:, i], cmap='viridis', marker='.')
        plt.colorbar(pm)
        plt.savefig(f'{prob_map_dir}/prob_map_{idx}_{iter_num}_{i}.png')
        plt.close('all')


def warp(ref_time, evt_vec, theta_c):
    '''

    :param ref_time:
    :param evt_vec:
    :param theta_c: theta of current cluster
    :return: new_evt_vec: shape:(evt_num, 2)  2dim:[x, y]
    '''
    new_evt_vec = np.ones([evt_vec.shape[0], 2])   # shape:(evt_num, 2)  2dim:[x, y]
    delta_t = evt_vec[:, 0] - ref_time

    # linear warp, 2 dof
    if theta_c.shape[0] == 2:
        new_evt_vec[:, 0] = evt_vec[:, 1] - theta_c[1]*delta_t
        new_evt_vec[:, 1] = evt_vec[:, 2] - theta_c[0]*delta_t
    # 4 dof
    elif theta_c.shape[0] == 4:
        raise NotImplementedError
    else:
        new_evt_vec = evt_vec

    return new_evt_vec


def update_assignments(weighted_iwe, evt_vec_warped, p):
    '''

    :param weighted_iwe: shape: [HEIGHT, WIDTH, C_NUM]
    :param evt_vec_warped: shape: [evt_num, 3, C_NUM]  3dim: [timestamp, x, y]
    :param p:
    :return:
    '''
    evt_num, evt_dim, c_num = evt_vec_warped.shape
    p_out = p
    for k in range(evt_num):
        sum_c = 0
        p_row_tmp = np.zeros([1, c_num])

        for j in range(c_num):
            x_warped = int(np.around(evt_vec_warped[k][1][j]))
            y_warped = int(np.around(evt_vec_warped[k][2][j]))

            if x_warped == 0 or y_warped == 0:
                continue

            if weighted_iwe[y_warped-1][x_warped-1][j] != 0:
                sum_c += weighted_iwe[y_warped][x_warped][j]
                p_row_tmp[0, j] = weighted_iwe[y_warped][x_warped][j]

        if sum_c != 0:
            p_out[k, :] = p_row_tmp / sum_c

    return p_out


def update_motion_param(weighted_iwe, evt_vec_warped, theta, p, ref_time):
    theta_out = np.zeros_like(theta)

    for l in range(configs['c_num']):
        grad_theta_i = find_grad(evt_vec_warped[:, :, l], weighted_iwe[:, :, l], p[:, l], ref_time)
        theta_out[:, l] = theta[:, l] + configs['c_step_mu'] * grad_theta_i

    return theta_out


def find_grad(evt_vec_warped_l, weighted_iwe_l, p_l, ref_time):
    '''

    :param evt_vec_warped_l: shape: [evt_num, 3], 3dim: [timestamp, x, y]
    :param weighted_iwe_l: shape: [HEIGHT, WIDTH]
    :param p_l:
    :param ref_time:
    :return:
    '''
    height, width, c_dist_thres = configs['height'], configs['width'], configs['c_dist_thres']
    x_min = int(np.around(max([np.min(evt_vec_warped_l[:, 1]), 1])))
    x_max = int(np.around(min([np.max(evt_vec_warped_l[:, 1]), width])))
    y_min = int(np.around(max([np.min(evt_vec_warped_l[:, 2]), 1])))
    y_max = int(np.around(min([np.max(evt_vec_warped_l[:, 2]), height])))
    grad_l_x = np.zeros([height, width])
    grad_l_y = np.zeros([height, width])

    mu_I_l = np.mean(weighted_iwe_l[y_min:y_max, x_min:x_max])

    # x_map, y_map are the x, y coordinates of each pixels
    # take HEIGHT=3, WIDTH=4 for example,
    # x_map is:         y_map is:
    # [[0 0 0 0]        [[0 1 2 3]
    #  [1 1 1 1]         [0 1 2 3]
    #  [2 2 2 2]]        [0 1 2 3]]
    # x_map = np.tile(np.arange(height).reshape(-1, 1), width)
    # y_map = np.tile(np.arange(width), height).reshape(height, width)

    # for k in range(evt_vec_warped_l.shape[0]):
    #     delta_t_k = evt_vec_warped_l[k, 0] - ref_time
    #     dist_x = x_map - evt_vec_warped_l[k, 1]
    #     dist_y = y_map - evt_vec_warped_l[k, 2]
    #     grad_l_x += p_l[k] * find_grad_delta(dist_x) * delta_t_k
    #     grad_l_y += p_l[k] * find_grad_delta(dist_y) * delta_t_k
    #
    # grad_l_x[np.where(np.abs(grad_l_x) > C_DIST_THRES)] = 0
    # grad_l_y[np.where(np.abs(grad_l_y) > C_DIST_THRES)] = 0

    # avg_grad_l_x = np.sum(grad_l_x, axis=1)/((x_max-x_min)*(y_max-y_min))
    # avg_grad_l_y = np.sum(grad_l_y, axis=0)/((x_max-x_min)*(y_max-y_min))

    grad_l_x, grad_l_y, avg_grad_l_x, avg_grad_l_y = cal_grad(x_min, x_max, y_min, y_max, grad_l_x, grad_l_y,
                                                              evt_vec_warped_l, p_l, ref_time, c_dist_thres)

    sum_x = 0
    sum_y = 0
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            sum_x += (weighted_iwe_l[y, x] - mu_I_l) * (grad_l_x[y, x] - avg_grad_l_x)
            sum_y += (weighted_iwe_l[y, x] - mu_I_l) * (grad_l_y[y, x] - avg_grad_l_y)

    grad_theta_i = 2 / ((x_max - x_min) * (y_max - y_min)) * np.array([sum_x, sum_y])

    return grad_theta_i


@nb.njit
def cal_grad(x_min, x_max, y_min, y_max, grad_l_x, grad_l_y, evt_vec_warped_l, p_l, ref_time, c_dist_thres):
    sum_grad_l_x = 0
    sum_grad_l_y = 0
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            for k in range(evt_vec_warped_l.shape[0]):
                delta_t_k = evt_vec_warped_l[k, 0] - ref_time
                dist_x = x - evt_vec_warped_l[k, 1]
                dist_y = y - evt_vec_warped_l[k, 2]
                if abs(dist_x) <= c_dist_thres:
                    grad_l_x[y, x] += p_l[k] * find_grad_delta(dist_x) * delta_t_k
                if abs(dist_y) <= c_dist_thres:
                    grad_l_y[y, x] += p_l[k] * find_grad_delta(dist_y) * delta_t_k

            sum_grad_l_x += grad_l_x[y, x]
            sum_grad_l_y += grad_l_y[y, x]

    avg_grad_l_x = sum_grad_l_x/((x_max-x_min)*(y_max-y_min))
    avg_grad_l_y = sum_grad_l_y/((x_max-x_min)*(y_max-y_min))

    return grad_l_x, grad_l_y, avg_grad_l_x, avg_grad_l_y


@nb.njit
def find_grad_delta(dist):
    '''

    :param dist:
    :return: gradient of approximated Dirac delta function
    '''
    return -dist*dist*dist

