import cv2
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from config.load_yaml import configs

# label对应的色彩, 三个维度为(r,g,b)
color_map = [[255, 253, 18],   # 镉黄色
             [106, 90, 205],   # 石板蓝
             [64, 224, 208],   # 青绿色
             [128, 128, 105],  # 暖灰色
             [255, 97, 3]]     # 镉橙色


def make_ellipses(gmm, ax):
    colors = ['navy', 'turquoise', 'darkorange']
    # colors = ['navy', 'turquoise']

    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        else:
            raise AttributeError(f'gmm.covariance_type should be '
                                 f'one of ["full", "tied", "diag", "spherical"], but got {gmm.covariance_type}]')
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


def plot_gmm(evt_xy_arr, label, output_path, gmm=None, means=None):
    # 画椭圆
    if gmm is not None:
        h = plt.subplot(1, 1, 1)
        make_ellipses(gmm, h)
    # 画坐标轴范围
    plt.xlim((0, configs['width']))
    plt.ylim((0, configs['height']))
    # 画事件
    plt.scatter(evt_xy_arr[:, 0], evt_xy_arr[:, 1], s=0.8, c=label)
    if means is not None:
        # 画椭圆中心
        colors = ['navy', 'turquoise', 'darkorange']
        for i in range(len(means)):
            plt.scatter(means[i, 0], means[i, 1], marker='x', c=colors[i])
    # 保存
    # plt.show()
    plt.savefig(f'{output_path}.png')
    plt.close('all')


def write_event_video(pic_list, video_path, audio_path=False, video_audio_path=False):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 1e6 / configs['packet_time']
    out = cv2.VideoWriter(video_path, fourcc, fps, (configs['width'], configs['height']))
    for pic in pic_list:
        pic = np.uint8(pic)
        pic[np.where(pic != 0)] = 255
        pic_3c = cv2.merge([pic, pic, pic])
        out.write(pic_3c)
    out.release()

    if audio_path and video_audio_path:
        os.system(f'ffmpeg -i {video_path} -i {audio_path} -c copy {video_audio_path}')


def write_label_video(label_list, label_path, output_image=False, audio_path=False, video_audio_path=False):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 1e6 / configs['packet_time']
    out = cv2.VideoWriter(label_path, fourcc, fps, (configs['width'], configs['height']))
    for i, label in enumerate(label_list):
        img_r = np.zeros((configs['height'], configs['width']), np.uint8)
        img_g = np.zeros((configs['height'], configs['width']), np.uint8)
        img_b = np.zeros((configs['height'], configs['width']), np.uint8)
        if label is not None:
            for item in label:
                img_r[configs['height'] - int(item[1])][int(item[0])] = color_map[item[2]][0]
                img_g[configs['height'] - int(item[1])][int(item[0])] = color_map[item[2]][1]
                img_b[configs['height'] - int(item[1])][int(item[0])] = color_map[item[2]][2]
            img_3c = cv2.merge([img_b, img_g, img_r])
            out.write(img_3c)
            if output_image:
                cv2.imwrite(f'{configs["label_dir"]}/output_{i}.png', img_3c)
        else:
            if output_image:
                img_3c = cv2.merge([img_b, img_g, img_r])
                cv2.imwrite(f'{configs["label_dir"]}/output_{i}.png', img_3c)

    out.release()

    if audio_path and video_audio_path:
        os.system(f'ffmpeg -i {label_path} -i {audio_path} -c copy {video_audio_path}')



