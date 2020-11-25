# 事件相机参数
HEIGHT = 264
WIDTH = 320

# 使用PACKET_TIME做为每帧图像积分的时间段
# 每个PACKET默认为5000us
PACKET_NUM = 5
PACKET_SIZE = 5000
PACKET_TIME = PACKET_NUM * PACKET_SIZE
ACCUMULATE_NUM = 5  # 一个事件最大叠加亮度的次数
FPS = 1000000/PACKET_TIME  # 导出视频的帧率
VELOCITY_THRESHOLD = [0.5, 1, 2]  # 速度分为4个档，分别对应用半音符，四分音符，八分音符和十六分音符的时间长度来演奏
NOTE_PER_FRAME = 5  # 每几帧图像判断是否有一个note

# 音乐参数
OCTAVE_NUM = 2  # 用几个八度的音，默认为2 (目前仅支持2)
SCALE_TYPE = 'Major'  # 音阶的类，默认为大调 （目前仅支持大调）
ROOT_PITCH = 48  # 每个音阶的起始音符，默认为C3
EVENT_NUM_THRESHOLD = 20  # 一个高度范围内能够触发音符所需的事件数
TOTAL_EVENT_THRESHOLD = 60 * PACKET_NUM  # 一张图内总噪声事件数，低于该值证明该帧图像全为噪声

# 系统参数
MODE = 'video'  # 'real_time' or 'video'

# 以下为事件高度差所对应的音高
PITCH_MAJOR = [
    [0],
    [0],
    [0],
    [0],
    [0, 4],
    [0, 4],
    [0, 4],
    [0, 4, 7],
    [0, 4, 7],
    [0, 4, 7],
    [0, 4, 7],
    [0, 4, 7, 11],
    [0, 4, 7, 12],
    [0, 4, 7, 12],
    [0, 4, 7, 12],
    [0, 4, 7, 12],
    [0, 4, 7, 12, 16],
    [0, 4, 7, 12, 16],
    [0, 4, 7, 12, 16],
    [0, 4, 7, 12, 16, 19],
    [0, 4, 7, 12, 16, 19],
    [0, 4, 7, 12, 16, 19],
    [0, 4, 7, 12, 16, 19],
    [0, 4, 7, 12, 16, 19, 23],
]
