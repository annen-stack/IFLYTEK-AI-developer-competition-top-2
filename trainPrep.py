############# 读取数据 ################
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pywt
import numpy
from scipy.interpolate import interp1d
from scipy.fftpack import fft, fftshift, ifft
import csv
import pandas as pd
from sklearn import preprocessing
import os
import shutil

# 生成临时文件夹
os.mkdir('logs')
os.mkdir('plots')
os.mkdir('temp')

warnings.filterwarnings('ignore') # 取消warning

# 读取数据
ECG = sio.loadmat('Data/Train/ECG_Train.mat')
data = ECG['DataTrain'][0][0][0] # 第一个字段：表示ECG采样数据
label = ECG['DataTrain'][0][0][1] # 第二个字段：数据的标签信息
fs = ECG['DataTrain'][0][0][2] # 第三个字段：采样率
fs = fs[0][0]


print('Data:', data.shape)
print('Label:', label.shape)
print('Sample Rate:', fs)

##去噪
def denoise(data):
    """
    :param data: 原始数据
    :return: 去噪平滑后的数据
    """
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata




# ECG数据生成 .csv 文件
def ECG2csv(ecgData, fileName, cropLen=None):
    """
    ecgData: 某患者的某个情绪对应的ECG数据
    cropLen: 设定的样本截取长度，若为None则表示只输出一个样本 .csv 文件；否则每 cropLen 长度剪出一段样本 .csv 文件
    fileName: 输出的.csv文件的文件名，若 cropLen 不为 None，则会以 fileName 为基础生成多个文件
    partsNum: 分割出来的 .csv 文件的个数
    """
    header1 = 'number'
    header2 = 'ecg_measurement'
    res = []
    ecg = []
    for i in ecgData:
        a = [i]
        res.append(a)
    ecg = res

    partsNum = 1
    if cropLen:
        partsNum = len(ecgData[::cropLen])
        for i in range(partsNum):
            cropData = ecgData[i * cropLen:(i + 1) * cropLen] if i != partsNum else ecgData[i * cropLen:(i + 1):]
            temp = np.array([[row] for row in range(1, 1 + len(cropData))])
            f = open(fileName + '_' + str(i) + '.csv', "w", newline="")
            output = "%s,%s\n" % (header1, header2)
            f.write(output)

            for j in range(len(cropData)):
                output = "%d,%10.6f\n" % (temp[j], cropData[j])
                f.write(output)

    else:
        temp = np.array([[row] for row in range(1, 1 + len(ecg))])
        f = open(fileName, "w", newline="")
        output = "%s,%s\n" % (header1, header2)
        f.write(output)
        for i in range(len(ecg)):
            output = "%d,%10.6f\n" % (temp[i], ecg[i][0])
            f.write(output)
    return partsNum


################ PQRST检测 ########################
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.signal import butter, lfilter

LOG_DIR = "logs/"
PLOT_DIR = "plots/"
class QRSDetectorOffline(object):
    """
    基于Pan-Tomkins算法的心电QRS检测
    """
    def __init__(self, ecg_data_path, log_data=False, plot_data=False, show_plot=False):
        """
        :param ecg_data_path:到ECG数据集的路径
        :param log_data:用于记录结果的标志
        :param plot_data:用于将将结果绘制文件中的标志
        :param show_plot:用于显示生成的结果图的标志
        """
        # 配置参数。
        self.ecg_data_path = ecg_data_path

        self.signal_frequency = 64  # 在这里设置心电图设备频率，每秒fs采样。

        self.filter_lowcut = 0.001
        self.filter_highcut = 5
        self.filter_order = 1

        self.integration_window = 15  # 频率按比例变化(在样本中)。

        self.findpeaks_limit = 0
        self.findpeaks_spacing = 10  #

        self.refractory_period = 120  # 频率按比例变化(在样本中)。
        self.qrs_peak_filtering_factor = 0.125
        self.noise_peak_filtering_factor = 0.125
        self.qrs_noise_diff_weight = 0.25

        # 心电图数据加载。
        self.ecg_data_raw = None

        # 测量值和计算值。
        self.filtered_ecg_measurements = None
        self.differentiated_ecg_measurements = None
        self.squared_ecg_measurements = None
        self.integrated_ecg_measurements = None
        self.detected_peakss_indices = None
        self.detected_peakss_values = None
        self.detected_peakr_indices = None
        self.detected_peakr_values = None
        self.detected_peakt_indices = None
        self.detected_peakt_values = None
        self.detected_peakp_indices = None
        self.detected_peakp_values = None
        self.detected_peakq_indices = None
        self.detected_peakq_values = None
        self.detected_peaks_indices = None
        self.detected_peaks_values = None

        self.qrs_peak_value = 0.0
        self.noise_peak_value = 0.0
        self.threshold_value = 0.0

        # 检测结果。
        self.qrs_peaks_indices = np.array([], dtype=int)
        self.noise_peaks_indices = np.array([], dtype=int)

        # 最终ECG数据和QRS检测结果检测到QRS的阵列样本标记为1。
        self.ecg_data_detected = None

        # 运行整个检测器流。
        self.load_ecg_data()
        self.detect_peaks()
        self.detect_qrs()
        self.findR()
        self.findS()
        self.findT()
        self.findP()
        self.findQ()

        if log_data:
            self.log_path = "{:s}QRS_offline_detector_log_{:s}.csv".format(LOG_DIR,
                                                                           strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            self.log_detection_data()

        if plot_data:
            self.plot_path = "{:s}QRS_offline_detector_plot_{:s}.png".format(PLOT_DIR,
                                                                             strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            self.plot_detection_data(show_plot=show_plot)

    def print_peakr_indices(self):

        return self.detected_peaks_indices, self.detected_peakr_indices, self.detected_peakt_indices, self.detected_peakp_indices, self.detected_peakq_indices

    def print_data(self):
        return self.ecg_data_raw[:, 1]

    """负荷心电图测量数据方法"""

    def load_ecg_data(self):

        self.ecg_data_raw = np.loadtxt(self.ecg_data_path, skiprows=1, delimiter=',')

    """心电图测量数据处理方法"""

    def detect_peaks(self):
        """
        负责通过测量处理从加载的心电测量数据中提取峰值的方法。
        """
        # 从加载的心电图数据中提取测量值
        ecg_measurements = self.ecg_data_raw[:, 1]

        # 测量滤波- 0- 15hz带通滤波器
        self.filtered_ecg_measurements = self.bandpass_filter(ecg_measurements, lowcut=self.filter_lowcut,
                                                              highcut=self.filter_highcut,
                                                              signal_freq=self.signal_frequency,
                                                              filter_order=self.filter_order)
        self.filtered_ecg_measurements[:5] = self.filtered_ecg_measurements[5]

        # 导数-提供QRS斜率信息
        self.differentiated_ecg_measurements = np.ediff1d(self.filtered_ecg_measurements)

        # 平方-加强在导数接收的值
        self.squared_ecg_measurements = self.differentiated_ecg_measurements ** 2

        # 移动窗积分
        self.integrated_ecg_measurements = np.convolve(self.squared_ecg_measurements, np.ones(self.integration_window))

        # 综合测量的基准标记-峰值检测
        self.detected_peakss_indices = self.findpeaks(data=self.integrated_ecg_measurements,
                                                     limit=self.findpeaks_limit,
                                                     spacing=self.findpeaks_spacing)




        self.detected_peakss_values = self.integrated_ecg_measurements[self.detected_peakss_indices]

    """QRS检测方法"""

    def detect_qrs(self):
        """
        负责将检测到的心电图测量峰值分类为噪声或QRS波(心跳)的方法。
        """
        for detected_peak_index, detected_peakss_value in zip(self.detected_peakss_indices, self.detected_peakss_values):

            try:
                last_qrs_index = self.qrs_peaks_indices[-1]
            except IndexError:
                last_qrs_index = 0

            # 在一个有效的QRS复合检测后，有200 ms的不应期才能检测到下一个
            if detected_peak_index - last_qrs_index > self.refractory_period or not self.qrs_peaks_indices.size:
                # 峰值必须分为噪声峰值或QRS峰值
                # 要归类为QRS峰值，它必须超过动态设置的阈值
                if detected_peakss_value > self.threshold_value:
                    self.qrs_peaks_indices = np.append(self.qrs_peaks_indices, detected_peak_index)

                    # 调整QRS峰值，稍后用于设置QRS噪声阈值
                    self.qrs_peak_value = self.qrs_peak_filtering_factor * detected_peakss_value + \
                                          (1 - self.qrs_peak_filtering_factor) * self.qrs_peak_value
                else:
                    self.noise_peaks_indices = np.append(self.noise_peaks_indices, detected_peak_index)

                    # 调整噪声峰值，稍后用于设置qrs噪声阈值
                    self.noise_peak_value = self.noise_peak_filtering_factor * detected_peakss_value + \
                                            (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

                # 根据先前检测到的QRS或噪声峰值值调整QRS-噪声阈值
                self.threshold_value = self.noise_peak_value + \
                                       self.qrs_noise_diff_weight * (self.qrs_peak_value - self.noise_peak_value)

        # 创建包含输入心电图测量数据和QRS检测指示列的阵列
        # 我们在“qrs_detected”日志列中用“1”标记QRS检测(否则为“0”)
        measurement_qrs_detection_flag = np.zeros([len(self.ecg_data_raw[:, 1]), 1])
        measurement_qrs_detection_flag[self.qrs_peaks_indices] = 1

        self.ecg_data_detected = np.append(self.ecg_data_raw, measurement_qrs_detection_flag, 1)

    """R点检测方法"""

    def findR(self):
        l = self.detected_peakss_indices
        r_ind = []
        index = 0
        # r点为QRS检测周期中的峰值点
        for i in range(len(l) - 1):
            m = 0
            for j in range(l[i], l[i + 1]):
                if self.ecg_data_detected[j, 1] > m:
                    index = j
                    m = self.ecg_data_detected[j, 1]
            r_ind.append(index)  # 记录R点坐标

        r_max = np.max(self.ecg_data_detected[r_ind, 1])
        r_min = np.min(self.ecg_data_detected[r_ind, 1])
        r_ind_0 = []
        # 检测QRS第一个周期前的r点
        for i in range(l[0]):
            if r_max + 2 > self.ecg_data_detected[i, 1] > r_min - 2:
                r_ind_0.append(i)
        if r_ind_0:
            re = r_ind_0[0]
            for j in r_ind_0:
                if self.ecg_data_detected[j, 1] > self.ecg_data_detected[re, 1]:
                    re = j
            r_ind.insert(0, re)

        r_value = []
        for i in r_ind:
            r_value.append(self.ecg_data_raw[i, 1])

        self.detected_peakr_indices = r_ind
        return self.detected_peakr_indices

    """S点检测方法"""

    def findS(self):
        global j
        r = self.detected_peakr_indices
        s_ind = []
        # 每个S点为每个R点之后的第一个谷值点
        for i in range(len(r) - 1):
            for j in range(r[i], r[i + 1]):
                if self.ecg_data_raw[j, 1] < self.ecg_data_raw[j + 1, 1] - 1:
                    break
            s_ind.append(j)

        self.detected_peaks_indices = s_ind
        s_value = []
        for i in s_ind:
            s_value.append(self.ecg_data_raw[i, 1])

        return self.detected_peaks_indices

    """T点检测方法"""

    def findT(self):
        r = self.detected_peakr_indices
        s = self.detected_peaks_indices
        t_ind = []
        index = 0
        D = []
        # T点为S点到R-R间距的50%的最大值
        for i in range(len(r) - 1):
            m = 0
            d = r[i + 1] - r[i]
            D.append(d)
            for j in range(s[i], r[i] + int(0.5 * d)):
                if self.ecg_data_detected[j, 1] > m:
                    index = j
                    m = self.ecg_data_detected[j, 1]
            t_ind.append(index)
        # 满足以下条件，检测出是否存在QRS检测周期之外的T点
        m = 0
        if len(r) == len(s) and r[len(r) - 1] + int(0.5 * min(D)) < 320:
            for j in range(s[len(r) - 1], 320):
                # T点为该区间的峰峰值
                if self.ecg_data_detected[j, 1] > m:
                    index = j
                    m = self.ecg_data_detected[j, 1]
            t_ind.append(index)

        self.detected_peakt_indices = t_ind
        t_value = []
        for i in t_ind:
            t_value.append(self.ecg_data_raw[i, 1])
        return self.detected_peakt_indices

    """P点检测方法"""

    def findP(self):
        # P点为R-R间距75%-83.3%的最大值
        r = self.detected_peakr_indices
        p_ind = []
        index = 0
        D = []
        for i in range(len(r) - 1):
            m = 0
            d = r[i + 1] - r[i]
            D.append(d)
            for j in range(r[i] + int(0.75 * d), r[i] + int(0.833 * d)):
                if self.ecg_data_detected[j, 1] > m:
                    index = j
                    m = self.ecg_data_detected[j, 1]
            p_ind.append(index)

        # 满足以下条件，检测出是否存在QRS检测周期之外的P点
        if D:
            m = 0
            if r[len(r) - 1] + 0.833 * min(D) < 230:  # 最后一个r点之后的区间
                for j in range(r[len(r) - 1] + int(0.75 * min(D)), r[len(r) - 1] + int(0.833 * min(D))):
                    if self.ecg_data_detected[j, 1] > m:
                        # P点为该区间的峰值
                        index = j
                        m = self.ecg_data_detected[j, 1]
                p_ind.append(index)

            m = 0
            if min(D) * 0.25 < r[0]:  # 第一个r点之前的区间
                for j in range(r[0] - int(0.25 * min(D)), r[0] - int(0.17 * min(D))):
                    if self.ecg_data_detected[j, 1] > m:
                        # T点为该区间的峰值
                        index = j
                        m = self.ecg_data_detected[j, 1]
                p_ind.insert(0, index)

        self.detected_peakp_indices = p_ind
        return self.detected_peakp_indices

    """Q点检测方法"""

    def findQ(self):
        r = self.detected_peakr_indices
        q_ind = []
        index = 0
        # Q点为R点之前的第一个谷点
        for i in range(len(r)):
            for j in range(r[i] - 1, 0, -1):
                if self.ecg_data_detected[j, 1] < self.ecg_data_detected[j + 1, 1]:
                    index = j
                else:
                    index = j + 1
                    break
            q_ind.append(index)

        self.detected_peakq_indices = q_ind

        q_value = []
        for i in q_ind:
            q_value.append(self.ecg_data_raw[i, 1])

        return self.detected_peakq_indices


    def log_detection_data(self):
        """
        方法负责将测得的心电图和检测结果记录成文件
        """
        with open(self.log_path, "wb") as fin:
            fin.write(b"timestamp,ecg_measurement,qrs_detected\n")
            np.savetxt(fin, self.ecg_data_detected, delimiter=",")

    def plot_detection_data(self, show_plot=False):
        """
        Method 负责绘制检测结果.
        :param bool show_plot: 标记用于绘制结果和显示图形
        """

        def plot_data(axis, data, title='', fontsize=10):
            axis.set_title(title, fontsize=fontsize)
            axis.grid(which='both', axis='both', linestyle='--')
            axis.plot(data, color="salmon", zorder=1)

        def plot_points(axis, values, indices):
            axis.scatter(x=indices, y=values[indices], c="black", s=50, zorder=2)

       #  绘制图像
       #  plt.close('all')
       #  fig, axarr = plt.subplots(6, sharex=True, figsize=(15, 18))
       #
       #  plot_data(axis=axarr[0], data=self.ecg_data_raw[:, 1], title='Raw ECG measurements')
       #  plot_data(axis=axarr[1], data=self.filtered_ecg_measurements, title='Filtered ECG measurements')
       #  plot_data(axis=axarr[2], data=self.differentiated_ecg_measurements, title='Differentiated ECG measurements')
       #  plot_data(axis=axarr[3], data=self.squared_ecg_measurements, title='Squared ECG measurements')
       #  plot_data(axis=axarr[4], data=self.integrated_ecg_measurements,
       #            title='Integrated ECG measurements with QRS peaks marked (black)')
       #  plot_points(axis=axarr[4], values=self.integrated_ecg_measurements, indices=self.detected_peakss_indices)
       #  plot_data(axis=axarr[5], data=self.ecg_data_raw[:, 1],
       #            title='Raw ECG measurements with QRS peaks marked (black)')
       #  plot_points(axis=axarr[5], values=self.ecg_data_raw[:, 1], indices=self.detected_peaks_indices)
       #  plot_points(axis=axarr[5], values=self.ecg_data_raw[:, 1], indices=self.detected_peakr_indices)
       #  plot_points(axis=axarr[5], values=self.ecg_data_raw[:, 1], indices=self.detected_peakp_indices)
       #  plot_points(axis=axarr[5], values=self.ecg_data_raw[:, 1], indices=self.detected_peakt_indices)
       #  plot_points(axis=axarr[5], values=self.ecg_data_raw[:, 1], indices=self.detected_peakq_indices)
       #
       #  plt.tight_layout()
       # # plt.show()
       #  fig.savefig(self.plot_path)
       #
       #  if show_plot:
       #      plt.show()
       #
       #  plt.close()



    def bandpass_filter(self, data, lowcut, highcut, signal_freq, filter_order):
        """
        :param data:原始数据
        :param lowcut:滤波器低切割频率值
        :param highcut:滤波高频值
        :param signal_freq:信号频率为每秒采样数
        :param filter_order:滤波器阶数
        :return:过滤后的数据
        """
        nyquist_freq = 0.6 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

    def findpeaks(self, data, spacing=1, limit=None):
        """
        :param data:输入数据
        :param spacing:到下一个峰值的最小间距(应该是1或更多)
        :param limit:峰值的值应该大于或等于
        :return:检测峰值索引数组
        """
        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)

        if limit is not None:
            ind = ind[data[ind] > limit]

        return ind



################ 特征提取 ########################
columns = ['ecgS-Mean', 'ecgS-Median', 'ecgS-Std', 'ecgS-Max', 'ecgS-Min',
       'ecgS-Range', 'ecgT-Mean', 'ecgT-Median', 'ecgT-Std', 'ecgT-Max',
       'ecgT-Min', 'ecgT-Range', 'ecgP-Mean', 'ecgP-Median', 'ecgP-Std',
       'ecgP-Max', 'ecgP-Min', 'ecgP-Range', 'ecgQ-Mean', 'ecgQ-Median',
       'ecgQ-Std', 'ecgQ-Max', 'ecgQ-Min', 'ecgQ-Range', 'ecgR-Mean',
       'ecgR-Median', 'ecgR-Std', 'ecgR-Max', 'ecgR-Min', 'ecgR-Range',
       'ecgPQ-Mean', 'ecgPQ-Median', 'ecgPQ-Std', 'ecgPQ-Max', 'ecgPQ-Min',
       'ecgPQ-Range', 'ecgQS-Mean', 'ecgQS-Median', 'ecgQS-Std', 'ecgQS-Max',
       'ecgQS-Min', 'ecgQS-Range', 'ecgST-Mean', 'ecgST-Median', 'ecgST-Std',
       'ecgST-Max', 'ecgST-Min', 'ecgST-Range', 'ecgRR-Mean', 'ecgRR-Median',
       'ecgRR-Std', 'ecgRR-Max', 'ecgRR-Min', 'ecgRR-Range', 'ecgHR-Mean',
       'ecgHR-Median', 'ecgHR-Std', 'ecgHR-Max', 'ecgHR-Min', 'ecgHR-Range',
       'ecgSDNN-Mean', 'ecgSDNN-Median', 'ecgSDNN-Std', 'ecgSDNN-Max',
       'ecgSDNN-Min', 'ecgSDNN-Range', 'ecgVF_Min', 'ecgVF_Max', 'ecgVF_Range',
       'ecgPF_v', 'ecgLF_Mean', 'ecgLF_Std', 'ecgLF_Median', 'ecgLF_Min',
       'ecgLF_Max', 'ecgLF_Range', 'ecgLF_u', 'ecgPF_l', 'ecgHF_Mean',
       'ecgHF_Std', 'ecgHF_Median', 'ecgHF_Min', 'ecgHF_Max', 'ecgHF_Range',
       'ecgHF_u', 'ecgPF_h', 'ecg_LFHF', 'ecgpample_mean', 'ecgpample_median',
       'ecgpample_std', 'ecgpample_max', 'ecgpample_min', 'ecgpample_range',
       'ecgqample_mean', 'ecgqample_median', 'ecgqample_std', 'ecgqample_max',
       'ecgqample_min', 'ecgqample_range', 'ecgrample_mean','ecgrample_median',
       'ecgrample_std', 'ecgrample_max', 'ecgrample_min',
       'ecgrample_range', 'ecgsample_mean', 'ecgsample_median',
       'ecgsample_std', 'ecgsample_max', 'ecgsample_min', 'ecgsample_range',
       'ecgtample_mean', 'ecgtample_median', 'ecgtample_std', 'ecgtample_max',
       'ecgtample_min', 'ecgtample_range', 'ecgtdown_mean', 'ecgtdown_median',
       'ecgtdown_std', 'ecgptdown_mean', 'ecgptdown_median', 'ecgptdown_std',
       'ecgabsdifftample_median', 'ecgdifframple_median', 'ecgdifframple_std',
       'ecgabsdifframple_mean ', 'ecgabsdifframple_median',
       'ecgabsdifframple_std', 'ecgdifftample_mean', 'ecgdifftample_median',
       'ecgdifftample_std', 'ecgabsdifftample_mean',
       'ecgabsdifftample_median.1', 'ecgabsdifftample_std', 'ecgpenegry_mean',
       'ecgpenegry_median', 'ecgpenegry_std', 'ecgtenegry_mean',
       'ecgtenegry_median', 'ecgtenegry_std', 'ecgqenegry_mean',
       'ecgqenegry_median', 'ecgqenegry_std ', 'rateupamp_mean',
       'rateupamp_median', 'rateupamp_std', 'rateupamp_min', 'rateupamp_max',
       'rateuptime_mean', 'rateuptime_median', 'rateuptime_std',
       'rateuptime_max', 'rateuptime_min', 'rateupr_mean', 'rateupr_median',
       'rateupr_std', 'rateupr_min', 'rateupr_max', 'rateampdiff_std',
       'rateampdiff_mean', 'nn_range', 'rateamp_std',
       'rateamp_std/rateampdiff_std', 'ratedownamp_mean', 'ratedownamp_median',
       'ratedownamp_std ', 'ratedownamp_min', 'ratedownamp_max',
       'ratedownr_mean', 'ratedownr_median', 'ratedownr_std', 'ratedownr_min',
       'ratedownr_max', 'label']
feature_total = []
for m in range(len(data)):#被试个数
    for n in range(len(data[0])):#情绪类别
        a = ECG2csv(data[m][n], 'temp/myTemp',320)#将心电图数据划分为320的小样本数据
        for i in range(a-1):
            feature = []
            # 检测每个小样本中的PQRST点
            qrs_detector = QRSDetectorOffline(ecg_data_path='temp/myTemp_'+str(i)+'.csv',
                                      log_data=True, plot_data=True, show_plot=False)
            s_indices, r_indices, t_indices, p_indices, q_indices = qrs_detector.print_peakr_indices()

            d = qrs_detector.print_data()
            """
            心电信号特征值选取
            """
            # 时域分析
            # 以所定位到的相邻P,Q,R,S,T这五种关键点的为基础，分别计算每个人每种情感状态下每一种关键点的Mean（平均值），Median（中间值），Std（标准差），MIN（最小值），MAX（最大值），以及Range
            # S点
            s = []
            for i in range(len(s_indices) - 1):
                s.append(s_indices[i + 1] - s_indices[i])
            ecgs_mean = np.mean(s)
            feature.append(ecgs_mean)
            ecgs_median = np.median(s)
            feature.append(ecgs_median)
            ecgs_std = np.std(s)
            feature.append(ecgs_std)
            if not s:
                ecgs_max = 0
                ecgs_min = 0
            else:
                ecgs_max = max(s)
                ecgs_min = min(s)
            feature.append(ecgs_max)
            feature.append(ecgs_min)
            ecgs_range = ecgs_max - ecgs_min
            feature.append(ecgs_range)

            # T点
            t = []
            for i in range(len(t_indices) - 1):
                t.append(t_indices[i + 1] - t_indices[i])
            ecgt_mean = np.mean(t)
            feature.append(ecgt_mean)
            ecgt_median = np.median(t)
            feature.append(ecgt_median)
            ecgt_std = np.std(t)
            feature.append(ecgt_std)
            if not t:
                ecgt_max = 0
                ecgt_min = 0
            else:
                ecgt_max = max(t)
                ecgt_min = min(t)
            feature.append(ecgt_max)
            feature.append(ecgt_min)
            ecgt_range = ecgt_max - ecgt_min
            feature.append(ecgt_range)

            # P点
            p = []
            for i in range(len(p_indices) - 1):
                p.append(p_indices[i + 1] - p_indices[i])
            ecgp_mean = np.mean(p)
            feature.append(ecgp_mean)
            ecgp_median = np.median(p)
            feature.append(ecgp_median)
            ecgp_std = np.std(p)
            feature.append(ecgp_std)
            if not p:
                ecgp_max = 0
                ecgp_min = 0
            else:
                ecgp_max = max(p)
                ecgp_min = min(p)
            feature.append(ecgp_max)
            feature.append(ecgt_min)
            ecgp_range = ecgp_max - ecgp_min
            feature.append(ecgp_range)

            # Q点
            q = []
            for i in range(len(q_indices) - 1):
                q.append(q_indices[i + 1] - q_indices[i])
            ecgq_mean = np.mean(q)
            feature.append(ecgq_mean)
            ecgq_median = np.median(q)
            feature.append(ecgq_median)
            ecgq_std = np.std(q)
            feature.append(ecgq_std)
            if not q:
                ecgq_max = 0
                ecgq_min = 0
            else:
                ecgq_max = max(q)
                ecgq_min = min(q)
            feature.append(ecgq_max)
            feature.append(ecgq_min)
            ecgq_range = ecgq_max - ecgq_min
            feature.append(ecgq_range)

            # R点
            r = []
            for i in range(len(r_indices) - 1):
                r.append(r_indices[i + 1] - r_indices[i])
            ecgr_mean = np.mean(r)
            feature.append(ecgr_mean)
            ecgr_median = np.median(r)
            feature.append(ecgr_median)
            ecgr_std = np.std(r)
            feature.append(ecgr_std)
            if not r:
                ecgr_max = 0
                ecgr_min = 0
            else:
                ecgr_max = max(r)
                ecgr_min = min(r)
            feature.append(ecgr_max)
            feature.append(ecgr_min)
            ecgr_range = ecgr_max - ecgr_min
            feature.append(ecgr_range)

            # 以P-Q、Q-S、S-T间隔为基础，分别计算的Mean（平均值），Median（中间值），Std（标准差），MIN（最小值），MAX（最大值），以及Range
            # PQ间隔
            pq = []
            if len(q_indices) == len(p_indices) and p_indices[0] < q_indices[0]:
                for i in range(len(q_indices)):
                    pq.append(q_indices[i] - p_indices[i])
            if len(q_indices) == len(p_indices) and p_indices[0] > q_indices[0]:
                for i in range(len(p_indices) - 1):
                    pq.append(q_indices[i + 1] - p_indices[i])
            if len(q_indices) == len(p_indices) + 1:
                for i in range(len(p_indices)):
                    pq.append(q_indices[i + 1] - p_indices[i])
            if len(q_indices) + 1 == len(p_indices):
                for i in range(len(q_indices)):
                    pq.append(q_indices[i] - p_indices[i])

            # QS间隔
            qs = []
            if len(q_indices) == len(s_indices):
                for i in range(len(q_indices)):
                    qs.append(s_indices[i] - q_indices[i])
            if len(q_indices) == len(s_indices) + 1:
                for i in range(len(s_indices)):
                    qs.append(s_indices[i] - q_indices[i])
            if len(q_indices) + 1 == len(s_indices):
                for i in range(len(q_indices)):
                    qs.append(s_indices[i + 1] - q_indices[i])

            # ST间隔
            st = []
            if len(s_indices) == len(t_indices):
                for i in range(len(t_indices)):
                    st.append(t_indices[i] - s_indices[i])
            if len(s_indices) == len(t_indices) + 1:
                for i in range(len(t_indices)):
                    st.append(t_indices[i] - s_indices[i])

            ecgpq_mean = np.mean(pq)
            feature.append(ecgpq_mean)
            ecgpq_median = np.median(pq)
            feature.append(ecgpq_median)
            ecgpq_std = np.std(pq)
            feature.append(ecgpq_std)
            if not pq:
                ecgpq_max = 0
                ecgpq_min = 0
            else:
                ecgpq_max = max(pq)
                ecgpq_min = min(pq)
            feature.append(ecgpq_max)
            feature.append(ecgpq_min)
            ecgpq_range = ecgpq_max - ecgpq_min
            feature.append(ecgpq_range)

            ecgqs_mean = np.mean(qs)
            feature.append(ecgqs_mean)
            ecgqs_median = np.median(qs)
            feature.append(ecgqs_median)
            ecgqs_std = np.std(qs)
            feature.append(ecgqs_std)
            if not qs:
                ecgqs_max = 0
                ecgqs_min = 0
            else:
                ecgqs_max = max(qs)
                ecgqs_min = min(qs)
            feature.append(ecgqs_max)
            feature.append(ecgqs_min)
            ecgqs_range = ecgqs_max - ecgqs_min
            feature.append(ecgqs_range)

            ecgst_mean = np.mean(st)
            feature.append(ecgst_mean)
            ecgst_median = np.median(st)
            feature.append(ecgst_median)
            ecgst_std = np.std(st)
            feature.append(ecgst_std)
            if not st:
                ecgst_max = 0
                ecgst_min = 0
            else:
                ecgst_max = max(st)
                ecgst_min = min(st)
            feature.append(ecgst_max)
            feature.append(ecgst_min)
            ecgst_range = ecgst_max - ecgst_min
            feature.append(ecgst_range)

            # 以R-R间期为基础计算HRV（心率变异率）时域下的特征值
            rr = []
            for i in range(len(r_indices) - 1):
                rr.append(r_indices[i + 1] - r_indices[i])  # R-R间距

            ecgrr_mean = np.mean(rr)
            feature.append(ecgrr_mean)
            ecgrr_median = np.median(rr)
            feature.append(ecgrr_median)
            ecgrr_std = np.std(rr)
            feature.append(ecgrr_std)
            if not rr:
                ecgrr_max = 0
            else:
                ecgrr_max = max(rr)
            feature.append(ecgrr_max)
            if not rr:
                ecgrr_min = 0
            else:
                ecgrr_min = min(rr)
            feature.append(ecgrr_min)
            ecgrr_range = ecgrr_max - ecgrr_min
            feature.append(ecgrr_range)

            # 以所有心率为基础的统计特征值hr
            hr = d / 60  # min
            ecghr_mean = np.mean(hr)
            feature.append(ecghr_mean)
            ecghr_median = np.median(hr)
            feature.append(ecghr_median)
            ecghr_std = np.std(hr)
            feature.append(ecghr_std)
            if not hr.all():
                ecghr_min = 0
                ecghr_max = 0
            else:
                ecghr_min = min(hr)
                ecghr_max = max(hr)
            feature.append(ecghr_max)
            feature.append(ecghr_min)
            ecghr_range = ecghr_max - ecghr_min
            feature.append(ecghr_range)

            # 以各个相邻RR间隔的差值的均方根为基础的统计特征值
            sd = []
            sdnn = []
            for i in range(len(rr) - 1):
                sd.append(abs(rr[i + 1] - rr[i]))
            for i in range(len(sd) - 1):
                sdnn.append(np.sqrt((sd[i] ** 2 + sd[i + 1] ** 2) / 2))
            ecgsdnn_mean = np.mean(sdnn)
            feature.append(ecgsdnn_mean)
            ecgsdnn_median = np.median(sdnn)
            feature.append(ecgsdnn_median)
            ecgsdnn_std = np.std(sdnn)
            feature.append(ecgsdnn_std)
            if not sdnn:
                ecgsdnn_max = 0
            else:
                ecgsdnn_max = max(sdnn)
            feature.append(ecgsdnn_max)
            if not sdnn:
                ecgsdnn_min = 0
            else:
                ecgsdnn_min = min(sdnn)
            feature.append(ecgsdnn_min)
            ecgsdnn_range = ecgsdnn_max - ecgsdnn_min
            feature.append(ecgsdnn_range)

            # 频域分析
            # HRV的频域分析是从心电图的RR间隔信号中提取频域方面的参数如峰值的频率带内功率等特征点进行统计
            r_indices_t = []
            for i in range(len(r_indices) - 1):
                r_indices_t.append(r_indices[i + 1] - r_indices[i])  # R-R间距

            if len(r_indices) > 1:
                # 首先对心电图RR间隔的信号进行重采样
                r_indices_x = r_indices[1:]
                # 利用一维插值函数将最初的RR信号转换为均匀采样的时间序列
                x = np.linspace(r_indices_x[0], r_indices_x[-1], r_indices_x[-1] - r_indices_x[0])
                if len(q_indices) == 1 or len(q_indices) == 0 or len(q_indices) == 2:
                    feature.extend([0] * 21)  # 检测不到特征点
                else:
                    f = interp1d(r_indices_x, r_indices_t, kind="linear")
                    if f(x) is not np.empty:
                        feature.extend([0] * 21)
                    else:
                        # 利用基于FFT的频谱计算方法进行频谱密度的估计
                        num_fft = 1024
                        fs = 64
                        Y = np.abs(fft(f(x), num_fft))
                        print(len(Y))
                        lsg = len(Y)
                        w = [i * fs / lsg for i in range(lsg)]
                        print(len(w))
                        # 功率谱
                        ps = Y ** 2 / num_fft
                        # 使用相关功率谱
                        cor_x = np.correlate(f(x), f(x), 'same')
                        cor_X = fft(cor_x, num_fft)
                        ps_cor = np.abs(cor_X)
                        ps_cor = ps_cor / np.max(ps_cor)

                        # 0–0.4 Hz 频段的功率
                        vlhf = ps_cor[:int(0.4 * lsg / fs) + 1]

                        # 0–0.04 Hz 频段的功率
                        vf = ps_cor[:int(0.04 * lsg / fs) + 1]
                        ecgvf_min = min(vf)
                        feature.append(ecgvf_min)
                        ecgvf_max = max(vf)
                        feature.append(ecgvf_max)
                        ecgvf_range = ecgvf_max - ecgvf_min
                        feature.append(ecgvf_range)
                        if ecgvf_max not in vf.tolist():
                            ecgpf_v = np.nan
                        else:
                            ecgpf_v = vf.tolist().index(ecgvf_max) * fs / lsg
                        feature.append(ecgpf_v)

                        # 以 0.04–0.15 Hz 频段的功率为基础的统计特征值
                        lf = ps_cor[int(0.04 * lsg / fs):int(0.15 * lsg / fs)]
                        ecglf_mean = np.mean(lf)
                        feature.append(ecglf_mean)
                        ecglf_std = np.std(lf)
                        feature.append(ecglf_std)
                        ecglf_median = np.median(lf)
                        feature.append(ecglf_median)
                        ecglf_min = min(lf)
                        feature.append(ecglf_min)
                        ecglf_max = max(lf)
                        feature.append(ecglf_max)
                        ecglf_range = ecglf_max - ecglf_min
                        feature.append(ecglf_range)
                        ecglf_u = sum(lf) / (sum(vlhf) - sum(vf)) * 100
                        feature.append(ecglf_u)
                        if ecglf_max not in lf.tolist():
                            ecgpf_l = np.nan
                        else:
                            ecgpf_l = lf.tolist().index(ecglf_max) * fs / lsg + 0.04
                        feature.append(ecgpf_l)

                        # 以 0.15–0.4 Hz 频段的功率为基础的统计特征值
                        hf = ps_cor[int(0.15 * lsg / fs):int(0.4 * lsg / fs)]
                        ecghf_mean = np.mean(hf)
                        feature.append(ecghf_mean)
                        ecghf_std = np.std(hf)
                        feature.append(ecghf_std)
                        ecghf_median = np.median(hf)
                        feature.append(ecghf_median)
                        ecghf_min = min(hf)
                        feature.append(ecghf_min)
                        ecghf_max = max(hf)
                        feature.append(ecghf_max)
                        ecghf_range = ecghf_max - ecghf_min
                        feature.append(ecghf_range)
                        ecghf_u = sum(hf) / (sum(vlhf) - sum(vf)) * 100
                        feature.append(ecghf_u)
                        if ecghf_max not in hf.tolist():
                            ecgpf_h = np.nan
                        else:
                            ecgpf_h = hf.tolist().index(ecghf_max) * fs / lsg + 0.15
                        feature.append(ecgpf_h)
                        ecg_lfhf = sum(ps_cor[int(0.04 * lsg / fs):int(0.15 * lsg / fs)]) / sum(
                            ps_cor[int(0.15 * lsg / fs):int(0.4 * lsg / fs)])
                        feature.append(ecg_lfhf)
            else:
                feature.extend([0] * 21)

            # 以所定位到的P,Q,R,S,T这五种关键点的振幅为基础，分别计算每个人每种情感状态下每一种关键点的Mean（平均值），Median（中间值），Std（标准差），MIN（最小值），MAX（最大值），以及Range
            # P点
            ecgpample_mean = np.mean(d[p_indices])
            feature.append(ecgpample_mean)
            ecgpample_median = np.median(d[p_indices])
            feature.append(ecgpample_median)
            ecgpample_std = np.std(d[p_indices])
            feature.append(ecgpample_std)
            if not d[p_indices].any():
                ecgpample_max = 0
                ecgpample_min = 0
            else:
                ecgpample_max = max(d[p_indices])
                ecgpample_min = min(d[p_indices])
            feature.append(ecgpample_max)
            feature.append(ecgpample_min)
            ecgpample_range = ecgpample_max - ecgpample_min
            feature.append(ecgpample_range)

            ecgqample_mean = np.mean(d[q_indices])
            feature.append(ecgqample_mean)
            ecgqample_median = np.median(d[q_indices])
            feature.append(ecgqample_median)
            ecgqample_std = np.std(d[q_indices])
            feature.append(ecgqample_std)
            if not d[q_indices].any():
                ecgqample_max = 0
                ecgqample_min = 0
            else:
                ecgqample_max = max(d[q_indices])
                ecgqample_min = min(d[q_indices])
            feature.append(ecgqample_max)
            feature.append(ecgqample_min)
            ecgqample_range = ecgqample_max - ecgqample_min
            feature.append(ecgqample_range)

            # R点
            ecgrample_mean = np.mean(d[r_indices])
            feature.append(ecgrample_mean)
            ecgrample_median = np.median(d[r_indices])
            feature.append(ecgrample_median)
            ecgrample_std = np.std(d[r_indices])
            feature.append(ecgrample_std)
            if not d[r_indices].any():
                ecgrample_max = 0
                ecgrample_min = 0
            else:
                ecgrample_max = max(d[r_indices])
                ecgrample_min = min(d[r_indices])
            feature.append(ecgrample_max)
            feature.append(ecgrample_min)
            ecgrample_range = ecgrample_max - ecgrample_min
            feature.append(ecgrample_range)

            # S点
            ecgsample_mean = np.mean(d[s_indices])
            feature.append(ecgsample_mean)
            ecgsample_median = np.median(d[s_indices])
            feature.append(ecgsample_median)
            ecgsample_std = np.std(d[s_indices])
            feature.append(ecgsample_std)
            if not d[s_indices].any():
                ecgsample_max = 0
                ecgsample_min = 0
            else:
                ecgsample_max = max(d[s_indices])
                ecgsample_min = min(d[s_indices])
            feature.append(ecgsample_max)
            feature.append(ecgsample_min)
            ecgsample_range = ecgsample_max - ecgsample_min
            feature.append(ecgsample_range)

            # T点
            ecgtample_mean = np.mean(d[t_indices])
            feature.append(ecgtample_mean)
            ecgtample_median = np.median(d[t_indices])
            feature.append(ecgtample_median)
            ecgtample_std = np.std(d[t_indices])
            feature.append(ecgtample_std)
            if not d[t_indices].any():
                ecgtample_max = 0
                ecgtample_min = 0
            else:
                ecgtample_max = max(d[t_indices])
                ecgtample_min = min(d[t_indices])
            feature.append(ecgtample_max)
            feature.append(ecgtample_min)
            ecgtample_range = ecgtample_max - ecgtample_min
            feature.append(ecgtample_range)

            # 以所定位的S、T点计算S-T振幅差相关特征值
            stample = []
            for i in range(len(t_indices)):
                stample.append(d[t_indices[i]] - d[s_indices[i]])
            ecgtdown_mean = np.mean(stample)
            feature.append(ecgtdown_mean)
            ecgtdown_median = np.median(stample)
            feature.append(ecgtdown_median)
            ecgtdown_std = np.std(stample)
            feature.append(ecgtdown_std)

            # 以所定位的P、T点计算P-T振幅差相关特征值
            ptample = []
            if p_indices and t_indices:
                if len(p_indices) == len(t_indices) and p_indices[0] > t_indices[0]:
                    for i in range(len(t_indices)):
                        ptample.append(d[p_indices[i]] - d[t_indices[i]])
                if len(p_indices) == len(t_indices) and p_indices[0] < t_indices[0]:
                    for i in range(len(t_indices) - 1):
                        ptample.append(d[p_indices[i + 1]] - d[t_indices[i]])

                if len(p_indices) == len(t_indices) + 1 and p_indices[0] < t_indices[0]:
                    for i in range(len(t_indices)):
                        ptample.append(d[p_indices[i + 1]] - d[t_indices[i]])
                if len(p_indices) + 1 == len(t_indices) and p_indices[0] > t_indices[0]:
                    for i in range(len(p_indices)):
                        ptample.append(d[p_indices[i]] - d[t_indices[i]])
            ecgptdown_mean = np.mean(ptample)
            feature.append(ecgptdown_mean)
            ecgptdown_median = np.median(ptample)
            feature.append(ecgptdown_median)
            ecgptdown_std = np.std(ptample)
            feature.append(ecgptdown_std)

            # 相邻RR波幅度差值的特征值
            difframple = []
            for i in range(len(r_indices) - 1):
                difframple.append(d[r_indices[i + 1]] - d[r_indices[i]])
            ecgdifframple_mean = np.mean(difframple)
            feature.append(ecgdifframple_mean)
            ecgdifframple_median = np.median(difframple)
            feature.append(ecgdifframple_median)
            ecgdifframple_std = np.std(difframple)
            feature.append(ecgdifframple_std)

            # 相邻RR波幅度差值的绝对值的特征值
            diffabsrample = []
            for i in range(len(r_indices) - 1):
                diffabsrample.append(abs(d[r_indices[i + 1]] - d[r_indices[i]]))
            ecgabsdifframple_mean = np.mean(diffabsrample)
            feature.append(ecgabsdifframple_mean)
            ecgabsdifframple_median = np.median(diffabsrample)
            feature.append(ecgabsdifframple_median)
            ecgabsdifframple_std = np.std(diffabsrample)
            feature.append(ecgabsdifframple_std)

            # 相邻TT波幅度差值的特征值
            difftample = []
            for i in range(len(t_indices) - 1):
                difftample.append(d[t_indices[i + 1]] - d[t_indices[i]])
            ecgdifftample_mean = np.mean(difftample)
            feature.append(ecgdifframple_mean)
            ecgdifftample_median = np.median(difftample)
            feature.append(ecgdifframple_median)
            ecgdifftample_std = np.std(difftample)
            feature.append(ecgdifframple_std)

            # 相邻TT波幅度差值的绝对值的特征值
            diffabstample = []
            for i in range(len(t_indices) - 1):
                diffabstample.append(abs(d[t_indices[i + 1]] - d[t_indices[i]]))
            ecgabsdifftample_mean = np.mean(diffabstample)
            feature.append(ecgabsdifftample_mean)
            ecgabsdifftample_median = np.median(diffabsrample)
            feature.append(ecgabsdifftample_median)
            ecgabsdifftample_std = np.std(diffabstample)
            feature.append(ecgabsdifftample_std)

            # P波能量的特征值
            penerge = []
            for i in range(len(p_indices) - 1):
                if p_indices[i + 1] == p_indices[i]:
                    penerge.append(0)
                else:
                    penerge.append((1 / (p_indices[i + 1] - p_indices[i])))
            ecgpenegry_mean = np.mean(penerge)
            feature.append(ecgpenegry_mean)
            ecgpenegry_median = np.median(penerge)
            feature.append(ecgpenegry_median)
            ecgpenegry_std = np.std(penerge)
            feature.append(ecgpenegry_std)

            # T波能量的特征值
            tenerge = []
            for i in range(len(t_indices) - 1):
                if t_indices[i + 1] == t_indices[i]:
                    tenerge.append(0)
                else:
                    tenerge.append((1 / (t_indices[i + 1] - t_indices[i])))
            ecgtenegry_mean = np.mean(tenerge)
            feature.append(ecgtenegry_mean)
            ecgtenegry_median = np.median(tenerge)
            feature.append(ecgtenegry_median)
            ecgtenegry_std = np.std(tenerge)
            feature.append(ecgtenegry_std)

            # q波能量的特征值
            qenerge = []
            for i in range(len(q_indices) - 1):
                if q_indices[i + 1] == q_indices[i]:
                    qenerge.append(0)
                else:
                    qenerge.append((1 / (q_indices[i + 1] - q_indices[i])))
            ecgqenegry_mean = np.mean(qenerge)
            feature.append(ecgqenegry_mean)
            ecgqenegry_median = np.median(qenerge)
            feature.append(ecgqenegry_median)
            ecgqenegry_std = np.std(qenerge)
            feature.append(ecgqenegry_std)

            # 心率上升沿的特征值
            rateupamp = []
            for i in range(len(t_indices)):
                rateupamp.append(d[t_indices[i]] - d[s_indices[i]])
            rateupamp_mean = np.mean(rateupamp)
            feature.append(rateupamp_mean)
            rateupamp_median = np.median(rateupamp)
            feature.append(rateupamp_median)
            rateupamp_std = np.std(rateupamp)
            feature.append(rateupamp_std)
            if not rateupamp:
                rateupamp_max = 0
                rateupamp_min = 0
            else:
                rateupamp_max = max(rateupamp)
                rateupamp_min = min(rateupamp)
            feature.append(rateupamp_min)
            feature.append(rateupamp_max)

            # 心率上升沿所用时间的特征值
            rateuptime = []
            for i in range(len(t_indices)):
                rateuptime.append(t_indices[i] - s_indices[i])
            rateuptime_mean = np.mean(rateuptime)
            feature.append(rateuptime_mean)
            rateuptime_median = np.median(rateuptime)
            feature.append(rateuptime_median)
            rateuptime_std = np.std(rateuptime)
            feature.append(rateuptime_std)
            if not rateuptime:
                rateuptime_max = 0
                rateuptime_min = 0
            else:
                rateuptime_max = max(rateuptime)
                rateuptime_min = min(rateuptime)
            feature.append(rateuptime_min)
            feature.append(rateuptime_max)

            # 心率上升沿斜率大小的特征值
            rateupr = []
            for i in range(len(t_indices)):
                rateupr.append((d[t_indices[i]] - d[s_indices[i]]) / (t_indices[i] - s_indices[i]))
            rateupr_mean = np.mean(rateupr)
            feature.append(rateupr_mean)
            rateupr_median = np.median(rateupr)
            feature.append(rateupr_median)
            rateupr_std = np.std(rateupr)
            feature.append(rateupr_std)
            if not rateupr:
                rateupr_max = 0
                rateupr_min = 0
            else:
                rateupr_max = max(rateupr)
                rateupr_min = min(rateupr)
            feature.append(rateupr_min)
            feature.append(rateupr_max)

            # 心率幅度相邻差值的特征值
            rateampdiff = []
            for i in range(len(r_indices) - 1):
                rateampdiff.append(d[r_indices[i + 1]] - d[r_indices[i]])
            rateampdiff_std = np.std(rateampdiff)
            feature.append(rateampdiff_std)
            rateampdiff_mean = np.mean(rateampdiff)
            feature.append(rateampdiff_mean)
            if not rateampdiff:
                rateampdiff_max = 0
                rateampdiff_min = 0
            else:
                rateampdiff_max = max(rateampdiff)
                rateampdiff_min = min(rateampdiff)
            nn_range = rateampdiff_max - rateampdiff_min
            feature.append(nn_range)

            # 心率幅度方差与心率幅度相邻差
            rateamp = []
            for i in range(len(r_indices)):
                rateamp.append(d[r_indices[i]])
            rateamp_std = np.std(rateamp)
            feature.append(rateamp_std)

            feature.append(rateamp_std / rateampdiff_std)

            # 心率下降沿幅度的特征值
            ratedownamp = []
            for i in range(len(s_indices)):
                ratedownamp.append(d[r_indices[i]] - d[s_indices[i]])
            ratedownamp_mean = np.mean(ratedownamp)
            feature.append(ratedownamp_mean)
            ratedownamp_median = np.median(ratedownamp)
            feature.append(ratedownamp_median)
            ratedownamp_std = np.std(ratedownamp)
            feature.append(ratedownamp_std)
            if not ratedownamp:
                ratedownamp_max = 0
                ratedownamp_min = 0
            else:
                ratedownamp_max = max(ratedownamp)
                ratedownamp_min = min(ratedownamp)

            feature.append(ratedownamp_min)
            feature.append(ratedownamp_max)

            # 心率下降斜率大小的特征值
            ratedownr = []
            for i in range(len(s_indices)):
                ratedownr.append((d[r_indices[i]] - d[s_indices[i]]) / (s_indices[i] - r_indices[i]))
            ratedownr_mean = np.mean(ratedownr)
            feature.append(ratedownr_mean)
            ratedownr_median = np.median(ratedownr)
            feature.append(ratedownr_median)
            ratedownr_std = np.std(ratedownr)
            feature.append(ratedownr_std)
            if not ratedownr:
                ratedownr_max = 0
                ratedownr_min = 0
            else:
                ratedownr_max = max(ratedownr)
                ratedownr_min = min(ratedownr)
            feature.append(ratedownr_min)
            feature.append(ratedownr_max)

            # 标签：情绪类别
            feature.append(n)

            feature_total.append(feature)


# 保存处数据理结果
output = pd.DataFrame(columns=columns,data=feature_total)
output.to_csv('train_data.csv',encoding='utf-8',float_format='%.9f',index=None)
# 删除临时文件
shutil.rmtree('logs')
shutil.rmtree('plots')
shutil.rmtree('temp')
