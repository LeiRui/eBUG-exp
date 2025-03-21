from simplification.cutil import (
    simplify_coords,
    simplify_coords_idx,
    simplify_coords_vw,
    simplify_coords_vw_idx,
    simplify_coords_vwp,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from csv import reader
import math
from scipy import signal
from tsdownsample2 import M4Downsampler,EveryNthDownsampler,LTTBDownsampler,LTOBDownsampler,\
    MinMaxDownsampler,LTDDownsampler,LTDOBDownsampler,LTOBETDownsampler,LTTBETDownsampler,\
    LTTBETGapDownsampler,LTOBETGapDownsampler,MinMaxGapDownsampler,LTTBETNewDownsampler,\
    LTTBETFurtherDownsampler,MinMaxFPLPDownsampler,LTSDownsampler,ILTSParallelDownsampler,\
    LTTBETFurtherRandomDownsampler,LTTBETFurtherMinMaxDownsampler,ILTSWaveletDownsampler,\
    ILTSLTDDownsampler
from polysimplify import VWSimplifier
from polysimplify import *
import cv2
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from skimage.util import invert 
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from matplotlib import image as mpimg
from PIL import Image
from matplotlib.transforms import IdentityTransform
import random
import scipy.ndimage as ndi
from textwrap import wrap
from matplotlib.ticker import ScalarFormatter
import os
import gc
import pywt
from scipy.fft import fft, ifft
from sklearn.decomposition import PCA
import re
import pickle
import copy
# import plotly.graph_objects as go; 
from tsdownsample2.waveletBucketTool import *
import subprocess
import shutil
import plotly.express as px 
import plotly.graph_objects as go; 
# from swab import *
from swab_withTimestamps import *
from datetime import datetime
import time

# fix type 3 error
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# colorsmap={
#     'ILTS':'tab:blue',
#     'LTTB':'tab:orange',
#     'MinMaxLTTB':'tab:red',
#     'MinMax':'tab:green',
#     'M4':'tab:purple',
#     'Uniform':'gold',
#     'FSW':'tab:pink',
#     'Visval':'tab:cyan',
#     'Sim-Piece':'tab:brown',
#     'PCA':'turquoise',
#     'DFT':'olive',
#     'OM3':'gray'
# }

# markersmap={
#     'ILTS':'*',
#     'LTTB':'v',
#     'MinMaxLTTB':'>',
#     'MinMax':'^',
#     'M4':'X',
#     'Uniform':'p',
#     'FSW':'s',
#     'Visval':'D',
#     'Sim-Piece':'<',
#     'PCA':'o',
#     'DFT':'P',
#     'OM3':'+'
# }

# colorsmap={
#     'eBUG-e5000':'tab:blue',
#     'eBUG-e∞':'tab:orange',
#     'eBUG-e0':'tab:red',
#     'SWAB-AD':'tab:green',
#     'BU-L2':'tab:purple',
#     'BU-L1':'gold',
#     'BU-L∞':'tab:pink',
#     'SWAB-L2':'tab:cyan',
#     'FSW':'tab:brown',
#     'MinMax-width':'turquoise',
#     'MinMax-freq':'olive',
#     'Rdp':'gray'
# }

colorsmap={
    'eBUG-e5000':'tab:blue',
    'eBUG-e500':'tab:blue',
    'eBUG-e∞':'tab:blue',
    'eBUG-e$\infty$':'tab:blue',
    'eBUG-e0':'tab:blue',
    'SWAB-AD':'tab:blue',
    'BU-L2':'tab:red',
    'BU-L1':'tab:red',
    'BU-L∞':'tab:red',
    'BU-L$\infty$':'tab:red',
    'SWAB-L2':'tab:red',
    'SWAB':'tab:red',
    'FSW':'tab:red',
    'Rdp':'tab:red',
    'MinMax-width':'tab:green',
    'MinMax-freq':'tab:green'
}

markersmap={
    'ILTS':'*',
    'LTTB':'v',
    'MinMaxLTTB':'>',
    'MinMax':'^',
    'M4':'X',
    'Uniform':'p',
    'FSW':'s',
    'Visval':'D',
    'Sim-Piece':'<',
    'PCA':'o',
    'DFT':'P',
    'OM3':'+'
}


def full_frame(width=None, height=None, dpi=None):
    import matplotlib as mpl
    # First we remove any padding from the edges of the figure when saved by savefig. 
    # This is important for both savefig() and show(). Without this argument there is 0.1 inches of padding on the edges by default.
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width/dpi, height/dpi) # so as to control pixel size exactly
    fig = plt.figure(figsize=figsize,dpi=dpi)
    # Then we set up our axes (the plot region, or the area in which we plot things).
    # Usually there is a thin border drawn around the axes, but we turn it off with `frameon=False`.
    ax = plt.axes([0,0,1,1], frameon=False)
    # Then we disable our xaxis and yaxis completely. If we just say plt.axis('off'),
    # they are still used in the computation of the image padding.
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Even though our axes (plot region) are set to cover the whole image with [0,0,1,1],
    # by default they leave padding between the plotted data and the frame. We use tigher=True
    # to make sure the data gets scaled to the full extents of the axes.
    plt.autoscale(tight=True)


def get_first_matched_file(folder_path,pattern):
    pattern = re.compile(pattern)
    files = os.listdir(folder_path)
    matched_files = [file for file in files if pattern.match(file)]
    if matched_files:
        return os.path.join(folder_path,matched_files[0])
    else:
        return None
        
def next_power_of_2(n):
    if n & (n - 1) == 0:
        return n
    power = 1
    while power < n:
        power <<= 1
    return power

def calculate_tree_height(n):
    if n <= 0:
        return 0 
    return math.floor(math.log2(n)) + 1

def calculate_values(input_list):
    xmin = input_list[0]
    xmax = input_list[1]
    cha1 = input_list[2]
    cha2 = input_list[3]
    if cha1<0:
        x1 = xmin
        x3 = x1 - cha1
    else:
        x3 = xmin
        x1 = x3 + cha1

    if cha2<0:
        x4 = xmax
        x2 = cha2 + x4
    else:
        x2 = xmax
        x4 = x2 - cha2

    return x1, x2, x3, x4


def VWSimplifier_PageWise(points,nout,pagePointNum,fast=True,mode='3',eaList=None):
    print('n=',len(points),', m=',nout,', pagePointNum=',pagePointNum,', fast=',fast)
    
    if fast==False: # 因为这个不分批运行很慢
        print("----------build thresholds: visval all--------------")
        ea_all,_=build_thresholds(points,mode=mode)

    print("----------build thresholds: visval batch--------------")
    if eaList is None:
        # 按照每p个点分批次，最后一个批次可能不足p
        pts_batches = [points[i:i + pagePointNum] for i in range(0, len(points), pagePointNum)]
        pages_num=len(pts_batches)
        print('pages num=',len(pts_batches))
        
        eaList=[]
        # eaBatches=[]
        for pts in pts_batches:
            ea,_=build_thresholds(pts,mode=mode)
            eaList+=list(ea)
            # eaBatches.append(ea)
        eaList=np.array(eaList)
    else:
        print('use given eaList')
        pages_num = (len(points) + pagePointNum - 1) // pagePointNum

    if fast==False: # 因为这个不分批运行很慢
        same_ea_count = sum(1 for a, b in zip(ea_all, eaList) if a == b) # 注意这两个是逐点对应的，是每个数据点的EA，所以这样按顺序一一比较
        print('points with the same EA:', same_ea_count, same_ea_count / len(ea_all) * 100, '%')

        print("----------sample based on threhsolds: visval all--------------")
        thresholds = sorted(ea_all,reverse=True)
        threshold_all = thresholds[int(nout)] # 从0开始编号
        print('threshold (all)=',threshold_all)
        sampled_all = points[ea_all>threshold_all]
        print(len(sampled_all))
    
    print("----------sample based on threhsolds: visval batch--------------")
    # 过滤掉 inf 和 -inf，也就是默认不采每个batch里的首尾点
    valid_indices = ~np.isinf(eaList)
    filtered_eaList = eaList[valid_indices]
    # 对过滤后的列表排序
    thresholds = sorted(filtered_eaList, reverse=True)
    threshold_batch = thresholds[int(nout)]
    print('threshold (batch)=',threshold_batch)
    sampled_batch = points[(eaList > threshold_batch) & valid_indices]
    sampled_batch = np.vstack([points[0], sampled_batch]) # 添加全局首点，不会重复因为首点EA一定是Inf
    sampled_batch = np.vstack([sampled_batch, points[-1]]) # 添加全局尾点，不会重复因为尾点EA一定是Inf
    print(len(sampled_batch))
    # print(sampled_batch)
    
    # eaBatches 统计一下有没有完全不采点的batch
    emptyBatchCnt=0
    for i in range(0, len(eaList), pagePointNum):
        emptyBatch=True
        for j in range(i, min(i + pagePointNum, len(eaList))): # 注意最后一个batch不要越界的处理
            ea=eaList[j]
            if ~np.isinf(ea) and ea>threshold_batch:
                emptyBatch=False
            if j==0 or j==len(eaList)-1: # 全局首尾点采
                emptyBatch=False
        if emptyBatch:
            emptyBatchCnt+=1
    print('prunedPageCnt=',emptyBatchCnt,', total pageNum=',pages_num, ', percentage(%)=', emptyBatchCnt/pages_num*100)
    
    if fast==False: # 因为这个不分批运行很慢
        # 将 sampled_all 和 sampled_batch 的第一列时间戳转换为集合，这里不是按照顺序逐一比较，因为有可能不同位置
        set_all_t = set(sampled_all[:, 0])
        set_batch_t = set(sampled_batch[:, 0])
        # 找到 set_all 和 set_batch 的相同点
        common_points = set_all_t.intersection(set_batch_t)
        # 统计相同点的个数
        common_points_count = len(common_points)
        print('Common sampled points:', common_points_count, ', nout=',nout, ', percentage(%)=', common_points_count / nout * 100, '%')


    t_visval_batch=np.array([x for x,y in sampled_batch])
    v_visval_batch=np.array([y for x,y in sampled_batch])
    if fast==False: # 因为这个不分批运行很慢
        t_visval_all=np.array([x for x,y in sampled_all])
        v_visval_all=np.array([y for x,y in sampled_all])
        return t_visval_all,v_visval_all,t_visval_batch,v_visval_batch,\
            ea_all,eaList,\
            same_ea_count/len(ea_all),emptyBatchCnt/pages_num,common_points_count/nout
    else:
        return None,None,t_visval_batch,v_visval_batch,\
            None,eaList,\
            float('inf'),float('inf'),float('inf')

def OM3_reverse(query_table_csv, n):
    data = pd.read_csv(query_table_csv) # default has header i,minvd,maxvd
    # assume sorted by i
    #     data = data.sort_values(by='i', ascending=True) 
    print(data)
    
    expected_ids = [-1] + list(range(1, next_power_of_2(n)//2)) 
    full_df = pd.DataFrame({'i': expected_ids})
    
    data = full_df.merge(data, on='i', how='left')
    data.fillna(0, inplace=True) 
    
    
    inp0 = data.iloc[0, 1:]
    values = []
    values.append([inp0[0], inp0[1]])
    cha = data.iloc[1:, 1:].values

    height = calculate_tree_height(len(cha))
    
    for i in range(height):
        start = (1 << i) - 1  
        end = (1 << (i + 1)) - 1  
#         print(start, end)
        tempcha = cha[start:end]
        tempva = values[start:end]
        for e in range(len(tempcha)):
            temp = [tempva[e][0], tempva[e][1], tempcha[e][0], tempcha[e][1]]
            result = calculate_values(temp)
            values.append([result[0], result[1]])
            values.append([result[2], result[3]])

    startend = (1 << height) - 1 
    endend = (1 << (height + 1)) - 1  

    tmp=values[startend:endend]
    flattened_list = [item for sublist in tmp for item in sublist]
    
    print(len(flattened_list))
    print('finish revserse transform')
    
    gc.collect()
    return flattened_list

def sortRaw(raw_csv):
    raw = pd.read_csv(raw_csv,header=None) # default no header
    second_column = raw.iloc[:,1]
    second_column = second_column.to_numpy('float')
    sorted_pairs = np.array([sorted(second_column[i:i+2]) for i in range(0, len(second_column), 2)])
    sorted_column = sorted_pairs.flatten()
    print(len(sorted_column))
    return sorted_column
    
def exp(t, A, lbda):
    r"""y(t) = A \cdot \exp(-\lambda t)"""
    return A * np.exp(-lbda * t)

def sine(t, omega, phi):
    r"""y(t) = \sin(\omega \cdot t + phi)"""
    return np.sin(omega * t + phi)

def damped_sine(t, A, lbda, omega, phi):
    r"""y(t) = A \cdot \exp(-\lambda t) \cdot \left( \sin \left( \omega t + \phi ) \right)"""
    return exp(t, A, lbda) * sine(t, omega, phi)

def _get_or_conv_mask(f1, f2, win_size: int = 11) -> np.ndarray:
    # get inverted img: 0 for original white background, 255 for original dark foreground
    img1 = cv2.imread(f1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img1 = invert(img1) # [0,255], np.uint8. This step to make a larger value means more data points.
    # img1 = img_as_float(img1) # [0,1]

    img2 = cv2.imread(f2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img2 = invert(img2) # [0,255], np.uint8. This step to make a larger value means more data points.
    # img2 = img_as_float(img2) # [0,1]

    joined=(img1+img2)>0
    # plt.imshow(joined)
    # plt.show()
    
    or_conv = signal.convolve2d(joined, np.ones((win_size, win_size)), mode="same")
    or_conv_mask = or_conv>0
    # plt.imshow(or_conv_mask)
    # plt.show()

    return or_conv_mask


def match(imfil1,imfil2,ws=None):    
    img1=cv2.imread(imfil1)    
    (h,w)=img1.shape[:2]    
    img2=cv2.imread(imfil2)    
    resized=cv2.resize(img2,(w,h))    
    (h1,w1)=resized.shape[:2]    
    img1=img_as_float(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)) # img_as_float: the dtype is uint8, means convert [0, 255] to [0, 1]
    img2=img_as_float(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
    if ws is None:
        return ssim(img1,img2,data_range=img1.max()-img1.min())
    else:
        return ssim(img1,img2,data_range=img1.max()-img1.min(),win_size=ws)

def match_masked(imfil1,imfil2,ws=11):
    img1=cv2.imread(imfil1)
    (h,w)=img1.shape[:2]    
    img2=cv2.imread(imfil2)    
    resized=cv2.resize(img2,(w,h))    
    (h1,w1)=resized.shape[:2]    
    img1=img_as_float(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)) # img_as_float: the dtype is uint8, means convert [0, 255] to [0, 1]
    img2=img_as_float(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))

    or_conv_mask = _get_or_conv_mask(imfil1,imfil2)

    ssim_kwgs = dict(win_size=ws, full=True, gradient=False)
    SSIM = ssim(img1, img2, data_range=img1.max()-img1.min(),**ssim_kwgs)[1]

    SSIM_masked = SSIM.ravel()[or_conv_mask.ravel()]
    return np.mean(SSIM_masked)


def mse_in_255_masked(imfil1, imfil2):
    # mse_in_255 use divider as the total number of global pixels
    # while here only considers masked pixels

    img1 = cv2.imread(imfil1)
    img2 = cv2.imread(imfil2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    squared_diff = (img1.astype("float") -img2.astype("float")) ** 2

    or_conv_mask = _get_or_conv_mask(imfil1,imfil2)

    mse_masked = squared_diff.ravel()[or_conv_mask.ravel()].mean() 
    # actually sum is the same as mse_in_255, the only difference is divider
    # mse_in_255 use divider as the total number of global pixels
    # while here only uses the number of masked pixels as divider

    return mse_masked


def mse_in_255(imfil1, imfil2): # mse=mse_in_255/(255*255)
    img1 = cv2.imread(imfil1)
    img2 = cv2.imread(imfil2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    squared_diff = (img1.astype("float") -img2.astype("float")) ** 2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
    err = summed / num_pix
    return err

def mse(imfil1,imfil2): # mse=mse_in_255/(255*255)
    img1 = cv2.imread(imfil1)
    img2 = cv2.imread(imfil2)
    img1 = img_as_float(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    img2 = img_as_float(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    squared_diff = (img1-img2) ** 2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
    err = summed / num_pix
    return err

def pem20(f1,f2):
    # Pixel Error Margin 20
    # Reference: Data Point Selection for Line Chart Visualization:Methodological Assessment and Evidence-Based Guidelines
    img1 = cv2.imread(f1)
    img2 = cv2.imread(f2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_diff = img1.astype("float") - img2.astype("float")
    res=0
    for iy, ix in np.ndindex(img_diff.shape):
        x=img_diff[iy, ix]
        if abs(x)>20:
            res=res+1
    num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
    res=res/num_pix
    return res

def compare_vis(f1,f2):
    img1 = cv2.imread(f1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img1 = invert(img1) # [0,255], np.uint8. This step to make a larger value means more data points.
    img1 = img_as_float(img1) # [0,1]
    
    img2 = cv2.imread(f2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img2 = invert(img2) # [0,255], np.uint8. This step to make a larger value means more data points.
    img2 = img_as_float(img2) # [0,1]
    
    plt.imshow(img1 - img2, cmap="gray")
    plt.title("{} \n vs \n {}\n Black: latter data, but no former data.\n White: former data, but no latter data".format(f1,f2))
    plt.show()

    
def compare_vis_color(f1,f2):
    # color -> data
    img1 = cv2.imread(f1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img1 = invert(img1) # [0,255], np.uint8. This step to make a larger value means more data points.
    img1 = img_as_float(img1) # [0,1]
    
    img2 = cv2.imread(f2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img2 = invert(img2) # [0,255], np.uint8. This step to make a larger value means more data points.
    img2 = img_as_float(img2) # [0,1]
    
    # data -> color
    plt.imshow(
        np.concatenate( # [w,h,3] shape
            (
                img1[:,:, None],  # RED
                img2[:,:, None],  # GREEN
                np.zeros((*img1.shape, 1)),
            ),
            axis=-1,
        ),
    )
    # GREEN 0-1-0
    # RED 1-0-0
    # YELLOW 1-1-0
    # BLACK 0-0-0
    plt.title("{} \n vs \n {}\n GREEN: latter data, but no former data \n RED: former data, but no latter data \n YELLOW: both latter and former data \n BLACK: neither latter nor former data".format(f1,f2))
    plt.tight_layout()
    plt.savefig('compare_vis_color.png')
    plt.show()

# def add_grid_column(ax, fig_width, fig_height, step, horizontal=False):
#     ax.set_frame_on(False)

#     # Minor ticks
#     ax.set_xticks(np.arange(-0.5, fig_width, step), minor=True)
#     if horizontal == True:
#         ax.set_yticks(np.arange(-0.5, fig_height, step), minor=True)

#     # Gridlines based on minor ticks
#     ax.grid(which="minor", color="red", linestyle="-", linewidth=0.3)
#     ax.tick_params(which="minor",bottom=False, left=False)

#     # hid the ticks but keep the grid
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.tick_params(which="major",bottom=False, left=False)
#     ax.set_ylim((fig_height - 0.5, -0.8)) # NOTE the special order because of image
#     ax.set_xlim((-0.5, fig_width-0.3))

def add_grid_column(ax, fig_width, fig_height, step, horizontal=False):
    ax.set_frame_on(False)

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, fig_width, step), minor=True)
    if horizontal == True:
        ax.set_yticks(np.arange(-0.5, fig_height, step), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="grey", linestyle="-", linewidth=2)
    ax.tick_params(which="minor",bottom=False, left=False)

    # hid the ticks but keep the grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(which="major",bottom=False, left=False)
    ax.set_ylim((fig_height - 0.5, -0.51)) # NOTE the special order because of image
    ax.set_xlim((-0.51, fig_width-0.49))


def subplt_myplot_random(width,height,dpi,name,anti,downsample,nout,lw,gridStep):
    full_frame(width,height,dpi)
    t = np.linspace(0, 5, num=3000)
    A = 1
    lbda = 1
    omega = 20 * np.pi
    phi = 0
    v = damped_sine(t, A, lbda, omega, phi)
    v_min=min(v)
    v_max=max(v)
    t_min=min(t)
    t_max=max(t)
    # MinMaxDownsampler, MinMaxLTTBDownsampler, M4Downsampler, EveryNthDownsampler, LTTBDownsampler
    if downsample == "MinMaxDownsampler":
        s_ds = MinMaxDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif downsample == "MinMaxLTTBDownsampler":
        s_ds = MinMaxLTTBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif downsample == "M4Downsampler":
        s_ds = M4Downsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif downsample == "EveryNthDownsampler":
        s_ds = EveryNthDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif downsample == "LTTBDownsampler":
        s_ds = LTTBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    
    plt.plot(t,v,'-',color='k',linewidth=lw,antialiased=anti,markersize=1)
    plt.xlim(t_min, t_max)
    plt.ylim(v_min, v_max)
    plt.savefig(name+'.png',backend='agg')
    plt.close()
    img=cv2.imread(name+'.png')
    plt.imshow(img)
    ax = plt.gca()
    add_grid_column(ax, height, width, gridStep)

def analyze(name,width,anti,downsample,lw,gridOn):
    # (width,height,dpi,name,anti,downsample,nout,lw,gridOn,gridStep)
    print(name+' rendered on a big 300*300 canvas:')
    myplot_random(300,300,96,name+'_rendered_big',anti,downsample,4*width,lw,gridOn,300/width)
    print(name+' rendered on the target small canvas:')
    myplot_random(width,width,96,name+'_rendered_small_target',anti,downsample,4*width,lw,gridOn,1)
    print('overlay '+ name + ' on the small rendered image:')
    from PIL import Image
    img = Image.open(name+'_rendered_big.png')
    back = Image.open(name+'_rendered_small_target.png')
    back = back.resize(img.size)
    blended_image = Image.blend(img, back, 0.7)
    plt.imshow(blended_image)
    plt.show()
    
def subplt_compare_vis(f1,f2):
    img1 = cv2.imread(f1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img1 = invert(img1) # [0,255], np.uint8. This step to make a larger value means more data points.
    img1 = img_as_float(img1) # [0,1]
    
    img2 = cv2.imread(f2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img2 = invert(img2) # [0,255], np.uint8. This step to make a larger value means more data points.
    img2 = img_as_float(img2) # [0,1]
    
    plt.imshow(img1 - img2, cmap="gray")
    plt.title("{} \n vs \n {}\n Black: latter data, but no former data.\n White: former data, but no latter data".format(f1,f2),
             fontsize = 10)

def subplt_compare_vis_color(f1,f2,gridStep):
    # color -> data
    img1 = cv2.imread(f1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img1 = invert(img1) # [0,255], np.uint8. This step to make a larger value means more data points.
    img1 = img_as_float(img1) # [0,1]
    
    img2 = cv2.imread(f2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img2 = invert(img2) # [0,255], np.uint8. This step to make a larger value means more data points.
    img2 = img_as_float(img2) # [0,1]
    
    # data -> color
    plt.imshow(
        np.concatenate( # [w,h,3] shape
            (
                img1[:,:, None],  # RED
                1-img2[:,:, None],  # GREEN
                np.zeros((*img1.shape, 1)),
            ),
            axis=-1,
        ),
    )
    plt.tick_params(axis='both', which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
    # GREEN 0-1-0
    # RED 1-0-0
    # YELLOW 1-1-0
    # BLACK 0-0-0
    # plt.title("{} \n vs \n {}\n GREEN: latter data, but no former data \n RED: former data, but no latter data \n YELLOW: both latter and former data \n BLACK: neither latter nor former data".format(f1,f2),
              # fontsize = 10)
    ax = plt.gca()
    add_grid_column(ax, img1.shape[1], img1.shape[0], gridStep, True)

def point_line_distance(x0,y0,x1,y1,x2,y2):
    # (x0,y0)到直线（经过(x1,y1)、(x2,y2)点的直线）的正交距离
    return abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1)) / math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

# dist = point_line_distance(p[third,0],p[third,1],p[first,0],p[first,1],p[second,0],p[second,1])
def simplify_reumann_witkam(p,tolerance):
    mask = np.zeros(len(p),dtype='bool')
    sp=0
    ep=1
    for i in range(2,len(p)):
        dist=point_line_distance(p[i,0],p[i,1],p[sp,0],p[sp,1],p[ep,0],p[ep,1])
        if dist>tolerance:
            mask[i-1]=True
            sp=i-1
            ep=i
    mask[0]=mask[-1]=True
    return mask

def vertical_distance(x0,y0,x1,y1,x2,y2):
    # (x0,y0)到直线（经过(x1,y1)、(x2,y2)点的直线）的y轴方向上的垂直距离
    return abs((y2-y1)/(x2-x1)*(x0-x1)+y1-y0)

def simplify_reumann_witkam_residual(p,tolerance):
    # 但是residual error没有vertical error那种error bound性质，所以其实这种改法是不太有意义的
    # 比如首二点斜率很大但是后面的点都是水平，导致后面少数的几个点就使得residual error达到阈值，
    # 而这样得到的分段与数据真实的residual error其实和首二点直线和数据的residual error差的很大
    # 虽然simplify_reumann_witkam用的也不是vertical error而是point-to-line正交距离也没啥意义

    # 固定首二点，这样计算residual error只需要增量地累加
    mask = np.zeros(len(p),dtype='bool')
    sp=0
    ep=1
    residual=0
    for i in range(2,len(p)):
        dist=vertical_distance(p[i,0],p[i,1],p[sp,0],p[sp,1],p[ep,0],p[ep,1])
        residual=residual+dist*dist
        if residual>tolerance:
            mask[i-1]=True
            sp=i-1
            ep=i
            residual=0
    mask[0]=mask[-1]=True
    return mask

def vertical_distance_interpolation(p,sp,ep):
    res=0
    x1=p[sp,0]
    y1=p[sp,1]
    x2=p[ep,0]
    y2=p[ep,1]
    for i in range(sp+1,ep):
        x0=p[i,0]
        y0=p[i,1]
        res=res+((y2-y1)/(x2-x1)*(x0-x1)+y1-y0)*((y2-y1)/(x2-x1)*(x0-x1)+y1-y0)
    return res

def simplify_interpolation_residual(p,tolerance):
    # simplify_interpolation_residual is very slow because it is not linear time complexity
    # 不是固定首二点，而是连接首尾点
    mask = np.zeros(len(p),dtype='bool')
    sp=0
    for i in range(2,len(p)):
        ep=i
        dist=vertical_distance_interpolation(p,sp,ep)
        if dist>tolerance:
            mask[i-1]=True
            sp=i-1
    mask[0]=mask[-1]=True
    return mask

# def simplify_shrinking_cone(p,tolerance):
#     mask=np.zeros(len(p),dtype='bool')

#     sp=0
#     i=1
#     upSlope = float(p[i,1]+tolerance-p[sp,1])/(p[i,0]-p[sp,0])
#     lowSlope = float(p[i,1]-tolerance-p[sp,1])/(p[i,0]-p[sp,0])

#     while i < len(p)-1:
#         i+=1
#         upV=upSlope*(p[i,0]-p[sp,0])+p[sp,1] # k1*(x-x0)+y0
#         lowV=lowSlope*(p[i,0]-p[sp,0])+p[sp,1] # k2*(x-x0)+y0
#         if lowV<=p[i,1]<=upV:
#             upSlope = min(upSlope,(float(p[i,1]+tolerance-p[sp,1])/(p[i,0]-p[sp,0])))
#             lowSlope = max(lowSlope,(float(p[i,1]-tolerance-p[sp,1])/(p[i,0]-p[sp,0])))
#         else:
#             mask[i-1]=True
#             sp=i-1
#             upSlope = float(p[i,1]+tolerance-p[sp,1])/(p[i,0]-p[sp,0])
#             lowSlope = float(p[i,1]-tolerance-p[sp,1])/(p[i,0]-p[sp,0])

#     mask[0]=mask[-1]=True
#     return mask

# upl = min(upl,(float(p[i,1]+tolerance-p[sp,1])/(p[i,0]-p[sp,0])))
# lowl = max(lowl,(float(p[i,1]-tolerance-p[sp,1])/(p[i,0]-p[sp,0])))
# if upl < lowl:
#     mask[i-1]=True
#     sp=i-1
#     i-=1
#     upl = float('inf')
#     lowl = float('-inf')

def simplify_shrinking_cone(p, tolerance):
    mask = np.zeros(len(p), dtype=bool)
    mask[0] = mask[-1] = True
    
    sp = 0
    i = 1

    # 提前计算批量的值
    # print(p)
    # print(tolerance)
    p_upper = p[:, 1] + tolerance
    p_lower = p[:, 1] - tolerance

    dx = p[i, 0] - p[sp, 0]
    upSlope = (p_upper[i] - p[sp, 1]) / dx
    lowSlope = (p_lower[i] - p[sp, 1]) / dx

    while i < len(p) - 1:
        i += 1
        dx = p[i, 0] - p[sp, 0]
        upV = upSlope * dx + p[sp, 1]
        lowV = lowSlope * dx + p[sp, 1]
        
        if lowV <= p[i, 1] <= upV:
            upSlope = min(upSlope, (p_upper[i] - p[sp, 1]) / dx)
            lowSlope = max(lowSlope, (p_lower[i] - p[sp, 1]) / dx)
        else:
            mask[i - 1] = True
            sp = i - 1
            dx = p[i, 0] - p[sp, 0] # note sp has changed
            upSlope = (p_upper[i] - p[sp, 1]) / dx
            lowSlope = (p_lower[i] - p[sp, 1]) / dx

    return mask

# def swinging_door(data, deviation=.1, mode=False, step=10):
#     #　https://github.com/chelaxe/SwingingDoor/blob/main/SwingingDoor.ipynb
#     # mode=True: take the previous point before the swing door doesn't parallel
#     # step: if mode=True, step is used additionally to add sampling point even when the door is fine

#     current_step = 0
#     upper_pivot = lower_pivot = current = (0., 0.)

#     sloping_upper_max = sloping_lower_min = 0. # TODO 会不会是这里没有更新

#     for i, item in enumerate(data):
#         if not i:
#             entrance = current = item

#             upper_pivot = (entrance[0], entrance[1] + deviation)
#             lower_pivot = (entrance[0], entrance[1] - deviation)

#             yield entrance

#             current_step = 0
#             continue

#         past, current = current, item

#         sloping_upper = float(current[1] - upper_pivot[1]) / (current[0] - upper_pivot[0])
#         sloping_lower = float(current[1] - lower_pivot[1]) / (current[0] - lower_pivot[0])

#         if not sloping_upper_max and not sloping_lower_min:
#             # print('continue')
#             sloping_upper_max = sloping_upper
#             sloping_lower_min = sloping_lower

#             current_step += 1
#             continue

#         if sloping_upper > sloping_upper_max:
#             sloping_upper_max = sloping_upper

#             if sloping_upper_max > sloping_lower_min:
#                 entrance = past if mode else ((past[0] + current[0]) / 2, (past[1] + current[1]) / 2 - (deviation / 2))

#                 yield entrance

#                 current_step = 0

#                 upper_pivot = entrance[0], entrance[1] + deviation
#                 lower_pivot = entrance[0], entrance[1] - deviation

#                 sloping_upper_max = float(current[1] - upper_pivot[1]) / (current[0] - upper_pivot[0])
#                 sloping_lower_min = float(current[1] - lower_pivot[1]) / (current[0] - lower_pivot[0])

#         if sloping_lower < sloping_lower_min: # bug原来在这里：原来用的是elif，没有考虑到完全有可能上下门斜率都更新的情况
#             sloping_lower_min = sloping_lower

#             if sloping_upper_max > sloping_lower_min:
#                 entrance = past if mode else ((past[0] + current[0]) / 2, (past[1] + current[1]) / 2 - (deviation / 2))

#                 yield entrance

#                 current_step = 0

#                 upper_pivot = entrance[0], entrance[1] + deviation
#                 lower_pivot = entrance[0], entrance[1] - deviation

#                 sloping_upper_max = float(current[1] - upper_pivot[1]) / (current[0] - upper_pivot[0])
#                 sloping_lower_min = float(current[1] - lower_pivot[1]) / (current[0] - lower_pivot[0])

#         if mode and current_step == step:
#             entrance = past

#             yield entrance

#             current_step = 0

#             upper_pivot = entrance[0], entrance[1] + deviation
#             lower_pivot = entrance[0], entrance[1] - deviation

#             sloping_upper_max = float(current[1] - upper_pivot[1]) / (current[0] - upper_pivot[0])
#             sloping_lower_min = float(current[1] - lower_pivot[1]) / (current[0] - lower_pivot[0])

#         else:
#             current_step += 1

#     yield current


def swinging_door(data, deviation=.1, mode=False, step=10):
    # mode=True: take the previous point before the swing door doesn't parallel
    # step: if mode=True, step is used additionally to add sampling point even when the door is fine

    current_step = 0
    upper_pivot = lower_pivot = current = (0., 0.)
    sloping_upper_max = sloping_lower_min = 0.
    
    sampled_points = []  # Initialize an empty list to store results

    for i, item in enumerate(data):
        if not i:
            entrance = current = item
            upper_pivot = (entrance[0], entrance[1] + deviation)
            lower_pivot = (entrance[0], entrance[1] - deviation)
            sampled_points.append(entrance)  # Append instead of yield
            current_step = 0
            continue

        past, current = current, item
        sloping_upper = float(current[1] - upper_pivot[1]) / (current[0] - upper_pivot[0])
        sloping_lower = float(current[1] - lower_pivot[1]) / (current[0] - lower_pivot[0])

        if not sloping_upper_max and not sloping_lower_min:
            sloping_upper_max = sloping_upper
            sloping_lower_min = sloping_lower
            current_step += 1
            continue

        if sloping_upper > sloping_upper_max:
            sloping_upper_max = sloping_upper
            if sloping_upper_max > sloping_lower_min:
                entrance = past if mode else ((past[0] + current[0]) / 2, (past[1] + current[1]) / 2 - (deviation / 2))
                sampled_points.append(entrance)
                current_step = 0
                upper_pivot = entrance[0], entrance[1] + deviation
                lower_pivot = entrance[0], entrance[1] - deviation
                sloping_upper_max = float(current[1] - upper_pivot[1]) / (current[0] - upper_pivot[0])
                sloping_lower_min = float(current[1] - lower_pivot[1]) / (current[0] - lower_pivot[0])

        if sloping_lower < sloping_lower_min:
            sloping_lower_min = sloping_lower
            if sloping_upper_max > sloping_lower_min:
                entrance = past if mode else ((past[0] + current[0]) / 2, (past[1] + current[1]) / 2 - (deviation / 2))
                sampled_points.append(entrance)
                current_step = 0
                upper_pivot = entrance[0], entrance[1] + deviation
                lower_pivot = entrance[0], entrance[1] - deviation
                sloping_upper_max = float(current[1] - upper_pivot[1]) / (current[0] - upper_pivot[0])
                sloping_lower_min = float(current[1] - lower_pivot[1]) / (current[0] - lower_pivot[0])

        if mode and current_step == step:
            entrance = past
            sampled_points.append(entrance)
            current_step = 0
            upper_pivot = entrance[0], entrance[1] + deviation
            lower_pivot = entrance[0], entrance[1] - deviation
            sloping_upper_max = float(current[1] - upper_pivot[1]) / (current[0] - upper_pivot[0])
            sloping_lower_min = float(current[1] - lower_pivot[1]) / (current[0] - lower_pivot[0])

        else:
            current_step += 1

    sampled_points.append(current)  # Append the final point
    return sampled_points  # Return the list


def getParam(nout, t, v, simplify_func, epsilon=2, mask=True, **kwargs):
    # important assumption: the larger the param, the fewer the sampled points!
    # epsilon: The maximum error between the actual number of sampled points and the target nout
    
    points = np.array(list(zip(t, v)))
    x = 1
    directLess = False
    directMore = False
    
    iterNum = 0
    maxIterNum=100
    
    # First binary search loop to find the initial range
    base=10 # bigger to quicker narrow the search space
    while True:
        iterNum += 1
        if iterNum > maxIterNum:  # Avoid infinite loops for special cases
            # print('reach maxIterNum')
            break
        res = simplify_func(points, x, **kwargs)
        if mask:
            t_result = points[res, 0]
            length=len(t_result)
        else:
            length=len(res)
            
        # print(length)
        if abs(length-nout)<=epsilon:
            # print('return')
            return x
        
        if length > nout:
            if directMore:
                # print('reach the first more')
                break
            if not directLess:
                directLess = True
                
            # print('need less')
            x *= base
            
        else:
            if directLess:
                # print('reach the first less')
                break
            if not directMore:
                directMore = True
                
            # print('need more')
            x /= base

    # Determine the initial left and right bounds
    if directLess:
        left, right = x / base, x
    elif directMore:
        left, right = x, x * base

    # Refine the range with binary search
    iterNum = 0
    while True:
        iterNum += 1
        if iterNum > maxIterNum:  # Avoid infinite loops
            # print('reach maxIterNum')
            break
            
        mid = (left + right) / 2
        res = simplify_func(points, mid, **kwargs)
        if mask:
            t_result = points[res, 0]
            length=len(t_result)
        else:
            length=len(res)
            
        # print(length)
        if abs(length-nout)<=epsilon:
            # print('return')
            return mid
        
        if length > nout:
            # print('need less')
            left = mid
        else:
            # print('need more')
            right = mid

    return (left + right) / 2


def getReuwiParam(nout, t, v, epsilon=2):
    # epsilon=max(1, int(nout * 0.01))
    return getParam(nout, t, v, simplify_func=simplify_reumann_witkam, epsilon=epsilon,mask=True)

def getShrinkingConeParam(nout, t, v, epsilon=2):
    # epsilon=max(1, int(nout * 0.01))
    return getParam(nout, t, v, simplify_func=simplify_shrinking_cone, epsilon=epsilon,mask=True)

def getRDPParam(nout, t, v, epsilon=2):
    # epsilon=max(1, int(nout * 0.01))
    return getParam(nout, t, v, simplify_func=simplify_coords, epsilon=epsilon,mask=False)

def getFastVisvalParam(nout, t, v, epsilon=2):
    # epsilon=max(1, int(nout * 0.01))
    return getParam(nout, t, v, simplify_func=simplify_coords_vw, epsilon=epsilon,mask=False)

def getReuwiResidualParam(nout, t, v, epsilon=2):
    # epsilon=max(1, int(nout * 0.01))
    return getParam(nout, t, v, simplify_func=simplify_reumann_witkam_residual, epsilon=epsilon,mask=True)

def getInterpolationResidualParam(nout, t, v, epsilon=2): 
    # epsilon=max(1, int(nout * 0.01))
    # note simplify_interpolation_residual is very slow
    return getParam(nout, t, v, simplify_func=simplify_interpolation_residual, epsilon=epsilon,mask=True)

def getSDTParam(nout, t, v, epsilon=2):
    # epsilon=max(1, int(nout * 0.01))
    return getParam(nout, t, v, simplify_func=swinging_door, mode=True, step=float('inf'), epsilon=epsilon, mask=False)

def getPySWABParam(nout, t, v, epsilon=2, joint=True, residual=True):
    # epsilon=max(1, int(nout * 0.01)) # 注意：这样的话传进来的参数没用
    return getParam(nout, t, v, simplify_func=seg_swab_withTimestamps,\
        epsilon=epsilon, mask=False, joint=joint, residual=residual, m=nout-1)


def myDFT(v,nout):
    signal_dft = fft(v)
    abs_flattened = np.abs(signal_dft)
    threshold = np.partition(abs_flattened, -nout)[-nout]
    # print(threshold)
    filtered_dft = np.where(np.abs(signal_dft) >= threshold, signal_dft, 0) # note abs
    print(np.count_nonzero(filtered_dft))
    v = ifft(filtered_dft).real # reconstruct before plotting!
    return v

def myDWT(v,nout):
    wavelet = 'haar'
    coeffs = pywt.wavedec(v, wavelet)
    tmp=[]
    for arr in coeffs:
        tmp.append(arr.tolist()[:])
    flattened_list = [item for sublist in tmp for item in sublist]
    abs_flattened = np.abs(flattened_list)
    threshold = np.partition(abs_flattened, -nout)[-nout]
    # print(threshold)
    cnt=0
    for i in range(len(coeffs)):
        for j in range(len(coeffs[i])):
            if np.abs(coeffs[i][j])<threshold: # note abs
                coeffs[i][j]=0
            else:
                cnt+=1
    print(cnt)
    v=pywt.waverec(coeffs, wavelet) # reconstruct before plotting!
    return v

def PCAUnit(v,n1):
    X=v.reshape(n1,-1) # 2500000*1->2000*1250
    pca_raw = PCA(copy=True, whiten=False, n_components=1)
    X_L=pca_raw.fit_transform(X) # 2000*1
    X_R = pca_raw.components_  # 1*1250
    X_means = pca_raw.mean_ # 1250*1
    return X_L,X_R,X_means

def PCARecon(XL,XR,XM):
    XL=XL.reshape(-1,1)
    XR=XR.reshape(1,-1)
    XM=XM.reshape(-1)
    return XL.dot(XR)+XM

def myPCA(v):
    if len(v)==2500000:
        n1=2000
        n2=50
        n3=50
        n4=50
        n5=10
        n6=8
        n7=8
        n8=10 
        n9=5
        n10=5
        n11=10 
        n12=5
        n13=5
    elif len(v)==500000:
        n1=1000
        n2=50
        n3=50
        n4=50
        n5=10
        n6=5
        n7=5
        n8=10 
        n9=5
        n10=5
        n11=10 
        n12=5
        n13=5
    else:
        print("not implemented")
        return

    XL,XR,XM=PCAUnit(v,n1)
    XLL,XLR,XLM=PCAUnit(XL,n2)
    XRL,XRR,XRM=PCAUnit(XR,n3)
    XML,XMR,XMM=PCAUnit(XM,n4)

    XLLL,XLLR,XLLM=PCAUnit(XLL,n5)
    XLRL,XLRR,XLRM=PCAUnit(XLR,n6)
    XLML,XLMR,XLMM=PCAUnit(XLM,n7)
    
    XRLL,XRLR,XRLM=PCAUnit(XRL,n8)
    XRRL,XRRR,XRRM=PCAUnit(XRR,n9)
    XRML,XRMR,XRMM=PCAUnit(XRM,n10)

    XMLL,XMLR,XMLM=PCAUnit(XML,n11)
    XMRL,XMRR,XMRM=PCAUnit(XMR,n12)
    XMML,XMMR,XMMM=PCAUnit(XMM,n13)

    XLLL_shape_0 = XLLL.shape[0]
    XLLR_shape_1 = XLLR.shape[1]
    XLLM_shape_0 = XLLM.shape[0]

    XLRL_shape_0 = XLRL.shape[0]
    XLRR_shape_1 = XLRR.shape[1]
    XLRM_shape_0 = XLRM.shape[0]

    XLML_shape_0 = XLML.shape[0]
    XLMR_shape_1 = XLMR.shape[1]
    XLMM_shape_0 = XLMM.shape[0]

    XRLL_shape_0 = XRLL.shape[0]
    XRLR_shape_1 = XRLR.shape[1]
    XRLM_shape_0 = XRLM.shape[0]

    XRRL_shape_0 = XRRL.shape[0]
    XRRR_shape_1 = XRRR.shape[1]
    XRRM_shape_0 = XRRM.shape[0]

    XRML_shape_0 = XRML.shape[0]
    XRMR_shape_1 = XRMR.shape[1]
    XRMM_shape_0 = XRMM.shape[0]

    XMLL_shape_0 = XMLL.shape[0]
    XMLR_shape_1 = XMLR.shape[1]
    XMLM_shape_0 = XMLM.shape[0]

    XMRL_shape_0 = XMRL.shape[0]
    XMRR_shape_1 = XMRR.shape[1]
    XMRM_shape_0 = XMRM.shape[0]

    XMML_shape_0 = XMML.shape[0]
    XMMR_shape_1 = XMMR.shape[1]
    XMMM_shape_0 = XMMM.shape[0]

    total_sum = (XLLL_shape_0 + XLLR_shape_1 + XLLM_shape_0) + \
                (XLRL_shape_0 + XLRR_shape_1 + XLRM_shape_0) + \
                (XLML_shape_0 + XLMR_shape_1 + XLMM_shape_0) + \
                (XRLL_shape_0 + XRLR_shape_1 + XRLM_shape_0) + \
                (XRRL_shape_0 + XRRR_shape_1 + XRRM_shape_0) + \
                (XRML_shape_0 + XRMR_shape_1 + XRMM_shape_0) + \
                (XMLL_shape_0 + XMLR_shape_1 + XMLM_shape_0) + \
                (XMRL_shape_0 + XMRR_shape_1 + XMRM_shape_0) + \
                (XMML_shape_0 + XMMR_shape_1 + XMMM_shape_0)

    print('original data size:', len(v))
    print('reduced data size:', total_sum)

    XLL_RECON=PCARecon(XLLL,XLLR,XLLM)
    XLR_RECON=PCARecon(XLRL,XLRR,XLRM)
    XLM_RECON=PCARecon(XLML,XLMR,XLMM)

    XRL_RECON = PCARecon(XRLL, XRLR, XRLM)
    XRR_RECON = PCARecon(XRRL, XRRR, XRRM)
    XRM_RECON = PCARecon(XRML, XRMR, XRMM)

    XML_RECON = PCARecon(XMLL, XMLR, XMLM)
    XMR_RECON = PCARecon(XMRL, XMRR, XMRM)
    XMM_RECON = PCARecon(XMML, XMMR, XMMM)

    XL_RECON=PCARecon(XLL_RECON,XLR_RECON,XLM_RECON)
    XR_RECON=PCARecon(XRL_RECON,XRR_RECON,XRM_RECON)
    XM_RECON=PCARecon(XML_RECON,XMR_RECON,XMM_RECON)

    X_RECON=PCARecon(XL_RECON,XR_RECON,XM_RECON)
    
    v2=X_RECON.reshape(1,-1)[0]
    return v2


def reshape_to_nearly_square(data):
    n = data.shape[0]
#     return data.reshape(-1,2)

    sqrt_n = int(np.sqrt(n))
    a, b = None, None

    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            a = n // i
            b = i
            if a >= b:
                return data.reshape(a, b)
    raise ValueError("cannot find suitable a and b")

def iterative_pca(data):
    x = reshape_to_nearly_square(data)

    num_components = 1
    pca = PCA(n_components=num_components)
    pca.fit(x)

    left_matrix = pca.transform(x).flatten()  # Left matrix (2000)
    right_matrix = pca.components_.flatten()  # Right matrix (1250)
    mean_matrix = pca.mean_  # Mean matrix (1250)

    n = left_matrix.shape[0] + right_matrix.shape[0] + mean_matrix.shape[0]

    print(left_matrix.shape[0])
    print(right_matrix.shape[0])
    print(mean_matrix.shape[0])
    return pca,left_matrix, right_matrix, mean_matrix, n


def reconstruct_data(left_matrix, right_matrix, mean_matrix):
    left_matrix_reshaped = left_matrix[:, np.newaxis]  # (2000, 1)
    reconstructed_data = np.dot(left_matrix_reshaped, right_matrix[np.newaxis, :]) + mean_matrix  # (2000, 1250)
    return reconstructed_data.flatten()

def myPCA_auto(data,nout):
    v0=None
    if len(data)%2!=0:
        v0=data[0]
        print(v0)
        data=data[1:len(data)]
    
    dicts = []
    pcasss = []
    dicts.append(data)
    ns_list = []
    closest_ns = None
    s = 0
    lastns= nout
    while True:
        ns = 0
        tempdict = dicts[(3 ** s - 1) // 2:(3 ** (s + 1) - 1) // 2]
        print('level %s' % s, len(tempdict))
        for i in range(len(tempdict)):
            pca, left_matrix, right_matrix, mean_matrix, n = iterative_pca(tempdict[i])
            pcasss.append(pca)
            dicts.append(left_matrix)
            dicts.append(right_matrix)
            dicts.append(mean_matrix)
            ns = ns + n
        print('*'*100)
        if len(ns_list)>0:
            print(ns,ns_list[-1])
        if len(ns_list) > 0 and abs(ns - lastns) > abs(ns_list[-1] - lastns):
            print('number of points:',ns_list[-1])
            ss= s+1
            out = dicts[(3 ** ss - 1) // 2:(3 ** (ss + 1) - 1) // 2]
            with open('out.pkl', 'wb') as f:
                pickle.dump(out, f)

            break
        ns_list.append(ns)
        s = s + 1

    with open('out.pkl', 'rb') as f:
        loaded_arrays = pickle.load(f)

    # reconstruct
    n1 = 0
    dicts=[]
    N=0
    rawN=len(data)
    while(N<rawN):
#         print(N)
        for i in range(int(len(loaded_arrays)/3)):
            tenpdata =reconstruct_data(loaded_arrays[n1], loaded_arrays[n1+1], loaded_arrays[n1+2])
            n1 = n1+3
            dicts.append(tenpdata)
#             print(tenpdata.shape)
        N=tenpdata.shape[0]
        loaded_arrays=dicts
        n1=0
        dicts=[]
#     print(tenpdata.shape)

#     if os.path.exists('out.pkl'):
#         os.remove('out.pkl')    
    
    if v0 is not None:
        tenpdata=np.insert(tenpdata,0,v0)
    return tenpdata

def subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,t,v,\
    gridVertical=False,gridHorizontal=False,gridStep=1,\
    isPlot=True,resultFile="",plotMarker=False,interact="",interact_ratio=0,lineColor='k',pngDir=".",
    transform=False,om3_v=None):
    full_frame(width,height,dpi)

    t_min=min(t)
    t_max=max(t)

    # adjust value range for zoom-in
    # calculate here because v are sampled later
    if interact == "zoomin": # interact_ratio should be smaller than 0.5
        idx1=np.where(t>=t_min+(t_max-t_min)*interact_ratio)[0][0]
        idx2=np.where(t<=t_max-(t_max-t_min)*interact_ratio)[0][-1]
        v_min=min(v[idx1:idx2+1])
        v_max=max(v[idx1:idx2+1])
    else:
        v_min=min(v)
        v_max=max(v)
    
    returnPreselection=False

    print("===================",downsample,"===================")

    if str(downsample) == "MinMaxDownsampler":
        s_ds = MinMaxDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "MinMaxFPLPDownsampler":
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "ILTSParallelDownsampler":
        s_ds = ILTSParallelDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    ############## MinMaxLTTB family ##################
    elif str(downsample) == "MinMaxLTTB1Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*1+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*1)
        # Select downsampled data
        t_pre = t[s_ds] # note that this is deep copy, so safe when later t is modified
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxLTTB2Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*2+2)
        # divide [t1,tn] plus global FP&LP
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*2) 
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        # divide [t1,tn]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxLTTB4Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*4+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*4)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxLTTB6Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*6+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*6)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxLTTB8Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*8+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*8)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    ############## MinMaxILTS family ##################
    elif str(downsample) == "MinMaxILTS1Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*1+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*1)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxILTS2Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*2+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*2)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxILTS4Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*4+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*4)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxILTS6Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*6+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*6)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxILTS8Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*8+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*8)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    ############## M4LTTB family ##################
    elif str(downsample) == "M4LTTB1Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*1)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4LTTB2Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*2)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4LTTB4Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*4)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4LTTB6Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*6)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4LTTB8Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*8)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    ############## M4ILTS family ##################
    elif str(downsample) == "M4ILTS1Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*1)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4ILTS2Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*2)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4ILTS4Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*4)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4ILTS6Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*6)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4ILTS8Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*8)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    ############## family end ##################
    elif str(downsample) == "MinMaxGapDownsampler":
        s_ds = MinMaxGapDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
#     elif downsample == "MinMaxLTTBDownsampler":
#         s_ds = MinMaxLTTBDownsampler().downsample(t, v, n_out=nout)
#         # Select downsampled data
#         t = t[s_ds]
#         v = v[s_ds]
    elif str(downsample) == "M4Downsampler":
        s_ds = M4Downsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
#     elif str(downsample) == "EveryNthDownsampler":
#         s_ds = EveryNthDownsampler().downsample(t, v, n_out=nout)
#         # Select downsampled data
#         t = t[s_ds]
#         v = v[s_ds]
    elif str(downsample) == "LTOBETDownsampler":
        s_ds = LTOBETDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    # elif str(downsample) == "LLBDownsampler":
    #     s_ds = LLBDownsampler().downsample(t, v, n_out=nout)
    #     # Select downsampled data
    #     t = t[s_ds]
    #     v = v[s_ds]
    elif str(downsample) == "LTTBETDownsampler":
        s_ds = LTTBETDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETFurtherDownsampler":
        s_ds = LTTBETFurtherDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "ILTSWaveletDownsampler":
        s_ds = ILTSWaveletDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "ILTSLTDDownsampler":
        s_ds = ILTSLTDDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETFurtherRandomDownsampler": # init with random points
        s_ds = LTTBETFurtherRandomDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETFurtherMinMaxDownsampler": # init with MinMaxFPLP
        s_ds = LTTBETFurtherMinMaxDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTSDownsampler":
        s_ds = LTSDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETNewDownsampler":
        s_ds = LTTBETNewDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETGapDownsampler":
        s_ds = LTTBETGapDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTOBETGapDownsampler":
        s_ds = LTOBETGapDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBDownsampler":
        s_ds = LTTBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTOBDownsampler":
        s_ds = LTOBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTDDownsampler":
        s_ds = LTDDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTDOBDownsampler":
        s_ds = LTDOBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "visval":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        simplifier = VWSimplifier(points)
        tmp=simplifier.from_number(nout)
        t=tmp[:,0]
        v=tmp[:,1]
    elif str(downsample) == "rdp":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        # tmp=np.array(rdp(points, epsilon=nout)) # python native rdp is too slow
        tmp=np.array(simplify_coords(points, nout)) # nout is used as epsilon
        t=tmp[:,0]
        v=tmp[:,1]
    elif str(downsample) == "fastVisval":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        tmp=np.array(simplify_coords_vw(points, nout)) # nout is used as area
        t=tmp[:,0]
        v=tmp[:,1]
    elif str(downsample) == "sdt":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        tmp=np.array(list(swinging_door(points, deviation=nout, mode=True, step=float('inf')))) #nout used as deviation
        t=tmp[:,0]
        v=tmp[:,1]
    # elif str(downsample) == "pla": # very very slow
    #     model = pwlf.PiecewiseLinFit(t,v)
    #     breakpoints = model.fit(nout)
    #     t=breakpoints
    #     v=model.predict(t)
    elif str(downsample) == "reuwi":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        points=np.array(points)
        mask = simplify_reumann_witkam(points,nout) # nout is used as perpendicular distance
        t=points[mask,0]
        v=points[mask,1]
    elif str(downsample) == "reuwi_residual":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        points=np.array(points)
        mask = simplify_reumann_witkam_residual(points,nout) # nout is used as perpendicular distance
        t=points[mask,0]
        v=points[mask,1]
    elif str(downsample) == "interpolation_residual":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        points=np.array(points)
        mask = simplify_interpolation_residual(points,nout) # nout is used as perpendicular distance
        t=points[mask,0]
        v=points[mask,1]
    elif str(downsample) == "SimPiece":
        df=pd.read_csv(resultFile,header=None) # assume no header by default
        t=df.iloc[:,0]
        v=df.iloc[:,1]
    elif str(downsample) == "FSW":
        df=pd.read_csv(resultFile,header=None) # assume no header by default
        t=df.iloc[:,0]
        v=df.iloc[:,1]
    elif str(downsample) == "swab": # from matlab swab result
        df=pd.read_csv(resultFile,header=None) # assume no header by default
        t=df.iloc[:,0]
        v=df.iloc[:,1]
    elif str(downsample) == "PySWAB": # from python version 
        points=np.array(list(zip(t,v)))
        max_error=getPySWABParam(nout, t, v, epsilon=2)
        num_segments=nout-1 # m是分段数，虽然SWAB不能直接控制分段数，但是也需要这个参数来确定buffer大小
        M=seg_swab_withTimestamps(points, max_error, num_segments) # 使用真实时间戳
        t=M[:,0]
        v=M[:,1]
    elif str(downsample) == "PyBottomUp": # from python version 
        points=np.array(list(zip(t,v)))
        num_segments=nout-1 # m是分段数，seg_bottom_up能直接控制分段数
        M=seg_bottom_up_withTimestamps(points, num_segments) # 使用真实时间戳
        t=M[:,0]
        v=M[:,1]
    elif str(downsample) == "uniform":
        t_tmp=t[0:len(t):int(len(t)/(nout-1))]
        v_tmp=v[0:len(t):int(len(t)/(nout-1))]
        t=np.array(t_tmp)
        v=np.array(v_tmp)
    elif str(downsample) == "uniformTime":
        bins = _get_bin_idxs(t, nout-1)
        t_tmp=[]
        v_tmp=[]
        for i in range(len(bins)-1):
            t_tmp.append(t[bins[i]])
            v_tmp.append(v[bins[i]])
        t_tmp.append(t[-1])
        v_tmp.append(v[-1])
        t=np.array(t_tmp)
        v=np.array(v_tmp)
    elif str(downsample)=="PAA":
        bins = _get_bin_idxs(t, nout)
        t_tmp=[]
        v_tmp=[]
        for lower, upper in zip(bins, bins[1:]):
            t_tmp.append(t[lower])
            v_tmp.append(np.mean(v[lower:upper]))
        t_tmp.append(t_max)
        v_tmp.append(v_tmp[-1])
        t=np.array(t_tmp)
        v=np.array(v_tmp)
    elif str(downsample)=="PyShrinkingCone":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        points=np.array(points)
        mask = simplify_shrinking_cone(points,nout) # nout is used as tolerance
        t=points[mask,0]
        v=points[mask,1]
    elif str(downsample) == "ShrinkingCone":
        df=pd.read_csv(resultFile,header=None) # assume no header by default
        t=df.iloc[:,0]
        v=df.iloc[:,1]
    elif str(downsample) == "PyDFT":
        v = myDFT(v,nout)
        # t = np.arange(0,len(v)) # NOTE this
        # t use original input 
    elif str(downsample) == "PyDWT":
        v = myDWT(v,nout)
        # t = np.arange(0,len(v)) # NOTE this
        # t use original input 
    elif str(downsample) == "PCA":
        v = myPCA(v)
        # t = np.arange(0,len(v)) # NOTE this
        # t use original input 
    elif str(downsample) == "PCA_auto":
        v = myPCA_auto(v,nout)
    elif str(downsample) == "OM3":
        v = om3_v
        t = np.arange(len(v))

    # elif str(downsample) == "DWT":
    #     df=pd.read_csv(resultFile,header=None) # assume no header by default
    #     t=df.iloc[:,0]
    #     v=df.iloc[:,1] # need reconstruct！
    else:
        downsample="original"

    # print("number of output points=", len(t))

    if transform==True:
        # transform
        # note that use t_min, t_max, v_min, v_max that are recored before sampling 
        # scale t -> x
        x=(t-t_min)/(t_max-t_min)*width
        x=np.floor(x)
        # scale v -> y
        y=(v-v_min)/(v_max-v_min)*height
        y=np.floor(y)

        t=x
        v=y

        t_min=0
        t_max=width
        v_min=0
        v_max=height
    
    if plotMarker: # deprecated
        plt.plot(t,v,'--o',color=lineColor,linewidth=lw,antialiased=anti,markersize=lw*20)
    else:
        if str(downsample)=="PAA": # customized plot for PAA
            # t_steps = np.concatenate([t, [t_max]])
            # v_steps = np.concatenate([v, [v[-1]]])
            # plt.step(t_steps,v_steps,where='post',linewidth=lw,antialiased=anti,color='k')
            plt.step(t,v,where='post',linewidth=lw,antialiased=anti,color=lineColor)
        else:
            plt.plot(t,v,'-',color=lineColor,linewidth=lw,antialiased=anti)

    print(interact)
    if interact == "lpan": # left panning 
        plt.xlim(t_min-(t_max-t_min)*interact_ratio, t_max-(t_max-t_min)*interact_ratio)
    elif interact == "rpan": # right panning
        plt.xlim(t_min+(t_max-t_min)*interact_ratio, t_max+(t_max-t_min)*interact_ratio)
    elif interact == "zoomin": # interact_ratio should be smaller than 0.5
        plt.xlim(t_min+(t_max-t_min)*interact_ratio, t_max-(t_max-t_min)*interact_ratio)
    elif interact == "zoomout":
        plt.xlim(t_min-(t_max-t_min)*interact_ratio, t_max+(t_max-t_min)*interact_ratio)
    else:
        plt.xlim(t_min, t_max)

    plt.ylim(v_min, v_max) # already adjusted at the beginning if zoom-in
    
    plt.savefig(os.path.join(pngDir,name+'.png'),backend='agg')
    plt.close()

    if isPlot:
        img=cv2.imread(os.path.join(pngDir,name+'.png'))
        plt.imshow(img)
        ax = plt.gca()
        if gridVertical==True:
            add_grid_column(ax, width, height, gridStep, gridHorizontal)
    
    if returnPreselection==False:
        return t,v
    else:
        return t,v,t_pre,v_pre

def _get_bin_idxs_gapAware(x: np.ndarray, nb_bins: int) -> np.ndarray:
    bins = np.searchsorted(x, np.linspace(x[0], x[-1], nb_bins + 1), side="left")
    bins[-1] = len(x)
    return bins

def _get_bin_idxs(x: np.ndarray, nb_bins: int) -> np.ndarray:
    bins=_get_bin_idxs_gapAware(x,nb_bins)
    return np.unique(bins)


def read_ucr(filename,col=0): # col starting from 0
    df=pd.read_csv(filename, header=None, delimiter=r"\s+")
    df2=df.T
    v=df2.iloc[1:,col]
    v=v.to_numpy(dtype='float')
    t = np.linspace(0, 10, num=len(v))
    return t,v

def my_get_bin_idxs(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # Thanks to the `linspace` the data is evenly distributed over the index-range
    # The searchsorted function returns the index positions
    bins = np.searchsorted(x, np.linspace(x[0], x[-1], nb_bins + 1), side="right")
    bins[0] = 0
    bins[-1] = len(x)
    return np.unique(bins)