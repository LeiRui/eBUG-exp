import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from csv import reader
import math
from scipy import signal
import pywt

def flatCoeffs(coeffs):
    tmp=[]
    for arr in coeffs:
        tmp.append(arr.tolist()[:])
    flattened_list = [item for sublist in tmp for item in sublist]
    return flattened_list

def padZero(v):
    target_length = 2**int(np.ceil(np.log2(len(v))))
    pad_length = target_length - len(v)
    data=np.pad(v, (0, pad_length), mode='constant', constant_values=0)
    return data

def plotDWTCoeff(coeffs,threshold=None):
    tmp=[]
    for arr in coeffs:
        tmp.append(arr.tolist()[:])
    flattened_list = [item for sublist in tmp for item in sublist]
    print(len(flattened_list))
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0,len(flattened_list)),np.abs(flattened_list))
    levels=len(coeffs)-1
    for i in range(1,levels+1):
        plt.axvline(x=len(flattened_list)/2**i,color='r')
    if threshold!=None:
        plt.axhline(y=threshold,color='r')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Frequency vs Power (DWT)')
    plt.show()
    plt.close()
    
def plotDWTCoeffPloty(coeffs,threshold=None,thresholdByLevel=None,log=False,simple=True):
    import plotly.graph_objects as go;
    fig = go.Figure()
    flattened_list=flatCoeffs(coeffs)
    levels=len(coeffs)-1
    if simple: # 不画成一条条的竖线
        fig.add_trace(go.Scatter(x=np.arange(0,len(flattened_list)),y=np.abs(flattened_list), \
                                 name="DWT coefficients"))
        if threshold is not None:
            fig.add_trace(go.Scatter(x=[0,len(flattened_list)],y=[threshold,threshold],name="threshold"))
        elif thresholdByLevel is not None:
            threshold=thresholdByLevel[:2] + [x for i, x in enumerate(thresholdByLevel[2:], start=1) for _ in range(2**i)]
            fig.add_trace(go.Scatter(x=np.arange(0,len(flattened_list)),y=threshold,name="threshold"))   
    else:
        if threshold is None and thresholdByLevel is None:
            fig.add_trace(go.Scatter(x=np.arange(0,len(flattened_list)),y=np.abs(flattened_list), \
                                     name="DWT coefficients",\
                                     mode="lines+markers",marker=dict(size=4, color='blue')))
        elif threshold is not None:
            # 绘制比 threshold 大的部分（蓝色竖线）
            above_threshold = np.abs(flattened_list) >= threshold
            for x, y in zip(np.arange(0, len(flattened_list))[above_threshold], np.abs(flattened_list)[above_threshold]):
                fig.add_trace(go.Scatter(
                    x=[x, x],  # 固定 x 值，表示竖线
                    y=[0, y],  # 从 y=0 到当前值 y
                    mode="lines",
                    line=dict(color='blue', width=2),  # 蓝色线
                    showlegend=False  # 不重复显示图例
                ))

            # 绘制比 threshold 小的部分（红色竖线）
            below_threshold = np.abs(flattened_list) < threshold
            for x, y in zip(np.arange(0, len(flattened_list))[below_threshold], np.abs(flattened_list)[below_threshold]):
                fig.add_trace(go.Scatter(
                    x=[x, x],  # 固定 x 值，表示竖线
                    y=[0, y],  # 从 y=0 到当前值 y
                    mode="lines",
                    line=dict(color='red', width=2),  # 红色线
                    showlegend=False  # 不重复显示图例
                ))
            # 横线threshold
            fig.add_trace(go.Scatter(x=[0,len(flattened_list)],y=[threshold,threshold],name="threshold"))

        elif thresholdByLevel is not None:
            threshold=thresholdByLevel[:2] + [x for i, x in enumerate(thresholdByLevel[2:], start=1) for _ in range(2**i)]
            above_threshold = np.abs(flattened_list) >= threshold
            for x, y in zip(np.arange(0, len(flattened_list))[above_threshold], np.abs(flattened_list)[above_threshold]):
                fig.add_trace(go.Scatter(
                    x=[x, x],  # 固定 x 值，表示竖线
                    y=[0, y],  # 从 y=0 到当前值 y
                    mode="lines",
                    line=dict(color='blue', width=2),  # 蓝色线
                    showlegend=False  # 不重复显示图例
                ))

            # 绘制比 threshold 小的部分（红色竖线）
            below_threshold = np.abs(flattened_list) < threshold
            for x, y in zip(np.arange(0, len(flattened_list))[below_threshold], np.abs(flattened_list)[below_threshold]):
                fig.add_trace(go.Scatter(
                    x=[x, x],  # 固定 x 值，表示竖线
                    y=[0, y],  # 从 y=0 到当前值 y
                    mode="lines",
                    line=dict(color='red', width=2),  # 红色线
                    showlegend=False  # 不重复显示图例
                ))
            # 横线threshold
            fig.add_trace(go.Scatter(x=np.arange(0,len(flattened_list)),y=threshold,name="threshold"))

        

    # 每层level分割虚线 按照aL,dL,dL-1,...,d1排列
    for i in range(1,levels+1):
        fig.add_trace(go.Scatter(x=[len(flattened_list) / 2**i,len(flattened_list) / 2**i],\
                                 y=[min(np.abs(flattened_list)),max(np.abs(flattened_list))],\
                                 mode="lines",
                                 name="scale{}".format(i),
                                 line=dict(dash="dash")))
    fig.update_layout(
        autosize=False,
        width=850,
        height=650,
    )
    if log:
        fig.update_xaxes(type="log")
    fig.show()

def DWTCoeffTree(data,coeffs_raw,threshold=None,thresholdByLevel=None,node_size=500,font_size=5):
    import networkx as nx
    
    # coeffs_values = [
    #     np.array([1.25]), 
    #     np.array([-2.75]), 
    #     np.array([3.8890873, 0.0]), 
    #     np.array([0.0, -5.5, 0.0, 0.0]), 
    #     np.array([0.0, 0.0, -7.77817459, 0.0, 0.0, 0.0, 0.0, 0.0])
    # ]
    # print(coeffs_raw)

    coeffs_values=copy.deepcopy(coeffs_raw)
    coeffs_values.append(data)
    # print(coeffs_values)

    # 将系数转换为保留两位小数的字符串
    coeffs_values_str = [', '.join(['{:.2f}'.format(val) for val in layer]) for layer in coeffs_values]
    # print(coeffs_values_str)

    # 小波系数节点列表
    # coeffs_name = [
    #     ['a41'],               # 第一层: [a41]
    #     ['d41'],               # 第二层: [d41]
    #     ['d31', 'd32'],        # 第三层: [d31, d32]
    #     ['d21', 'd22', 'd23', 'd24'],  # 第四层: [d21, d22, d23, d24]
    #     ['d11', 'd12', 'd13', 'd14', 'd15', 'd16', 'd17', 'd18']  # 第五层: [d11, d12, ...]
    # ]
    levelNum=len(coeffs_raw)-1
    print(levelNum)
    coeffs_name = []
    coeffs_name.append([f"a{levelNum}_1"])
    # for i, layer_values in enumerate(coeffs_values):
    for i in range(1,levelNum+1):
        layer_values=coeffs_values[i]
        layer_name = []
        for j, value in enumerate(layer_values):
            node_name = f"d{levelNum-i+1}_{j+1}"
            layer_name.append(node_name)
        coeffs_name.append(layer_name)
#     print(coeffs_name)

    # 原始数据
    layer_name = []
    for i in range(1,len(data)+1):
        node_name = f"v{i}"
        layer_name.append(node_name)
    coeffs_name.append(layer_name)

    # 创建图
    G = nx.DiGraph()

    # 添加根节点
    G.add_node(coeffs_name[0][0])

    # 连接节点
    previous_layer = coeffs_name[0]  # 起始层是a41

    # 手动设置节点位置
    pos = {}

    # 每层的节点数
    layer_sizes = [len(layer) for layer in coeffs_name]

    # 计算每层的x坐标偏移量
    # x_offsets = [sum(layer_sizes[:i+1]) for i in range(len(layer_sizes))]
    x_offsets=np.zeros(len(layer_sizes))

    # 设置根节点位置
    pos[coeffs_name[0][0]] = (0, 0)

    # 计算其他节点位置
    for i in range(1, len(coeffs_name)):
        current_layer = coeffs_name[i]
        for j, node in enumerate(current_layer):
            # 计算当前节点的x坐标
            x = x_offsets[i] - (len(current_layer) - 1) / 2 + j
            # 计算当前节点的y坐标
            y = -i
            pos[node] = (x, y)
            # 添加边
            if i == 1:
                G.add_edge(coeffs_name[0][0], node)
            else:
                parent_node_1 = coeffs_name[i-1][j//2]
                G.add_edge(parent_node_1, node)

    # print(pos)

    # 绘制树形结构
    labels = {}
    for i in range(0, len(coeffs_name)):
        for node in coeffs_name[i]:
    #         print(node,coeffs_name[i].index(node),coeffs_values_str[i])
            labels[node] = coeffs_values_str[i].split(', ')[coeffs_name[i].index(node)]
    # print(labels)

    # 改变带有"v"的节点颜色为红色，或者不带"v"但对应的浮点数绝对值大于5的节点也标红
    node_colors = []
    for node in G.nodes():
        if 'v' in node:
            node_colors.append('green')
        else:
            for i, (layer, values) in enumerate(zip(coeffs_name, coeffs_values)):
                if node in layer:
#                     print(node,i,layer,values)
                    if threshold is not None:
                        if abs(values[layer.index(node)]) >= threshold:
                            node_colors.append('red')
                        else:
                            node_colors.append('skyblue')
                    elif thresholdByLevel is not None:
                        if abs(values[layer.index(node)]) >= thresholdByLevel[i]:
                            node_colors.append('red')
                        else:
                            node_colors.append('skyblue')
                    else:
                        node_colors.append('skyblue')
    # print(node_colors)

    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=node_size, node_color=node_colors, \
        font_size=font_size, font_weight='bold', arrows=False)

    # 显示图
    plt.title('Wavelet Coefficients Tree Visualization')
    plt.show()

    
def plotWaveletRecon(data,coeffs,wavelet,twoColumn=False):
    # 初始化重建的信号列表
    trend_reconstructed = []
    anomaly_reconstructed = []

    # 除去一个a之后的d的个数等于分解层数，验证pywt.dwt_max_level(len(data), 'db1')查看自动分解层数
    levels=len(coeffs)-1
    print(levels)

    # 逐层重建趋势和异常
    for i in range(levels): #从最下面一层往上，并且最下面的a单独重建一下不加
        coeffs_trend = [c if j <= i else np.zeros_like(c) for j, c in enumerate(coeffs)]
        coeffs_anomaly = [c if j == i+1 else np.zeros_like(c) for j, c in enumerate(coeffs)]

        trend_reconstructed.append(pywt.waverec(coeffs_trend, wavelet))
        anomaly_reconstructed.append(pywt.waverec(coeffs_anomaly, wavelet))

    # 绘制结果
    plt.figure(figsize=(12, 15))
    
    if twoColumn: # original, (aL,dL), (aL-1,dL-1), ..., (a1,d1)
        plt.subplot(1+levels, 1, 1)
        plt.plot(data, label='Original Data')
        plt.title('Original Data')

        for i, (trend, anomaly) in enumerate(zip(trend_reconstructed, anomaly_reconstructed)):
            plt.subplot(1+levels, 2, 2*i+3)
            plt.plot(trend)
            plt.ylabel(f'a{levels-i}')

            plt.subplot(1+levels, 2, 2*i+4)
            plt.plot(anomaly)
            plt.ylabel(f'd{levels-i}')

        plt.tight_layout()
        plt.show()
    else: # original, aL, dL, dL-1, ..., d1
        plt.subplot(2+levels, 1, 1)
        plt.plot(data, label='Original Data')
        plt.title('Original Data')

        plt.subplot(2+levels, 1, 2)
        plt.plot(trend_reconstructed[0])
        plt.ylabel(f'a{levels}')

        for i in range(len(anomaly_reconstructed)):
            plt.subplot(2+levels, 1, i+3)
            plt.plot(anomaly_reconstructed[i])
            plt.ylabel(f'd{levels-i}')

        plt.tight_layout()
        plt.show()
     
    
def detailCoeffMappedTime(data,coeffs,threshold=None,thresholdByLevel=None):
    # 注意coeffs是原始的没有经过thresholding的
    levels=len(coeffs)-1
    
    fig=plt.figure(figsize=(12, 18))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(levels+2, 1, figure=fig)
    
    ###################################################
    # # all reserved coefficients mapped to the original time domain
    ax1 = fig.add_subplot(gs[0:2, 0])
    plt.sca(ax1)
#     select_coeffs=[]
    cmap = plt.cm.get_cmap('tab10')
    for i in range(1,levels+2): # d1,d2,...,dL,aL频域从高到低
        y = np.abs(coeffs[-i])
        x = np.arange(len(y))
        x_mapped = x / len(y) * len(data) # 映射对应到原始数据time轴位置
        if threshold is not None:
            above_threshold = y >= threshold
        else:
            above_threshold = y >= thresholdByLevel[-i]
        plt.vlines(x_mapped[above_threshold], 0, y[above_threshold], colors=cmap(i/levels), linewidth=1,label="d{}".format(i))
#         select_coeffs.append([x_mapped[above_threshold],y[above_threshold]])
            
    plt.plot(data-max(data))
    plt.title('All reserved coefficients')
    plt.legend(ncol=levels, loc='upper center', bbox_to_anchor=(0.5, 1.05))

    ###################################################
    # reserved coefficients of each level mapped to the original time domain
    for i in range(1,levels+1):
        ax = fig.add_subplot(gs[i+1, 0])
        plt.sca(ax)
        y = np.abs(coeffs[-i])
        x = np.arange(len(y)) 
        x_mapped = x / len(y) * len(data) # 映射对应到原始数据time轴位置

        if threshold is not None:
            above_threshold = y >= threshold
        else:
            above_threshold = y >= thresholdByLevel[-i]
        plt.vlines(x_mapped[above_threshold], 0, y[above_threshold], colors='b', linewidth=1)

        if threshold is not None:
            below_threshold = y < threshold
        else:
            below_threshold = y < thresholdByLevel[-i]
        plt.vlines(x_mapped[below_threshold], 0, y[below_threshold], colors='r', linewidth=1)
            
        plt.plot(data-max(data))
        plt.ylabel(f'd{i}')

    ###################################################
    fig.tight_layout()
    fig.show()
    
#     return select_coeffs

def detailCoeffMappedTimePloty(data,coeffs,threshold=None,thresholdByLevel=None):
    # 注意coeffs是原始的没有经过thresholding的
    import plotly.express as px 
    import plotly.graph_objects as go; 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0,len(data)),y=data, name="original"))
    levels=len(coeffs)-1
    for i in range(1,levels+2): # d1,d2,...,dL,aL频域从高到低
        y = np.abs(coeffs[-i])
        x = np.arange(len(y))
        x_mapped = x / len(y) * len(data) # 映射对应到原始数据time轴位置

        if threshold is not None:
            above_threshold = y >= threshold
        else:
            above_threshold = y >= thresholdByLevel[-i]

        color = np.random.rand(3)
        color_str = 'rgb({}, {}, {})'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))  # 转换为 rgb 字符串
        
        for x, y in zip(x_mapped[above_threshold],y[above_threshold]):
            fig.add_trace(go.Scatter(
                x=[x, x], 
                y=[0, -y], 
                mode="lines",
                line=dict(color=color_str, width=2), 
                showlegend=False  
            ))
            
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # 空的 x 和 y 来创建图例条目
            mode='lines', 
            line=dict(
                color=color_str,  
                width=3  
            ),
            name="d{}".format(i),  
            showlegend=True  
        ))
    fig.update_layout(
        autosize=False,
        width=850,
        height=650,
    )
    fig.show()

    
def threshold_naive(coeffs_input,nout):
    # 这种做法应该就是类似于OM3里的haar做法吧，类似某一粒度的低频average平滑结果
    coeffs=copy.deepcopy(coeffs_input)
    cnt=0
    for i in range(len(coeffs)):
        for j in range(len(coeffs[i])):
            if cnt>nout:
                coeffs[i][j]=0
            else:
                cnt+=1
    print('【abs naive】',cnt)
    return coeffs

def threshold_abs(coeffs_input,nout):
    coeffs=copy.deepcopy(coeffs_input) # 注意deepcopy
    flattened_list=flatCoeffs(coeffs)
    tmp2=np.sort(np.abs(flattened_list))
    tmp2=tmp2[::-1]
    threshold = tmp2[nout-1] # 从0开始编号
    # print(threshold)

    if threshold==0:
        # 当很多0，threshold取到0，只统计非零的
        if tmp2[0]>0:
            threshold=tmp2[np.where(tmp2>0)[0][-1]]
        else: # tmp2[0]=0也就是说全部都是0
            threshold=1
    # print(threshold)
    
    cnt=0
    for i in range(len(coeffs)):
        for j in range(len(coeffs[i])):
            # print(i,j,np.abs(coeffs[i][j]))
            if np.abs(coeffs[i][j])<threshold: #不要漏了abs!!!!!!!!!!!!!!!!!!!
                coeffs[i][j]=0
            else:
                cnt+=1
    print('【abs global】',cnt)
    
    # if plot:
    #     plotDWTCoeffPloty(coeffs_input,threshold=threshold,log=False)
    
    return coeffs,threshold

def threshold_abs_levelwise(coeffs_input,nout):
    coeffs=copy.deepcopy(coeffs_input) # 注意deepcopy

    N=len(flatCoeffs(coeffs))
    r=nout/N
    
    thresholdByLevel=[]
    cnt=0
    for i in range(len(coeffs)):
        tmp=coeffs[i]
        tmp2=np.sort(np.abs(tmp))
        tmp2=tmp2[::-1]

        nout=int(np.ceil(len(tmp)*r)) # ceil的话每层至少保留一个系数
        threshold=tmp2[nout-1]

        # nout=int(np.round(len(tmp)*r))
        # if nout==0:
        #     threshold=tmp2[0]+1 # larger than any one
        # else:
        #     threshold=tmp2[nout-1]
        
        if threshold==0:
            # 当很多0，threshold取到0，只统计非零的
            if tmp2[0]>0:
                threshold=tmp2[np.where(tmp2>0)[0][-1]]
            else: # tmp2[0]=0也就是说全部都是0
                threshold=1
        thresholdByLevel.append(threshold)
        for j in range(len(coeffs[i])):
            if np.abs(coeffs[i][j])<threshold: #不要漏了abs!!!!!!!!!!!!!!!!!!!
                coeffs[i][j]=0
            else: # >=threshold>0
                cnt+=1
    print('【abs levelwise】',cnt)
    
    # if plot:
    #     plotDWTCoeffPloty(coeffs_input,thresholdByLevel=thresholdByLevel,log=False)
    
    return coeffs,thresholdByLevel

def threshold_abs_mixed(coeffs_input,nout):
    coeffs=copy.deepcopy(coeffs_input) # 注意deepcopy

    N=len(flatCoeffs(coeffs))
    r=nout/N
    
    thresholdByLevel=[]
    cnt=0
    for i in range(len(coeffs)):
        tmp=coeffs[i]
        tmp2=np.sort(np.abs(tmp))
        tmp2=tmp2[::-1]

        noutTmp=int(np.ceil(len(tmp)*r)) # ceil的话每层至少保留一个系数
        threshold=tmp2[noutTmp-1]
        
        if threshold==0:
            # 当很多0，threshold取到0，只统计非零的
            if tmp2[0]>0:
                threshold=tmp2[np.where(tmp2>0)[0][-1]]
            else: # tmp2[0]=0也就是说全部都是0
                threshold=1
        thresholdByLevel.append(threshold)
        for j in range(len(coeffs[i])):
            if np.abs(coeffs[i][j])<threshold: #不要漏了abs!!!!!!!!!!!!!!!!!!!
                coeffs[i][j]=0
            else: # >=threshold>0
                cnt+=1
    print('abs mixed1',cnt)

    flattened_list=flatCoeffs(coeffs_input)
    tmp2=np.sort(np.abs(flattened_list))
    tmp2=tmp2[::-1]
    threshold = tmp2[nout-1] # 从0开始编号
    if threshold==0:
        # 当很多0，threshold取到0，只统计非零的
        if tmp2[0]>0:
            threshold=tmp2[np.where(tmp2>0)[0][-1]]
        else: # tmp2[0]=0也就是说全部都是0
            threshold=1
    print(threshold)
    for i in range(len(coeffs)):
        for j in range(len(coeffs[i])):
            if np.abs(coeffs_input[i][j])>=threshold: #不要漏了abs!!!!!!!!!!!!!!!!!!!
                coeffs[i][j]=coeffs_input[i][j]
                cnt+=1

    print('abs mixed2',cnt)
    
    return coeffs,thresholdByLevel

def decayWeights(tmpWeights,startPos,endPos):
    from scipy.ndimage import gaussian_filter1d
    smoothed_weights = gaussian_filter1d(tmpWeights, sigma=0.5) # sigma越大越平滑
    smoothed_weights /= smoothed_weights.sum() # 归一化以保证累加和为 1

    # 原本的区域平均化，构造平台
    tmp=sum(smoothed_weights[startPos:endPos])
    smoothed_weights[startPos:endPos]=tmp/(endPos-startPos)

    return smoothed_weights

def waveletWeights(coeffs,decay=True):
    N=len(flatCoeffs(coeffs))
    weights=np.zeros(N)
    levels=len(coeffs)-1
    for i in range(1,levels+1):
        rangeLen=np.power(2,levels-i+1)
        for j in range(len(coeffs[i])):
            if coeffs[i][j]!=0:
                # print(i,j)
                startPos=j*rangeLen
                endPos=(j+1)*rangeLen
                tmpWeights=np.zeros(N)
                tmpWeights[startPos:endPos]=1/rangeLen # 注意此时总和为1
                    
                if decay: #and i==levels:
                    # # 衰减,而不是左右两边直接降为0
                    # smoothed_weights = gaussian_filter1d(tmpWeights, sigma=0.5) # sigma越大越平滑
                    # smoothed_weights /= smoothed_weights.sum() # 归一化以保证累加和为 1
                    # tmpWeights=smoothed_weights
                    tmpWeights=decayWeights(tmpWeights,startPos,endPos)
                    # print(i,j,tmpWeights)
                    # plt.plot(tmpWeights,'o-')
                    # plt.title("{},{}".format(i,j))
                    # plt.show()
                
                weights+=tmpWeights

        # print(i,weights)
        # plt.plot(weights,'o-')
        # plt.title(i)
        # plt.show()
                
    # 单独处理第一个approximate coefficient
    if coeffs[0][0]!=0:
        rangeLen=np.power(2,levels)
        startPos=0
        endPos=rangeLen
        tmpWeights=np.zeros(N)
        tmpWeights[startPos:endPos]=1/rangeLen
        weights+=tmpWeights
    
    return weights

def waveletBucketFromCoeffs(coeffs,decay=True,plot=False):
    # coeffs是已经经过thresholding后的
    weights=waveletWeights(coeffs,decay=decay)
    cumulative_weights = np.cumsum(weights)
    m=round(cumulative_weights[-1]) # 等于coeffs中非零系数的个数
#     print(m)
    
    buckets = [0]
    for i in range(1,m):
        # 寻找累计权重数组中第一个大于或等于i+0.5的位置
        bucket_end = np.searchsorted(cumulative_weights, i+0.5, side='left')
        buckets.append(bucket_end)
    
    buckets.append(len(flatCoeffs(coeffs))) # 左闭右开
    buckets=np.unique(buckets)
    
    if plot:
        fig, ax = plt.subplots()
        plt.plot(weights)
        plt.title('weights')
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        if(len(buckets)<=150):
            plt.plot(np.arange(len(weights)),cumulative_weights,'o-')
            for i in range(len(buckets)):
                plt.axvline(x=buckets[i],color='r')
            for i in range(1,len(buckets)):
                plt.axhline(y=i+0.5,color='r')
        else:
            plt.plot(np.arange(len(weights)),cumulative_weights)

        plt.title('cumsum weights & buckets')
        plt.show()
        plt.close()
        
    return buckets,weights

def waveletBucketFromData(data,nout,wavelet='haar',decay=True,byLevel=True,plot=False):
    coeffs_raw = pywt.wavedec(data, wavelet) #,level=3
    if byLevel:
        print('byLevel')
        coeffs,thresholdByLevel=threshold_abs_levelwise(coeffs_raw,nout) 
        if plot:
            plotDWTCoeffPloty(coeffs_raw,thresholdByLevel=thresholdByLevel,log=False,simple=True)
            detailCoeffMappedTime(data,coeffs_raw,thresholdByLevel=thresholdByLevel)
            # detailCoeffMappedTimePloty(data,coeffs_raw,thresholdByLevel=thresholdByLevel)
    else:
        print('global')
        coeffs,threshold=threshold_abs(coeffs_raw,nout) 
        if plot:
            plotDWTCoeffPloty(coeffs_raw,threshold=threshold,log=False,simple=True)
            detailCoeffMappedTime(data,coeffs_raw,threshold=threshold)
            # detailCoeffMappedTimePloty(data,coeffs_raw,threshold=threshold)

    # coeffs是已经经过thresholding后的
    buckets,weights=waveletBucketFromCoeffs(coeffs,decay=decay,plot=plot)
        
    return buckets,weights