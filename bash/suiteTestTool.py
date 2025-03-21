from myfuncs import *
from zoomSSIMTool import *
from tsdownsample2.waveletBucketTool import *
import plotly.express as px 
import plotly.graph_objects as go; 
import time

def onlyGroundTruth(v,operations,width=240,height=240,dpi=12,rootName="D://mytest"):
    t=np.arange(0,len(v))

    # 画ground truth
    print('=================ground truth=================')
    truthDirPath=os.path.join(rootName,f"tmp-truth")
    _=random_scale_ground_truth(t,v, width, height, truthDirPath, operations, dpi=dpi)
    print('finish')

# def suiteTest1(v,nout,operations,wavelet='haar',isPlot=False,width=240,height=240,dpi=12,rootName="D://mytest"):
#     t=np.arange(0,len(v))

#     # 画ground truth
#     print('=================ground truth=================')
#     truthDirPath=os.path.join(rootName,f"tmp-truth")
#     tmin,tmax,vmin,vmax=random_scale_ground_truth(t,v, width, height, truthDirPath, operations, dpi=dpi)

#     # 画sampled result，并且和ground truth比较
#     res=innerSuiteTest1(tmin,tmax,vmin,vmax,truthDirPath,\
#         t,v,nout,operations,wavelet=wavelet,isPlot=isPlot,width=width,height=height,dpi=dpi,rootName=rootName)
#     return res

def my_build_thresholds(points,pagePointNum,mode='3'):
    # 按照每p个点分批次，最后一个批次可能不足p
    pts_batches = [points[i:i + pagePointNum] for i in range(0, len(points), pagePointNum)]
    print('pages num=',len(pts_batches))
    
    eaList=[]
    for pts in pts_batches:
        ea,_=build_thresholds(pts,mode=mode) # note this if mine or not
        eaList+=list(ea)
    eaList=np.array(eaList)
    return eaList


def innerSuiteTest1(tmin,tmax,vmin,vmax,truthDirPath,\
    t,v,nout,operations,wavelet='haar',isPlot=False,width=240,height=240,dpi=12,\
    rootName="D://mytest",fsw_jar_path="MySample_fsw_UCR-jar-with-dependencies.jar",fsw_dir="fsw",\
    eaListMine=None,eaListAll=None,seriesTimeColumn=True):
    print('-----------------------------------------------')
    print('n=',len(v),', m=',nout,', wavelet=',wavelet,', width=',width,', height=',height, ', truthDirPath=', truthDirPath)
    print('fsw_jar_path=',fsw_jar_path,', fsw_dir=',fsw_dir)

    # print operations
    print(operations)
    # df=pd.read_csv(operations,header=0)
    # opType=df.iloc[:,1]
    # startTime=df.iloc[:,2]
    # endTime=df.iloc[:,3]
    # for i in range(len(opType)):
    #     print("<<<<<<<<<<",i,opType[i],startTime[i],endTime[i])

    noutWarningRatio=0.1

    print('=================WAVELET ABS=================')
    # coeffs_raw = pywt.wavedec(v, wavelet) #,level=3
    # coeffs_abs,threshold=threshold_abs(coeffs_raw,nout)
    # data_recon_abs=pywt.waverec(coeffs_abs, wavelet) # 小波会padding一些点，但是影响不大，只要不用到最末尾pad的点就可以

    # # plotDWTCoeffPloty(coeffs_raw,threshold=threshold,log=False)
    # # detailCoeffMappedTime(data,coeffs_raw,threshold=threshold)
    # # # detailCoeffMappedTimePloty(data,coeffs_raw,threshold=threshold)
    # # waveletBucketFromCoeffs(coeffs_abs,plot=True)

    dwt_v = myDWT(v,nout)

    # # 为了AD面积计算公平让大家都选中全局首尾点
    # dwt_v[0]=v[0]
    # dwt_v[len(v)-1]=v[-1]

    ssim_result_dwt = random_scale_ssim(tmin,tmax,vmin,vmax,truthDirPath,t,v,\
        t,dwt_v,width,height,\
        os.path.join(rootName,f"tmp-dwt-{nout}"),"dwt",operations,dpi=dpi)
    gc.collect()

    # print('+++++++WAVELET ABS LEVELWISE++++++++++')
    # coeffs_abs_levelwise,thresholdByLevel=threshold_abs_levelwise(coeffs_raw,nout)
    # data_recon_abs_levelwise=pywt.waverec(coeffs_abs_levelwise, wavelet)

    # print('=================WAVELET naive=================')
    # coeffs_naive=threshold_naive(coeffs_raw,nout)
    # data_recon_naive=pywt.waverec(coeffs_naive, wavelet)

    print('=================DFT=================')
    dft_v = myDFT(v,nout)

    # # 为了AD面积计算公平让大家都选中全局首尾点
    # dft_v[0]=v[0]
    # dft_v[len(v)-1]=v[-1]

    ssim_result_dft = random_scale_ssim(tmin,tmax,vmin,vmax,truthDirPath,t,v,\
        t,dft_v,width,height,\
        os.path.join(rootName,f"tmp-dft-{nout}"),"dft",operations,dpi=dpi)
    gc.collect()

    print('=================ILTS=================')
    s_ds = LTTBETFurtherDownsampler().downsample(t, v, n_out=nout)
    t1 = t[s_ds]
    v1 = v[s_ds]
    print('【ilts】',len(t1))
    if abs(len(t1)-nout)>nout*noutWarningRatio:
        # raise ValueError("ATTENTION: not match nout")
        print("!!!!!!!!!!ATTENTION: not match nou")
    ssim_result_ilts = random_scale_ssim(tmin,tmax,vmin,vmax,truthDirPath,t,v,\
        t1,v1, width,height,\
        os.path.join(rootName,f"tmp-ilts-{nout}"),"ILTS",operations,dpi=dpi)
    gc.collect()

    print('=================LTD=================')
    # s_ds = ILTSLTDDownsampler().downsample(t, v, n_out=nout)
    s_ds = LTDDownsampler().downsample(t, v, n_out=nout)
    t2 = t[s_ds]
    v2 = v[s_ds]
    print('【original LTD】',len(t2))
    if abs(len(t2)-nout)>nout*noutWarningRatio:
        # raise ValueError("ATTENTION: not match nout")
        print("!!!!!!!!!!ATTENTION: not match nou")
    ssim_result_ltd = random_scale_ssim(tmin,tmax,vmin,vmax,truthDirPath,t,v,\
        t2,v2, width,height,\
        os.path.join(rootName,f"tmp-ltd-{nout}"),"ltd",operations,dpi=dpi)
    gc.collect()

    print('=================visval=================')
    area=getFastVisvalParam(nout,t,v)
    points=[]
    for x, y in zip(t,v):
        points.append((x,y))
    tmp=np.array(simplify_coords_vw(points, area))
    t3=tmp[:,0]
    v3=tmp[:,1]
    print('【visval】',len(t3))
    if abs(len(t3)-nout)>nout*noutWarningRatio:
        # raise ValueError("ATTENTION: not match nout")
        print("!!!!!!!!!!ATTENTION: not match nou")
    ssim_result_visval = random_scale_ssim(tmin,tmax,vmin,vmax,truthDirPath,t,v,\
        t3,v3, width,height,\
        os.path.join(rootName,f"tmp-visval-{nout}"),"visval",operations,dpi=dpi)
    gc.collect()

    # print('=================SC=================')
    # epsilon=getShrinkingConeParam(nout,t,v)
    # points=[]
    # for x, y in zip(t,v):
    #     points.append((x,y))
    # points=np.array(points)
    # mask = simplify_shrinking_cone(points,epsilon) # nout is used as tolerance
    # t4=points[mask,0]
    # v4=points[mask,1]
    # print('【sc】',len(t4))
    # if abs(len(t4)-nout)>nout*noutWarningRatio:
    #     raise ValueError("ATTENTION: not match nout")
    #     # print("!!!!!!!!!!ATTENTION: not match nou")
    # ssim_result_sc = random_scale_ssim(tmin,tmax,vmin,vmax,truthDirPath,t,v,\
    #     t4,v4, width,height,\
    #     os.path.join(rootName,f"tmp-sc-{nout}"),"sc",operations,dpi=dpi)
    # gc.collect()

    print('=================FSW=================')
    print(fsw_jar_path,': mind if latest!')
    # 构造命令
    command = [
        "java", "-jar", fsw_jar_path, 
        fsw_dir, 
        str(nout),
        str(False), # hasHeader
        str(seriesTimeColumn) # seriesTimeColumn
    ]
    try:
        subprocess.run(command, check=True)
        print("命令执行成功！")
    except subprocess.CalledProcessError as e:
        print("命令执行失败：", e)

    datasetNames = os.listdir(fsw_dir)
    print(len(datasetNames))
    for appendix in datasetNames:
        if '-segment' in appendix:
            continue
        filename=os.path.join(fsw_dir,appendix)
        print(filename)
        break
    pattern=r"{dataset}-segment-{method}.csv"
    resultFile=os.path.join(fsw_dir,pattern.format(dataset=os.path.split(filename)[1].split(".")[0],\
                                      method="fsw"))
    print(resultFile)
    df=pd.read_csv(resultFile,header=None) # assume no header by default
    t1=df.iloc[:,0]
    v1=df.iloc[:,1]
    print('【fsw】',len(t1))
    if abs(len(t1)-nout)>nout*noutWarningRatio:
        # raise ValueError("ATTENTION: not match nout")
        print("!!!!!!!!!!ATTENTION: not match nou")
    ssim_result_fsw = random_scale_ssim(tmin,tmax,vmin,vmax,truthDirPath,t,v,\
        t1,v1, width,height,\
        os.path.join(rootName,f"tmp-fsw-{nout}"),"fsw",operations,dpi=dpi)
    gc.collect()


    print('=================rdp=================')
    # normalize to [0,1], same scale of normalized t&v
    rdptmin=min(t) # 不要和输入参数列表tmin混淆
    rdptmax=max(t)
    rdpvmin=min(v)
    rdpvmax=max(v)
    t_normalized=(t-rdptmin)/(rdptmax-rdptmin)
    v_normalized=(v-rdpvmin)/(rdpvmax-rdpvmin)

    epsilon=getRDPParam(nout,t_normalized,v_normalized)
    points=[]
    for x, y in zip(t_normalized,v_normalized):
        points.append((x,y))
    tmp=np.array(simplify_coords(points, epsilon))
    t6=tmp[:,0]
    v6=tmp[:,1]

    # back unnormalized
    t6=t6*(rdptmax-rdptmin)+rdptmin
    v6=v6*(rdpvmax-rdpvmin)+rdpvmin
    print('【rdp】',len(t6))
    if abs(len(t6)-nout)>nout*noutWarningRatio:
        # raise ValueError("ATTENTION: not match nout")
        print("!!!!!!!!!!ATTENTION: not match nou")
    ssim_result_rdp = random_scale_ssim(tmin,tmax,vmin,vmax,truthDirPath,t,v,\
        t6,v6, width,height,\
        os.path.join(rootName,f"tmp-rdp-{nout}"),"rdp",operations,dpi=dpi)
    gc.collect()


    print('=================MinMax=================')
    s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout) # 注意这里用了FPLP版本，为了AD公平让大家都选中全局首尾点
    t8 = t[s_ds]
    v8 = v[s_ds]
    print('【MinMaxFPLP】',len(t8))
    if abs(len(t8)-nout)>nout*noutWarningRatio:
        # raise ValueError("ATTENTION: not match nout")
        print("!!!!!!!!!!ATTENTION: not match nou")
    ssim_result_minmax = random_scale_ssim(tmin,tmax,vmin,vmax,truthDirPath,t,v,\
        t8,v8, width,height,
        os.path.join(rootName,f"tmp-minmax-{nout}"),"minmax",operations,dpi=dpi)
    gc.collect()

    print('=================vs batch mine=================')
    pagePointNum=10000
    fast=True # True to not running the slow version Visval
    useAll=False

    pts_all = np.array(list(zip(t, v)))

    # 记录开始时间
    start_time = time.time()

    if eaListMine is None:
        eaListMine = my_build_thresholds(pts_all,pagePointNum,mode='4')

    sample_all_t,sample_all_v,sample_batch_t,sample_batch_v,\
        ea_all,eaList,sameEApercent,prunedPagePercent,commonPointsPercent=\
        VWSimplifier_PageWise(pts_all,nout,pagePointNum,fast=True,eaList=eaListMine)

    # 记录结束时间
    end_time = time.time()

    # 计算耗时（秒），并转化为毫秒
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"vs batch mine代码运行时间: {elapsed_time_ms:.3f} 毫秒")

    print('【visval-batch-useMine】',len(sample_batch_t))
    if abs(len(sample_batch_t)-nout)>nout*noutWarningRatio:
        # raise ValueError("ATTENTION: not match nout")
        print("!!!!!!!!!!ATTENTION: not match nou")

    ssim_result_visval_batch_mine = random_scale_ssim(tmin,tmax,vmin,vmax,truthDirPath,t,v,\
        sample_batch_t,sample_batch_v, width,height,
        os.path.join(rootName,f"tmp-visval_batch_mine-{nout}"),"visval_batch_mine",operations,dpi=dpi)
    gc.collect()

    print('=================vs batch all just like bottomUp=================')
    pagePointNum=10000
    fast=True # True to not running the slow version Visval
    useAll=True

    pts_all = np.array(list(zip(t, v)))

    # 记录开始时间
    start_time = time.time()

    if eaListAll is None:
        eaListAll = my_build_thresholds(pts_all,pagePointNum,mode='all')

    sample_all_t,sample_all_v,sample_batch_t,sample_batch_v,\
        ea_all,eaList,sameEApercent,prunedPagePercent,commonPointsPercent=\
        VWSimplifier_PageWise(pts_all,nout,pagePointNum,fast=True,eaList=eaListAll)

    # 记录结束时间
    end_time = time.time()

    # 计算耗时（秒），并转化为毫秒
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"vs batch all代码运行时间: {elapsed_time_ms:.3f} 毫秒")

    print('【visval-batch-useAll】',len(sample_batch_t))
    if abs(len(sample_batch_t)-nout)>nout*noutWarningRatio:
        # raise ValueError("ATTENTION: not match nout")
        print("!!!!!!!!!!ATTENTION: not match nou")

    ssim_result_visval_batch_all = random_scale_ssim(tmin,tmax,vmin,vmax,truthDirPath,t,v,\
        sample_batch_t,sample_batch_v, width,height,
        os.path.join(rootName,f"tmp-visval_batch_all-{nout}"),"visval_batch_all",operations,dpi=dpi)
    gc.collect()


    # TODO 暂时占用ssim_masked的位置计算areal displacement
    print('ilts:areal displacement,data-rmse,pixel-ssim,pixel-mse',\
        ssim_result_ilts[0],ssim_result_ilts[2],ssim_result_ilts[4],ssim_result_ilts[6])
    print('dwt:areal displacement,data-rmse,pixel-ssim,pixel-mse',\
        ssim_result_dwt[0],ssim_result_dwt[2],ssim_result_dwt[4],ssim_result_dwt[6])
    print('ltd:areal displacement,data-rmse,pixel-ssim,pixel-mse',\
        ssim_result_ltd[0],ssim_result_ltd[2],ssim_result_ltd[4],ssim_result_ltd[6])
    # print('sc:areal displacement,data-rmse,pixel-ssim,pixel-mse',\
    #     ssim_result_sc[0],ssim_result_sc[2],ssim_result_sc[4],ssim_result_sc[6])
    print('fsw:areal displacement,data-rmse,pixel-ssim,pixel-mse',\
        ssim_result_fsw[0],ssim_result_fsw[2],ssim_result_fsw[4],ssim_result_fsw[6])
    print('visval:areal displacement,data-rmse,pixel-ssim,pixel-mse',\
        ssim_result_visval[0],ssim_result_visval[2],ssim_result_visval[4],ssim_result_visval[6])
    print('rdp:areal displacement,data-rmse,pixel-ssim,pixel-mse',\
        ssim_result_rdp[0],ssim_result_rdp[2],ssim_result_rdp[4],ssim_result_rdp[6])
    print('MinMax:areal displacement,data-rmse,pixel-ssim,pixel-mse',\
        ssim_result_minmax[0],ssim_result_minmax[2],ssim_result_minmax[4],ssim_result_minmax[6])
    print('visval-batch-mine:areal displacement,data-rmse,pixel-ssim,pixel-mse',\
        ssim_result_visval_batch_mine[0],ssim_result_visval_batch_mine[2],ssim_result_visval_batch_mine[4],ssim_result_visval_batch_mine[6])
    print('dft:areal displacement,data-rmse,pixel-ssim,pixel-mse',\
        ssim_result_dft[0],ssim_result_dft[2],ssim_result_dft[4],ssim_result_dft[6])
    print('visval-batch-all:areal displacement,data-rmse,pixel-ssim,pixel-mse',\
        ssim_result_visval_batch_all[0],ssim_result_visval_batch_all[2],ssim_result_visval_batch_all[4],ssim_result_visval_batch_all[6])


    gc.collect()

    return ssim_result_ilts,ssim_result_dwt,ssim_result_ltd,\
        ssim_result_fsw,\
        ssim_result_visval,ssim_result_rdp,ssim_result_minmax,ssim_result_visval_batch_mine,ssim_result_dft,\
        ssim_result_visval_batch_all


def suiteTest2_varyM(points,noutMin,noutMax,numStep,operations,wavelet='haar',width=240,height=240,dpi=12,\
    rootName="D://mytest",fsw_jar_path="MySample_fsw_UCR-jar-with-dependencies.jar",fsw_dir="fsw",seriesTimeColumn=True):
    ilts=[]
    wavelet_abs=[]
    ltd=[]
    # sc=[]
    fsw=[]
    visval=[]
    rdp=[]
    minmax=[]
    visval_batch_mine=[]
    dft=[]
    visval_batch_all=[]
    
    # noutList=range(noutMin,noutMax,noutStep)
    noutList=np.logspace(np.log10(noutMin), np.log10(noutMax), num=numStep, dtype=int)
    # 将 noutList 中的所有数变为偶数，因为MinMaxDownsampler要求
    noutList = np.where(noutList % 2 == 0, noutList, noutList + 1)
    print('noutList=',noutList)

    # 画ground truth
    print('=================ground truth=================')
    # t=np.arange(0,len(data))
    t=points[:,0]
    data=points[:,1]
    truthDirPath=os.path.join(rootName,f"tmp-truth")
    tmin,tmax,vmin,vmax=random_scale_ground_truth(t,data, width, height, truthDirPath, operations, dpi=dpi)

    pagePointNum=10000
    pts_all = np.array(list(zip(t, data)))
    print('=================build threshold batch mine=================')
    start_time = time.time()
    eaListMine = my_build_thresholds(pts_all,pagePointNum,mode='4')
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"vs batch mine代码运行时间: {elapsed_time_ms:.3f} 毫秒")
    print('=================build threshold batch all=================')
    start_time = time.time()
    eaListAll = my_build_thresholds(pts_all,pagePointNum,mode='all')
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"vs batch all代码运行时间: {elapsed_time_ms:.3f} 毫秒")

    for nout in noutList:
        # res=suiteTest1(data,nout,operations,wavelet=wavelet,isPlot=False,width=width,height=height,dpi=dpi,rootName=rootName)
        res=innerSuiteTest1(tmin,tmax,vmin,vmax,truthDirPath,\
            t,data,nout,operations,wavelet=wavelet,isPlot=False,width=width,height=height,dpi=dpi,rootName=rootName,\
            fsw_jar_path=fsw_jar_path,fsw_dir=fsw_dir,eaListMine=eaListMine,eaListAll=eaListAll,seriesTimeColumn=seriesTimeColumn)
        # 对于某个nout下某个方法的某个指标，取用在所有缩放操作operations下的平均结果
        k=0
        # The higher the better: pixel_ssim_masked,1/data rmse, pixel ssim, 1/pixel mse
        # TODO 暂时占用ssim_masked的位置计算areal displacement
        ilts.append([1/res[k][0],1/res[k][2],res[k][4],1/res[k][6]])
        k+=1
        wavelet_abs.append([1/res[k][0],1/res[k][2],res[k][4],1/res[k][6]])
        k+=1
        ltd.append([1/res[k][0],1/res[k][2],res[k][4],1/res[k][6]])
        k+=1
        # sc.append([1/res[k][0],1/res[k][2],res[k][4],1/res[k][6]])
        fsw.append([1/res[k][0],1/res[k][2],res[k][4],1/res[k][6]])
        k+=1
        visval.append([1/res[k][0],1/res[k][2],res[k][4],1/res[k][6]])
        k+=1
        rdp.append([1/res[k][0],1/res[k][2],res[k][4],1/res[k][6]])
        k+=1
        minmax.append([1/res[k][0],1/res[k][2],res[k][4],1/res[k][6]])
        k+=1
        visval_batch_mine.append([1/res[k][0],1/res[k][2],res[k][4],1/res[k][6]])
        k+=1
        dft.append([1/res[k][0],1/res[k][2],res[k][4],1/res[k][6]])
        k+=1
        visval_batch_all.append([1/res[k][0],1/res[k][2],res[k][4],1/res[k][6]])

    mapping = {
        # 1: "canvas-rmse-(smaller better)",
        # TODO 暂时占用ssim_masked的位置计算areal displacement
        1: "inverse of areal displacement", # note
        2: "inverse of data-rmse", # note
        3: "pixel-ssim",
        4: "inverse of pixel-mse" # note
    }

    # 创建4个子图
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列
    axs = axs.flatten()  # 将子图数组展平，方便索引

    titlepos=-0.43

    font=18
    lw=2.5
    slw=1
    ms=9
    columnspacing=0.9
    handletextpad=0.5

    # Remove space between axes
    fig.subplots_adjust(wspace=0.4) # hspace竖向,wspace横向
    fig.subplots_adjust(hspace=0.55) # hspace竖向,wspace横向

    # 迭代每个子图
    for idx in range(4):
        plt.sca(axs[idx])
        axs[idx].plot(noutList, [sublist[idx] for sublist in ilts], 'o-', label='ilts',markersize=ms,linewidth=lw)
        axs[idx].plot(noutList, [sublist[idx] for sublist in wavelet_abs], 'o-', label='wavelet_abs',markersize=ms,linewidth=lw)
        axs[idx].plot(noutList, [sublist[idx] for sublist in ltd], 'o-', label='ltd',markersize=ms,linewidth=lw)
        # axs[idx].plot(noutList, [sublist[idx] for sublist in sc], 'o-', label='sc',markersize=ms,linewidth=lw)
        axs[idx].plot(noutList, [sublist[idx] for sublist in fsw], 'o-', label='fsw',markersize=ms,linewidth=lw)
        axs[idx].plot(noutList, [sublist[idx] for sublist in visval], 'o-', label='visval',markersize=ms,linewidth=lw)
        axs[idx].plot(noutList, [sublist[idx] for sublist in rdp], 'o-', label='rdp',markersize=ms,linewidth=lw)
        axs[idx].plot(noutList, [sublist[idx] for sublist in minmax], 'o-', label='MinMax',markersize=ms,linewidth=lw)
        axs[idx].plot(noutList, [sublist[idx] for sublist in visval_batch_mine], 'o-', label='visval_batch_mine',markersize=ms,linewidth=lw)
        # axs[idx].plot(noutList, [sublist[idx] for sublist in dft], 'o-', label='dft',markersize=ms,linewidth=lw) # 不画dft因为太差
        axs[idx].plot(noutList, [sublist[idx] for sublist in visval_batch_all], 'o-', label='visval_batch_all',markersize=ms,linewidth=lw)
        axs[idx].set_xlabel('m',fontsize=font)
        axs[idx].set_ylabel(mapping[idx + 1], fontsize=font)
        axs[idx].set_xscale('log')
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        plt.title(mapping[idx + 1],y=titlepos,fontsize=font)

        # 保存每个子图的数据为CSV文件
        csv_filename = os.path.join(rootName, f'metric_{mapping[idx + 1]}.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            labels=[line.get_label() for line in axs[idx].get_lines()]
            labels.insert(0, 'nout') 
            writer.writerow(labels)
            for i in range(len(noutList)):
                # NOTE this!!!!!!!!!!!!!!!!!!!!!
                row = [noutList[i],
                       ilts[i][idx],
                       wavelet_abs[i][idx],
                       ltd[i][idx],
                       # sc[i][idx],
                       fsw[i][idx],
                       visval[i][idx],
                       rdp[i][idx],
                       minmax[i][idx],
                       visval_batch_mine[i][idx],
                       dft[i][idx],
                       visval_batch_all[i][idx],
                       ]
                writer.writerow(row)

    labels=[line.get_label() for line in axs[0].get_lines()]
    fig.legend(fontsize=font, labels=labels, ncol=4,bbox_to_anchor=(0.467,1.03), loc='upper center',\
               columnspacing=columnspacing,handletextpad=handletextpad)
    fig.savefig(os.path.join(rootName,'combined.eps'),bbox_inches='tight')
    fig.savefig(os.path.join(rootName,'combined.png'),bbox_inches='tight')
    fig.show()

    gc.collect()

    return ilts,wavelet_abs,ltd,\
        fsw,\
        visval,rdp,minmax,visval_batch_mine,\
        dft,\
        visval_batch_all