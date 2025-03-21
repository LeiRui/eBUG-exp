from myfuncs import *
from scipy.interpolate import interp1d
from ts_areal_displacement import *

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
def plot_and_save(t,v, t_min, t_max, v_min, v_max, filename, width, height, dpi=12):
    full_frame(width,height,dpi)
    plt.plot(t,v,'-',color='k',linewidth=1,antialiased=True)
    plt.xlim(t_min, t_max)
    plt.ylim(v_min, v_max)
    plt.savefig(filename,backend='agg')
    plt.close()

def random_scale_ground_truth(truth_t, truth_v, width, height, dirPath, operations, dpi=12):
    import shutil
    if os.path.exists(dirPath):
        shutil.rmtree(dirPath)
        
    # get the start and end data time of each operation
    df=pd.read_csv(operations,header=0)
    opType=df.iloc[:,1]
    startTime=df.iloc[:,2]
    endTime=df.iloc[:,3]
    
    tmin=[]
    tmax=[]
    vmin=[]
    vmax=[]
    cnt=0
    for op,start,end in zip(opType,startTime,endTime):
        print("<<<<<<<<<<",cnt,op,start,end)
        
        ##### 【获取ground truth的数据范围和画布显示范围】 #####
        idx1=np.where(truth_t>=start)[0][0]
        idx2=np.where(truth_t<end)[0][-1]
        idx2+=1 # 因为下面左开右闭
        if idx2-idx1<=1: # 只有一个点，不用画图比较了
            print('continue')
            continue

        truth_segment_t = truth_t[idx1:idx2]
        truth_segment_v = truth_v[idx1:idx2]

        segment_t_min=min(truth_segment_t)
        segment_t_max=max(truth_segment_t)
        segment_v_min=min(truth_segment_v) # 自动缩放y轴为当前数据的最小值和最大值！
        segment_v_max=max(truth_segment_v) # 自动缩放y轴为当前数据的最小值和最大值！

#             # 暂时debug: 如果画图y轴范围始终固定是全部数据的min-max value range，
#             # 那么canvas-rmse和data-rmse的排序一致，因为只是比例系数scale不一样
#             segment_v_min=min(truth_v) # 自动缩放y轴为当前数据的最小值和最大值
#             segment_v_max=max(truth_v) # 自动缩放y轴为当前数据的最小值和最大值

        # Attempting to set identical low and high ylims makes transformation singular; automatically expanding
        if segment_v_min==segment_v_max:
            print("vmin=vmax, automatically expanding")
#             Attempting to set identical low and high ylims makes transformation singular; automatically expanding.
#             segment_v_min=min(last_v_min,segment_v_min)
#             segment_v_max=max(last_v_max,segment_v_max)
            delta=1e-2 # 这种做法符合python画图的时候遇到ylim一样大的时候automatically expanding的处理
            segment_v_min-=delta
            segment_v_max+=delta

        # 画图的统一边界
        tmin.append(segment_t_min)
        tmax.append(segment_t_max)
        vmin.append(segment_v_min)
        vmax.append(segment_v_max)


        # 为了体现左右两边和画面外的点的连线，但是不要改动上面的x&y轴范围
        if idx1>0: # 左闭
            idx1-=1
        if idx2<len(truth_t): # 右开 
            idx2+=1
        truth_segment_t = truth_t[idx1:idx2]
        truth_segment_v = truth_v[idx1:idx2]
#             print('truth',idx1,idx2)

        # Save the segments as images and calculate SSIM
        create_folder(dirPath)
        truth_filename = os.path.join(dirPath,f'{cnt}_{op}_{start}_{end}-truth.png')
        plot_and_save(truth_segment_t,truth_segment_v,\
                      segment_t_min,segment_t_max,segment_v_min,segment_v_max,\
                      truth_filename,width,height,dpi=dpi)

        cnt+=1
        gc.collect()

    return tmin,tmax,vmin,vmax

def random_scale_ssim(tmin,tmax,vmin,vmax, truthDirPath, truth_t,truth_v,\
    sample_t, sample_v, width, height, dirPath, dataName, operations, dpi=72, onlySSIM=False):
    # 如果不计算插值的data rmse，不需要truth_t,truth_v

    # 打印输入参数
    # print("Function: multi_scale_canvas")
    # print("------------------------------------------------------")
    # print(f"truth_t: {len(truth_t)}")
    # print(f"truth_v: {len(truth_v)}")
    # print(f"sample_t: {len(sample_t)}")
    # print(f"sample_v: {len(sample_v)}")
    # print(f"width: {width}")
    # print(f"height: {height}")
    # print(f"dirPath: {dirPath}")
    # print(f"dataName: {dataName}")
    # print(f"operations: {operations}")
    # print("------------------------------------------------------")
    
    import shutil
    if os.path.exists(dirPath):
        shutil.rmtree(dirPath)
        
    # get the start and end data time of each operation
    df=pd.read_csv(operations,header=0)
    opType=df.iloc[:,1]
    startTime=df.iloc[:,2]
    endTime=df.iloc[:,3]
    
    # 由于后面要对sample数据插值，所以这里先把里面重复时间戳的点去掉，因为ILTS有可能全局首点和第一个桶的点相同
    sample_t, indices = np.unique(sample_t, return_index=True)
    sample_v = sample_v[indices]

    
    all_scores_pixel_ssim_masked=[]
    all_scores_data_rmse=[]
    all_scores_pixel_ssim=[]
    all_scores_pixel_mse=[]
    
    cnt=0
    for op,start,end in zip(opType,startTime,endTime):
        # print("<<<<<<<<<<",cnt,op,start,end)
        
        ##### 【获取ground truth的数据范围为了后面插值计算data rmse】 #####
        idx1=np.where(truth_t>=start)[0][0]
        idx2=np.where(truth_t<end)[0][-1]
        idx2+=1 # 因为下面左开右闭
        if idx2-idx1<=1: # 只有一个点，不用画图比较了
            # 注意这里不可少，和ground truth的continue对应！
            print('continue')
            continue

        # truth_segment_t = truth_t[idx1:idx2]
        # truth_segment_v = truth_v[idx1:idx2]

        # 为了体现左右两边和画面外的点的连线，但是不要改动上面的x&y轴范围
        if idx1>0: # 左闭
            idx1-=1
        if idx2<len(truth_t): # 右开 
            idx2+=1
        truth_segment_t = truth_t[idx1:idx2]
        truth_segment_v = truth_v[idx1:idx2]
        
        ##### 【获取ground truth的画布显示范围和图片路径】 #####
        segment_t_min=tmin[cnt]
        segment_t_max=tmax[cnt]
        segment_v_min=vmin[cnt]
        segment_v_max=vmax[cnt]
        truth_filename = os.path.join(truthDirPath,f'{cnt}_{op}_{start}_{end}-truth.png')

        ##### 【获取sample data的数据范围】 #####
        idx1=np.where(sample_t>=start)[0][0]
        idx2=np.where(sample_t<end)[0][-1]
        idx2+=1 # 因为下面左开右闭

        # 考虑到采样点稀疏的时候左右两边连线不能断开即便在画面之外
        if idx1>0: # 左闭
            idx1-=1
        if idx2<len(sample_t): # 右开 
            idx2+=1

        # 有可能这一段没有点，但是idx1和idx2连线覆盖这一段，所以不能continue
        if idx2-idx1<=1:
            print(idx1,idx2,'attention')
        #     continue

        sample_segment_t = sample_t[idx1:idx2]
        sample_segment_v = sample_v[idx1:idx2]
#             print('sample',idx1,idx2)

        # 【使用插值方法对齐时间戳】 
        # 暂时不需要计算这个data_rmse了
        # if not onlySSIM: 
            # interpolate_func = interp1d(sample_t, sample_v, kind='linear', fill_value='extrapolate')
            # aligned_values = interpolate_func(truth_segment_t)
        
        ##### 【计算分数】 #####
        # segment_v_min==segment_v_max的情况已经提前处理好了
        # TODO 注意truth一条水平线然后SC在画面之外的情况
        # 把v值映射到画面上的y值 y=(v-vmin)/(vmax-vmin)*height
        # tmp_truth_v=copy.deepcopy(truth_segment_v)
        # tmp_sample_v=copy.deepcopy(aligned_values)
        # tmp_truth_v=(tmp_truth_v-segment_v_min)/(segment_v_max-segment_v_min)*height
        # tmp_sample_v=(tmp_sample_v-segment_v_min)/(segment_v_max-segment_v_min)*height
        # score_canvas_rmse=np.sqrt(np.mean((tmp_truth_v-tmp_sample_v) ** 2))

        # Save the segments as images and calculate SSIM
        create_folder(dirPath)
        sample_filename = os.path.join(dirPath,f'{cnt}_{op}_{start}_{end}-{dataName}.png')
        plot_and_save(sample_segment_t,sample_segment_v,\
                      segment_t_min,segment_t_max,segment_v_min,segment_v_max,\
                      sample_filename,width,height,dpi=dpi)

        # print(truth_filename,sample_filename)


        if not onlySSIM:
            # score_pixel_ssim_masked=match_masked(truth_filename, sample_filename)
            # TODO 暂时占用ssim_masked的位置计算areal displacement
            points1=np.array(list(zip(truth_segment_t,truth_segment_v)))
            points2=np.array(list(zip(sample_segment_t,sample_segment_v)))
            # 为了AD面积计算公平让大家都选中全局首尾点，而且为了至少形成一个闭合形状
            points2[0]=points1[0]
            points2[-1]=points1[-1]
            score_pixel_ssim_masked=total_areal_displacement(points1,points2)

            # score_data_rmse=np.sqrt(np.mean((truth_segment_v-aligned_values) ** 2))
            score_data_rmse=0

            # 鉴于SSIM没有体现距离远近的能力，试试pixel MSE指标：也没有
            # print("use mse_in_255")
            # score_pixel_mse = mse_in_255(truth_filename, sample_filename)
            score_pixel_mse=0
        else:
            score_pixel_ssim_masked=0 
            score_data_rmse=0 
            score_pixel_mse=0 


        #放大到后面画面很多都是背景白色，导致即便黑色曲线差异很大SSIM还很大0.95之类，所以改用masked看看
        # # print("use match_masked")
        # score_pixel_ssim = match_masked(truth_filename, sample_filename)
        score_pixel_ssim = match(truth_filename, sample_filename)

        # 把指标结果加入到sample png名字中
        new_filename = os.path.join(dirPath,\
                                    f'{cnt}_{op}_{start}_{end}-{dataName}-{len(sample_t)}-{score_pixel_ssim_masked:.4f}-{score_data_rmse:.4f}-{score_pixel_ssim:.4f}-{score_pixel_mse:.4f}.png')
        os.rename(sample_filename, new_filename)

        all_scores_pixel_ssim_masked.append(score_pixel_ssim_masked)
        all_scores_data_rmse.append(score_data_rmse)
        all_scores_pixel_ssim.append(score_pixel_ssim)
        all_scores_pixel_mse.append(score_pixel_mse)
        cnt+=1

        gc.collect()

    # 保存原始结果到csv
    # TODO 暂时占用ssim_masked的位置计算areal displacement
    headers = ['id','areal_displacement', 'data_rmse', 'pixel_ssim', 'pixel_mse']
    with open(os.path.join(dirPath,f'metrics-of-all-operations-by-{dataName}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for i in range(len(all_scores_pixel_ssim)):
            writer.writerow([i, all_scores_pixel_ssim_masked[i], all_scores_data_rmse[i], all_scores_pixel_ssim[i], all_scores_pixel_mse[i]])

    return np.mean(all_scores_pixel_ssim_masked),all_scores_pixel_ssim_masked,\
            np.mean(all_scores_data_rmse),all_scores_data_rmse,\
            np.mean(all_scores_pixel_ssim),all_scores_pixel_ssim,\
            np.mean(all_scores_pixel_mse),all_scores_pixel_mse

# def multi_scale_ssim(truth_t, truth_v, sample_t, sample_v, width, height, dirPath, dataName, omitOne=False, zoomLevel=6, type="SSIM"):
#     import shutil
#     if os.path.exists(dirPath):
#         shutil.rmtree(dirPath)
    
#     step_size=max(truth_t)-min(truth_t)
#     minStep=step_size/np.power(2,zoomLevel) # 即step_size会以1/2的倍率缩小zoomLevel次（不包括初始）
    
#     ssim_scores = []
#     all_scores=[]
#     all_offsetCnts=[]
#     all_stepsizes=[]
#     while True:
#         if step_size<minStep:
#             break
            
#         print('>>>>>>>>>>>>>>step_size',step_size)
            
#         scale_ssim = 0
#         offset_cnt=0
#         all_cnt=0
#         for start in np.arange(min(truth_t), max(truth_t), step_size):
#             end=start+step_size
# #             print(start,end)
            
#             idx1=np.where(truth_t>=start)[0][0]
#             idx2=np.where(truth_t<end)[0][-1]
#             idx2+=1 # 因为下面左开右闭
#             if idx2-idx1<=1: # 只有一个点，不用画图比较了
#                 continue
#             truth_segment_t = truth_t[idx1:idx2]
#             truth_segment_v = truth_v[idx1:idx2]

#             segment_t_min=min(truth_segment_t)
#             segment_t_max=max(truth_segment_t)
            
#             segment_v_min=min(truth_segment_v) # 自动缩放y轴为当前数据的最小值和最大值
#             segment_v_max=max(truth_segment_v) # 自动缩放y轴为当前数据的最小值和最大值
#             # segment_v_min=min(truth_v) # 注意改成：y轴固定全局范围，不自动缩放
#             # segment_v_max=max(truth_v) # 注意改成：y轴固定全局范围，不自动缩放

#             # 和下面的处理一致，但是不要改动上面的x&y轴范围
#             if idx1>0: # 左闭
#                 idx1-=1
#             if idx2<len(truth_t): # 右开 
#                 idx2+=1
#             truth_segment_t = truth_t[idx1:idx2]
#             truth_segment_v = truth_v[idx1:idx2]

            
#             idx1=np.where(sample_t>=start)[0][0]
#             idx2=np.where(sample_t<end)[0][-1]
#             idx2+=1 # 因为下面左开右闭
                
#             # 考虑到采样点稀疏的时候左右两边连线不能断开即便在画面之外
#             if idx1>0: # 左闭
#                 idx1-=1
#             if idx2<len(sample_t): # 右开 
#                 idx2+=1

#             # 有可能这一段没有点，但是idx1和idx2连线覆盖这一段，所以不能continue
#             if idx2-idx1<=1:
#                 print(idx1,idx2,'attention')
#             #     continue
                
#             sample_segment_t = sample_t[idx1:idx2]
#             sample_segment_v = sample_v[idx1:idx2]

#             # Save the segments as images and calculate SSIM
#             create_folder(os.path.join(dirPath,str(step_size)))
#             truth_filename = os.path.join(dirPath,str(step_size),f'{dataName}-{step_size}_{start}-truth.png')
#             sample_filename = os.path.join(dirPath,str(step_size),f'{dataName}-{step_size}_{start}-sample.png')
            
#             plot_and_save(truth_segment_t,truth_segment_v,\
#                           segment_t_min,segment_t_max,segment_v_min,segment_v_max,\
#                           truth_filename,width,height)
#             plot_and_save(sample_segment_t,sample_segment_v,\
#                           segment_t_min,segment_t_max,segment_v_min,segment_v_max,\
#                           sample_filename,width,height)

#             # score = match(truth_filename, sample_filename)

#             #放大到后面画面很多都是背景白色，导致即便黑色曲线差异很大SSIM还很大0.95之类，所以改用masked看看
#             # # print("use match_masked")
#             # score = match_masked(truth_filename, sample_filename)

#             # 鉴于SSIM没有体现距离远近的能力，试试pixel MSE指标：也没有
#             # print("use mse_in_255")
#             if type=="SSIM":
#                 score = match_masked(truth_filename, sample_filename)
#             else: # MSE
#                 score = mse_in_255(truth_filename, sample_filename)

#             # 把ssim结果加入到png名字中
#             new_filename = os.path.join(dirPath,str(step_size),f'{dataName}-{step_size}_{start}-sample-{score:.4f}.png')
#             os.rename(sample_filename, new_filename)
            
#             all_scores.append(score) # 这个不管是不是SSIM=1都列进来
#             all_cnt+=1

#             if omitOne:
#                 if score==1:
#                     continue
                    
#             scale_ssim += score
#             offset_cnt+=1
        
#         # Average SSIM for the current scale
#         all_offsetCnts.append(all_cnt)
#         all_stepsizes.append(step_size)
#         if offset_cnt>0:
#             scale_ssim /= offset_cnt
#             ssim_scores.append(scale_ssim)
        
#         step_size /= 2

#     # Average SSIM across all scales
# #     print(ssim_scores)
#     avg_ssim = np.mean(ssim_scores)
#     return avg_ssim,all_scores,np.cumsum(all_offsetCnts),all_stepsizes

# # Example usage
# a = np.random.rand(2**4)  # Example sequence a with 2^6 points
# b = np.random.rand(2**4)  # Example sequence b with 2^6 points
# w, h = 640, 480  # Canvas size

# ssim_result = multi_scale_ssim(np.arange(0,len(a)),a, np.arange(0,len(a)),b, w, h, "D://tmp","a")
# print(f"Multi-scale SSIM: {ssim_result}")

def valueRange_distribution(truth_t, truth_v, zoomLevel):
    step_size=max(truth_t)-min(truth_t)
    step_size/=np.power(2,zoomLevel)
    print('>>>>>>>>>>>>>>step_size',step_size)

    ranges=[]
    for start in np.arange(min(truth_t), max(truth_t), step_size):
        end=start+step_size
#         print(start,end)

        ##### 【获取ground truth的数据范围和画布显示范围】 #####
        idx1=np.where(truth_t>=start)[0][0]
        idx2=np.where(truth_t<end)[0][-1]
        idx2+=1 # 因为下面左开右闭
        if idx2-idx1<=1: # 只有一个点，不用画图比较了
            continue

        truth_segment_t = truth_t[idx1:idx2]
        truth_segment_v = truth_v[idx1:idx2]

        segment_v_min=min(truth_segment_v) # 自动缩放y轴为当前数据的最小值和最大值！
        segment_v_max=max(truth_segment_v) # 自动缩放y轴为当前数据的最小值和最大值！

        ranges.append(segment_v_max-segment_v_min)
        
        
    ranges_non_zero = [r for r in ranges if r > 0]
    # print(f"剔除 0 后的范围: {len(ranges_non_zero)}")

    ranges_non_zero_log = np.log(ranges_non_zero)

    q1 = np.percentile(ranges_non_zero_log, 50)  # 25%分位数

    return np.exp(q1),ranges

# def multi_scale_canvas(truth_t, truth_v, sample_t, sample_v, width, height, dirPath, dataName, zoomLevel=6):
#     # 打印输入参数
#     print("Function: multi_scale_canvas")
#     print("------------------------------------------------------")
#     print(f"truth_t: {len(truth_t)}")
#     print(f"truth_v: {len(truth_v)}")
#     print(f"sample_t: {len(sample_t)}")
#     print(f"sample_v: {len(sample_v)}")
#     print(f"width: {width}")
#     print(f"height: {height}")
#     print(f"dirPath: {dirPath}")
#     print(f"dataName: {dataName}")
#     print(f"zoomLevel: {zoomLevel}")
#     print("------------------------------------------------------")
    
#     import shutil
#     if os.path.exists(dirPath):
#         shutil.rmtree(dirPath)


#     # 由于后面要对sample数据插值，所以这里先把里面重复时间戳的点去掉
#     sample_t, indices = np.unique(sample_t, return_index=True)
#     sample_v = sample_v[indices]
        
#     step_size=max(truth_t)-min(truth_t)
#     minStep=step_size/np.power(2,zoomLevel) # 即step_size会以1/2的倍率缩小zoomLevel次（不包括初始）
    
#     level_scores_canvas_rmse=[] # average of average of level scores
#     all_scores_canvas_rmse=[] # all scores irrelevant of levels
    
#     level_scores_data_rmse=[]
#     all_scores_data_rmse=[]
    
#     level_scores_pixel_ssim=[]
#     all_scores_pixel_ssim=[]
    
#     level_scores_pixel_mse=[]
#     all_scores_pixel_mse=[]
    
#     all_offsetCnts=[]

#     # smallValueRangeThreshold,_=valueRange_distribution(truth_t,truth_v,zoomLevel)
#     # print('smallValueRangeThreshold=',smallValueRangeThreshold)

#     level_cnt=0
#     while True:
#         if step_size<minStep:
#             break
            
#         print('>>>>>>>>>>>>>>step_size',step_size)
            
#         sum_score_canvas_rmse = 0
#         sum_score_data_rmse = 0
#         sum_score_pixel_ssim = 0
#         sum_score_pixel_mse = 0
#         all_cnt=0
#         last_v_min=min(truth_v)
#         last_v_max=max(truth_v)
#         for start in np.arange(min(truth_t), max(truth_t), step_size):
#             end=start+step_size
# #             print(start,end)
            
#             ##### 【获取ground truth的数据范围和画布显示范围】 #####
#             idx1=np.where(truth_t>=start)[0][0]
#             idx2=np.where(truth_t<end)[0][-1]
#             idx2+=1 # 因为下面左开右闭
#             if idx2-idx1<=1: # 只有一个点，不用画图比较了
#                 continue
                
#             truth_segment_t = truth_t[idx1:idx2]
#             truth_segment_v = truth_v[idx1:idx2]
            
#             segment_t_min=min(truth_segment_t)
#             segment_t_max=max(truth_segment_t)
#             segment_v_min=min(truth_segment_v) # 自动缩放y轴为当前数据的最小值和最大值！
#             segment_v_max=max(truth_segment_v) # 自动缩放y轴为当前数据的最小值和最大值！

# #             # 暂时debug: 如果画图y轴范围始终固定是全部数据的min-max value range，
# #             # 那么canvas-rmse和data-rmse的排序一致，因为只是比例系数scale不一样
# #             segment_v_min=min(truth_v) # 自动缩放y轴为当前数据的最小值和最大值
# #             segment_v_max=max(truth_v) # 自动缩放y轴为当前数据的最小值和最大值
            
#             # Attempting to set identical low and high ylims makes transformation singular; automatically expanding
#             # TODO 也许改成取周围邻居分段的range
#             # TODO 改成用采样段的range看看
#             if segment_v_min==segment_v_max:
#                 continue
# # #                 delta=1e-1
# # #                 segment_v_min -= delta
# # #                 segment_v_max += delta
# #                 segment_v_min=min(last_v_min,segment_v_min)
# #                 segment_v_max=max(last_v_max,segment_v_max)
#             # if segment_v_max-segment_v_min<smallValueRangeThreshold:
#             #     print(start,end,': too small value range, so skip')
#             #     continue
            
#             last_v_min=segment_v_min
#             last_v_max=segment_v_max
            

#             # 和下面的处理一致，但是不要改动上面的x&y轴范围
#             if idx1>0: # 左闭
#                 idx1-=1
#             if idx2<len(truth_t): # 右开 
#                 idx2+=1
#             truth_segment_t = truth_t[idx1:idx2]
#             truth_segment_v = truth_v[idx1:idx2]
# #             print('truth',idx1,idx2)

#             ##### 【获取sample data的数据范围】 #####
#             idx1=np.where(sample_t>=start)[0][0]
#             idx2=np.where(sample_t<end)[0][-1]
#             idx2+=1 # 因为下面左开右闭
                
#             # 考虑到采样点稀疏的时候左右两边连线不能断开即便在画面之外
#             if idx1>0: # 左闭
#                 idx1-=1
#             if idx2<len(sample_t): # 右开 
#                 idx2+=1

#             # 有可能这一段没有点，但是idx1和idx2连线覆盖这一段，所以不能continue
#             if idx2-idx1<=1:
#                 print(idx1,idx2,'attention')
#             #     continue
                
#             sample_segment_t = sample_t[idx1:idx2]
#             sample_segment_v = sample_v[idx1:idx2]
# #             print('sample',idx1,idx2)
            
#             # 假设t已经对齐
# #             score=np.sqrt(np.mean((truth_segment_v-sample_segment_v) ** 2))

#             # 【使用插值方法对齐时间戳】
#             interpolate_func = interp1d(sample_t, sample_v, kind='linear', fill_value='extrapolate')
#             aligned_values = interpolate_func(truth_segment_t)
            
#             ##### 【计算分数】 #####
#             # segment_v_min==segment_v_max的情况已经提前处理好了
#             # TODO 注意truth一条水平线然后SC在画面之外的情况
#             # 把v值映射到画面上的y值
#             # y=(v-vmin)/(vmax-vmin)*height
#             tmp_truth_v=copy.deepcopy(truth_segment_v)
#             tmp_sample_v=copy.deepcopy(aligned_values)
#             tmp_truth_v=(tmp_truth_v-segment_v_min)/(segment_v_max-segment_v_min)*height
#             tmp_sample_v=(tmp_sample_v-segment_v_min)/(segment_v_max-segment_v_min)*height
#             score_canvas_rmse=np.sqrt(np.mean((tmp_truth_v-tmp_sample_v) ** 2))
                
            
#             score_data_rmse=np.sqrt(np.mean((truth_segment_v-aligned_values) ** 2))

            
#             # Save the segments as images and calculate SSIM
#             create_folder(os.path.join(dirPath,str(step_size)))
#             truth_filename = os.path.join(dirPath,str(step_size),f'{dataName}-{step_size}_{start}-truth.png')
#             sample_filename = os.path.join(dirPath,str(step_size),f'{dataName}-{step_size}_{start}-sample.png')
            
#             plot_and_save(truth_segment_t,truth_segment_v,\
#                           segment_t_min,segment_t_max,segment_v_min,segment_v_max,\
#                           truth_filename,width,height)
#             plot_and_save(sample_segment_t,sample_segment_v,\
#                           segment_t_min,segment_t_max,segment_v_min,segment_v_max,\
#                           sample_filename,width,height)

#             #放大到后面画面很多都是背景白色，导致即便黑色曲线差异很大SSIM还很大0.95之类，所以改用masked看看
#             # # print("use match_masked")
#             score_pixel_ssim = match_masked(truth_filename, sample_filename)

#             # 鉴于SSIM没有体现距离远近的能力，试试pixel MSE指标：也没有
#             # print("use mse_in_255")
#             score_pixel_mse = mse_in_255(truth_filename, sample_filename)

#             # 把ssim结果加入到png名字中
#             new_filename = os.path.join(dirPath,str(step_size),\
#                                         f'{dataName}-{step_size}_{start}-sample-{score_canvas_rmse:.4f}-{score_data_rmse:.4f}-{score_pixel_ssim:.4f}-{score_pixel_mse:.4f}.png')
#             os.rename(sample_filename, new_filename)
            
#             all_cnt+=1
            
#             all_scores_canvas_rmse.append(score_canvas_rmse) # 这个不管是不是SSIM=1都列进来
#             sum_score_canvas_rmse += score_canvas_rmse
            
#             all_scores_data_rmse.append(score_data_rmse) # 这个不管是不是SSIM=1都列进来
#             sum_score_data_rmse += score_data_rmse
            
#             all_scores_pixel_ssim.append(score_pixel_ssim) # 这个不管是不是SSIM=1都列进来
#             sum_score_pixel_ssim += score_pixel_ssim
            
#             all_scores_pixel_mse.append(score_pixel_mse) # 这个不管是不是SSIM=1都列进来
#             sum_score_pixel_mse += score_pixel_mse
            
        
#         # Average SSIM for the current scale
#         all_offsetCnts.append(all_cnt)
#         if all_cnt>0:
#             level_scores_canvas_rmse.append(sum_score_canvas_rmse/all_cnt)
#             level_scores_data_rmse.append(sum_score_data_rmse/all_cnt)
#             level_scores_pixel_ssim.append(sum_score_pixel_ssim/all_cnt)
#             level_scores_pixel_mse.append(sum_score_pixel_mse/all_cnt)
        
#         step_size /= 2 # 减半，相当于放大一倍
#         level_cnt+=1

#     # Average across all scales
#     # 0,1 canvas rme:1 越小越好
#     # 2,3 data rmse:2 越小越好
#     # 4,5 pixel ssim:3 越大越好
#     # 6,7 pixel mse:4 越小越好
#     # 8 offset
#     return np.mean(level_scores_canvas_rmse),all_scores_canvas_rmse,\
#             np.mean(level_scores_data_rmse),all_scores_data_rmse,\
#             np.mean(level_scores_pixel_ssim),all_scores_pixel_ssim,\
#             np.mean(level_scores_pixel_mse),all_scores_pixel_mse,\
#             np.cumsum(all_offsetCnts)


import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import io
from PIL import Image
import dash
from dash import dcc, html, Input, Output, State

class SSIMApp:
    def __init__(self,t,data,t_recon,data_recon):
        self.t = t
        self.data = data
        self.t_recon = t_recon
        self.data_recon = data_recon
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        # 创建初始图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.t, y=self.data, name="Original"))
        fig.add_trace(go.Scatter(x=self.t_recon, y=self.data_recon, name="Reconstructed"))
        fig.update_layout(title="Interactive Plot with SSIM Calculation")
        
        # 定义布局
        self.app.layout = html.Div([
            dcc.Graph(id='interactive-plot', figure=fig),
            html.Div([
                "Width: ", dcc.Input(id="width-input", type="number", value=850, style={"margin-right": "20px"}),
                "Height: ", dcc.Input(id="height-input", type="number", value=650)
            ]),
            html.Button("Update Size", id="update-size-button"),
            html.Button("Calculate SSIM", id="calculate-button", n_clicks=0),
            html.Div(id="ssim-result", style={"marginTop": "20px", "fontSize": "18px"})
        ])
        self.fig = fig

    def _setup_callbacks(self):
        @self.app.callback(
            Output("ssim-result", "children"),
            Input("calculate-button", "n_clicks"),
            State("interactive-plot", "relayoutData"),
            State("width-input", "value"),  # 添加获取宽度的State
            State("height-input", "value"),  # 添加获取高度的State
        )
        def calculate_ssim(n_clicks, relayout_data,width, height):
            # 如果没有缩放过，使用完整的数据范围
            if not relayout_data or "xaxis.range[0]" not in relayout_data:
                x_start, x_end = 0, len(self.data)  # 使用整个数据集的范围
                y_start, y_end = min(self.data), max(self.data)  # 使用 y 轴的最小值和最大值
            else:
                # 获取缩放范围
                x_start = relayout_data["xaxis.range[0]"]
                x_end = relayout_data["xaxis.range[1]"]
                print(x_start,x_end)
                y_start = relayout_data.get("yaxis.range[0]", None)  # 获取 y 轴最小值
                y_end = relayout_data.get("yaxis.range[1]", None)    # 获取 y 轴最大值
                print(y_start,y_end)

            # 提取缩放范围内的数据
            idx1=np.where(self.t>=x_start)[0][0]
            idx2=np.where(self.t<x_end)[0][-1]
            idx2+=1 # 因为下面左开右闭
            # 考虑到采样点稀疏的时候左右两边连线不能断开即便在画面之外
            if idx1>0: # 左闭
                idx1-=1
            if idx2<len(self.data): # 右开 
                idx2+=1
            if idx2-idx1<=1: # 只有一个点，不用画图比较了
                return "no points"
            print(idx1,idx2)
            zoomed_data = self.data[idx1:idx2]
            zoomed_t=self.t[idx1:idx2]
            t_min=min(zoomed_t)
            t_max=max(zoomed_t)
            v_min=min(zoomed_data)
            v_max=max(zoomed_data)
            if y_start is None:
                y_start=v_min
                y_end=v_max

            idx1=np.where(self.t_recon>=x_start)[0][0]
            idx2=np.where(self.t_recon<x_end)[0][-1]
            idx2+=1 # 因为下面左开右闭
            # 考虑到采样点稀疏的时候左右两边连线不能断开即便在画面之外
            if idx1>0: # 左闭
                idx1-=1
            if idx2<len(self.data_recon): # 右开 
                idx2+=1
            # 有可能这一段没有点，但是idx1和idx2连线覆盖这一段，所以不能continue
            if idx2-idx1<=1:
                print(idx1,idx2,'attention')
            print(idx1,idx2)
            zoomed_recon_data = self.data_recon[idx1:idx2]
            zoomed_recon_t = self.t_recon[idx1:idx2]

            # 保存为图像到内存
        #     def save_plot_to_png(data, filename):
        #         plt.figure(figsize=(4, 2))
        #         plt.plot(data)
        #         plt.axis('off')
        #         buf = io.BytesIO()
        #         plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        #         buf.seek(0)
        #         plt.close()
        #         return np.array(Image.open(buf).convert('L'))  # 返回灰度图像数据

            # 保存为图像到磁盘
        #     def save_plot_to_png(data, filename):
        #         plt.figure(figsize=(4, 2))
        #         plt.plot(data)
        #         plt.axis('off')
        #         plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0)
        #         plt.close()
        #         return np.array(Image.open(filename).convert('L'))  # 返回灰度图像数据

        #     img1 = save_plot_to_png(zoomed_data, "zoomed_original.png")
        #     img2 = save_plot_to_png(zoomed_recon, "zoomed_reconstructed.png")
        
#             width=240
#             height=240
            filename1="zoomed_original.png"
        #     plot_and_save(zoomed_t,zoomed_data, t_min, t_max, v_min, v_max, filename1, width, height)
            plot_and_save(zoomed_t,zoomed_data, x_start, x_end, y_start, y_end, filename1, width, height)
            filename2="zoomed_recon.png"
        #     plot_and_save(zoomed_recon_t,zoomed_recon_data, t_min, t_max, v_min, v_max, filename2, width, height)
            plot_and_save(zoomed_recon_t,zoomed_recon_data, x_start, x_end, y_start, y_end, filename2, width, height)

            # 计算 SSIM
            # score=match(filename1, filename2)
            score=match_masked(filename1, filename2)
            print(f"SSIM for the current zoomed range: {score:.4f}")

            print("--------------------------------")
            return f"SSIM for the current zoomed range: {score:.4f}"

        @self.app.callback(
            Output("interactive-plot", "figure"),
            Input("update-size-button", "n_clicks"),
            [State("width-input", "value"),
             State("height-input", "value")]
        )
        def update_plot_size(n_clicks, width, height):
            self.fig.update_layout(
                autosize=False,
                width=width,
                height=height,
            )
            return self.fig

    def run(self, debug=True):
        self.app.run_server(debug=debug)

# # 示例运行
# if __name__ == '__main__':
#     data = np.sin(np.linspace(0, 20, 500))
#     t = np.arange(len(data))
#     data_recon = np.cos(np.linspace(0, 20, 500))
#     t_recon = np.arange(len(data_recon))

#     app = SSIMApp(data, t, data_recon, t_recon)
#     app.run()
