import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
import gc

def seg_bottomUp_maxerror_withTimestamps(points, maxError, joint, residual):
    if len(points) == 1:
        segment = [{'lx': 0, 'rx': 0, 'mc': float('inf')}]
        return segment
    
    if len(points) == 0:
        return []
    
    # initialization with the finest segments
    if not joint:
        left_x = np.arange(0, len(points)-1, 2)
        right_x = left_x + 1
        right_x[-1] = len(points)-1  # for odd number of points
        number_of_segments = len(left_x)
    else:
        left_x = np.arange(0, len(points)-1)
        right_x = left_x + 1
        number_of_segments = len(left_x)

    segment = []
    for i in range(number_of_segments):
        segment.append({'lx': left_x[i], 'rx': right_x[i], 'mc': float('inf')})

    # Initialize merge cost of adjacent segments
    for i in range(number_of_segments - 1):
        segment[i]['mc'] = myLA_withTimestamps(points, segment[i]['lx'], segment[i+1]['rx'], joint, residual)

    # Bottom-up merging
    while min([s['mc'] for s in segment]) < maxError and len(segment) > 1:
        # Find cheapest pair to merge
        min_mc_index = np.argmin([s['mc'] for s in segment])
        
        # Merge them
        if min_mc_index > 0 and min_mc_index < len(segment) - 2:
            # Update right index of the merged segment
            segment[min_mc_index]['rx'] = segment[min_mc_index + 1]['rx']
            
            # Update merge cost
            segment[min_mc_index]['mc'] = myLA_withTimestamps(points, segment[min_mc_index]['lx'],\
                segment[min_mc_index + 2]['rx'], joint, residual)
            
            # Remove the old segment being merged
            segment.pop(min_mc_index + 1)
            
            # Update merge cost of the previous segment
            min_mc_index -= 1
            segment[min_mc_index]['mc'] = myLA_withTimestamps(points, segment[min_mc_index]['lx'],\
                segment[min_mc_index + 1]['rx'], joint, residual)
        
        elif min_mc_index == 0:  # the first segment
            segment[min_mc_index]['rx'] = segment[min_mc_index + 1]['rx']
            
            # Update merge cost of the new merged segment
            if min_mc_index + 2 < len(segment):
                segment[min_mc_index]['mc'] = myLA_withTimestamps(points, segment[min_mc_index]['lx'],\
                    segment[min_mc_index + 2]['rx'], joint, residual)
            else:
                segment[min_mc_index]['mc'] = float('inf')
            
            # Remove the old segment being merged
            segment.pop(min_mc_index + 1)
        
        else:  # the last segment
            segment[min_mc_index]['rx'] = segment[min_mc_index + 1]['rx']
            segment[min_mc_index]['mc'] = float('inf')
            
            # Remove the old segment being merged
            segment.pop(min_mc_index + 1)
            
            # Update merge cost of the previous segment
            min_mc_index -= 1
            segment[min_mc_index]['mc'] = myLA_withTimestamps(points, segment[min_mc_index]['lx'],\
                segment[min_mc_index + 1]['rx'], joint, residual)

    return segment

def myLA_withTimestamps(points, lx, rx, joint, residual): 
    t=points[:,0]
    data=points[:,1]

    # lx~rx 左闭右闭
    """
    Linear Approximation Function: returns the error of fit for a segment.
    joint=True to fit by linear interpolation, False for linear regression.
    residual=True to use the sum of squared vertical errors, False for maximal absolute vertical error.
    """
    if joint:
        x1 = t[lx]
        y1 = data[lx]
        x2 = t[rx]
        y2 = data[rx]
        k = (y2 - y1) / (x2 - x1)
        b = (y1 * x2 - y2 * x1) / (x2 - x1)
        # best = (k * np.arange(x1, x2 + 1)) + b
        best = (k * t[lx:rx+1]) + b
    else:
        # coef = np.polyfit(np.arange(lx, rx + 1), data[lx:rx+1], 1)
        coef = np.polyfit(t[lx:rx+1], data[lx:rx+1], 1)
        # best = (coef[0] * np.arange(lx, rx + 1)) + coef[1]
        best = (coef[0] * t[lx:rx+1]) + coef[1]

    if residual:
        # Sum of squares of vertical errors

        # print('use sum of squared error')
        mc = np.sum((data[lx:rx+1] - best) ** 2) # 平方会偏好大误差

        # print('use sum of absolute error')
        # mc = np.sum(np.abs(data[lx:rx+1] - best)) # 这个可能更接近areal displacement
    else:
        # Maximal absolute vertical error
        mc = np.max(np.abs(data[lx:rx+1] - best))

    return mc

def next_slidingWindow_withTimestamps(points, max_error, joint, residual):
    if len(points) <= 1:
        return len(points)-1
    i = 2
    while i < len(points) and myLA_withTimestamps(points, 0, i, joint, residual) < max_error: # 0~i 左闭右闭
        i += 1
    return i - 1 # python从0开始，并且是从0到i-1的左闭右闭

def getPointsFromSegments_withTimestamps(points,segment,joint):
    t=points[:,0]
    data=points[:,1]

    M = []
    if not joint:
        temp = []
        for seg in segment:
            # coef = np.polyfit(range(seg["lx"], seg["rx"] + 1), data[seg["lx"]:seg["rx"] + 1], 1)
            coef = np.polyfit(t[seg["lx"]:seg["rx"] + 1], data[seg["lx"]:seg["rx"] + 1], 1)
            # best = coef[0] * np.arange(seg["lx"], seg["rx"] + 1) + coef[1]
            best = coef[0] * t[seg["lx"]:seg["rx"] + 1] + coef[1]
            seg["ly"], seg["ry"] = best[0], best[-1]
            M.append([seg["lx"], seg["ly"]])
            M.append([seg["rx"], seg["ry"]])
    else:
        for seg in segment:
            # x1, y1 = seg["lx"], data[seg["lx"]]
            x1, y1 = t[seg["lx"]], data[seg["lx"]]
            M.append([x1, y1])
            
        seg=segment[-1]
        # x2, y2 = seg["rx"], data[seg["rx"]]
        x2, y2 = t[seg["rx"]], data[seg["rx"]]
        M.append([x2, y2])
    
    M = np.array(M)
    return M

def seg_swab_withTimestamps(points, max_error, m, joint=True, residual=True, returnSegment=False, debug=False):
    # returnSegment=False意为直接返回采样点而不是分段的形式

    # 这里已经取用真实的时间戳，而不再是时间戳默认认为是从0开始的编号了

    # data=points[:,1]

    # originalData=data # 因为后面会对data重新赋值
    originalPoints=points

    if debug:
        print('data length=',len(points),', max_error=',max_error,', joint=',joint, ', residual=',residual, ', m=',m)
    
    # m在这里的含义是分段数，虽然SWAB不能直接控制分段数，但是也需要这个参数来确定buffer大小
    bs = math.floor(len(points) / m * 6)  # init buffer size
    lb, ub = bs / 2, bs * 2
    if debug:
        print('bs=',bs,', lb=',lb,', ub=',ub)
    
    # w = data[:bs] # bs个数
    # data = data[bs:]

    w=points[:bs]
    points = points[bs:]
    if debug:
        print('buffer data=',len(w),', remaining data=',len(points))
    
    seg_ts = []
    
    base = 0 # for global use
    if debug:
        print('base=',base)

    need_bottom_up = True # false when only output no input

    while True:
        if debug:
            print('#####################################################')
            print('buffer data=',len(w),', remaining data=',len(points))
            print('reBottomUp:',need_bottom_up)
        
        if need_bottom_up:
            # Call the classic Bottom-Up algorithm
            # Note that index starts from 0 (python) after calling seg_bottomUp_maxerror, 
            # so base is used for global position
            if debug:
                print('>>>>bottom up on the buffer')
            segment = seg_bottomUp_maxerror_withTimestamps(w, max_error, joint, residual)
            
        if debug:
            print('number of segments=', len(segment))

            print('---------------------------------')

        if segment:
            # 左边输出一个segment
            if debug:
                print('>>>>left output a segment')
            segment[0]["lx"] += base # note that no need to add base later for segment(1)
            segment[0]["rx"] += base # note that no need to add base later for segment(1)
            seg_ts.append(segment[0])

            if debug:
                print('output len=',segment[0]["rx"]-segment[0]["lx"]+1,f': [{segment[0]["lx"]},{segment[0]["rx"]}]')

            if len(segment) == 1:
                # 更新buffer w
                w = [w[-1]] if joint else []

                # 更新base
                # NOTE python start from 0 not 1
                # NOTE segment(0) rx already add base! i.e., segment(0) rx is already global position
                base = segment[0]["rx"] if joint else segment[0]["rx"]+1

                if debug:
                    print('base=',base)
            else:
                # 更新buffer w
                out = segment[0]["rx"] - segment[0]["lx"] + 1
                w = w[out-1:] if joint else w[out:] # note that python start from 0, so joint at the (out-1)-th point

                # 更新base
                # NOTE python start from 0 not 1
                # prepared for calling seg_bottomUp_maxerror in the future starting from 0
                base = base + segment[1]["lx"] # 这个通用于joint和disjoint

                if debug:
                    print('base=',base)
                
        if debug:
            print('buffer data=',len(w),', remaining data=',len(points))

            print('---------------------------------')
        # Add remaining segments and break if no more input data
        if len(points) == 0: # no more point added into buffer w
            if debug:
                print('>>>>no more data remaining')
            if len(segment) > 1: # 注意此时segment并没有把上面“左边输出一个segment”的第一个segment pop出去！
                if debug:
                    print('>>>>so just left output all remaining segments')

                # 更新base
                # 这次不需要re-bottomUp了，所以回退到本次分段起点，这样后面的相对位置加上base才是绝对位置
                # NOTE python start from 0 not 1
                # NOTE segments does NOT have their index recalculated by bottom-up to start from 0, 
                # therefore minus the previously added segment[1]["lx"]
                base = base - segment[1]["lx"] 
                if debug:
                    print('base=',base,': no need re-bottomUp, so revert')
                for k in range(1, len(segment)): 
                    # 注意此时segment并没有把上面“左边输出一个segment”的第一个segment pop出去！
                    # 所以要从第二个segment开始继续输出，同时注意python从0开始计数
                    segment[k]["lx"] += base
                    segment[k]["rx"] += base
                    seg_ts.append(segment[k])
                    if debug:
                        print('output len=',segment[k]["rx"]-segment[k]["lx"]+1,f': [{segment[k]["lx"]},{segment[k]["rx"]}]')
           
            break # 结束
            

        if len(w) >= lb: # buffer w上面已经更新了，已经是去除了左边输出的第一个segment之后的剩余的点
            if debug:
                print('>>>>no need adding the sliding segment from right because len(w) >= lb')
            need_bottom_up = False

            # 更新base
            # 这次不需要re-bottomUp了，所以回退到本次分段起点，这样后面的相对位置加上base才是绝对位置
            # NOTE python start from 0 not 1
            # NOTE segments does NOT have their index recalculated by bottom-up to start from 0, 
            # therefore minus the previously added segment[1]["lx"]
            base = base - segment[1]["lx"]
            if debug:
                print('base=',base,': no need re-bottomUp, so revert')

            # 注意此步骤不要在更新base之前执行！！！因为上面更新base时假设“左边输出一个segment”的第一个segment还没有pop
            # 注意这里真的把上面“左边输出一个segment”的第一个segment pop出去了，
            # 因为下一轮迭代要直接用剩余的segment
            segment.pop(0) # next iteration uses the remaining segment without re-bottomUp
            
            # assume lb>1, at least two segments
        else:
            if debug:
                print('>>>>adding the sliding segment from right because len(w) < lb')
            need_bottom_up = True # 加了新的数据进来意味着下一轮里就要re-bottomUp
            
            while len(w) < lb and len(points) > 0: # avoid less than lb
                # python从0开始，并且rx是从0开始的右闭 [0,rx]一共rx+1个点
                rx = next_slidingWindow_withTimestamps(points, max_error, joint, residual)
                if len(w) + (rx + 1) > ub: # avoid more than ub
                    rx = ub - len(w) - 1
                w = np.concatenate((w, points[:rx+1])) # 更新buffer w
                points = points[rx+1:] # 更新data
                if debug:
                    print('input len=',rx+1) # python从0开始，并且rx是从0开始的右闭

        if debug:
            print('buffer data=',len(w),', remaining data=',len(points))
            if len(w)<lb:
                print('warn less')
            if len(w)>ub:
                print('warn more')

    if returnSegment:
        return seg_ts
    else:
        return getPointsFromSegments_withTimestamps(originalPoints,seg_ts,joint)

def seg_bottom_up_withTimestamps(points, num_segments, joint=True,residual=True,returnSegment=False):
    popOrder=[]

    # Initialization with the finest segments
    if not joint:
        # print("disjoint segments from linear regression")
        left_x = np.arange(0, len(points)-1, 2)
        right_x = left_x + 1 # 最后一个分段可能是2个点或者3个点
        right_x[-1] = len(points)-1
    else:
        # print("joint segments from linear interpolation")
        left_x = np.arange(0, len(points)-1, 1)
        right_x = left_x + 1

    number_of_segments = len(left_x)

    # Initialize segments
    segments = []
    for i in range(number_of_segments):
        segments.append({
            'lx': left_x[i],
            'rx': right_x[i],
            'mc': np.inf
        })

    # Initialize merge costs of adjacent segments
    for i in range(number_of_segments - 1):
        segments[i]['mc'] = myLA_withTimestamps(points, segments[i]['lx'], segments[i + 1]['rx'], joint, residual)

    # Bottom-up merging
    while len(segments) > num_segments:
        # Find the cheapest pair to merge
        costs = [segment['mc'] for segment in segments]
        i = np.argmin(costs)
        popOrder.append(segments[i]['rx']) #注意这个区别，merge两个相邻的segments等价于把中间的点淘汰掉

        # Merge segments
        if 0 < i < len(segments) - 2: # 注意python从0开始编号
            # 即前面至少还有一个分段、后面至少还有两个分段
            # 本次要合并本分段和后面一个分段
            # 也就是一共有四个分段受到波及：
            # 合并i&i+1分段、更新i-1分段的merge cost、更新新的i分段的merge cost（与原来的i+2分段merge）
            # Update the right index of the new merged segment
            segments[i]['rx'] = segments[i + 1]['rx']

            # Update merge cost of the new merged segment
            segments[i]['mc'] = myLA_withTimestamps(points, segments[i]['lx'], segments[i + 2]['rx'], joint, residual)

            # Remove the old segment being merged
            segments.pop(i + 1)

            # Update merge cost of the previous segment with the new segment
            i = i - 1
            segments[i]['mc'] = myLA_withTimestamps(points, segments[i]['lx'], segments[i+1]['rx'], joint, residual)

        elif i == 0:  # The first segment
            # Update the right index of the new merged segment
            segments[i]['rx'] = segments[i + 1]['rx']

            # Update merge cost of the new merged segment
            if i < len(segments) - 2: # 即后面至少还有第i+2分段
                segments[i]['mc'] = myLA_withTimestamps(points, segments[i]['lx'], segments[i + 2]['rx'], joint, residual)
            else:
                segments[i]['mc'] = np.inf # 因为右边没有分段可以merge了，这个新的分段就是最后一个分段了

            # Remove the old segment being merged
            segments.pop(i + 1)

        else:  # 倒数第二个分段
            # Update the right index of the new merged segment
            segments[i]['rx'] = segments[i + 1]['rx']

            # Update merge cost of the new merged segment
            segments[i]['mc'] = np.inf # 因为右边没有分段可以merge了，这个新的分段就是最后一个分段了

            # Remove the old segment being merged
            segments.pop(i + 1)

            # Update merge cost of the previous segment with the new segment
            # 一定有i>=1，否则就走上一个elif i==0分支了
            i = i - 1
            segments[i]['mc'] = myLA_withTimestamps(points, segments[i]['lx'], segments[i+1]['rx'], joint, residual)

    if returnSegment:
        return segments
    else:
        return getPointsFromSegments_withTimestamps(points,segments,joint),popOrder