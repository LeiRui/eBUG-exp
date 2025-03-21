from typing import Union

import numpy as np
import random

from .downsampling_interface import AbstractDownsampler
from .waveletBucketTool import *

def _effective_area(x_bucket: np.ndarray, y_bucket: np.ndarray, sampled_x: np.ndarray):
    total_area=0
    for i in range(1,len(sampled_x)-1):
        l=sampled_x[i-1]
        c=sampled_x[i]
        r=sampled_x[i+1]

        t1=x_bucket[l]
        t2=x_bucket[c]
        t3=x_bucket[r]

        v1=y_bucket[l]
        v2=y_bucket[c]
        v3=y_bucket[r]

        total_area+=np.abs((t1-t3)*(v2-v3)-(t2-t3)*(v1-v3))/2

    return total_area


def _calculate_average_point(x_bucket: np.ndarray, y_bucket: np.ndarray):
    return np.sum(x_bucket)/len(x_bucket),np.sum(y_bucket)/len(y_bucket)

def _calculate_linear_regression_coefficients(x_bucket: np.ndarray, y_bucket: np.ndarray):
    avg_x,avg_y=_calculate_average_point(x_bucket,y_bucket)
   
    aNumerator = 0.0
    aDenominator = 0.0
    for i in range(len(x_bucket)):
        aNumerator += (x_bucket[i] - avg_x) * (y_bucket[i] - avg_y)
        aDenominator += (x_bucket[i] - avg_x) * (x_bucket[i] - avg_x)

    a = aNumerator / aDenominator
    b = avg_y - a*avg_x 

    return a,b

def _calculate_sse_for_bucket(x_bucket: np.ndarray, y_bucket: np.ndarray):
    a,b=_calculate_linear_regression_coefficients(x_bucket,y_bucket)
    sse = 0.0
    
    # # 加入窗口数值范围大小的因素
    # vmax=max(y_bucket)
    # vmin=min(y_bucket)
    # if vmax-vmin==0: # 水平线
    #     delta=1e-3
    #     vmax+=delta
    #     vmin-=delta

    for i in range(len(x_bucket)):
        error = y_bucket[i] - (a*x_bucket[i] + b)

        # # 加入窗口数值范围大小的因素
        # # y=(v-vmin)/(vmax-vmin)*height
        # # y2-y1=(v2-v1)/(vmax-vmin)*height
        # if error!=0:
        #     error/=(vmax-vmin)

        sse += error * error

    # return np.sqrt(sse/len(x_bucket))
    return sse


def _get_ltd_bin_idxs_faster(x: np.ndarray, y: np.ndarray, n_out: int) -> np.ndarray:
    # 加速：借鉴bottom-up line segmentation matlab代码，增量式地更新受影响的部分，而不是每轮都重新计算全部的。
    # 相同点：每次操作只会影响周围所以可以增量更新
    # 区别：bottom-up line segmentation matlab代码是bottom-up只管合并相邻的两个segments，LTD则是一来就在分好的指定桶数基础上
    # 一边merge一边还要split来保持桶数不变。
    # 此外，LTD在计算sse的时候多用了桶前后一个点来参与回归计算，前者则没有这个设计。
    # 此外，LTD merge选择依据的就是直接把两个桶各自的sse加起来，而不是真的计算merge之后的sse，
    # LTD用的是桶本身的近似误差来决策下一步的merge和split（因为LTD一来就在指定桶数的等时间间隔分好的桶的基础上），
    # bottom-up用的是相邻两个桶合并之后的近似误差来决策

    # Bucket size. Leave room for start and end data points
    block_size = (y.shape[0] - 2) / (n_out - 2)
    # Note this 'astype' cast must take place after array creation (and not with the
    # aranage() its dtype argument) or it will cast the `block_size` step to an int
    # before the arange array creation
    offset = np.arange(start=1, stop=y.shape[0], step=block_size).astype(np.int64)
    # nout-2 buckets
    # number of intervals = floor((stop-start)/step)=n-2+floor((n-2)/(N-2))=n-2!
    # 1, 1+bs, 1+2*bs, ..., 1+(nout-2)*bs=y.shape[0]-1

    # numIterations = 1000
    # numIterations = min(1000,int(len(x)/n_out*10))
    # numIterations = max(1000,int(len(x)/n_out*10))
    numIterations = int(len(x)/n_out*10)
    # numIterations = max(10000,int(len(x)/n_out*10))
    print('LTD numIterations=',numIterations)
    # numIterations = max(nout,int(len(x)/n_out*10)) # 如果nout=n/2，那复杂度就是n^2了

    sse=np.zeros(n_out-2)
    
    # initialization
    for i in range(n_out - 2):
            x_bucket = x[offset[i]-1 : offset[i + 1]+1]
            y_bucket = y[offset[i]-1 : offset[i + 1]+1]
            sse[i]=_calculate_sse_for_bucket(x_bucket,y_bucket) # replace all data
            
    for c in range(numIterations):
        # find the bucket to be split
        maxSSEIndex = -1
        maxSSE = np.finfo(np.float64).min
        for i in range(n_out - 2):
            if offset[i+1]-offset[i] <= 1:
                continue
            if sse[i] > maxSSE:
                maxSSE = sse[i]
                maxSSEIndex = i
        if maxSSEIndex < 0:
            print(c)
            print(maxSSEIndex)
            print('break max')
            break

        # find the buckets to be merged
        minSSEIndex = -1
        minSSE = np.finfo(np.float64).max
        for i in range(n_out-3):
            if i==maxSSEIndex or i+1==maxSSEIndex:
                continue
            if sse[i]+sse[i+1] < minSSE:
                minSSE = sse[i]+sse[i+1]
                minSSEIndex = i
        if minSSEIndex < 0:
            print(c)
            print(minSSEIndex)
            print('break min')
            break

        # split
        startIdx = offset[maxSSEIndex]
        endIdx = offset[maxSSEIndex+1]
        middleIdx = int(np.floor((startIdx+endIdx)/2))
        tmp=list(offset)
        tmp.insert(maxSSEIndex+1,middleIdx)
        offset=np.array(tmp)
        # update sse affected by split
        i=maxSSEIndex
        x_bucket = x[offset[i]-1 : offset[i + 1]+1]
        y_bucket = y[offset[i]-1 : offset[i + 1]+1]
        sse[i]=_calculate_sse_for_bucket(x_bucket,y_bucket)

        i=maxSSEIndex+1
        x_bucket = x[offset[i]-1 : offset[i + 1]+1]
        y_bucket = y[offset[i]-1 : offset[i + 1]+1]
        ssetmp=_calculate_sse_for_bucket(x_bucket,y_bucket)
        tmp=list(sse)
        tmp.insert(maxSSEIndex+1,ssetmp)
        sse=np.array(tmp)


        # merge
        if minSSEIndex > maxSSEIndex: 
            minSSEIndex += 1 # note when the index add one
        tmp=list(offset)
        tmp.pop(minSSEIndex+1)
        offset=np.array(tmp)
        # update sse affected by merge
        i=minSSEIndex
        x_bucket = x[offset[i]-1 : offset[i + 1]+1]
        y_bucket = y[offset[i]-1 : offset[i + 1]+1]
        sse[i]=_calculate_sse_for_bucket(x_bucket,y_bucket)

        tmp=list(sse)
        tmp.pop(minSSEIndex+1)
        sse=np.array(tmp)

    # print('final offset',offset)
    # print('sse of final offset',sse)
    return offset

def _get_ltd_bin_idxs(x: np.ndarray, y: np.ndarray, n_out: int) -> np.ndarray:
    # Bucket size. Leave room for start and end data points
    block_size = (y.shape[0] - 2) / (n_out - 2)
    # Note this 'astype' cast must take place after array creation (and not with the
    # aranage() its dtype argument) or it will cast the `block_size` step to an int
    # before the arange array creation
    offset = np.arange(start=1, stop=y.shape[0], step=block_size).astype(np.int64)
    # nout-2 buckets
    # number of intervals = floor((stop-start)/step)=n-2+floor((n-2)/(N-2))=n-2!
    # 1, 1+bs, 1+2*bs, ..., 1+(nout-2)*bs=y.shape[0]-1

    numIterations = 1000

    sse=np.zeros(n_out-2)
    for c in range(numIterations):
        for i in range(n_out - 2):
            x_bucket = x[offset[i]-1 : offset[i + 1]+1]
            y_bucket = y[offset[i]-1 : offset[i + 1]+1]
            sse[i]=_calculate_sse_for_bucket(x_bucket,y_bucket) # replace all data

        maxSSEIndex = -1
        maxSSE = np.finfo(np.float64).min
        for i in range(n_out - 2):
            if offset[i+1]-offset[i] <= 1:
                continue
            if sse[i] > maxSSE:
                maxSSE = sse[i]
                maxSSEIndex = i
        if maxSSEIndex < 0:
            print(c)
            print(maxSSEIndex)
            print('break max')
            break

        minSSEIndex = -1
        minSSE = np.finfo(np.float64).max
        for i in range(n_out-3):
            if i==maxSSEIndex or i+1==maxSSEIndex:
                continue
            if sse[i]+sse[i+1] < minSSE:
                minSSE = sse[i]+sse[i+1]
                minSSEIndex = i
        if minSSEIndex < 0:
            print(c)
            print(minSSEIndex)
            print('break min')
            break


        startIdx = offset[maxSSEIndex]
        endIdx = offset[maxSSEIndex+1]
        middleIdx = int(np.floor((startIdx+endIdx)/2))
        tmp=list(offset)
        tmp.insert(maxSSEIndex+1,middleIdx)
        offset=np.array(tmp)

        if minSSEIndex > maxSSEIndex: 
            minSSEIndex += 1
        tmp=list(offset)
        tmp.pop(minSSEIndex+1)
        offset=np.array(tmp)

    for i in range(n_out - 2):
        x_bucket = x[offset[i]-1 : offset[i + 1]+1]
        y_bucket = y[offset[i]-1 : offset[i + 1]+1]
        sse[i]=_calculate_sse_for_bucket(x_bucket,y_bucket) # replace all data
    print('final offset',offset)
    print('sse of final offset',sse)
    return offset

def _get_bin_idxs_nofirstlast_gapAware_deprecated(x: np.ndarray, nb_bins: int) -> np.ndarray:
    bins = np.searchsorted(x, np.linspace(x[1], x[-2], nb_bins + 1), side="left")
    bins[-1] = len(x)-1
    return bins

def _get_bin_idxs_stepLTTBET(x: np.ndarray, t2:int, tn:int, nb_bins: int) -> np.ndarray:
    bins = np.searchsorted(x, np.linspace(t2, tn, nb_bins + 1), side="left")
    return np.unique(bins)

def _get_bin_idxs_nofirstlast_gapAware(x: np.ndarray, nb_bins: int) -> np.ndarray:
    bins = np.searchsorted(x, np.linspace(x[1], x[-1], nb_bins + 1), side="left")
    return bins

def _get_bin_idxs_nofirstlast(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # divide [t2,tn)
    bins=_get_bin_idxs_nofirstlast_gapAware(x,nb_bins)
    return np.unique(bins)

def _get_bin_idxs_nofirstlast_new(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # divide [t2,t_{n-1}]
    bins=_get_bin_idxs_nofirstlast_gapAware_new(x,nb_bins)
    return np.unique(bins)

def _get_bin_idxs_nofirstlast_gapAware_new(x: np.ndarray, nb_bins: int) -> np.ndarray:
    bins = np.searchsorted(x, np.linspace(x[1], x[-2], nb_bins + 1), side="left")
    bins[-1] = len(x)-1 # make the last bucket right include x[-2]
    return bins

def _get_bin_idxs_gapAware(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # divide [t1,tn]
    bins = np.searchsorted(x, np.linspace(x[0], x[-1], nb_bins + 1), side="left")
    bins[-1] = len(x)
    return bins

def _get_bin_idxs(x: np.ndarray, nb_bins: int) -> np.ndarray:
    bins=_get_bin_idxs_gapAware(x,nb_bins)
    return np.unique(bins)

class LTOBETGapDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(x_bucket, y_bucket) -> int:
        arr=np.column_stack((x_bucket, y_bucket))
        result = np.empty((len(arr)-2,),arr.dtype) 
        # calculate the triangle area for the points of the current bucket
        # except the padded first and last points borrowed from the prior or afterward buckets
        p1 = arr[:-2]
        p2 = arr[1:-1]
        p3 = arr[2:]
        #an accumulators to avoid unnecessary intermediate arrays
        accr = result #Accumulate directly into result
        acc1 = np.empty_like(accr)
        np.subtract(p2[:,1], p3[:,1], out = accr) # v2-v3
        np.multiply(p1[:,0], accr,    out = accr) # t1*(v2-v3)
        np.subtract(p3[:,1], p1[:,1], out = acc1  ) # v3-v1
        np.multiply(p2[:,0], acc1,    out = acc1  ) # t2*(v3-v1)
        np.add(acc1, accr,            out = accr) # t1*(v2-v3)+t2*(v3-v1)
        np.subtract(p1[:,1], p2[:,1], out = acc1  ) # v1-v2
        np.multiply(p3[:,0], acc1,    out = acc1  ) # t3*(v1-v2)
        np.add(acc1, accr,            out = accr) # t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)
        np.abs(accr, out = accr) # |t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|
        accr /= 2. # 1/2*|t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|


        return result.argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for
            # testing
            x = np.arange(y.shape[0])

        bins = _get_bin_idxs_nofirstlast_gapAware(x, n_out-2)

        # Construct the output list
        sampled_x = []
        sampled_x.append(0)

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        for i in range(len(bins)-1):
            if bins[i]==bins[i + 1]:
                continue

            x_bucket=x[bins[i]-1 : bins[i + 1]+1].copy()
            y_bucket=y[bins[i]-1 : bins[i + 1]+1].copy()

            if i>0 and bins[i-1]==bins[i]:
                
                sampled_x.append(bins[i])
                
                x_bucket[0]=x[bins[i]]
                y_bucket[0]=y[bins[i]]

            if i<len(bins)-2 and bins[i+1]==bins[i+2]:
                
                sampled_x.append(bins[i+1]-1)
                
                x_bucket[-1]=x[bins[i+1]-1]
                y_bucket[-1]=y[bins[i+1]-1]


            a = (
                LTOBETGapDownsampler._argmax_area(
                    x_bucket=x_bucket,
                    y_bucket=y_bucket,
                )
                + bins[i]
            )
            sampled_x.append(a)

        sampled_x.append(len(x)-1)

        # Returns the sorted unique elements of an array.
        return np.unique(sampled_x)

class LTOBETDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(x_bucket, y_bucket) -> int:
        arr=np.column_stack((x_bucket, y_bucket))
        result = np.empty((len(arr)-2,),arr.dtype) 
        # calculate the triangle area for the points of the current bucket
        # except the padded first and last points borrowed from the prior or afterward buckets
        p1 = arr[:-2]
        p2 = arr[1:-1]
        p3 = arr[2:]
        #an accumulators to avoid unnecessary intermediate arrays
        accr = result #Accumulate directly into result
        acc1 = np.empty_like(accr)

        np.subtract(p2[:,1], p3[:,1], out = accr) # v2-v3
        np.multiply(p1[:,0], accr,    out = accr) # t1*(v2-v3)
        np.subtract(p3[:,1], p1[:,1], out = acc1  ) # v3-v1
        np.multiply(p2[:,0], acc1,    out = acc1  ) # t2*(v3-v1)
        np.add(acc1, accr,            out = accr) # t1*(v2-v3)+t2*(v3-v1)
        np.subtract(p1[:,1], p2[:,1], out = acc1  ) # v1-v2
        np.multiply(p3[:,0], acc1,    out = acc1  ) # t3*(v1-v2)
        np.add(acc1, accr,            out = accr) # t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)
        np.abs(accr, out = accr) # |t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|
        accr /= 2. # 1/2*|t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|

        return result.argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for
            # testing
            x = np.arange(y.shape[0])

        bins = _get_bin_idxs_nofirstlast(x, n_out-2)

        sampled_x = []
        sampled_x.append(0)

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        for lower, upper in zip(bins, bins[1:]):
            if upper==lower:
                continue

            a = (
                LTOBETDownsampler._argmax_area(
                    x_bucket=x[lower-1 : upper+1].copy(),
                    y_bucket=y[lower-1 : upper+1].copy(),
                    # offset starts from 1, ends at len(x)-1
                    # so lower-1 >= 0, upper+1 <= len(x)
                    # and x[] is left included and right excluded, so correct.
                    # -1 +1 for padding as the triangle areas of the first and last points of the current bucket needs calculation
                    # otherwise might result in points that has the max triangle area but at the border of the bucket not selected
                )
                + lower
            )
            sampled_x.append(a)

        sampled_x.append(len(x)-1)

        # Returns the sorted unique elements of an array.
        return np.unique(sampled_x)

class LTOBDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(x_bucket, y_bucket) -> int:
        arr=np.column_stack((x_bucket, y_bucket))
        result = np.empty((len(arr)-2,),arr.dtype) 
        # calculate the triangle area for the points of the current bucket
        # except the padded first and last points borrowed from the prior or afterward buckets
        p1 = arr[:-2]
        p2 = arr[1:-1]
        p3 = arr[2:]
        #an accumulators to avoid unnecessary intermediate arrays
        accr = result #Accumulate directly into result
        acc1 = np.empty_like(accr)
        np.subtract(p2[:,1], p3[:,1], out = accr) # v2-v3
        np.multiply(p1[:,0], accr,    out = accr) # t1*(v2-v3)
        np.subtract(p3[:,1], p1[:,1], out = acc1  ) # v3-v1
        np.multiply(p2[:,0], acc1,    out = acc1  ) # t2*(v3-v1)
        np.add(acc1, accr,            out = accr) # t1*(v2-v3)+t2*(v3-v1)
        np.subtract(p1[:,1], p2[:,1], out = acc1  ) # v1-v2
        np.multiply(p3[:,0], acc1,    out = acc1  ) # t3*(v1-v2)
        np.add(acc1, accr,            out = accr) # t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)
        np.abs(accr, out = accr) # |t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|
        accr /= 2. # 1/2*|t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|

        return result.argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # Bucket size. Leave room for start and end data points
        block_size = (y.shape[0] - 2) / (n_out - 2)
        # Note this 'astype' cast must take place after array creation (and not with the
        # aranage() its dtype argument) or it will cast the `block_size` step to an int
        # before the arange array creation
        offset = np.arange(start=1, stop=y.shape[0], step=block_size).astype(np.int64)

        # Construct the output array
        sampled_x = np.empty(n_out, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(n_out - 2):
            a = (
                LTOBDownsampler._argmax_area(
                    x_bucket=x[offset[i]-1 : offset[i + 1]+1].copy(),
                    y_bucket=y[offset[i]-1 : offset[i + 1]+1].copy(),
                    # offset starts from 1, ends at y.shape[0]-1
                    # so offset[i]-1 >= 0, offset[i + 1]+1<=y.shape[0]
                    # and x[] is left included and right excluded, so correct.
                    # -1 +1 for padding as the triangle areas of the first and last points of the current bucket needs calculation
                    # otherwise might result in points that has the max triangle area but at the border of the bucket not selected
                )
                + offset[i]
            )
            sampled_x[i + 1] = a

        return sampled_x

class LTDOBDownsampler(AbstractDownsampler):
    # LTOB based on dynamic buckets
    @staticmethod
    def _argmax_area(x_bucket, y_bucket) -> int:
        arr=np.column_stack((x_bucket, y_bucket))
        result = np.empty((len(arr)-2,),arr.dtype) 
        # calculate the triangle area for the points of the current bucket
        # except the padded first and last points borrowed from the prior or afterward buckets
        p1 = arr[:-2]
        p2 = arr[1:-1]
        p3 = arr[2:]
        #an accumulators to avoid unnecessary intermediate arrays
        accr = result #Accumulate directly into result
        acc1 = np.empty_like(accr)
        np.subtract(p2[:,1], p3[:,1], out = accr) # v2-v3
        np.multiply(p1[:,0], accr,    out = accr) # t1*(v2-v3)
        np.subtract(p3[:,1], p1[:,1], out = acc1  ) # v3-v1
        np.multiply(p2[:,0], acc1,    out = acc1  ) # t2*(v3-v1)
        np.add(acc1, accr,            out = accr) # t1*(v2-v3)+t2*(v3-v1)
        np.subtract(p1[:,1], p2[:,1], out = acc1  ) # v1-v2
        np.multiply(p3[:,0], acc1,    out = acc1  ) # t3*(v1-v2)
        np.add(acc1, accr,            out = accr) # t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)
        np.abs(accr, out = accr) # |t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|
        accr /= 2. # 1/2*|t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|

        return result.argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        offset = _get_ltd_bin_idxs_faster(x,y,n_out)

        # Construct the output array
        sampled_x = np.empty(n_out, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(n_out - 2):
            a = (
                LTDOBDownsampler._argmax_area(
                    x_bucket=x[offset[i]-1 : offset[i + 1]+1].copy(),
                    y_bucket=y[offset[i]-1 : offset[i + 1]+1].copy(),
                    # offset starts from 1, ends at y.shape[0]-1
                    # so offset[i]-1 >= 0, offset[i + 1]+1<=y.shape[0]
                    # and x[] is left included and right excluded, so correct.
                    # -1 +1 for padding as the triangle areas of the first and last points of the current bucket needs calculation
                    # otherwise might result in points that has the max triangle area but at the border of the bucket not selected
                )
                + offset[i]
            )
            sampled_x[i + 1] = a

        return sampled_x

class LTSDownsampler(AbstractDownsampler):
    # optimal
    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out <=8, "n_out must be smaller than 8, for toy example of LTS"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        # assert len(x)<100, "len(x) must be smaller than 100, for toy example of LTS"

        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # bins = _get_bin_idxs_nofirstlast(x, n_out-2)
        bins = _get_bin_idxs(x, n_out-2)

        nbins=len(bins)-1

        assert nbins == 6, "nbins must be 6, for toy example of LTS"

        # Construct the output list
        optimal_x = np.empty(nbins+2, dtype="int64")
        largest_area = -1

        sampled_x = np.empty(nbins+2, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        for i1 in np.arange(bins[0],bins[1]):
            sampled_x[1]=i1
            for i2 in np.arange(bins[1],bins[2]):
                sampled_x[2]=i2
                for i3 in np.arange(bins[2],bins[3]):
                    sampled_x[3]=i3
                    for i4 in np.arange(bins[3],bins[4]):
                        sampled_x[4]=i4
                        for i5 in np.arange(bins[4],bins[5]):
                            sampled_x[5]=i5
                            for i6 in np.arange(bins[5],bins[6]):
                                sampled_x[6]=i6
                                area=_effective_area(x,y,sampled_x)
                                if area > largest_area:
                                    largest_area = area
                                    optimal_x = np.array(sampled_x) # NOTE: use deep copy!

        print('LTSDownsampler max area',largest_area)
        print(optimal_x)
        return optimal_x

class StepLTTBETDownsampler_deprecated(AbstractDownsampler):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        bins = _get_bin_idxs_stepLTTBET(x, 100, 2100, n_out-2)

        # Construct the output list
        sampled_x = []
        sampled_x.append(0)

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(len(bins)-2): 
            a = (
                LTTBETDownsampler._argmax_area(
                        prev_x=x[a],
                        prev_y=y[a],

                        # 这里不用担心桶里没有点，因为提前合并过空桶了
                        avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                        avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),

                        x_bucket=x[bins[i] : bins[i + 1]].copy(),
                        y_bucket=y[bins[i] : bins[i + 1]].copy(),
                    )
                + bins[i]
            )
            sampled_x.append(a)

        # ------------ EDGE CASE ------------
        # next-average of last bucket = last point
        sampled_x.append(
            LTTBETDownsampler._argmax_area(
                prev_x=x[a],
                prev_y=y[a],

                avg_next_x=x[-1],  # last point
                avg_next_y=y[-1],  # last point

                x_bucket=x[bins[-2] : bins[-1]].copy(),
                y_bucket=y[bins[-2] : bins[-1]].copy(),
            )
            + bins[-2]
        )

        sampled_x.append(x.shape[0] - 1)
        return np.unique(sampled_x)

class LTTBETDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # bins = _get_bin_idxs_nofirstlast_new(x, n_out-2)
        # bins = _get_bin_idxs_nofirstlast(x, n_out-2)
        bins = _get_bin_idxs(x, n_out-2)
        # print(bins)

        # Construct the output list
        sampled_x = []
        sampled_x.append(0)

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(len(bins)-2): 
            a = (
                LTTBETDownsampler._argmax_area(
                        prev_x=x[a],
                        prev_y=y[a],

                        # note this!!! float to avoid overflow in _argmax_area
                        avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                        avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),

                        x_bucket=x[bins[i] : bins[i + 1]].copy(),
                        y_bucket=y[bins[i] : bins[i + 1]].copy(),
                    )
                + bins[i]
            )
            sampled_x.append(a)

        # ------------ EDGE CASE ------------
        # next-average of last bucket = last point
        # print(x[bins[-2] : bins[-1]])
        sampled_x.append(
            LTTBETDownsampler._argmax_area(
                prev_x=x[a],
                prev_y=y[a],

                # note this!!! float to avoid overflow in _argmax_area
                avg_next_x=float(x[-1]),  # last point
                avg_next_y=float(y[-1]),  # last point

                x_bucket=x[bins[-2] : bins[-1]].copy(),
                y_bucket=y[bins[-2] : bins[-1]].copy(),
            )
            + bins[-2]
        )

        sampled_x.append(x.shape[0] - 1)

        # return np.unique(sampled_x)
        return np.sort(sampled_x)

class LTTBETFurtherDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # bins = _get_bin_idxs_nofirstlast_new(x, n_out-2)
        bins = _get_bin_idxs(x, n_out-2)

        nbins=len(bins)-1

        # Construct the output list
        sampled_x = np.empty(nbins+2, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        # if x.dtype == np.bool_:
        #     x = x.astype(np.int8)
        # if y.dtype == np.bool_:
        #     y = y.astype(np.int8)

        numIterations=8
        # print('numIterations=',numIterations)

        areas=np.zeros(numIterations)
        for num in range(numIterations):
            a = 0
            if num==0:
                for i in range(nbins-1): 
                    a = (
                        LTTBETFurtherDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # note this!!! float to avoid overflow in _argmax_area
                                avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                                avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )
                    sampled_x[i+1] = a
            else:
                for i in range(nbins-1): 
                    a = (
                        LTTBETFurtherDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # note this!!! float to avoid overflow in _argmax_area
                                avg_next_x=float(x[sampled_x[i+2]]),
                                avg_next_y=float(y[sampled_x[i+2]]),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )

                    sampled_x[i+1] = a

            # ------------ EDGE CASE ------------
            # next-average of last bucket = last point
            sampled_x[-2]=(
                LTTBETFurtherDownsampler._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],

                    # note this!!! float to avoid overflow in _argmax_area
                    avg_next_x=float(x[-1]),  # last point
                    avg_next_y=float(y[-1]),  # last point

                    x_bucket=x[bins[-2] : bins[-1]].copy(),
                    y_bucket=y[bins[-2] : bins[-1]].copy(),
                )
                + bins[-2]
            )

            areas[num]=_effective_area(x,y,sampled_x)

        print('LTTBETFurtherDownsampler effective area of all iterations',areas)
        return sampled_x



class ILTSWaveletDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # bins = _get_bin_idxs_nofirstlast_new(x, n_out-2)
        # bins = _get_bin_idxs(x, n_out-2)
        print('padding to 2^n and x is ids')

        bins,_ = waveletBucketFromData(y,n_out-2)
        # print(bins)

        nbins=len(bins)-1

        # Construct the output list
        sampled_x = np.empty(nbins+2, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        # if x.dtype == np.bool_:
        #     x = x.astype(np.int8)
        # if y.dtype == np.bool_:
        #     y = y.astype(np.int8)

        numIterations=8
        # print('numIterations=',numIterations)

        areas=np.zeros(numIterations)
        for num in range(numIterations):
            a = 0
            if num==0:
                for i in range(nbins-1): 
                    a = (
                        ILTSWaveletDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # note this!!! float to avoid overflow in _argmax_area
                                avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                                avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )
                    sampled_x[i+1] = a
            else:
                for i in range(nbins-1): 
                    a = (
                        ILTSWaveletDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # note this!!! float to avoid overflow in _argmax_area
                                avg_next_x=float(x[sampled_x[i+2]]),
                                avg_next_y=float(y[sampled_x[i+2]]),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )

                    sampled_x[i+1] = a

            # ------------ EDGE CASE ------------
            # next-average of last bucket = last point
            sampled_x[-2]=(
                ILTSWaveletDownsampler._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],

                    # note this!!! float to avoid overflow in _argmax_area
                    avg_next_x=float(x[-1]),  # last point
                    avg_next_y=float(y[-1]),  # last point

                    x_bucket=x[bins[-2] : bins[-1]].copy(),
                    y_bucket=y[bins[-2] : bins[-1]].copy(),
                )
                + bins[-2]
            )

            areas[num]=_effective_area(x,y,sampled_x)

        print('ILTSWaveletDownsampler effective area of all iterations',areas)
        return sampled_x


class ILTSLTDDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # bins = _get_bin_idxs_nofirstlast_new(x, n_out-2)
        # bins = _get_bin_idxs(x, n_out-2)

        bins = _get_ltd_bin_idxs_faster(x,y,n_out)
        # print(bins)

        nbins=len(bins)-1

        # Construct the output list
        sampled_x = np.empty(nbins+2, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        # if x.dtype == np.bool_:
        #     x = x.astype(np.int8)
        # if y.dtype == np.bool_:
        #     y = y.astype(np.int8)

        numIterations=8
        # print('numIterations=',numIterations)

        areas=np.zeros(numIterations)
        for num in range(numIterations):
            a = 0
            if num==0:
                for i in range(nbins-1): 
                    a = (
                        ILTSLTDDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # note this!!! float to avoid overflow in _argmax_area
                                avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                                avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )
                    sampled_x[i+1] = a
            else:
                for i in range(nbins-1): 
                    a = (
                        ILTSLTDDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # note this!!! float to avoid overflow in _argmax_area
                                avg_next_x=float(x[sampled_x[i+2]]),
                                avg_next_y=float(y[sampled_x[i+2]]),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )

                    sampled_x[i+1] = a

            # ------------ EDGE CASE ------------
            # next-average of last bucket = last point
            sampled_x[-2]=(
                ILTSLTDDownsampler._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],

                    # note this!!! float to avoid overflow in _argmax_area
                    avg_next_x=float(x[-1]),  # last point
                    avg_next_y=float(y[-1]),  # last point

                    x_bucket=x[bins[-2] : bins[-1]].copy(),
                    y_bucket=y[bins[-2] : bins[-1]].copy(),
                )
                + bins[-2]
            )

            areas[num]=_effective_area(x,y,sampled_x)

        print('ILTSLTDDownsampler effective area of all iterations',areas)
        return sampled_x


class LTTBETFurtherRandomDownsampler(AbstractDownsampler): 
    # init with random points
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        bins = _get_bin_idxs_nofirstlast(x, n_out-2)

        nbins=len(bins)-1

        # Construct the output list
        sampled_x = np.empty(nbins+2, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        # if x.dtype == np.bool_:
        #     x = x.astype(np.int8)
        # if y.dtype == np.bool_:
        #     y = y.astype(np.int8)

        numIterations=8
        # print('numIterations=',numIterations)

        areas=np.zeros(numIterations)
        for num in range(numIterations):
            a = 0
            if num==0:
                for i in range(nbins-1): 
                    ran = random.randrange(bins[i + 1],bins[i + 2]) # left close right open
                    a = (
                        LTTBETFurtherRandomDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # note this!!! float to avoid overflow in _argmax_area
                                avg_next_x = float(x[ran]),
                                avg_next_y = float(y[ran]),
                                # avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                                # avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )
                    sampled_x[i+1] = a
            else:
                for i in range(nbins-1): 
                    a = (
                        LTTBETFurtherRandomDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # note this!!! float to avoid overflow in _argmax_area
                                avg_next_x=float(x[sampled_x[i+2]]),
                                avg_next_y=float(y[sampled_x[i+2]]),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )

                    sampled_x[i+1] = a

            # ------------ EDGE CASE ------------
            # next-average of last bucket = last point
            sampled_x[-2]=(
                LTTBETFurtherRandomDownsampler._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],

                    # note this!!! float to avoid overflow in _argmax_area
                    avg_next_x=float(x[-1]),  # last point
                    avg_next_y=float(y[-1]),  # last point

                    x_bucket=x[bins[-2] : bins[-1]].copy(),
                    y_bucket=y[bins[-2] : bins[-1]].copy(),
                )
                + bins[-2]
            )

            areas[num]=_effective_area(x,y,sampled_x)

        print('LTTBETFurtherRandomDownsampler effective area of all iterations',areas)
        return sampled_x

class LTTBETFurtherMinMaxDownsampler(AbstractDownsampler): 
    # init with random points
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        bins = _get_bin_idxs_nofirstlast(x, n_out-2)

        nbins=len(bins)-1

        # Construct the output list
        sampled_x = np.empty(nbins+2, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        # if x.dtype == np.bool_:
        #     x = x.astype(np.int8)
        # if y.dtype == np.bool_:
        #     y = y.astype(np.int8)

        numIterations=8
        # print('numIterations=',numIterations)

        # init with MinMax
        s_ds = MinMaxFPLPNoUniqueDownsampler().downsample(x, y, n_out=n_out)

        areas=np.zeros(numIterations)
        for num in range(numIterations):
            a = 0
            if num==0:
                for i in range(nbins-1): 
                    # ran = random.randrange(bins[i + 1],bins[i + 2]) # left close right open
                    a = (
                        LTTBETFurtherMinMaxDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # note this!!! float to avoid overflow in _argmax_area
                                # avg_next_x = float(x[ran]),
                                # avg_next_y = float(y[ran]),
                                # avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                                # avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),
                                avg_next_x = float(x[s_ds[i+2]]),
                                avg_next_y = float(y[s_ds[i+2]]),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )
                    sampled_x[i+1] = a
            else:
                for i in range(nbins-1): 
                    a = (
                        LTTBETFurtherMinMaxDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # note this!!! float to avoid overflow in _argmax_area
                                avg_next_x=float(x[sampled_x[i+2]]),
                                avg_next_y=float(y[sampled_x[i+2]]),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )

                    sampled_x[i+1] = a

            # ------------ EDGE CASE ------------
            # next-average of last bucket = last point
            sampled_x[-2]=(
                LTTBETFurtherMinMaxDownsampler._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],

                    # note this!!! float to avoid overflow in _argmax_area
                    avg_next_x=float(x[-1]),  # last point
                    avg_next_y=float(y[-1]),  # last point

                    x_bucket=x[bins[-2] : bins[-1]].copy(),
                    y_bucket=y[bins[-2] : bins[-1]].copy(),
                )
                + bins[-2]
            )

            areas[num]=_effective_area(x,y,sampled_x)

        print('LTTBETFurtherMinMaxDownsampler effective area of all iterations',areas)
        return sampled_x


class ILTSParallelDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        bins = _get_bin_idxs_nofirstlast(x, n_out-2)
        print('ILTSParallel bins=',bins)

        nbins=len(bins)-1

        # Construct the output list
        sampled_x = np.empty(nbins+2, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Construct the result of last round
        lastIter_sampled_x = np.empty(nbins+2, dtype="int64")

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        numIterations=8
        areas=np.zeros(numIterations)
        for num in range(numIterations):
            a = 0
            if num==0:
                sampled_x[1]=(
                    ILTSParallelDownsampler._argmax_area(
                        prev_x=x[0],
                        prev_y=y[0],

                        avg_next_x=np.mean(x[bins[1] : bins[2]]),
                        avg_next_y=y[bins[1] : bins[2]].mean(),

                        x_bucket=x[bins[0] : bins[1]].copy(),
                        y_bucket=y[bins[0] : bins[1]].copy(),
                    )
                    + bins[0]
                )
                for i in range(1,nbins-1): 
                    a = (
                        ILTSParallelDownsampler._argmax_area(
                                prev_x=np.mean(x[bins[i-1] : bins[i]]),
                                prev_y=y[bins[i-1] : bins[i]].mean(),

                                avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                                avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )
                    sampled_x[i+1] = a

                # next-average of last bucket = last point 
                sampled_x[-2]=(
                    ILTSParallelDownsampler._argmax_area(
                        prev_x=np.mean(x[bins[-3] : bins[-2]]),
                        prev_y=y[bins[-3] : bins[-2]].mean(),

                        avg_next_x=x[-1],  # last point
                        avg_next_y=y[-1],  # last point

                        x_bucket=x[bins[-2] : bins[-1]].copy(),
                        y_bucket=y[bins[-2] : bins[-1]].copy(),
                    )
                    + bins[-2]
                )

            else:
                for i in range(nbins): 
                    a = (
                        ILTSParallelDownsampler._argmax_area(
                                prev_x=x[lastIter_sampled_x[i]],
                                prev_y=y[lastIter_sampled_x[i]],

                                avg_next_x=x[lastIter_sampled_x[i+2]],
                                avg_next_y=y[lastIter_sampled_x[i+2]],

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    )

                    sampled_x[i+1] = a

            print('ILTSParallelDownsampler sampling result of each iteration',num+1,sampled_x)
            print(_effective_area(x,y,sampled_x))
            areas[num]=_effective_area(x,y,sampled_x)

            lastIter_sampled_x=np.array(sampled_x)

        print('ILTSParallelDownsampler effective area of all iterations',areas)
        return sampled_x

class LTTBETNewDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        bins = _get_bin_idxs_nofirstlast(x, n_out-2)

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        # Construct the output list
        sampled_x = []
        sampled_x.append(bins[0] + y[bins[0] : bins[1]].argmin())
        sampled_x.append(bins[0] + y[bins[0] : bins[1]].argmax())

        a = 0
        for i in range(1,len(bins)-1):
            a = (
                LTTBETNewDownsampler._argmax_area(
                        prev_x=x[sampled_x[-2]],
                        prev_y=y[sampled_x[-2]],

                        avg_next_x=x[sampled_x[-1]],
                        avg_next_y=y[sampled_x[-1]],

                        x_bucket=x[bins[i] : bins[i + 1]].copy(),
                        y_bucket=y[bins[i] : bins[i + 1]].copy(),
                    )
                + bins[i]
            )
            sampled_x.append(a)

        sampled_x.append(x.shape[0] - 1)
        return np.unique(sampled_x)

class LTTBETGapDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        bins = _get_bin_idxs_nofirstlast_gapAware(x, n_out-2)

        # Construct the output list
        sampled_x = []
        sampled_x.append(0) # 第一个点

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(len(bins)-2): 
            if bins[i]==bins[i + 1]:
                a=bins[i]
                sampled_x.append(a)
                continue

            if bins[i + 1]==bins[i + 2]:
                avg_next_x=x[bins[i + 1]-1]
                avg_next_y=y[bins[i + 1]-1]
                sampled_x.append(bins[i + 1]-1)
            else:
                avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]])
                avg_next_y=y[bins[i + 1] : bins[i + 2]].mean()


            a = (
                LTTBETGapDownsampler._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],

                    avg_next_x=avg_next_x,
                    avg_next_y=avg_next_y,

                    x_bucket=x[bins[i] : bins[i + 1]].copy(),
                    y_bucket=y[bins[i] : bins[i + 1]].copy(),
                ) 
                + bins[i]
            )
            sampled_x.append(a)

        # ------------ EDGE CASE ------------
        sampled_x.append(
            LTTBETGapDownsampler._argmax_area(
                prev_x=x[a],
                prev_y=y[a],

                avg_next_x=x[-1],  # last point
                avg_next_y=y[-1],  # last point

                x_bucket=x[bins[-2] : bins[-1]].copy(),
                y_bucket=y[bins[-2] : bins[-1]].copy(),
            )
            + bins[-2]
        )

        sampled_x.append(x.shape[0] - 1)
        return np.unique(sampled_x)

class LTTBDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # Bucket size. Leave room for start and end data points
        block_size = (y.shape[0] - 2) / (n_out - 2)
        offset = np.arange(start=1, stop=y.shape[0], step=block_size).astype(np.int64)

        # Construct the output array
        sampled_x = np.empty(n_out, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(n_out - 3):
            a = (
                LTTBDownsampler._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],

                    avg_next_x=np.mean(x[offset[i + 1] : offset[i + 2]]),
                    avg_next_y=y[offset[i + 1] : offset[i + 2]].mean(),

                    x_bucket=x[offset[i] : offset[i + 1]].copy(),
                    y_bucket=y[offset[i] : offset[i + 1]].copy(),
                )
                + offset[i]
            )
            sampled_x[i + 1] = a

        # ------------ EDGE CASE ------------
        # next-average of last bucket = last point
        sampled_x[-2] = (
            LTTBDownsampler._argmax_area(
                prev_x=x[a],
                prev_y=y[a],

                avg_next_x=x[-1],  # last point
                avg_next_y=y[-1],  # last point

                x_bucket=x[offset[-2] : offset[-1]].copy(),
                y_bucket=y[offset[-2] : offset[-1]].copy(),
            )
            + offset[-2]
        )
        return sampled_x

class LTDDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        offset = _get_ltd_bin_idxs_faster(x,y,n_out)
        # print(offset)

        # Construct the output array
        sampled_x = np.empty(n_out, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(n_out - 3):
            a = (
                LTDDownsampler._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],

                    avg_next_x=np.mean(x[offset[i + 1] : offset[i + 2]]),
                    avg_next_y=y[offset[i + 1] : offset[i + 2]].mean(),

                    x_bucket=x[offset[i] : offset[i + 1]].copy(),
                    y_bucket=y[offset[i] : offset[i + 1]].copy(),
                )
                + offset[i]
            )
            sampled_x[i + 1] = a

        # ------------ EDGE CASE ------------
        # next-average of last bucket = last point
        sampled_x[-2] = (
            LTDDownsampler._argmax_area(
                prev_x=x[a],
                prev_y=y[a],

                avg_next_x=x[-1],  # last point
                avg_next_y=y[-1],  # last point

                x_bucket=x[offset[-2] : offset[-1]].copy(),
                y_bucket=y[offset[-2] : offset[-1]].copy(),
            )
            + offset[-2]
        )
        return sampled_x

class MinMaxDownsampler(AbstractDownsampler):
    """Aggregation method which performs binned min-max aggregation over fully
    overlapping windows.
    """

    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out % 2 == 0, "n_out must be a multiple of 2"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        bins = _get_bin_idxs(x, n_out // 2)

        rel_idxs = []
        for lower, upper in zip(bins, bins[1:]):
            y_slice = y[lower:upper]
            if not len(y_slice):
                continue
            # calculate the argmin(slice) & argmax(slice)
            rel_idxs.append(lower + y_slice.argmin())
            rel_idxs.append(lower + y_slice.argmax())

        # Returns the sorted unique elements of an array.
        # return np.unique(rel_idxs)
        return np.sort(rel_idxs)

class MinMaxFPLPDownsampler(AbstractDownsampler):
    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert (n_out-2) % 2 == 0, "n_out-2 must be a multiple of 2"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        # bins = _get_bin_idxs_nofirstlast(x, int((n_out-2)/2))
        bins = _get_bin_idxs(x, int((n_out-2)/2))

        rel_idxs = []
        rel_idxs.append(0)
        rel_idxs.append(x.shape[0] - 1)

        for lower, upper in zip(bins, bins[1:]):
            y_slice = y[lower:upper]
            if not len(y_slice):
                continue
            # calculate the argmin(slice) & argmax(slice)
            rel_idxs.append(lower + y_slice.argmin())
            rel_idxs.append(lower + y_slice.argmax())

        # Returns the sorted unique elements of an array.
        # return np.unique(rel_idxs)
        return np.sort(rel_idxs)


class MinMaxFPLPNoUniqueDownsampler(AbstractDownsampler):
    # not using np.unique, just np.sort
    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert (n_out-2) % 2 == 0, "n_out-2 must be a multiple of 2"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        bins = _get_bin_idxs_nofirstlast(x, int((n_out-2)/2))

        rel_idxs = []
        rel_idxs.append(0)
        rel_idxs.append(x.shape[0] - 1)

        for lower, upper in zip(bins, bins[1:]):
            y_slice = y[lower:upper]
            if not len(y_slice):
                continue
            # calculate the argmin(slice) & argmax(slice)
            rel_idxs.append(lower + y_slice.argmin())
            rel_idxs.append(lower + y_slice.argmax())

        # Returns the sorted unique elements of an array.
        return np.sort(rel_idxs)


class MinMaxGapDownsampler(AbstractDownsampler):

    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out % 2 == 0, "n_out must be a multiple of 2"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        bins = _get_bin_idxs_gapAware(x, n_out // 2)

        rel_idxs = []
        for i in range(len(bins)-1):
            if bins[i]==bins[i+1]:
                continue

            if i>0 and bins[i-1]==bins[i]:
                rel_idxs.append(bins[i])

            if i<len(bins)-2 and bins[i+1]==bins[i+2]:
                rel_idxs.append(bins[i+1]-1)

            y_slice = y[bins[i]:bins[i+1]]
            # calculate the argmin(slice) & argmax(slice)
            rel_idxs.append(bins[i] + y_slice.argmin())
            rel_idxs.append(bins[i] + y_slice.argmax())

        # Returns the sorted unique elements of an array.
        return np.unique(rel_idxs)

class M4Downsampler(AbstractDownsampler):

    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out % 4 == 0, "n_out must be a multiple of 4"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        bins = _get_bin_idxs(x, n_out // 4)

        rel_idxs = []
        for lower, upper in zip(bins, bins[1:]):
            y_slice = y[lower:upper]
            if not len(y_slice):
                continue

            # calculate the min(idx), argmin(slice), argmax(slice), max(idx)
            rel_idxs.append(lower)
            rel_idxs.append(lower + y_slice.argmin())
            rel_idxs.append(lower + y_slice.argmax())
            rel_idxs.append(upper - 1)

        # NOTE: we do not use the np.unique so that all indices are retained
        # return np.array(sorted(rel_idxs))
        # Returns the sorted unique elements of an array.
        # return np.unique(rel_idxs)
        return np.sort(rel_idxs)

class EveryNthDownsampler(AbstractDownsampler):
    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **_
    ) -> np.ndarray:
        if x is not None:
            name = self.__class__.__name__
            warnings.warn(
                f"x is passed to downsample method of {name}, but is not taken "
                "into account by the current implementation of the EveryNth algorithm."
            )
        step = max(1, len(y) / n_out)
        return np.arange(start=0, stop=len(y) - 0.1, step=step).astype(np.uint)
