from enum import Enum
import numpy as np
from ts_areal_displacement import *

def _get_bin_idxs_equalFrequency_internal(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # 计算每个桶的大小（点数）
    n_points = len(x)
    points_per_bin = n_points // nb_bins  # 每个桶的基础点数
    remainder = n_points % nb_bins       # 分桶后剩余的点数
    
    # 生成每个桶的边界索引
    bins = [0]
    for i in range(nb_bins):
        # 前 remainder 个桶都分配 points_per_bin + 1 个点
        # 余下的 bins 分配 points_per_bin个点
        current_bin_size = points_per_bin + (1 if i < remainder else 0)
        bins.append(bins[-1] + current_bin_size)
    return np.array(bins)

def get_bin_idxs_equalFrequency(x: np.ndarray, nb_bins: int) -> np.ndarray:
    s_ds=_get_bin_idxs_equalFrequency_internal(x[:-1],nb_bins)
    return s_ds

def get_bin_idxs_equalTime(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # bins = _get_bin_idxs(t, nout-1)
    bins = np.searchsorted(x, np.linspace(x[0], x[-1], nb_bins + 1), side="left") # 第一个大于等于edge的点的位置，全局尾点是保留的
    return np.unique(bins) # 后续equal-width bucket是直接取这些bins的边界点的，因此全局尾点也是取到的，并不是用于左闭右开的分桶内采点，而是直接取分桶边界点


class ERROR(Enum):
    L1 = 1
    L2 = 2
    L_infy = 3
    area = 4


# https://justinwillmert.com/articles/2014/bellman-k-segmentation-algorithm/
# https://github.com/CrivelliLab/Protein-Structure-DL
# TODO 改造成PLA：每一段用linear interpolation近似，段与段之间joint，误差用L1/L2/L_infy/area度量
# "The principle of dynamic programming is to think top-down (i.e recursively) but solve bottom up."
# "Of course, divide and conquer will be highly ineffecient because every subproblem encountered 
# in the associated recursion tree will be solved again even if it has already been found and solved. 
# This is where DP differs: each time you encounter a subproblem, you solve it and store its solution 
# in a table. Later, when you encounter again that subproblem, you access in O(1)
# time its solution instead of solving it again. Since the number of overlapping subproblems 
# is typically bounded by a polynomial, and the time required to solve one subproblem is polynomial 
# as well (otherwise DP can not provide a cost-efficient solution), you achieve in general a polynomial solution."
# ""cut down redundant enumerating with help of storing useful value already enumerated"."
def prepare_ksegments(points, errorType, debug=False):
    '''
    '''
    # t = points[:, 0]
    # v = points[:, 1]
    N = len(points)
    dists = np.zeros((N, N))

    # 外层循环i：partial sum的元素个数
    # 内层循环j: 矩阵逐行
    for i in range(1, N + 1):
        for j in range(N - i):  # r=j+i<=N-1
            # j = left boundary, r = right boundary
            r = i + j
            if debug:
                print('>>>>', f'i={i},j={j},r={r}')

            # 从lx=j到rx=r（左闭右闭）的linear interpolation的SSE误差
            lx = j
            rx = r

            mc=joint_segment_error(points, lx, rx, errorType)

            # if errorType!=ERROR.area:
            #     x1 = t[lx]
            #     y1 = v[lx]
            #     x2 = t[rx]
            #     y2 = v[rx]
            #     k = (y2 - y1) / (x2 - x1)
            #     b = (y1 * x2 - y2 * x1) / (x2 - x1)
            #     best = (k * t[lx:rx + 1]) + b  # 注意是真实时间戳

            # if errorType == ERROR.L2:
            #     mc = np.sum((v[lx:rx + 1] - best) ** 2)  # 平方会偏好大误差
            # elif errorType == ERROR.L1:
            #     mc = np.sum(np.abs(v[lx:rx + 1] - best))
            # elif errorType == ERROR.L_infy:
            #     mc = np.max(np.abs(v[lx:rx + 1] - best))
            # else:
            #     mc = total_areal_displacement(points[lx:rx + 1],points[[lx,rx]],debug=False,plot=False,ax=None)

            dists[j, r] = mc # lx=j,rx=r
            if debug:
                print('dists=\n', dists)
                print('---------------------')
    return dists


def regress_ksegments(points, k, errorType, debug=False):
    '''
    '''
    N = len(points)

    # Get pre-computed distances and means for single-segment spans over any
    # arbitrary subsequence series(i:j). The costs for these subsequences will
    # be used *many* times over, so a huge computational factor is saved by
    # just storing these ahead of time.
    dists = prepare_ksegments(points, errorType, debug)
    if debug:
        print('dists=\n', dists)
        print('----------------------------------')

    # Keep a matrix of the total segmentation costs for any p-segmentation of
    # a subsequence series[1:n] where 1<=p<=k and 1<=n<=N. The extra column at
    # the beginning is an effective zero-th row which allows us to index to
    # the case that a (k-1)-segmentation is actually disfavored to the
    # whole-segment average.
    k_seg_dist = np.zeros((k, N + 1)) # 历史遗留问题，现在joint模式下似乎不需要这个第一列ghost来表达少采点，所以其实是多余的，还使得下面的代码为了它要index+1细节

    # Initialize the case k=1 directly from the pre-computed distances
    # 注意python从0开始，所以是0 index第一行代表分段数1
    k_seg_dist[0, 1:] = dists[0, :]
    if debug:
        print('k_seg_dist=\n', k_seg_dist)

    # Also store a pointer structure which will allow reconstruction of the
    # regression which matches. (Without this information, we'd only have the
    # cost of the regression.)
    # The index into the matrix is the right (inclusive) boundary of a segment,
    # and the value it contains is the left (exclusive) boundary.
    k_seg_path = np.zeros((k, N), dtype=int)

    # Any path with only a single segment has a right (non-inclusive) boundary
    # at the zeroth element.
    # 第一行只分一段，所以不管终点是哪个（列），左边的闭合起点都是0
    k_seg_path[0, :] = 0

    # Then for p segments through p elements, the right boundary for the (p-1)
    # case must obviously be (p-1).
    # for i in range(k):
    #     k_seg_path[i,:] = i
    # TODO 不知道这个的作用，是否可以删掉!
    np.fill_diagonal(k_seg_path, np.arange(k))

    if debug:
        print('k_seg_path=\n', k_seg_path)

    # 外层循环i：分段数，注意python从0开始，所以实际是i+1个分段 go through all remaining subcases 2 <= p <= k
    for i in range(1, k):
        # 内层循环j：从第一个点开始到j终点（闭合）的序列
        # 所以含义是：找到从第一个点开始到j终点（闭合）的序列的分成(i+1)段的最佳分段方案（误差最小）
        for j in range(0, N):
            if debug:
                print('>>>', f'分段数i+1={i + 1},end pos j={j}')
            # 动态规划
            # TODO 注意linear interpolation的话似乎不需要单点成为一个分段的情况
            # k_seg_dist[i-1, x]: 注意k_seg_dist第一列是ghost！！所以是从0:x-1(左闭右闭)的序列分成i-1段的最佳分段的SSE误差
            # dist[x,j]: 最后一个分段x:j(左闭右闭)，近似误差
            # Enumerate the choices and pick the best one. Encodes the recursion
            # for even the case where j=1 by adding an extra boundary column on the
            # left side of k_seg_dist. The j-1 indexing is then correct without
            # subtracting by one since the real values need a plus one correction.
            if errorType == ERROR.L_infy:
                choices = []
                for xtmp in range(1, j + 2):
                    choices.append(max(k_seg_dist[i - 1, xtmp], dists[xtmp - 1, j])) # TODO 注意max而不是累加
            else:
                choices = k_seg_dist[i - 1, 1:(j + 2)] + dists[:(j + 1), j]  # 注意保持joint而不是disjoint
                # e.g., j=2
                # x1=1,x2=0: 0-0; 0-2 这种单点就留着看它自己选
                # x1=2,x2=1: 0-1; 1-2
                # x1=3,x2=2: 0-2; 2-2 这种单点就留着看它自己选

            if debug:
                for x in range(j + 1):  # 遍历从 0 到 j 的每个元素
                    print(
                        f"  (k_seg_dist[{i - 1}, {x+1}] = {k_seg_dist[i - 1, x+1]}) + (dists[{x}, {j}] = {dists[x, j]}) --> {k_seg_dist[i - 1, x+1] + dists[x, j]}")

            best_index = np.argmin(choices)
            best_val = np.min(choices)

            # 从0:j的序列分成i段的最佳分段结果吗
            # Store the sub-problem solution. For the path, store where the (p-1)
            # case's right boundary is located.
            # TODO 这里的含义应该是后一个分段的左边闭起点？
            k_seg_path[i, j] = best_index

            # print(f'k_seg_path[{i},{j}] = best_index = {best_index}')
            # Then remember to offset the distance information due to the boundary
            # (ghost) cells in the first column.
            k_seg_dist[i, j + 1] = best_val

            if debug:
                print(f'k_seg_dist[{i},{j + 1}] = best_val = {best_val}')
                print('k_seg_dist=\n', k_seg_dist)
                print('k_seg_path=\n', k_seg_path)

    # Eventual complete regression
    # reg = np.zeros(series.shape)
    s_ds = []

    # Now use the solution information to reconstruct the optimal regression.
    # Fill in each segment reg(i:j) in pieces, starting from the end where the
    # solution is known.
    rhs = len(points) - 1
    s_ds.append(rhs)

    for i in reversed(range(k)):
        # Get the corresponding previous boundary
        lhs = k_seg_path[i, rhs]  # TODO 这个含义是最后一个分段的左闭合起点

        # The pair (lhs,rhs] is now a half-open interval, so set it appropriately
        s_ds.append(lhs)

        if debug:
            print("====", f'i={i},lhs={lhs},rhs={rhs},s_ds={s_ds}')

        # Update the right edge pointer
        rhs = lhs  

    if debug:
        print(s_ds[::-1])

        print(">>>>>dp[][]=\n",k_seg_dist[:,1:].T)
        print(">>>>>path[][]=\n",(k_seg_path+1).T)

    return s_ds[::-1]

def joint_segment_error(points, lx, rx, errorType): 
    t=points[:,0]
    v=points[:,1]

    # lx~rx 左闭右闭
    # 默认就是joint linear interpolation连接首尾点
    
    if errorType!=ERROR.area:
        x1 = t[lx]
        y1 = v[lx]
        x2 = t[rx]
        y2 = v[rx]
        k = (y2 - y1) / (x2 - x1)
        b = (y1 * x2 - y2 * x1) / (x2 - x1)
        # best = (k * np.arange(x1, x2 + 1)) + b
        best = (k * t[lx:rx+1]) + b

    if errorType == ERROR.L2:
        mc = np.sum((v[lx:rx+1] - best) ** 2) # 平方会偏好大误差
    elif errorType == ERROR.L1:
        mc = np.sum(np.abs(v[lx:rx+1] - best)) # 这个可能更接近areal displacement
    elif errorType == ERROR.L_infy:
        mc = np.max(np.abs(v[lx:rx + 1] - best))
    else:
        # 因为默认是linear interpolation这种joint，所以可以分段计算多边形面积因为两端交点闭合
        mc = total_areal_displacement(points[lx:rx + 1],points[[lx,rx]],debug=False,plot=False,ax=None)

    return mc
    
def error(points,s_ds,errorType):
    # if errorType == ERROR.area:
    #     res=total_areal_displacement(points,points[s_ds],debug=False,plot=False,ax=None)
    #     return res

    res=0
    for i in range(len(s_ds)-1):
        lx=s_ds[i]
        rx=s_ds[i+1]
        
        # 因为默认是linear interpolation这种joint，所以可以分段计算多边形面积因为两端交点闭合
        error=joint_segment_error(points, lx, rx, errorType)
        
        if errorType == ERROR.L_infy:
            res=max(res,error) # 注意max而不是累加
        else:
            res+=error
            
    return res