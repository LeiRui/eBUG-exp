'''
Visvalingam-Whyatt method of poly-line vertex reduction

Visvalingam, M and Whyatt J D (1993)
"Line Generalisation by Repeated Elimination of Points", Cartographic J., 30 (1), 46 - 51

Described here:
http://web.archive.org/web/20100428020453/http://www2.dcs.hull.ac.uk/CISRG/publications/DPs/DP10/DP10.html

=========================================

The MIT License (MIT)

Copyright (c) 2014 Elliot Hallmark

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

================================
'''

from numpy import array, argmin
import numpy as np
import matplotlib.pyplot as plt
from ts_areal_displacement import *

def triangle_area(p1,p2,p3):
    """
    calculates the area of a triangle given its vertices
    """
    return abs(p1[0]*(p2[1]-p3[1])+p2[0]*(p3[1]-p1[1])+p3[0]*(p1[1]-p2[1]))/2.

def triangle_areas_from_array(arr):
    '''
    take an (N,2) array of points and return an (N,1)
    array of the areas of those triangles, where the first
    and last areas are np.inf

    see triangle_area for algorithm
    '''

    result = np.empty((len(arr),),arr.dtype)
    result[0] = np.inf; result[-1] = np.inf

    p1 = arr[:-2]
    p2 = arr[1:-1]
    p3 = arr[2:]
    
    #an accumulators to avoid unnecessary intermediate arrays
    accr = result[1:-1] #Accumulate directly into result
    acc1 = np.empty_like(accr)

    np.subtract(p2[:,1], p3[:,1], out = accr)
    np.multiply(p1[:,0], accr,    out = accr)
    np.subtract(p3[:,1], p1[:,1], out = acc1  )
    np.multiply(p2[:,0], acc1,    out = acc1  )
    np.add(acc1, accr,            out = accr)
    np.subtract(p1[:,1], p2[:,1], out = acc1  )
    np.multiply(p3[:,0], acc1,    out = acc1  )
    np.add(acc1, accr,            out = accr)
    np.abs(accr, out = accr)
    accr /= 2.
    #Notice: accr was writing into result, so the answer is in there
    return result

#the final value in thresholds is np.inf, which will never be
# the min value.  So, I am safe in "deleting" an index by
# just shifting the array over on top of it
def remove(s,i):
    '''
    Quick trick to remove an item from a numpy array without
    creating a new object.  Rather than the array shape changing,
    the final value just gets repeated to fill the space.

    ~3.5x faster than numpy.delete
    '''
    s[i:-1]=s[i+1:]

def build_thresholds(pts,mode='3',debug=False):
    if mode=='3':
    	print('standard visval: using three points')
    elif mode=='4':
    	print('modified visval: using four points')
    else:
    	print('modified visval: using all points within the to-be-merged segments')

    '''compute the area value of each vertex, which one would
    use to mask an array of points for any threshold value.

    returns a numpy.array (length of pts)  of the areas.
    '''
    # real_areas: records the dominating area of all points, to be returned as results
    # area and i: records the dominating areas and indexes of remaining points during bottom-up elimination
    # through i we can know the adjacency of remaining points easily.
    # original source: https://github.com/Permafacture/Py-Visvalingam-Whyatt
    # original implementation eliminates the left point when both the left and right points are dominated by the last eliminated point.
    # right now after my revision: eliminates the point with the smaller updated area when both the left and right points are dominated by the last eliminated point.

    popOrder=[]

    nmax = len(pts)
    real_areas = triangle_areas_from_array(pts)

    #destructable copies 
    #ARG! areas=real_areas[:] doesn't make a copy!
    areas = np.copy(real_areas)
    i=list(range(nmax))
    
    #pick first point and set up for loop
    min_vert = argmin(areas)
    this_area = areas[min_vert]
    #  areas and i are modified for each point finished
    remove(areas,min_vert)   #faster
    #areas = np.delete(areas,min_vert) #slower
    pos=i.pop(min_vert)
    popOrder.append(pos)

    while this_area<np.inf: # when equals np.inf, means only endPoint left in i and the firstPoint is selected
       # this_area will only get larger
       # print(this_area) 
       '''min_vert was removed from areas and i.  Now,
       adjust the adjacent areas and remove the new 
       min_vert.

       Now that min_vert was filtered out, min_vert points 
       to the point after the deleted point.'''
       
       skip = False  #modified area may be the next minvert

       if debug:
         print('--pop ',pos,',EA=',this_area)
       
       if min_vert <= len(i)-2: # note that now i already pop out min_vert
         if mode=='3':
             right_area = triangle_area(pts[i[min_vert-1]],
                            pts[i[min_vert]],pts[i[min_vert+1]])
         elif mode=='4':
             # 注意：这里因为只是额外考虑最近淘汰的一个点，这个点刚好就是触发右边点要更新的点，所以他们之间的相对关系是固定如下的。
             points=np.array([pts[i[min_vert-1]],pts[pos],pts[i[min_vert]],pts[i[min_vert+1]]])
             points2=np.array([pts[i[min_vert-1]],pts[i[min_vert+1]]])
             right_area=total_areal_displacement(points,points2)
         else:
             points=pts[i[min_vert-1]:i[min_vert+1]+1]
             points2=np.array([pts[i[min_vert-1]],pts[i[min_vert+1]]])
             right_area=total_areal_displacement(points,points2)
         
         if debug:
             print('right=',right_area)
           
         right_area_real=right_area # not dominated by this_area

         right_idx = i[min_vert]
         if right_area < this_area: # TODO TMP
             #even if the point now has a smaller area,
             # it ultimately is not more significant than
             # the last point, which needs to be removed
             # first to justify removing this point.
             # Though this point is the next most significant
             right_area = this_area # dominated

             #min_vert refers to the point to the right of 
             # the previous min_vert, so we can leave it
             # unchanged if it is still the min_vert
             skip = min_vert

         #update both collections of areas
         real_areas[right_idx] = right_area
         areas[min_vert] = right_area
       
       if min_vert > 1:
         # left proprity
         #cant try/except because 0-1=-1 is a valid index
         if mode=='3':
             left_area = triangle_area(pts[i[min_vert-2]],
                           pts[i[min_vert-1]],pts[i[min_vert]])
         elif mode=='4':
             # 注意：这里因为只是额外考虑最近淘汰的一个点，这个点刚好就是触发左边点要更新的点，所以他们之间的相对关系是固定如下。
             points=np.array([pts[i[min_vert-2]],pts[i[min_vert-1]],pts[pos],pts[i[min_vert]]])
             points2=np.array([pts[i[min_vert-2]],pts[i[min_vert]]])
             left_area=total_areal_displacement(points,points2)
         else:
             points=pts[i[min_vert-2]:i[min_vert]+1]
             points2=np.array([pts[i[min_vert-2]],pts[i[min_vert]]])
             left_area=total_areal_displacement(points,points2)
         
         if debug:
             print('left=',left_area)
             
         if left_area < this_area: # TODO TMP
             #same justification as above
             if skip!=False: # means right point area is smaller than this_area, then compare left and right
                if left_area<=right_area_real: # otherwise keep skip right point
                    skip=min_vert-1
             else: # just left point area is smaller than this_area
                skip=min_vert-1

             left_area = this_area # dominated

         real_areas[i[min_vert-1]] = left_area
         areas[min_vert-1] = left_area

       #only argmin if we have too.
       min_vert = skip or argmin(areas)
       pos=i.pop(min_vert)
       popOrder.append(pos)
       this_area = areas[min_vert]
       #areas = np.delete(areas,min_vert) #slower
       remove(areas,min_vert)  #faster
       '''if sum(np.where(areas==np.inf)[0]) != sum(list(reversed(range(len(areas))))[:cntr]):
         print "broke:",np.where(areas==np.inf)[0],cntr
         break
       cntr+=1
       #if real_areas[0]<np.inf or real_areas[-1]<np.inf:
       #  print "NO!", real_areas[0], real_areas[-1]
       '''
    popOrder.append(len(pts)-1)
    return real_areas,popOrder

def find_neighbors(points, popOrder, t):
    """
    找出在删除第 t 个点时，左边和右边最近的两个点。
    """
    # 找左边最近的两个点
    left_neighbors = []
    # 找右边最近的两个点
    right_neighbors = []

    idx=popOrder[t]
    for i in range(idx-1,-1,-1):
        if i not in set(popOrder[:t]):
            left_neighbors.append(i)
        if len(left_neighbors)>=2:
            break
    for i in range(idx+1,len(points)):
        if i not in set(popOrder[:t]):
            right_neighbors.append(i)
        if len(right_neighbors)>=2:
            break
    
    return left_neighbors[::-1], right_neighbors

def comparePopOrder(points, popOrder, target, mode='3', saveFig="test",debug=False):
    """
    points: 二维坐标点的数组
    popOrder: 淘汰顺序的索引列表
    target: 要淘汰的点的数量（可以是一个整数或列表）比如compareVisval(points,popOrder,[15,16,17,18])
    """
    # 如果 target 不是列表，转为列表
    if not isinstance(target, list):
        target = [target]
    
    num_subplots = len(target)
    cols = 3  # 每行最多显示的子图数量
    rows = -(-num_subplots // cols)  # 向上取整，计算行数

    plt.figure(figsize=(4 * cols, 3 * rows))

    # plt.suptitle(saveFig, fontsize=16, y=1.02)
    
    for i, t in enumerate(target):
        # 创建子图
        plt.subplot(rows, cols, i + 1)
        plt.title(f"{saveFig}:Remaining after {t} eliminations")
        
        # 原始点及连线
        plt.plot(points[:, 0], points[:, 1], 'o-', label="data",linewidth=1)

        if len(points)<50:
            # 标出全局淘汰顺序
            # cnt = 1
            # for j in popOrder:
            #     plt.text(points[j, 0], points[j, 1], str(cnt), fontsize=8)
            #     cnt += 1

            # 标出点编号
            for j in np.arange(len(points)):
                plt.text(points[j, 0], points[j, 1], str(j+1), fontsize=8)
            
        # 删除指定数量的点后剩余点
        idx = popOrder[:t]
        remaining_points = np.delete(points, idx, axis=0)
        x_coords = remaining_points[:, 0]
        y_coords = remaining_points[:, 1]
        plt.plot(x_coords, y_coords, 'o-', label="remaining",linewidth=2)

        plt.scatter(points[popOrder[t-1]][0], points[popOrder[t-1]][1],\
                    marker='x', color='k', s=100)  # s=100 控制点的大小

        left_neighbors, right_neighbors = find_neighbors(points, popOrder, t-1)
        if debug:
            print('--pop ',popOrder[t-1],',left ',left_neighbors,',right ',right_neighbors)
        if len(right_neighbors)>1:
            if mode=='3':
                tmp_points=np.array([points[left_neighbors[-1]],\
                             points[right_neighbors[0]],points[right_neighbors[1]]])
            elif mode=='4':
                tmp_points=np.array([points[left_neighbors[-1]],points[popOrder[t-1]],\
                                 points[right_neighbors[0]],points[right_neighbors[1]]])
            else:
                tmp_points=points[left_neighbors[-1]:right_neighbors[1]+1]
            tmp_points2=np.array([points[left_neighbors[-1]],points[right_neighbors[1]]])
            right_area=total_areal_displacement(tmp_points,tmp_points2)
            polygon = []
            polygon.extend(tmp_points)
            polygon.extend(tmp_points2[::-1])
            polygon = np.array(polygon)
            plt.fill(polygon[:, 0], polygon[:, 1], alpha=0.3)
            plt.scatter(points[right_neighbors[0]][0], points[right_neighbors[0]][1],\
                        color='none', edgecolor='blue', s=100)
            if debug:
                print('right=',right_area)
        if len(left_neighbors)>1:
            if mode=='3':
                tmp_points=np.array([points[left_neighbors[-2]],points[left_neighbors[-1]],\
                             points[right_neighbors[0]]])
            elif mode=='4':
                tmp_points=np.array([points[left_neighbors[-2]],points[left_neighbors[-1]],\
                                 points[popOrder[t-1]],points[right_neighbors[0]]])
            else:
                tmp_points=points[left_neighbors[-2]:right_neighbors[0]+1]
            tmp_points2=np.array([points[left_neighbors[-2]],points[right_neighbors[0]]])
            left_area=total_areal_displacement(tmp_points,tmp_points2)
            polygon = []
            polygon.extend(tmp_points)
            polygon.extend(tmp_points2[::-1])
            polygon = np.array(polygon)
            plt.fill(polygon[:, 0], polygon[:, 1], alpha=0.3,color='r')
            plt.scatter(points[left_neighbors[-1]][0], points[left_neighbors[-1]][1],\
                        color='none', edgecolor='r', s=100)
            if debug:
                print('left=',left_area)
            
        plt.legend()
    

    plt.tight_layout()
    plt.savefig(f"{saveFig}.png")
    plt.savefig(f"{saveFig}.eps")
    plt.show()
    print("figure saved to ",f"{saveFig}.png")


class VWSimplifier(object):

    def __init__(self,pts):
        '''Initialize with points. takes some time to build 
        the thresholds but then all threshold filtering later 
        is ultra fast'''
        self.pts = np.array(pts)
        self.thresholds = self.build_thresholds()
        self.ordered_thresholds = sorted(self.thresholds,reverse=True)

    def build_thresholds(self):
        '''compute the area value of each vertex, which one would
        use to mask an array of points for any threshold value.

        returns a numpy.array (length of pts)  of the areas.
        '''
        # real_areas: records the dominating area of all points, to be returned as results
        # area and i: records the dominating areas and indexes of remaining points during bottom-up elimination
        # through i we can know the adjacency of remaining points easily.
        # original source: https://github.com/Permafacture/Py-Visvalingam-Whyatt
        # original implementation eliminates the left point when both the left and right points are dominated by the last eliminated point.
        # right now after my revision: eliminates the point with the smaller updated area when both the left and right points are dominated by the last eliminated point.

        pts = self.pts
        nmax = len(pts)
        real_areas = triangle_areas_from_array(pts)

        #destructable copies 
        #ARG! areas=real_areas[:] doesn't make a copy!
        areas = np.copy(real_areas)
        i=list(range(nmax))
        
        #pick first point and set up for loop
        min_vert = argmin(areas)
        this_area = areas[min_vert]
        #  areas and i are modified for each point finished
        remove(areas,min_vert)   #faster
        #areas = np.delete(areas,min_vert) #slower
        i.pop(min_vert)

        while this_area<np.inf: # when equals np.inf, means only endPoint left in i and the firstPoint is selected
           # this_area will only get larger
           # print(this_area) 
           '''min_vert was removed from areas and i.  Now,
           adjust the adjacent areas and remove the new 
           min_vert.

           Now that min_vert was filtered out, min_vert points 
           to the point after the deleted point.'''
           
           skip = False  #modified area may be the next minvert
           
           if min_vert <= len(i)-2: # note that now i already pop out min_vert
             right_area = triangle_area(pts[i[min_vert-1]],
                            pts[i[min_vert]],pts[i[min_vert+1]])
             right_area_real=right_area # not dominated by this_area

             right_idx = i[min_vert]
             if right_area <= this_area:
                 #even if the point now has a smaller area,
                 # it ultimately is not more significant than
                 # the last point, which needs to be removed
                 # first to justify removing this point.
                 # Though this point is the next most significant
                 right_area = this_area # dominated

                 #min_vert refers to the point to the right of 
                 # the previous min_vert, so we can leave it
                 # unchanged if it is still the min_vert
                 skip = min_vert

             #update both collections of areas
             real_areas[right_idx] = right_area
             areas[min_vert] = right_area
           
           if min_vert > 1:
             # left proprity
             #cant try/except because 0-1=-1 is a valid index
             left_area = triangle_area(pts[i[min_vert-2]],
                           pts[i[min_vert-1]],pts[i[min_vert]])
             if left_area <= this_area:
                 #same justification as above
                 if skip!=False: # means right point area is smaller than this_area, then compare left and right
                    if left_area<=right_area_real: # otherwise keep skip right point
                        skip=min_vert-1
                 else: # just left point area is smaller than this_area
                    skip=min_vert-1

                 left_area = this_area # dominated

             real_areas[i[min_vert-1]] = left_area
             areas[min_vert-1] = left_area

           #only argmin if we have too.
           min_vert = skip or argmin(areas)
           i.pop(min_vert)
           this_area = areas[min_vert]
           #areas = np.delete(areas,min_vert) #slower
           remove(areas,min_vert)  #faster
           '''if sum(np.where(areas==np.inf)[0]) != sum(list(reversed(range(len(areas))))[:cntr]):
             print "broke:",np.where(areas==np.inf)[0],cntr
             break
           cntr+=1
           #if real_areas[0]<np.inf or real_areas[-1]<np.inf:
           #  print "NO!", real_areas[0], real_areas[-1]
           '''
        return real_areas

    def from_threshold(self,threshold):
        return self.pts[self.thresholds >= threshold]

    def from_number(self,n):
        thresholds = self.ordered_thresholds
        try:
          threshold = thresholds[int(n)]
        except IndexError:
          return self.pts
        return self.pts[self.thresholds > threshold]

    def from_ratio(self,r):
        if r<=0 or r>1:
          raise ValueError("Ratio must be 0<r<=1")
        else:
          return self.from_number(r*len(self.thresholds))


def fancy_parametric(k):
    ''' good k's: .33,.5,.65,.7,1.3,1.4,1.9,3,4,5'''
    cos = np.cos
    sin = np.sin
    xt = lambda t: (k-1)*cos(t) + cos(t*(k-1))
    yt = lambda t: (k-1)*sin(t) - sin(t*(k-1))
    return xt,yt

if __name__ == "__main__":

   from time import time
   n = 5000
   thetas = np.linspace(0,16*np.pi,n)
   xt,yt = fancy_parametric(1.4)  
   pts = np.array([[xt(t),yt(t)] for t in thetas])
   start = time()
   simplifier = VWSimplifier(pts)
   pts = simplifier.from_number(1000)
   end = time()
   # print "%s vertices removed in %02f seconds"%(n-len(pts), end-start)
   
   import matplotlib
   matplotlib.use('AGG')
   import matplotlib.pyplot as plot
   plot.plot(pts[:,0],pts[:,1],color='r')
   plot.savefig('visvalingam.png')
   # print "saved visvalingam.png"
   #plot.show()