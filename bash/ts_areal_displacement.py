import numpy as np
import plotly.graph_objects as go; 

def calculate_polygon_area(points):
    """
    使用鞋带公式计算多边形面积。

    参数:
        points: numpy 数组，形状为 (n, 2)，表示多边形顶点 [[x1, y1], [x2, y2], ...]

    返回:
        面积的绝对值。
    """
    
    # 将输入转换为高精度浮点数
    points = points.astype(np.float64)

    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def cross_product(x1, y1, x2, y2):
    # >0: (x2,y2)在(x1,y1)的逆时针方向
    # <0: (x2,y2)在(x1,y1)的顺时针方向
    # =0: 平行或共线
    return x1 * y2 - y1 * x2
    
def line_intersection(L1, L2):
    """
    判断两条线段是否相交并计算交点。

    参数:
        L1: [(x1, y1), (x2, y2)]，第一条线段的两个端点
        L2: [(x3, y3), (x4, y4)]，第二条线段的两个端点

    返回:
        (is_intersect, intersection_point):
        - is_intersect: 布尔值，是否相交
        - intersection_point: 如果相交，返回交点 [x, y]；否则返回 None
    """
    x1, y1 = L1[0]
    x2, y2 = L1[1]
    x3, y3 = L2[0]
    x4, y4 = L2[1]

    # 判断是否相交（检查是否存在方向交替）
    d1 = cross_product(x2 - x1, y2 - y1, x3 - x1, y3 - y1)
    d2 = cross_product(x2 - x1, y2 - y1, x4 - x1, y4 - y1)
    d3 = cross_product(x4 - x3, y4 - y3, x1 - x3, y1 - y3)
    d4 = cross_product(x4 - x3, y4 - y3, x2 - x3, y2 - y3)

    # 判断两个线段是否有交点
    # d1*d2<0意味着P3、P4分别在L12的两边
    # d3*d4<0意味着P1、P2分别在L34的两边
    # 两个同时满足说明有交点
    if d1 * d2 < 0 and d3 * d4 < 0:
        # 计算交点
        denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1) # 不可能为0（平行或共线），因为已经判断相交了
        t1 = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
        x = x1 + t1 * (x2 - x1)
        y = y1 + t1 * (y2 - y1)
        
        # print(f"交点: ({x}, {y})")  # 打印交点
        return True, [x, y]
    
    # 检查是否起点或终点重合
    if (x1, y1) == (x3, y3) or (x1, y1) == (x4, y4):
        return True, [x1, y1]
    if (x2, y2) == (x3, y3) or (x2, y2) == (x4, y4):
        return True, [x2, y2]

    
    return False, None


def total_areal_displacement(points,points2,debug=False,plot=False,ax=None,lw=2.5,slw=1,ms=6,\
    hatch='/',facecolor='none',edgecolor='black',plotSampled=True,originalcolor="tab:blue",sampleColor="tab:orange"):
    # 假设时间戳严格递增
    """
    同时扫描两个时间序列，找到交点并计算围成多边形的总面积。

    参数:
        points: numpy 数组，形状为 (n, 2)，表示时间序列 1 [[t1, y1], [t2, y2], ...]
        points2: numpy 数组，形状为 (m, 2)，表示时间序列 2 [[t1, y1], [t2, y2], ...]

    返回:
        总面积。
    """
    
    total_area = 0
    i, j = 0, 0
    prev_intersection = None
    prev_i = None
    prev_j = None

    intersection_coords=[] # 为了画图的时候标记
    area_list=[] # 为了画图的时候标记

    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='lines',
            name='original',
            line=dict(color='blue')
        ))

    if ax is not None:
        ax.plot(points[:, 0],points[:, 1],'o-',color=originalcolor,label="original",lw=slw,markersize=ms)


    while i < len(points) - 1 and j < len(points2) - 1:
        if debug:
            print('---------',i,j,'------------')
        
        # 当前线段
        L1 = [points[i], points[i + 1]]
        L2 = [points2[j], points2[j + 1]]

        # 判断是否有交点
        is_intersect, intersection = line_intersection(L1, L2)
        # print(is_intersect, intersection)

        if is_intersect:
            intersection_coords.append(intersection)
                
            if prev_intersection is not None:
                # 构造多边形点集
                polygon = [prev_intersection]
                if debug:
                    print('- start intersection:', prev_intersection[0])
                polygon.extend(points[prev_i : i+1])  # 添加当前线段的点
                if debug:
                    print('- one side:',[x[0] for x in points[prev_i : i+1]])
                polygon.append(intersection)
                if debug:
                    print('- end intersection:',intersection[0])
                polygon.extend(points2[prev_j : j+1][::-1])  # 添加另一序列的点
                if debug:
                    print('- another side:',[x[0] for x in points2[prev_j : j+1][::-1]])
                polygon = np.array(polygon)

                area=calculate_polygon_area(polygon)
                if debug:
                    print('area=',area)
                # 计算多边形面积并累加
                total_area += area
                area_list.append(area)
                
                if plot:
                    # 添加封闭的多边形（区域填充）
                    fig.add_trace(go.Scatter(
                        x=polygon[:, 0],
                        y=polygon[:, 1],
                        fill='toself',
                        name='enclosed polygons',
                        opacity=0.5,
                        line=dict(width=0),
                        fillcolor='rgba(100, 150, 200, 0.3)',
                        showlegend=False  # 不显示在图例中
                    ))
                if ax is not None:
                    if plotSampled:
                        # '/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
                        ax.fill(polygon[:, 0],polygon[:, 1],hatch=hatch,facecolor=facecolor,edgecolor=edgecolor)
                    

            prev_intersection = intersection
            prev_i = i+1
            prev_j = j+1
            if debug:
                print('this intersection=',intersection,', next polygon:', 'side1=',prev_i,',side2=',prev_j)
        
        current_i = i  # 临时存储i
        current_j = j  # 临时存储j
        if points[current_i + 1][0] <= points2[current_j + 1][0]:
            i += 1 # 基于时间戳严格递增的假设，Line不会回头或者垂直
        if points[current_i + 1][0] >= points2[current_j + 1][0]: # 注意不要用更新之后的i
            j += 1 # 基于时间戳严格递增的假设，Line不会回头或者垂直

    if debug:
        print(area_list)
        
    if plot:
        fig.add_trace(go.Scatter(
            x=points2[:, 0],
            y=points2[:, 1],
            mode='lines+markers',
            name='sampled',
            line=dict(color='red')
        ))

        fig.add_trace(go.Scatter(
            x=[coord[0] for coord in intersection_coords],
            y=[coord[1] for coord in intersection_coords],
            mode='markers+text',
            name='intersections',
            marker=dict(color='green', size=8, symbol='circle-open'),
            # text=[f"{round(area, 2)}" for area in area_list] if len(area_list) < 1000 else None,
            # text=[f"{x},{round(area_list[x], 4)}" for x in np.arange(len(area_list))] if len(area_list) < 1000 else None,
            # 注意area_list个数等于交点intersection_coord个数减一
            # 如果面积为0就不标注在图里了，避免太乱
            text=[f"{round(area_list[x], 2)}" if area_list[x]!=0 else None for x in np.arange(len(area_list))],
            textposition='bottom right',
            textfont=dict(color='blue', size=15)
        ))

        # if len(points) < 50:
        #     fig.add_trace(go.Scatter(
        #         x=points[:, 0],
        #         y=points[:, 1],
        #         mode='text',
        #         text=[str(i) for i in range(len(points))],
        #         textposition='top center',
        #         textfont=dict(size=8, color='green'),
        #         showlegend=False  # 不显示在图例中
        #     ))


        # 图例、布局调整
        fig.update_layout(
            title="Time Series and Enclosed Areas",
            xaxis_title="Time",
            yaxis_title="Value",
            legend=dict(
                x=1,
                y=0.5,
                xanchor="left",
                yanchor="middle",
                orientation="v"
            ),
            margin=dict(r=150),  # 为图例留出足够空间
            template="plotly_white", 
            width=900,
            height=600
        )
        fig.show()
        
    if ax is not None:
        if plotSampled:
            # 绘制 "sampled" 数据
            ax.plot(points2[:, 0], points2[:, 1], 'o-', label="sampled", color=sampleColor,lw=lw,markersize=ms,\
                    markeredgecolor='black')
        
        # # 绘制 "intersections" 数据
        # # 由于绘制交点，添加坐标值作为文本
        # for i, coord in enumerate(intersection_coords):
        #     ax.scatter(coord[0], coord[1], color='green', s=50, edgecolor='black', marker='o')  # 绘制交点

        
        # # 如果 area_list 的长度小于1000，添加文本
        # if len(area_list) < 1000:
        #     for i in range(len(area_list)):
        #         if area_list[i]>0:
        #             coord=intersection_coords[i]
        #             ax.text(coord[0], coord[1], f"{round(area_list[i], 2)}", color='black',\
        #                     fontsize=10, ha='left', va='bottom')
        
        # # 如果点数小于50，显示点的编号
        # if len(points) < 50:
        #     for i, point in enumerate(points):
        #         ax.text(point[0], point[1], str(i), color='green', fontsize=8, ha='center', va='top')
        
        # 设置标题和坐标轴标签
        # ax.set_title("Time Series and Enclosed Areas")
        # ax.set_xlabel("time")
        # ax.set_ylabel("value")
        
        # 设置图例
        # ax.legend(fontsize=10)
        

    return total_area


# def total_areal_displacement(points,points2,debug=False,plot=False,ax=None):
#     # 假设时间戳严格递增
#     """
#     同时扫描两个时间序列，找到交点并计算围成多边形的总面积。

#     参数:
#         points: numpy 数组，形状为 (n, 2)，表示时间序列 1 [[t1, y1], [t2, y2], ...]
#         points2: numpy 数组，形状为 (m, 2)，表示时间序列 2 [[t1, y1], [t2, y2], ...]

#     返回:
#         总面积。
#     """
    
#     total_area = 0
#     i, j = 0, 0
#     prev_intersection = None
#     prev_i = None
#     prev_j = None

#     intersection_coords=[] # 为了画图的时候标记
#     area_list=[] # 为了画图的时候标记

#     if plot:
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=points[:, 0],
#             y=points[:, 1],
#             mode='lines',
#             name='original',
#             line=dict(color='blue')
#         ))

#     if ax is not None:
#         ax.plot(points[:, 0],points[:, 1],'o-',color='blue',label="original")


#     while i < len(points) - 1 and j < len(points2) - 1:
#         if debug:
#             print('---------',i,j,'------------')
        
#         # 当前线段
#         L1 = [points[i], points[i + 1]]
#         L2 = [points2[j], points2[j + 1]]

#         # 判断是否有交点
#         is_intersect, intersection = line_intersection(L1, L2)
#         # print(is_intersect, intersection)

#         if is_intersect:
#             intersection_coords.append(intersection)
                
#             if prev_intersection is not None:
#                 # 构造多边形点集
#                 polygon = [prev_intersection]
#                 if debug:
#                     print('- start intersection:', prev_intersection[0])
#                 polygon.extend(points[prev_i : i+1])  # 添加当前线段的点
#                 if debug:
#                     print('- one side:',[x[0] for x in points[prev_i : i+1]])
#                 polygon.append(intersection)
#                 if debug:
#                     print('- end intersection:',intersection[0])
#                 polygon.extend(points2[prev_j : j+1][::-1])  # 添加另一序列的点
#                 if debug:
#                     print('- another side:',[x[0] for x in points2[prev_j : j+1][::-1]])
#                 polygon = np.array(polygon)

#                 area=calculate_polygon_area(polygon)
#                 if debug:
#                     print('area=',area)
#                 # 计算多边形面积并累加
#                 total_area += area
#                 area_list.append(area)
                
#                 if plot:
#                     # 添加封闭的多边形（区域填充）
#                     fig.add_trace(go.Scatter(
#                         x=polygon[:, 0],
#                         y=polygon[:, 1],
#                         fill='toself',
#                         name='enclosed polygons',
#                         opacity=0.5,
#                         line=dict(width=0),
#                         fillcolor='rgba(100, 150, 200, 0.3)',
#                         showlegend=False  # 不显示在图例中
#                     ))
#                 if ax is not None:
#                     ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.3) # label="enclosed polygons"
                    

#             prev_intersection = intersection
#             prev_i = i+1
#             prev_j = j+1
#             if debug:
#                 print('this intersection=',intersection,', next polygon:', 'side1=',prev_i,',side2=',prev_j)
        
#         current_i = i  # 临时存储i
#         current_j = j  # 临时存储j
#         if points[current_i + 1][0] <= points2[current_j + 1][0]:
#             i += 1 # 基于时间戳严格递增的假设，Line不会回头或者垂直
#         if points[current_i + 1][0] >= points2[current_j + 1][0]: # 注意不要用更新之后的i
#             j += 1 # 基于时间戳严格递增的假设，Line不会回头或者垂直

#     if debug:
#         print(area_list)
        
#     if plot:
#         fig.add_trace(go.Scatter(
#             x=points2[:, 0],
#             y=points2[:, 1],
#             mode='lines+markers',
#             name='sampled',
#             line=dict(color='red')
#         ))

#         fig.add_trace(go.Scatter(
#             x=[coord[0] for coord in intersection_coords],
#             y=[coord[1] for coord in intersection_coords],
#             mode='markers+text',
#             name='intersections',
#             marker=dict(color='green', size=8, symbol='circle-open'),
#             # text=[f"{round(area, 2)}" for area in area_list] if len(area_list) < 1000 else None,
#             # text=[f"{x},{round(area_list[x], 4)}" for x in np.arange(len(area_list))] if len(area_list) < 1000 else None,
#             # 注意area_list个数等于交点intersection_coord个数减一
#             # 如果面积为0就不标注在图里了，避免太乱
#             text=[f"{round(area_list[x], 2)}" if area_list[x]!=0 else None for x in np.arange(len(area_list))],
#             textposition='bottom right',
#             textfont=dict(color='blue', size=15)
#         ))

#         # if len(points) < 50:
#         #     fig.add_trace(go.Scatter(
#         #         x=points[:, 0],
#         #         y=points[:, 1],
#         #         mode='text',
#         #         text=[str(i) for i in range(len(points))],
#         #         textposition='top center',
#         #         textfont=dict(size=8, color='green'),
#         #         showlegend=False  # 不显示在图例中
#         #     ))


#         # 图例、布局调整
#         fig.update_layout(
#             title="Time Series and Enclosed Areas",
#             xaxis_title="Time",
#             yaxis_title="Value",
#             legend=dict(
#                 x=1,
#                 y=0.5,
#                 xanchor="left",
#                 yanchor="middle",
#                 orientation="v"
#             ),
#             margin=dict(r=150),  # 为图例留出足够空间
#             template="plotly_white", 
#             width=900,
#             height=600
#         )
#         fig.show()
        
#     if ax is not None:
#         # 绘制 "sampled" 数据
#         ax.plot(points2[:, 0], points2[:, 1], 'o-', label="sampled", color='red')
        
#         # # 绘制 "intersections" 数据
#         # # 由于绘制交点，添加坐标值作为文本
#         # for i, coord in enumerate(intersection_coords):
#         #     ax.scatter(coord[0], coord[1], color='green', s=50, edgecolor='black', marker='o')  # 绘制交点

        
#         # # 如果 area_list 的长度小于1000，添加文本
#         # if len(area_list) < 1000:
#         #     for i in range(len(area_list)):
#         #         if area_list[i]>0:
#         #             coord=intersection_coords[i]
#         #             ax.text(coord[0], coord[1], f"{round(area_list[i], 2)}", color='black',\
#         #                     fontsize=10, ha='left', va='bottom')
        
#         # # 如果点数小于50，显示点的编号
#         # if len(points) < 50:
#         #     for i, point in enumerate(points):
#         #         ax.text(point[0], point[1], str(i), color='green', fontsize=8, ha='center', va='top')
        
#         # 设置标题和坐标轴标签
#         # ax.set_title("Time Series and Enclosed Areas")
#         # ax.set_xlabel("time")
#         # ax.set_ylabel("value")
        
#         # 设置图例
#         # ax.legend(fontsize=10)
        

#     return total_area

# def total_areal_displacement(points,points2,debug=False,plot=False):
#     # 假设时间戳严格递增
#     """
#     同时扫描两个时间序列，找到交点并计算围成多边形的总面积。

#     参数:
#         points: numpy 数组，形状为 (n, 2)，表示时间序列 1 [[t1, y1], [t2, y2], ...]
#         points2: numpy 数组，形状为 (m, 2)，表示时间序列 2 [[t1, y1], [t2, y2], ...]

#     返回:
#         总面积。
#     """
    
#     total_area = 0
#     i, j = 0, 0
#     prev_intersection = None
#     prev_i = None
#     prev_j = None

#     if plot:
#         intersection_coords=[] # 为了画图的时候标记
#         area_list=[] # 为了画图的时候标记

#         # plt.figure(figsize=(10, 6))
#         # plt.plot(points[:, 0], points[:, 1], label="original", color="blue")
#         # plt.plot(points2[:, 0], points2[:, 1], label="sampled", color="orange")

#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=points[:, 0],
#             y=points[:, 1],
#             mode='lines',
#             name='original',
#             line=dict(color='blue')
#         ))


#     while i < len(points) - 1 and j < len(points2) - 1:
#         if debug:
#             print('---------',i,j,'------------')
        
#         # 当前线段
#         L1 = [points[i], points[i + 1]]
#         L2 = [points2[j], points2[j + 1]]

#         # 判断是否有交点
#         is_intersect, intersection = line_intersection(L1, L2)
#         # print(is_intersect, intersection)

#         if is_intersect:
#             if plot:
#                 intersection_coords.append(intersection)
                
#             if prev_intersection is not None:
#                 # 构造多边形点集
#                 polygon = [prev_intersection]
#                 if debug:
#                     print('- start intersection:', prev_intersection[0])
#                 polygon.extend(points[prev_i : i+1])  # 添加当前线段的点
#                 if debug:
#                     print('- one side:',[x[0] for x in points[prev_i : i+1]])
#                 polygon.append(intersection)
#                 if debug:
#                     print('- end intersection:',intersection[0])
#                 polygon.extend(points2[prev_j : j+1][::-1])  # 添加另一序列的点
#                 if debug:
#                     print('- another side:',[x[0] for x in points2[prev_j : j+1][::-1]])
#                 polygon = np.array(polygon)

#                 area=calculate_polygon_area(polygon)
#                 if debug:
#                     print('area=',area)

#                 # 计算多边形面积并累加
#                 total_area += area

#                 if plot:
#                     area_list.append(area)
#                     # plt.fill(polygon[:, 0], polygon[:, 1], alpha=0.3) # label="enclosed polygons"
#                     # 添加封闭的多边形（区域填充）
#                     fig.add_trace(go.Scatter(
#                         x=polygon[:, 0],
#                         y=polygon[:, 1],
#                         fill='toself',
#                         name='enclosed polygons',
#                         opacity=0.5,
#                         line=dict(width=0),
#                         fillcolor='rgba(100, 150, 200, 0.3)',
#                         showlegend=False  # 不显示在图例中
#                     ))

#             prev_intersection = intersection
#             prev_i = i+1
#             prev_j = j+1
#             if debug:
#                 print('this intersection=',intersection,', next polygon:', 'side1=',prev_i,',side2=',prev_j)
        
#         current_i = i  # 临时存储i
#         current_j = j  # 临时存储j
#         if points[current_i + 1][0] <= points2[current_j + 1][0]:
#             i += 1 # 基于时间戳严格递增的假设，Line不会回头或者垂直
#         if points[current_i + 1][0] >= points2[current_j + 1][0]: # 注意不要用更新之后的i
#             j += 1 # 基于时间戳严格递增的假设，Line不会回头或者垂直

#         # # check 下面有bug，当连续两段是重叠的，会造成第二段的交点跳过计算直接复用第一段的交点了，所以还是不优化了，计算面积0就计算
#         # # 为了避免无意义的衔接存在点交点面积0的计算，下面手动推进执行一个循环，手动略过计算相交多边形的逻辑
#         # if np.array_equal(points[i],points2[j]): # proceed one more time
#         #     if debug:
#         #         print('------',i,j,'-------')
#         #     if i < len(points) - 1 and j < len(points2) - 1: #由于这里是手动推进执行一个循环所以这里要手动处理
#         #         # 相当于执行上述循环里的步骤，除了判断是否相交的部分，因为这里上一个多边形的末尾交点就是下一个多边形的开头交点
#         #         # 没必要重复计算这个0面积的多边形
#         #         # prev_intersection不动，还是这个交点
#         #         prev_i = i+1
#         #         prev_j = j+1
#         #         if debug:
#         #             print('[jump]this intersection=',intersection,', next polygon:','side1=',prev_i,',side2=',prev_j)
                
#         #         current_i = i  # 临时存储i
#         #         current_j = j  # 临时存储j
#         #         if points[current_i + 1][0] <= points2[current_j + 1][0]:
#         #             i += 1
#         #         if points[current_i + 1][0] >= points2[current_j + 1][0]: # 注意不要用更新之后的i
#         #             j += 1
#         #     else: #由于这里是手动提前执行一个循环所以这里要手动处理
#         #         break

#     if plot:
#         # plt.scatter(*zip(*intersection_coords), color="red", label="intersections")
#         # # if len(points)<50:
#         # #     for i in range(len(points)):
#         # #         plt.text(points[i, 0], points[i, 1], str(i), fontsize=8)
#         # if len(area_list)<50:
#         #     for i in range(len(area_list)):
#         #         plt.text(intersection_coords[i][0], intersection_coords[i][1], \
#         #                  round(area_list[i], 2), fontsize=8,color='blue')
#         # print(len(area_list))
#         # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#         # plt.show()

#         fig.add_trace(go.Scatter(
#             x=points2[:, 0],
#             y=points2[:, 1],
#             mode='lines+markers',
#             name='sampled',
#             line=dict(color='red')
#         ))

#         fig.add_trace(go.Scatter(
#             x=[coord[0] for coord in intersection_coords],
#             y=[coord[1] for coord in intersection_coords],
#             mode='markers+text',
#             name='intersections',
#             marker=dict(color='green', size=8, symbol='circle-open'),
#             # text=[f"{round(area, 2)}" for area in area_list] if len(area_list) < 1000 else None,
#             text=[f"{x},{round(area_list[x], 4)}" for x in np.arange(len(area_list))] if len(area_list) < 1000 else None,
#             textposition='top center',
#             textfont=dict(color='blue', size=10)
#         ))

#         if debug:
#             print(area_list)

#         if len(points) < 50:
#             fig.add_trace(go.Scatter(
#                 x=points[:, 0],
#                 y=points[:, 1],
#                 mode='text',
#                 text=[str(i) for i in range(len(points))],
#                 textposition='top center',
#                 textfont=dict(size=8, color='green'),
#                 showlegend=False  # 不显示在图例中
#             ))


#         # 图例、布局调整
#         fig.update_layout(
#             title="Time Series and Enclosed Areas",
#             xaxis_title="Time",
#             yaxis_title="Value",
#             legend=dict(
#                 x=1,
#                 y=0.5,
#                 xanchor="left",
#                 yanchor="middle",
#                 orientation="v"
#             ),
#             margin=dict(r=150),  # 为图例留出足够空间
#             template="plotly_white", 
#             width=900,
#             height=600
#         )
#         fig.show()

#     return total_area

# # 测试示例
# # points = np.array([[0, 0], [1, 1], [2, 0]])
# # points2 = np.array([[0, 1], [1, 0], [2, 1]])
# result,areas = total_areal_displacement(points, points2,plot=True)
# print(f"总面积: {result}")