# This is the tool for plotting the critical difference diagram with Wilcoxon-Holm test,
# adapted from https://github.com/hfawaz/cd-diagram.
# 
# About https://github.com/hfawaz/cd-diagram:
#   critical difference diagram tool
#   Author: Hassan Ismail Fawaz <hassan.ismail-fawaz@uha.fr>
#         Germain Forestier <germain.forestier@uha.fr>
#         Jonathan Weber <jonathan.weber@uha.fr>
#         Lhassane Idoumghar <lhassane.idoumghar@uha.fr>
#         Pierre-Alain Muller <pierre-alain.muller@uha.fr>
#   License: GPL3

import numpy as np
import pandas as pd
import matplotlib

# matplotlib.use('agg')
import matplotlib.pyplot as plt

# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = 'Arial'

import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx

# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=3, textspace=1, reverse=False, filename=None, labels=False, colors=None, 
                **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    # print(avranks)
        
    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # space_between_names = 0.24
    space_between_names = 0.55

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    # height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant
    height = cline + ((k + 1) / 2) * space_between_names + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    # 画点之间的连线，除了可以画一条线段（即2个点之间的），还可以画折线（即3个点之间的）
    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def lines(points, colorsInClique, color='k', marker='x', markersize=3, **kwargs):
        """
        Input: A list of points [(x1, y1), (x2, y2), ...] sorted by x.
        Draws the line connecting the points and marks the points with circles.
        """
        # Extract x and y coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_coords_wfl=wfl(x_coords)
        y_coords_hfl=hfl(y_coords)

        # Plot the line
        ax.plot(x_coords_wfl, y_coords_hfl, color=color, **kwargs, zorder=1000)
        
        # Plot the points
        for i in range(len(x_coords)):
            ax.scatter(x_coords_wfl[i], y_coords_hfl[i], color=colorsInClique[i], edgecolor='black', marker=marker, s=markersize, **kwargs, zorder=1000)


    def text(x, y, s, size=38, *args, **kwargs):
        ax.text(wf * x, hf * y, s, size=size, *args, **kwargs)


    # 画主轴上的ticks
    bigtick = 0.3
    smalltick = 0.15
    linewidth = 3.0
    linewidth_sign = 4.0

    # 画主轴
    line([(textspace, cline), (width - textspace, cline)], linewidth=linewidth)

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=2)
    
    # 标注主轴上主要ticks的数字
    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom")

    k = len(ssums)

    def filter_names(name):
        return name

    # 画左半部分的折线及其标注
    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth, color=colors.get(filter_names(nnames[i]), 'k') if colors is not None else 'k')
        if labels:
            # 把具体的平均rank标注出来
            text(textspace + 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="right", va="center")
        if "eBUG" in nnames[i]:
            text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center",weight='bold')
        else:
            text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center")

    # 画右半部分的折线及其标注
    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth, color=colors.get(filter_names(nnames[i]), 'k') if colors is not None else 'k')
        if labels:
            # 把具体的平均rank标注出来
            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="left", va="center")
        if "eBUG" in nnames[i]:
            text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
                ha="left", va="center",weight='bold')
        else:
            text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
                ha="left", va="center")
    
    start = cline + 0.2 # 第一条insignificant line的高度
    side = 0 # 画出头的长度
    height = 0.3 # insignificant lines之间的高度间隔

    # 画连线表示这些方法之间没有显著区别
    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    max_idx=None
    for clq in cliques:
        if len(clq) == 1:
            continue
        print('clq:',clq)

        if max_idx is not None and np.array(clq).min()>max_idx: # non-overlapping
            start-=height

        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            # 代表到了右半部分的画先重置一下高度，而不是在之前的高度基础上
            start = cline + 0.2
            achieved_half = True

        # line([(rankpos(ssums[min_idx]) - side, start),
        #       (rankpos(ssums[max_idx]) + side, start)],
        #      linewidth=linewidth_sign)

        points=[]
        colorsInClique=[]
        for idx in clq:
            points.append([rankpos(ssums[idx]), start])
            colorsInClique.append(colors.get(filter_names(nnames[idx]), 'k') if colors is not None else 'k')

        lines(points, colorsInClique, marker="o", markersize=80, linewidth=linewidth_sign)
        start += height



def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        # e.g., p=('clf1', 'clf3', 0.0001220703125, True)
        if p[3] == False:
            # names控制顺序，按照rank从高到低（越低越好）的排序的
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False, colors=None, outPath=None):
    """
    Draws the critical difference diagram given the list of pairwise classifiers that are
    significant or not
    """
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha)

    # print(average_ranks)

    # for p in p_values:
    #     print(p)

    # average_ranks.values从大到小（越小越好）排序，average_ranks.keys是对应的名字，p_values表示各个方法两两之间的区分度
    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=True, width=15, textspace=1.5, labels=labels, colors=colors)

    font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 22,
        }
    if title:
        plt.title(title,fontdict=font, y=0.9, x=0.5)
        
    plt.tight_layout()
    if outPath is not None:
        plt.savefig(outPath+".png",bbox_inches='tight')
        plt.savefig(outPath+".eps",bbox_inches='tight')
        plt.savefig(outPath+".pdf",bbox_inches='tight')
    else:
        plt.savefig('cd-diagram.png',bbox_inches='tight')
        plt.savefig('cd-diagram.eps',bbox_inches='tight')
        plt.savefig('cd-diagram.pdf',bbox_inches='tight')
    plt.show()

# def draw_cd_diagram_inline(df_perf=None, alpha=0.05, title=None, labels=False, colors=None, outPath=None):
#     """
#     Draws the critical difference diagram given the list of pairwise classifiers that are
#     significant or not
#     """
#     p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha)

#     # print(average_ranks)

#     # for p in p_values:
#     #     print(p)

#     # average_ranks.values从大到小（越小越好）排序，average_ranks.keys是对应的名字，p_values表示各个方法两两之间的区分度
#     graph_ranks_inline(average_ranks.values, average_ranks.keys(), p_values,
#                 cd=None, reverse=True, width=9, textspace=1.5, labels=labels, colors=colors)

#     font = {'family': 'sans-serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 22,
#         }
#     if title:
#         plt.title(title,fontdict=font, y=0.9, x=0.5)
        
#     # plt.tight_layout()
#     # if outPath is not None:
#     #     plt.savefig(outPath+".png",bbox_inches='tight')
#     #     plt.savefig(outPath+".eps",bbox_inches='tight')
#     # else:
#     #     plt.savefig('cd-diagram.png',bbox_inches='tight')
#     #     plt.savefig('cd-diagram.eps',bbox_inches='tight')
#     # plt.show()

def wilcoxon_holm(alpha=0.05, df_perf=None):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis 
    """
#     print(pd.unique(df_perf['classifier_name']))

    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # print(df_counts)
    
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])
    
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1] # [1]表示只提取检验结果中的 p 值
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        print('the null hypothesis over the entire classifiers cannot be rejected')
        exit()
        
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_1]['accuracy']
                          , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_2]
                              ['accuracy'], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])
    # get the rank data
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=
    np.unique(sorted_df_perf['dataset_name']))

    # number of wins
    dfff = df_ranks.rank(ascending=False)
#     print(dfff[dfff == 1.0].sum(axis=1))

    # average the ranks 并且从高到低排序（越低排名越好）
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    print(average_ranks)

    print(p_values)

    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets