from iotdb.Session import Session
from iotdb.utils.IoTDBConstants import TSDataType, TSEncoding, Compressor
from iotdb.utils.Tablet import Tablet

from matplotlib import pyplot as plt
import numpy as np
import csv
import datetime
import pandas as pd
import time
import argparse
import sys
import os
import math
import re


read_method=str(os.environ['READ_METHOD'])
print(read_method)

# read data---------------------------------------------------------------------------
disk_file_path=str(os.environ['local_FILE_PATH'])

column_names = ['Time', 'Value']

start = time.time_ns()
df = pd.read_csv(disk_file_path,engine="pyarrow",skiprows=1,names=column_names) # the first line is header; use engine="pyarrow" to accelerate read_csv otherwise is slow
convert_dict = {
	df.columns[0]:np.int64,
	df.columns[1]:np.double,
}
df = df.astype(convert_dict)
if read_method == "pre":
    df = df.sort_values(by=df.columns[0])  # 按照第一列（时间列）递增排序
parse_time = time.time_ns()-start
print(f"[1-ns]parse_data,{parse_time}") # print metric

# plot data---------------------------------------------------------------------------
x=df[df.columns[0]] # time
y=df[df.columns[1]] # value
r, c = df.shape
print(r) #number of points
print(c) #two columns: time and value

fig=plt.figure(1,dpi=120)
start = time.time_ns()
plt.plot(x,y,linewidth=0.5)
end = time.time_ns()
plt.savefig(os.environ['TRI_VISUALIZATION_EXP']+"/python-exp/dataset-{}-{}.png".format(read_method,os.environ['m']),bbox_inches='tight') #specify absolute fig path
print(f"[1-ns]plot_data,{end - start}") # print metric

plt.close(fig)