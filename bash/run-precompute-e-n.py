import sys
import gc
from datetime import datetime
from myfuncs import *
from suiteTestTool import *
from zoomSSIMTool import *
from dynamicProgramming import *

def process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
    method_name,
    png_dir, appendix, 
    min_base, writer, outfile):
    
    # 运行采样程序，提取运行时间
    command = [
        "java", "-Xmx10G", "-jar", sample_jar,
        str(inputFile),
        str(hasHeader),
        str(timeIdx),
        str(valueIdx),
        str(N),
        str(nout),
        str(lastParam), # 这个参数不同方法的含义不一样
        # 不给outDir参数,表示和inputFile保存在一个文件夹里
    ]
    try:
        result=subprocess.run(command, check=True,text=True, capture_output=True)
        print("命令执行成功！")
    except subprocess.CalledProcessError as e:
        print("命令执行失败：", e)
    print(result.stdout)
    output_lines = result.stdout.splitlines()  # 按行分割输出内容
    output_file_path = None
    for line in output_lines:
        if "Output file:" in line:
            output_file_path = line.split(":", 1)[1].strip()  # 只分割第一个冒号
        if "Time taken" in line:
            runtime=line.split(":", 1)[1].strip()  # 只分割第一个冒号
            runtime = runtime.replace('ms', '')  # 去掉 'ms'
            runtime=float(runtime) # ms
            runtime=runtime/1000 # s
    if output_file_path:
        print("提取的输出文件路径为：", output_file_path)
    else:
        print("未找到输出文件路径")

    # # 获取在线采样结果
    # df=pd.read_csv(output_file_path,header=0) # 表头为z,x,y
    # t1=df['x']
    # v1=df['y']
    # t1=t1.to_numpy(dtype='float')
    # v1=v1.to_numpy(dtype='float')
    # print(f'【{method_name}:m=】', len(t1)) # 结果点数 预计算模式下等于输入点数
    
    # 记录结果
    writer.writerow([method_name, 
                     nout, N,
                     filename, 
                     1 / (runtime+min_base), # for CD diagram, the larger the better
                     runtime # for recording raw runtime，注意此时runtime也是每个不同的seed下都会重新运行测试
                    ])
    gc.collect()
    outfile.flush()


#################################################################
ucr_dir="/root/starLightCurve_enlarge"
# ucr_dir="D:\\datasets\\starLightCurve_enlarge"
datasetNames = os.listdir(ucr_dir)
print(len(datasetNames))

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
pngDir=f"/root/tmpUCR_{current_time}"
# pngDir=f"D://tmp2//tmpUCR_{current_time}"
if not os.path.exists(pngDir):
    os.makedirs(pngDir)
print(pngDir)

current_script = os.path.abspath(__file__)
target_folder=pngDir
target_path = os.path.join(target_folder, os.path.basename(current_script))
shutil.copy(current_script, target_path)

width=1000
height=250
dpi=72

def getelist(n):
    eList=[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.6, 2]
    for i in range(len(eList)):
        eList[i]=int(eList[i]*n)
    return eList
    
targetDatasets=["StarLightCurves_enlarge"]
out=os.path.join(pngDir,f"parameter-evaluation-vary-n.csv")

with open(out, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    # for CD diagram, the larger the better
    row=['classifier_name', 'm','n','dataset_name','1/runtime','runtime'] 
    writer.writerow(row)
    minBase=sys.float_info.epsilon # 避免1/x当x=0
    
    for appendix in datasetNames: # 注意appendix自带.csv

        if not any(targetDataset in appendix for targetDataset in targetDatasets):  # 如果没有任何一个targetDataset在appendix中
            continue  # 跳过当前appendix，继续下一个

        if '-segment' in appendix:
            continue

        filename=os.path.join(ucr_dir,appendix)
        print(filename)
        df=pd.read_csv(filename, header=None)

        v_all=df.iloc[:,1]
        v_all=v_all.to_numpy(dtype='float')
        
        # 制造irregular时间戳
        if "steel_" in appendix:
            t_all=df.iloc[:,0]
            t_all=t_all.to_numpy(dtype='float64')
        else:
            np.random.seed(28)
            degree=5 # 从degree*len(v)里留下len(v)个数字作为时间戳，degree=1就是uniform的
            print('generate non-uniform timestamp, degree=',degree)
            t_all= np.arange(len(v_all)*degree)
            selected_indices = np.random.choice(len(t_all), size=len(v_all), replace=False)
            selected_indices = np.sort(selected_indices)
            t_all=t_all[selected_indices]
        
        nout = 2 # m=2对于BUG来说是预计算模式，计算全部点

        sourceDir=os.path.join(pngDir, appendix, "source")
        if not os.path.exists(sourceDir):
            os.makedirs(sourceDir)
        inputFile=os.path.join(sourceDir, f'{appendix.split(".")[0]}-raw-{len(t_all)}.csv')
        hasHeader=False
        df = pd.DataFrame({'t':t_all, 'v': v_all})
        df.to_csv(inputFile,header=hasHeader,index=False) #  保存为 CSV 文件，不包含表头和索引
        timeIdx=0
        valueIdx=1

        nList=[0.2, 0.4, 0.6, 0.8, 1]
        for n_per in nList:
            
            n=int(n_per*len(v_all))
            
            # t=t_all[0:n]
            # v=v_all[0:n]
            
            print('---------------n_per=',n_per,', n=',n,', m=',nout,'----------------')
            
            # sourceDir=os.path.join(pngDir, appendix, "source")
            # if not os.path.exists(sourceDir):
            #     os.makedirs(sourceDir)
            # inputFile=os.path.join(sourceDir, f'{appendix.split(".")[0]}-raw-{n}.csv')
            # hasHeader=False
            # df = pd.DataFrame({'t':t, 'v': v})
            # df.to_csv(inputFile,header=hasHeader,index=False) #  保存为 CSV 文件，不包含表头和索引
            # timeIdx=0
            # valueIdx=1
    
            ##################################
            basicName="eBUG"
            sample_jar=f"../jars/sample_{basicName}-jar-with-dependencies.jar"
            # N=-1 # 代表读每个文件的全部行
            N=n
            eList=getelist(len(t_all))
            print(eList)
            for eParam in eList:
                lastParam = eParam
                method_name=f"{basicName}-{lastParam}"
                print(f'>>>>>>>>{method_name}<<<<<<<<')

                process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
                    method_name,
                    pngDir, appendix, 
                    minBase, writer, outfile)
    

print('finish')
print(out)