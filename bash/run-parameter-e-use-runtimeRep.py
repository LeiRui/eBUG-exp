import sys
import gc
from datetime import datetime
from myfuncs import *
from suiteTestTool import *
from zoomSSIMTool import *
from dynamicProgramming import *
import time 

def process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
    method_name, noutWarningRatio,
    png_dir, appendix,
    t, v, 
    min_base, writer, outfile):

    rep=10
    timeList=[]

    # increase repetition time to measure elapsed time
    for i in range(rep):
        command = [
            "java", "-jar", sample_jar,
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

        timeList.append(runtime)
        print(runtime)
        
        gc.collect()
        time.sleep(2)

    runtime=np.mean(timeList)
    # 记录结果
    writer.writerow([method_name, 
                nout, len(t),
                filename + f'-seed{seed}',runtime
                ])
    outfile.flush()
    gc.collect()


#################################################################
ucr_dir="/root/UCRsets-single/UCRsets-single"
# ucr_dir="D://datasets//regular//UCRsets-single//UCRsets-single"
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
    eList=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
    for i in range(len(eList)):
        eList[i]=int(eList[i]*n)
    return eList
    
targetDatasets=["steel_REDU","Mallat","MixedShapesRegularTrain","StarLightCurves"]
out=os.path.join(pngDir,f"parameter-evaluation-vary-e.csv")

with open(out, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    # for CD diagram, the larger the better
    row=['classifier_name', 'm','n','dataset_name','runtime'] 
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

        v=df.iloc[:,1]
        v=v.to_numpy(dtype='float')
        
        # 制造irregular时间戳
        if "steel_" in appendix:
            t=df.iloc[:,0]
            t=t.to_numpy(dtype='float64')
        else:
            np.random.seed(28)
            degree=5 # 从degree*len(v)里留下len(v)个数字作为时间戳，degree=1就是uniform的
            print('generate non-uniform timestamp, degree=',degree)
            t= np.arange(len(v)*degree)
            selected_indices = np.random.choice(len(t), size=len(v), replace=False)
            selected_indices = np.sort(selected_indices)
            t=t[selected_indices]
        
        points=np.array(list(zip(t,v)))

        nout=int(len(v)*0.3)
        nout = (nout // 4) * 4 # for if M4 (MinMax) requires at least integer multiply of four (two)
        
        print('n=',len(v),', m=',nout)
        
        # 生成对应时间范围内的随机缩放操作序列，参考MinMaxCache
        from_range = int(min(t)) #左闭
        to_range = int(max(t)+1) #右开
        seq_count = 50
        q = 0.1
        if "steel_" in appendix:
            zoom_factor = 5
            jar_path="../jars/experiments-tool-ZI.jar"
        elif "StarLightCurves" in appendix:
            zoom_factor = 10
            jar_path="../jars/experiments-tool.jar"
        else:
            zoom_factor = 5
            jar_path="../jars/experiments-tool.jar"
        min_shift = 0.1
        max_shift = 0.5

        seedMap={}
        seedRange=1
    
        noutWarningRatio=0.1
        
        sourceDir=os.path.join(pngDir, appendix, "source")
        if not os.path.exists(sourceDir):
            os.makedirs(sourceDir)
        inputFile=os.path.join(sourceDir, f'{appendix.split(".")[0]}-raw.csv')
        hasHeader=False
        df = pd.DataFrame({'t':t, 'v': v})
        df.to_csv(inputFile,header=hasHeader,index=False) #  保存为 CSV 文件，不包含表头和索引
        timeIdx=0
        valueIdx=1

        ##################################
        basicName="eBUG"
        sample_jar=f"../jars/sample_{basicName}-jar-with-dependencies.jar"
        N=-1 # 代表读每个文件的全部行
        eList=getelist(len(t))
        print(eList)
        for eParam in eList:
            lastParam = eParam
            for seed in range(seedRange):
                print('------------')
                print('seed', seed)

                method_name=f"{basicName}-{lastParam}"
                print(f'>>>>{method_name}<<<<')

                process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
                    method_name, noutWarningRatio,
                    pngDir, appendix, 
                    t, v,
                    minBase, writer, outfile)

            print('===================seed finish==================')

print('finish')
print(out)