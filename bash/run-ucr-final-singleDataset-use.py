import sys
import gc
from datetime import datetime
from myfuncs import *
from suiteTestTool import *
from zoomSSIMTool import *
from dynamicProgramming import *

def process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
    method_name, noutWarningRatio,
    png_dir, appendix, seed, seq_count, from_range, to_range, q_str, zoom_factor_str, 
    min_shift, max_shift, 
    t, v, seed_map, width, height, dpi, 
    min_base, writer, outfile):
    
    # 运行采样程序，提取运行时间
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

    # 获取在线采样结果
    df=pd.read_csv(output_file_path,header=0) # 表头为z,x,y
    t1=df['x']
    v1=df['y']
    t1=t1.to_numpy(dtype='float')
    v1=v1.to_numpy(dtype='float')
    print(f'【{method_name}:m=】', len(t1)) # 结果点数
    if abs(len(t1) - nout) > nout * noutWarningRatio:
        # raise ValueError("ATTENTION: not match nout")
        print("ATTENTION: not match nout")

    # 进行作图精度比较
    root_name = os.path.join(png_dir, appendix, f'seed{seed}')
#     operations = f"op-cnt{seq_count}-from{from_range}-to{to_range}-q{q_str}-zoom{zoom_factor_str}-shift{min_shift}-{max_shift}-seed{seed}.csv"
    operations = "myop.csv"

    truth_dir_path = os.path.join(root_name, f"tmp-truth")
    tmin, tmax, vmin, vmax = seed_map
    ssim_result = random_scale_ssim(tmin, tmax, vmin, vmax, truth_dir_path, t, v, t1, v1, width, height,
                                    os.path.join(root_name, f"tmp-{method_name}-{nout}"), method_name, operations, dpi=dpi, 
                                    onlySSIM=True) # 只需要ssim，其他关闭减少耗时
    
    # 记录结果
    writer.writerow([method_name, filename + f'-seed{seed}', 
                     1 / (ssim_result[0] + min_base),  # for CD diagram, the larger the better
                     1 / (ssim_result[2] + min_base),  # for CD diagram, the larger the better
                     ssim_result[4], 
                     1 / (ssim_result[6] + min_base), # for CD diagram, the larger the better
                     1 / (runtime+min_base), # for CD diagram, the larger the better
                     runtime # for recording raw runtime，注意此时runtime也是每个不同的seed下都会重新运行测试
                     ])
    gc.collect()
    outfile.flush()


#################################################################
# ucr_dir="/root/UCRsets-single/UCRsets-single"
ucr_dir="D://datasets//regular//UCRsets-single//UCRsets-single"
datasetNames = os.listdir(ucr_dir)
print(len(datasetNames))

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# pngDir=f"/root/tmpUCR_{current_time}"
pngDir=f"D://tmp//tmpUCR_{current_time}"
if not os.path.exists(pngDir):
    os.makedirs(pngDir)
print(pngDir)

current_script = os.path.abspath(__file__)
target_folder=pngDir
target_path = os.path.join(target_folder, os.path.basename(current_script))
shutil.copy(current_script, target_path)

width=500
height=250
dpi=288
                
out=os.path.join(pngDir,"UCR_bench.csv")

with open(out, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    # for CD diagram, the larger the better
    row=['classifier_name', 'dataset_name', '1/accuracy-ad','1/accuracy-datarmse','accuracy-pixelssim','1/accuracy-pixelmse',
        '1/runtime','runtime'] 
    writer.writerow(row)
    minBase=sys.float_info.epsilon # 避免1/x当x=0
    
    for appendix in datasetNames: # 注意appendix自带.csv
        if '-segment' in appendix:
            continue

        if 'SwedishLeaf' not in appendix:
            continue

        filename=os.path.join(ucr_dir,appendix)
        print(filename)
        df=pd.read_csv(filename, header=None)

        v=df.iloc[:,1]
        v=v.to_numpy(dtype='float')
        
        # 制造irregular时间戳
        np.random.seed(28)
        degree=5 # 从degree*len(v)里留下len(v)个数字作为时间戳，degree=1就是uniform的
        print('generate non-uniform timestamp, degree=',degree)
        t= np.arange(len(v)*degree)
        selected_indices = np.random.choice(len(t), size=len(v), replace=False)
        selected_indices = np.sort(selected_indices)
        t=t[selected_indices]
        
        points=np.array(list(zip(t,v)))

        nout=10000
        nout = (nout // 4) * 4 # for if M4 (MinMax) requires at least integer multiply of four (two)

        print('n=',len(v),', m=',nout)
        
        # 生成对应时间范围内的随机缩放操作序列，参考MinMaxCache
        # java -jar experiments-tool.jar -c tool -out . -seqCount 50 -fromRange 0 -toRange 8197 -q 0.1 -zoomFactor 2 -minShift 0.1 -maxShift 0.5 -seed 42
        from_range = min(t) #左闭
        to_range = max(t)+1 #右开
        seq_count = 20 # TODO 重写operations
        q = 0.1
        zoom_factor = 10
        min_shift = 0.1
        max_shift = 0.5
        jar_path="experiments-tool.jar"

        # 变化seed
        seedMap={}

        seed = 1
        # print('seed',seed)
    
        command = [
            "java", "-jar", jar_path,
            "-c", "tool",
            "-out", ".",
            "-seqCount", str(seq_count),
            "-fromRange", str(from_range),
            "-toRange", str(to_range),
            "-q", str(q),
            "-zoomFactor", str(zoom_factor),
            "-minShift", str(min_shift),
            "-maxShift", str(max_shift),
            "-seed", str(seed)
        ]
        try:
            subprocess.run(command, check=True)
            print("operations生成成功！")
        except subprocess.CalledProcessError as e:
            print("operations生成失败：", e)

        zoom_factor_str = f"{zoom_factor:.1f}" if zoom_factor == int(zoom_factor) else str(zoom_factor)
        q_str = f"{q:.1f}" if q == int(q) else str(q) # let q=1 named as 1.0
#         operations = f"op-cnt{seq_count}-from{from_range}-to{to_range}-q{q_str}-zoom{zoom_factor_str}-shift{min_shift}-{max_shift}-seed{seed}.csv"
        operations = "myop.csv"


        print('=================ground truth=================')
        rootName=os.path.join(pngDir,appendix,f'seed{seed}')
        truthDirPath=os.path.join(rootName,f"tmp-truth")
        tmin,tmax,vmin,vmax=random_scale_ground_truth(t,v, width, height, truthDirPath, operations, dpi=dpi)
        seedMap=[tmin,tmax,vmin,vmax]
        print('==========================================')


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
        # for eBUG family
        basicName="eBUG"
        sample_jar=f"sample_{basicName}-jar-with-dependencies.jar"
        N=-1 # 代表读每个文件的全部行
        for eParam in [0,5000,100000000]:
            lastParam = eParam
            print('------------')
            print('seed', seed)

            method_name=f"{basicName}-{lastParam}"
            print(f'>>>>{method_name}<<<<')

            process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
                method_name, noutWarningRatio,
                pngDir, appendix, seed, seq_count, from_range, to_range, q_str, zoom_factor_str, 
                min_shift, max_shift, 
                t, v, seedMap, width, height, dpi, 
                minBase, writer, outfile)


        ##################################
        # for BUYdiff family
        basicName="BUYdiff"
        sample_jar=f"sample_{basicName}-jar-with-dependencies.jar"
        N=-1 # 代表读每个文件的全部行
        for error in ["L2"]:
            lastParam = error
            print('------------')
            print('seed', seed)

            method_name=f"{basicName}-{lastParam}"
            print(f'>>>>{method_name}<<<<')

            process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
                method_name, noutWarningRatio,
                pngDir, appendix, seed, seq_count, from_range, to_range, q_str, zoom_factor_str, 
                min_shift, max_shift, 
                t, v, seedMap, width, height, dpi, 
                minBase, writer, outfile)


        ##################################
        # for MinMax bucket family
        basicName="minmax"
        sample_jar=f"sample_{basicName}-jar-with-dependencies.jar"
        N=-1 # 代表读每个文件的全部行
        for bucket in ["width"]:
            lastParam = bucket
            print('------------')
            print('seed', seed)

            method_name=f"{basicName}-{lastParam}"
            print(f'>>>>{method_name}<<<<')

            process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
                method_name, noutWarningRatio,
                pngDir, appendix, seed, seq_count, from_range, to_range, q_str, zoom_factor_str, 
                min_shift, max_shift, 
                t, v, seedMap, width, height, dpi, 
                minBase, writer, outfile)



        ##################################
        # for fsw 
        basicName="fsw"
        sample_jar=f"sample_{basicName}-jar-with-dependencies.jar"
        N=-1 # 代表读每个文件的全部行
        tolerantRatio=0.001
        lastParam = tolerantRatio
        print('------------')
        print('seed', seed)

        method_name=f"{basicName}-{lastParam}"
        print(f'>>>>{method_name}<<<<')

        process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
            method_name, noutWarningRatio,
            pngDir, appendix, seed, seq_count, from_range, to_range, q_str, zoom_factor_str, 
            min_shift, max_shift, 
            t, v, seedMap, width, height, dpi, 
            minBase, writer, outfile)



        ##################################
        # for rdp
        basicName="rdp"
        sample_jar=f"sample_{basicName}-jar-with-dependencies.jar"
        N=-1 # 代表读每个文件的全部行
        tolerantRatio=0.001
        lastParam = tolerantRatio
        print('------------')
        print('seed', seed)

        method_name=f"{basicName}-{lastParam}"
        print(f'>>>>{method_name}<<<<')

        process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
            method_name, noutWarningRatio,
            pngDir, appendix, seed, seq_count, from_range, to_range, q_str, zoom_factor_str, 
            min_shift, max_shift, 
            t, v, seedMap, width, height, dpi, 
            minBase, writer, outfile)



        ##################################
        # for swab
        basicName="swab"
        sample_jar=f"sample_{basicName}-jar-with-dependencies.jar"
        N=-1 # 代表读每个文件的全部行
        tolerantRatio=0.01
        lastParam = tolerantRatio
        print('------------')
        print('seed', seed)

        method_name=f"{basicName}-{lastParam}"
        print(f'>>>>{method_name}<<<<')

        process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
            method_name, noutWarningRatio,
            pngDir, appendix, seed, seq_count, from_range, to_range, q_str, zoom_factor_str, 
            min_shift, max_shift, 
            t, v, seedMap, width, height, dpi, 
            minBase, writer, outfile)



        ##################################
        # for swab ad
        basicName="swab_ad"
        sample_jar=f"sample_{basicName}-jar-with-dependencies.jar"
        N=-1 # 代表读每个文件的全部行
        tolerantRatio=0.01
        lastParam = tolerantRatio
        print('------------')
        print('seed', seed)

        method_name=f"{basicName}-{lastParam}"
        print(f'>>>>{method_name}<<<<')

        process_method(sample_jar, inputFile, hasHeader, timeIdx, valueIdx, N, nout, lastParam, 
            method_name, noutWarningRatio,
            pngDir, appendix, seed, seq_count, from_range, to_range, q_str, zoom_factor_str, 
            min_shift, max_shift, 
            t, v, seedMap, width, height, dpi, 
            minBase, writer, outfile)


print('finish')