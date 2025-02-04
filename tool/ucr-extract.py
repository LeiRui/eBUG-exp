from myfuncs import *
import os
import sys
import gc

if len(sys.argv) != 3:
    print("Usage: python script.py <input_directory> <output_directory>")
    sys.exit(1)
    
mydir = sys.argv[1]
outDir = sys.argv[2]

datasetNames = os.listdir(mydir)
print(len(datasetNames))

if not os.path.exists(outDir):
    os.makedirs(outDir)
                
cnt=0
classCnt=0
for suffix in datasetNames:
    classCnt+=1
    for appendix in {"_TEST.tsv"}:
        filename=os.path.join(mydir,suffix,suffix+appendix)
        print(filename)
        df=pd.read_csv(filename, header=None, delimiter=r"\s+").T.iloc[1:, :] # the first column is class
        npts=df.shape[0]
        ncol=df.shape[1]
        print(df.shape)

        startT=0
        for j in np.arange(ncol):
            v=df.iloc[:,j]
            if v.isnull().any():
                continue

            cnt+=1
            v=v.to_numpy(dtype='float')
            t=np.arange(startT,startT+len(v))
            startT+=len(v)

            output_df = pd.DataFrame({'t': t, 'v': v})
            output_csv = os.path.join(outDir,suffix+".csv")
            if not os.path.isfile(output_csv):
                output_df.to_csv(output_csv, mode='w', index=False, header=False)
            else:
                output_df.to_csv(output_csv, mode='a', index=False, header=False)
            print(f"{j} has been saved to {output_csv}")
            gc.collect()
        
print('finish')