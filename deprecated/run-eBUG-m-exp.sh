#!/bin/bash

HOME_PATH=/root/vary-m
#cp $HOME_PATH/ProcessResult.* .

DEVICE_RAW=root.sg.d5
DEVICE_PRE=root.sg.d6
n=16867328
mlist="0.1 0.3 0.5 0.7 0.9"
e=3373465

echo 3 |sudo tee /proc/sys/vm/drop_cache
free -m
echo "Begin experiment!"

# [query data]
echo "raw data query"
cd $HOME_PATH
mkdir raw
cd raw
cp $HOME_PATH/ProcessResult.* .
i=0
# device measurement timestamp_precision dataMinTime dataMaxTime range m approach save_query_result save_query_path
$HOME_PATH/query_experiment.sh 3 ${DEVICE_RAW} ${m_int} $e >> result_${i}.txt
java ProcessResult result_${i}.txt result_${i}.out ../sumResult_raw.csv


echo "querying precomputed table"
cd $HOME_PATH
mkdir precomputed
cd precomputed
cp $HOME_PATH/ProcessResult.* .
i=0
for m_per in $mlist
do
  result=$(echo "${m_per} * $n" | bc)  # 使用 bc 计算
  m_int=$(printf "%.0f" "$result")  # 将结果转换为整数
  echo "[[[[[[[[[[[[[m_percent=${m_per}, m=${m_int}]]]]]]]]]]]]]"

  # Note the following command print info is appended into result_${i}.txt for query latency exp
  find $HOME_PATH -type f -iname "*.sh" -exec chmod +x {} \;

  # device measurement timestamp_precision dataMinTime dataMaxTime range m approach save_query_result save_query_path
  $HOME_PATH/query_experiment.sh 2 ${DEVICE_PRE} ${m_int} $e >> result_${i}.txt
  java ProcessResult result_${i}.txt result_${i}.out ../sumResult_precomputed.csv
  let i+=1
done

echo "online sampling"
cd $HOME_PATH
mkdir online
cd online
cp $HOME_PATH/ProcessResult.* .
i=0
for m_per in $mlist
do
  result=$(echo "${m_per} * $n" | bc)  # 使用 bc 计算
  m_int=$(printf "%.0f" "$result")  # 将结果转换为整数
  echo "[[[[[[[[[[[[[m_percent=${m_per}, m=${m_int}]]]]]]]]]]]]]"

  # Note the following command print info is appended into result_${i}.txt for query latency exp
  find $HOME_PATH -type f -iname "*.sh" -exec chmod +x {} \;

  # device measurement timestamp_precision dataMinTime dataMaxTime range m approach save_query_result save_query_path
  $HOME_PATH/query_experiment.sh 1 ${DEVICE_RAW} ${m_int} $e >> result_${i}.txt
  java ProcessResult result_${i}.txt result_${i}.out ../sumResult_online.csv
  let i+=1
done


#cd $HOME_PATH
#(cut -f 2,11,12,28,35 -d "," sumResult_MinMax.csv) > tmp1.csv
#(cut -f 2,11,12,28,35 -d "," sumResult_M4.csv| paste -d, tmp1.csv -) > tmp2.csv
#(cut -f 2,11,12,28,35 -d "," sumResult_LTTB.csv| paste -d, tmp2.csv -) > tmp3.csv
#(cut -f 2,11,12,28,35 -d "," sumResult_MinMaxLTTB.csv| paste -d, tmp3.csv -) > tmp4.csv
#(cut -f 2,11,12,28,35 -d "," sumResult_ILTS.csv| paste -d, tmp4.csv -) > tmp5.csv
#(cut -f 2,11,12,28,35 -d "," sumResult_OM3.csv| paste -d, tmp5.csv -) > tmp6.csv

#echo "MinMax(ns),A_METADATAS(ns),B_CHUNK(ns),readChunkNum,traversedPointNum,\
#M4(ns),A_METADATAS(ns),B_CHUNK(ns),readChunkNum,traversedPointNum,\
#LTTB(ns),A_METADATAS(ns),B_CHUNK(ns),readChunkNum,traversedPointNum,\
#MinMaxLTTB(ns),A_METADATAS(ns),B_CHUNK(ns),readChunkNum,traversedPointNum,\
#ILTS(ns),A_METADATAS(ns),B_CHUNK(ns),readChunkNum,traversedPointNum,\
#OM3(ns),A_METADATAS(ns),B_CHUNK(ns),readChunkNum,traversedPointNum"\
# > $HOME_PATH/res-${DATASET}-efficiency.csv
#
#sed '1d' tmp6.csv >> $HOME_PATH/res-${DATASET}-efficiency.csv
#rm tmp*.csv
#
## add varied parameter value and the corresponding estimated chunks per interval for each line
## estimated chunks per interval = range/m/(totalRange/(pointNum/chunkSize))
## range=totalRange, estimated chunks per interval=(pointNum/chunkSize)/m
#sed -i -e 1's/^/m,estimated chunks per interval,/' $HOME_PATH/res-${DATASET}-efficiency.csv
#line=2
#
#for m in $mlist
#do
#  #let c=${pointNum}/${chunkSize}/$m # note bash only does the integer division
#  c=$((echo scale=3 ; echo ${TOTAL_POINT_NUMBER}/${IOTDB_CHUNK_POINT_SIZE}/$m) | bc )
#  sed -i -e ${line}"s/^/${m},${c},/" $HOME_PATH/res-${DATASET}-efficiency.csv
#  let line+=1
#done

echo "ALL FINISHED!"
echo 3 |sudo tee /proc/sys/vm/drop_caches
free -m