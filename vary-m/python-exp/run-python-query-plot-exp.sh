#!/bin/bash

HOME_PATH=/root/vary-m

export READ_METHOD=raw # raw/online/pre
export TRI_VISUALIZATION_EXP=${HOME_PATH}
export remote_TRI_VISUALIZATION_EXP=${HOME_PATH}
export remote_IOTDB_HOME_PATH=${HOME_PATH}
export remote_ip=127.0.0.1
export remote_user_name=root
export remote_passwd='root' # do not use double quotes

#######################################
# below are local client configurations
export PYTHON_READ_PLOT_PATH=$TRI_VISUALIZATION_EXP/python-exp/python-read-plot.py
export EXPERIMENT_PATH=$TRI_VISUALIZATION_EXP/python-exp/python_query_plot_experiment.sh
export repetition=1
export PROCESS_QUERY_PLOT_JAVA_PATH=$TRI_VISUALIZATION_EXP/python-exp/ProcessQueryPlotResult.java
export local_FILE_PATH=$TRI_VISUALIZATION_EXP/python-exp/localData.csv

# below are remote data server configurations
export remote_IOTDB_SBIN_HOME=$remote_IOTDB_HOME_PATH/iotdb-server-0.12.4/sbin
export remote_IOTDB_CONF_PATH=$remote_IOTDB_HOME_PATH/iotdb-server-0.12.4/conf/iotdb-engine.properties
export remote_IOTDB_START=$remote_IOTDB_SBIN_HOME/start-server.sh
export remote_IOTDB_STOP=$remote_IOTDB_SBIN_HOME/stop-server.sh
export remote_IOTDB_EXPORT_CSV_TOOL=$remote_IOTDB_HOME_PATH/iotdb-cli-0.12.4/tools
export remote_iotdb_port=6667
export remote_iotdb_username=root
export remote_iotdb_passwd=root
export remote_tool_bash=$remote_TRI_VISUALIZATION_EXP/python-exp/tool.sh
# export remote_TRI_FILE_PATH=$remote_TRI_VISUALIZATION_EXP/python-exp/res.csv

echo "begin"

# prepare ProcessQueryPlotResult tool
sed '/^package/d' ProcessQueryPlotResult.java > ProcessQueryPlotResult2.java
rm ProcessQueryPlotResult.java
mv ProcessQueryPlotResult2.java ProcessQueryPlotResult.java
javac ProcessQueryPlotResult.java

n=16867328

if [ "$READ_METHOD" = "raw" ]; then
    m_per_list=(0.1)  # m does not affect raw
else
    m_per_list=(0.1 0.3 0.5)
fi

#for m_per in 0.1 0.3 0.5 0.7 0.9
for m_per in "${m_per_list[@]}"; do
	result=$(echo "${m_per} * $n" | bc)  # 使用 bc 计算
  	m_int=$(printf "%.0f" "$result")  # 将结果转换为整数

	echo "m=${m_int}"
	export m=${m_int}

	$EXPERIMENT_PATH >result-${READ_METHOD}_${m_int}.txt #> is overwrite, >> is append

	java ProcessQueryPlotResult result-${READ_METHOD}_${m_int}.txt result-${READ_METHOD}_${m_int}.out sumResult-${READ_METHOD}.csv ${m_int}
done

echo "ALL FINISHED!"
echo 3 |sudo tee /proc/sys/vm/drop_caches
free -m