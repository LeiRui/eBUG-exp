#!/bin/bash

echo 3 | sudo tee /proc/sys/vm/drop_caches

sshpass -p "${remote_passwd}" ssh ${remote_user_name}@${remote_ip} "sed -i -e 's/\r$//' $remote_tool_bash"


cd $IOTDB_SBIN_HOME
for((i=0;i<$repetition;i++)) do
    echo "repetition No.$i"

    # client get data from data server 
    # start remote iotdb server
    sshpass -p "${remote_passwd}" ssh ${remote_user_name}@${remote_ip} "bash ${remote_IOTDB_START} >/dev/null 2>&1 &" >/dev/null 2>&1 &
    sleep 10s

    # query at remote server and transfom query result to csv
    sshpass -p "${remote_passwd}" ssh ${remote_user_name}@${remote_ip} "python3 ${remote_TRI_VISUALIZATION_EXP}/python-exp/remote-query-to-csv.py -r ${READ_METHOD} -t ${remote_IOTDB_EXPORT_CSV_TOOL} -m ${m}"
    
    # stop remote iotdb server
    sshpass -p "${remote_passwd}" ssh ${remote_user_name}@${remote_ip} "bash ${remote_IOTDB_STOP}"
    sleep 3s

    sshpass -p "${remote_passwd}" ssh ${remote_user_name}@${remote_ip} "echo 3 | sudo tee /proc/sys/vm/drop_caches"
    sleep 4s
    
    # transfer csv from server to client
    path=${remote_IOTDB_EXPORT_CSV_TOOL}/dump0.csv
    if [ "$READ_METHOD" == "pre" ]; then # 去掉第一列idx列
        cut -d',' -f2- "${path}" > tmp.csv
        mv tmp.csv "${path}"
    fi

    ts=$(date +%s%N)
    sshpass -p "${remote_passwd}" scp ${remote_user_name}@${remote_ip}:$path ${local_FILE_PATH}
    transfer_time=$((($(date +%s%N) - $ts)))
    echo "[1-ns]transfer_data,$transfer_time" # print metric

    # client local read and plot
    python3 $PYTHON_READ_PLOT_PATH

    echo 3 | sudo tee /proc/sys/vm/drop_caches
    sleep 4s
done
