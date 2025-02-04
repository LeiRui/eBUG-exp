#!/bin/bash

HOME_PATH=/root/vary-m

IOTDB_SBIN_HOME=$HOME_PATH/iotdb-server-0.12.4/sbin
QUERY_JAR_PATH=$HOME_PATH/QueryEBUG*.jar # check

# only true in run-more-baselines.sh for saving query result csv for DSSIM exp
REP_ONCE=false
#SAVE_QUERY_RESULT_PATH=NONE

echo 3 | sudo tee /proc/sys/vm/drop_caches
cd $IOTDB_SBIN_HOME

echo $REP_ONCE

#if [ $# -eq 8 ]
if $REP_ONCE
then
  a=1
else # default
  a=1
fi
echo "rep=$a"

for((i=0;i<a;i++)) do
    echo $i
    ./start-server.sh /dev/null 2>&1 &
    sleep 10s

    java -jar $QUERY_JAR_PATH $1 $2 $3 $4

    ./stop-server.sh
    sleep 5s
    echo 3 | sudo tee /proc/sys/vm/drop_caches
    sleep 5s
done
