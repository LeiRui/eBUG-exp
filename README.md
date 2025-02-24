# Experimental Guidance

## Preparations

### Install Packages

```shell
sudo apt-get update
sudo apt-get upgrade

apt install unzip
pip install simplification
pip install matplotlib
pip install pandas
pip install scipy
pip install opencv-python
pip install scikit-image
pip install opencv-python-headless
pip install scikit-learn
```

### Install Java

Please make sure the JAVA_HOME environment path has been set. You can follow the steps below to install and configure Java.


```bash
sudo apt install openjdk-11-jdk-headless

# configure
vim /etc/profile
# add the following two lines to the end of /etc/profile
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
# save and exit vim, and let the configuration take effect
source /etc/profile
```


### Prepare Datasets

1.   Download the [UCR time series classification archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/). Suppose that the downloaded path is `/root/UCRArchive_2018`.
2.   Go to `tool` directory, and run `python3 ucr-extract.py /root/UCRArchive_2018 /root/UCRsets-single `, where `/root/UCRArchive_2018` is the directory of the downloaded UCR datasets in the first step, and `/root/UCRsets-single` is the output directory of the concatenated long series. This step will take some time, please be patient.
3.   Copy the concatenated "Mallat", "MixedShapesRegularTrain", "StarLightCurves" csv into `datasets` directory.
4.   "steel_REDU" csv is already in the `datasets` directory.
5.   Go to `jars` directory, and run `kaggle datasets download xxx123456789/ebug-jars` to download the necessary JAR files. Due to their large size, these JAR files are hosted externally instead of being included in the Git repository.

## Vis Example
Go to `bash` directory, and run `run-ucr-final-singleDataset-use.py`

## UCR Bench

Go to `bash` directory, and run `run-ucr-final.py`.


## Vary e
Go to `bash` directory, and run `run-parameter-e-use.py`


## Vary n
1. Go to `tool` directory, run `enlarge.py` to prepare `/root/starLightCurve_enlarge/StarLightCurves_enlarge.csv`.
2. Go to `bash` directory, and run `bash/run-precompute-e-n.py`.

## Vary m

This experiments involves communication between two nodes and is a bit more complicated than the previous sections in terms of installation preparation. Assume that the server and client nodes have the following IP addresses, usernames, and passwords.

|            | Database Server Node | Rendering Client Node |
| ---------- | -------------------- | --------------------- |
| IP address | A                    | B                     |
| Username   | server               | client                |
| Password   | x                    | y                     |

### (1) Environment Setup for Both Nodes

-   Download Java as instructed earlier.

-   Download `vary-m` repository.

-   Download sshpass:

    ```shell
    sudo apt-get install sshpass
    ```


-   **After downloading sshpass, run `sshpass -p 'x' ssh server@A "echo 'a'"` on the client node to verify if sshpass works. If sshpass works, you will see an "a" printed on the screen. Otherwise, try executing `ssh server@A "echo 'a'"` on the client node, and then reply "yes" to the prompt ("Are you sure you want to continue connecting (yes/no/[fingerprint])?") and enter the password 'x' manually. Then run again `sshpass -p 'x' ssh server@A "echo 'a'"` on the client node to verify if sshpass works.**

-   Download the Python packages to be used:

    ```shell
    sudo apt install python3-pip
    pip install matplotlib
    pip install thrift
    pip install pandas
    pip install pyarrow
    
    pip show matplotlib # this is to check where python packages are installed. 
    
    cd /root/lts-exp/python-exp
    # In the following, we assume that python packages are installed in "/usr/local/lib/python3.8/dist-packages"
    cp -r iotdb /usr/local/lib/python3.8/dist-packages/. # this step installs iotdb-python-connector
    ```


### (2) Populate the Database Server Node

Before doing experiments, follow the steps below to populate the database server with test data.

1. Get precomputed data:
```shell
java -jar sample_eBUG-jar-with-dependencies.jar "/root/starLightCurve_enlarge/StarLightCurves_enlarge.csv" false 0 1 -1 2 3373465 "/root/vary-m/"
```
2. Write raw data and precomputed data into IoTDB:
    - Go to `vary-m/iotdb-server-0.12.4/sbin`, and run `start-server.sh`.
    - Run the following command to write precomputed data:
    ```shell
   java -jar WritePrecomputedTable-jar-with-dependencies.jar "root.sg.d6" "ms" 10000 "/root/vary-m/StarLightCurves_enlarge-eBUG-e3373465-n16867328-m2.csv" -1 true true
   ```
   - Run the following command to write raw data:
    ```shell
   java -jar WritePrecomputedTable-jar-with-dependencies.jar "root.sg.d5" "ms" 10000 "/root/starLightCurve_enlarge/StarLightCurves_enlarge.csv" -1 false false
    ```
   - Go to `vary-m/iotdb-server-0.12.4/sbin`, and run `stop-server.sh`.


### (3) Experiments on the Rendering Client Node

Go to the rendering client node. Enter the `python-exp` folder within the downloaded `lts-exp` repository and then:

1.   Make all scripts executable by executing `chmod +x *.sh`.

2.   Update `run-python-query-plot-exp.sh` as follows:

     -   Update `READ_METHOD` as `raw`/`online`/`pre`.
         -   `raw`: using the raw data query at the database server.
         
         -   `online`: using online sampling query at the database server without precomputation.

         -   `pre`: querying the precomputed table directly.
         
     -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of the `vary-m` repository on the client node.

     -   Update `remote_TRI_VISUALIZATION_EXP` as the downloaded path of the `vary-m` repository on the server node.

     -   Update `remote_IOTDB_HOME_PATH` to the same path as the "HOME_PATH" set in the "(2) Populate the Database Server Node" section of this README.

     -   Update `remote_ip` as the IP address of the database server node.

     -   Update `remote_user_name` as the login username of the database server node.

     -   Update `remote_passwd` as the login password of the database server node.

3.   Run experiments using `nohup ./run-python-query-plot-exp.sh 2>&1 &`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`. 

4.   When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in `sumResult-[READ_METHOD].csv`, where `[READ_METHOD]` is `raw`/`online`/`pre`.
