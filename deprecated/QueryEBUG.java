package org.apache.iotdb.jarCode;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.iotdb.rpc.IoTDBConnectionException;
import org.apache.iotdb.rpc.StatementExecutionException;
import org.apache.iotdb.session.Session;
import org.apache.iotdb.session.SessionDataSet;
import org.apache.iotdb.session.SessionDataSet.DataIterator;
import org.apache.iotdb.tsfile.read.common.RowRecord;
import org.apache.thrift.TException;

public class QueryEBUG {

    private static final String onlineSQL =
            "SELECT EBUG(s1,'m'='%d','e'='%d') FROM %s";

    private static final String precomputeSQL =
            "SELECT pre_t,pre_v from %s limit %d";

    private static final String rawSQL =
            "SELECT s1 FROM %s";

    public static Session session;

    public static void main(String[] args)
            throws IoTDBConnectionException, StatementExecutionException, TException, IOException {
        int argIdx = 0;

        int precompute = Integer.parseInt(args[argIdx++]); // 1: online, 2:precompute, 3:raw
        System.out.println("[QueryData] precompute=" + precompute);

        String device = args[argIdx++];
        System.out.println("[QueryData] device=" + device);

        int m;
        if (args.length - 1 >= argIdx) {
            m = Integer.parseInt(args[argIdx++]);
            System.out.println("[QueryData] m=" + m);
        } else {
            m = -1; // not used
        }

        int e;
        if (args.length - 1 >= argIdx) {
            e = Integer.parseInt(args[argIdx++]);
            System.out.println("[QueryData] e=" + e);
        } else {
            e = -1; // not used
        }

        String sql;
        if (precompute == 2) {
            sql = String.format(precomputeSQL, device, m);
        } else if (precompute == 1) {
            sql = String.format(onlineSQL, m, e, device);
        } else {
            sql = String.format(rawSQL, device);
        }
        System.out.println("[QueryData] sql=" + sql);

        session = new Session("127.0.0.1", 6667, "root", "root");
        session.open(false);

        // Set it big to avoid multiple fetch, which is very important.
        // Because the IOMonitor implemented in IoTDB does not cover the fetchResults operator yet.
        // As M4 already does data reduction, so even the w is very big such as 8000, the returned
        // query result size is no more than 8000*4=32000.
        session.setFetchSize(20000000);

        long c = 0;
        long startTime = System.nanoTime();
        SessionDataSet dataSet = session.executeQueryStatement(sql);
        DataIterator ite = dataSet.iterator();
        while (ite.next()) { // this way avoid constructing rowRecord
            c++;
        }
        long elapsedTimeNanoSec = System.nanoTime() - startTime;
        System.out.println("[1-ns]ClientElapsedTime," + elapsedTimeNanoSec);
        System.out.println("number of result points:" + c);

        dataSet.closeOperationHandle();
        session.close();
    }
}
