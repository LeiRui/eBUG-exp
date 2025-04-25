package org.apache.iotdb.jarCode;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.iotdb.rpc.IoTDBConnectionException;
import org.apache.iotdb.rpc.StatementExecutionException;
import org.apache.iotdb.session.Session;
import org.apache.iotdb.tsfile.file.metadata.enums.TSDataType;
import org.apache.iotdb.tsfile.file.metadata.enums.TSEncoding;
import org.apache.iotdb.tsfile.write.record.Tablet;
import org.apache.iotdb.tsfile.write.schema.MeasurementSchema;

public class WritePrecomputedTable {

    /**
     * Before writing data, make sure check the server parameter configurations.
     */
    public static void main(String[] args)
            throws IoTDBConnectionException, StatementExecutionException, IOException {
        int argIdx = 0;
        String device = args[argIdx++];
        System.out.println("[WriteData] device=" + device);
        // device default to be three levels such as "root.HouseTwenty.targetDevice"
        // the first two levels form storage group, while the last level is device
        int dotCount = 0;
        for (int i = 0; i < device.length(); i++) {
            if (device.charAt(i) == '.') {
                dotCount++;
            }
        }
        if (dotCount != 2) { // not three levels
            throw new IOException("wrong device!");
        }

//    String measurement = args[1];
//    System.out.println("[WriteData] measurement=" + measurement);

        String timestamp_precision = args[argIdx++]; // ns, us, ms
        System.out.println("[WriteData] timestamp_precision=" + timestamp_precision);
        if (!timestamp_precision.toLowerCase().equals("ns") && !timestamp_precision.toLowerCase()
                .equals("us") && !timestamp_precision.toLowerCase().equals("ms")) {
            throw new IOException("timestamp_precision only accepts ns,us,ms.");
        }

//        String dataType = args[argIdx++]; // double
//        System.out.println("[WriteData] dataType=" + dataType);
//        TSDataType tsDataType;
//        if (dataType.toLowerCase().equals("double")) {
//            tsDataType = TSDataType.DOUBLE;
//        } else {
//            throw new IOException("Data type only accepts double right now.");
//        }

//        // value encoder
//        String valueEncoding = args[argIdx++]; // RLE, GORILLA, PLAIN
//        System.out.println("[WriteData] valueEncoding=" + valueEncoding);

        int iotdb_chunk_point_size = Integer.parseInt(args[argIdx++]);
        System.out.println("[WriteData] iotdb_chunk_point_size=" + iotdb_chunk_point_size);

        String filePath = args[argIdx++];
        System.out.println("[WriteData] filePath=" + filePath);

//    int cntOther = 20;
//    if (args.length >= 8) {
//      cntOther = Integer.parseInt(args[7]);
//    }
//    System.out.println("[WriteData] cntOther=" + cntOther);

        int maxPointWritten = -1;
        maxPointWritten = Integer.parseInt(args[argIdx++]);
        System.out.println("[WriteData] maxPointWritten=" + maxPointWritten);

//        String otherMeasurement = "otherSensor";

        boolean hasHeader = Boolean.parseBoolean(args[argIdx++]);
        System.out.println("[WriteData] hasHeader=" + hasHeader);

        boolean precompute = Boolean.parseBoolean(args[argIdx++]); // true to write id,x,y, false to write t,v
        System.out.println("[WriteData] precompute=" + precompute);

        Session session = new Session("127.0.0.1", 6667, "root", "root");
        session.open(false);

        String[] measurements = {
                "pre_t", "pre_v", "s1"
        };

        if (precompute) {
            String createSql = String.format("CREATE TIMESERIES %s.%s WITH DATATYPE=%s, ENCODING=%s",
                    device,
                    measurements[0],
                    TSDataType.DOUBLE,
//                    TSDataType.INT64,
                    "PLAIN"
//                    "TS_2DIFF"
            );
            session.executeNonQueryStatement(createSql);

            createSql = String.format("CREATE TIMESERIES %s.%s WITH DATATYPE=%s, ENCODING=%s",
                    device,
                    measurements[1],
                    TSDataType.DOUBLE, "PLAIN"
//                    "GORILLA"
            );
            session.executeNonQueryStatement(createSql);
        } else {
            String createSql = String.format("CREATE TIMESERIES %s.%s WITH DATATYPE=%s, ENCODING=%s",
                    device,
                    measurements[2],
                    TSDataType.DOUBLE, "PLAIN"
//                    "GORILLA"
            );
            session.executeNonQueryStatement(createSql);
        }

//        // device default to be three levels such as "root.HouseTwenty.targetDevice"
//        String storageGroup = device.substring(0, device.lastIndexOf('.'));
//        String otherDevice = storageGroup + ".otherDevice"; // same storage group but different devices
//        for (int i = 1; i <= cntOther; i++) { // note sensor name start from 1
//            String createOtherSql = String.format(
//                    "CREATE TIMESERIES %s%d.%s WITH DATATYPE=%s, ENCODING=%s",
//                    otherDevice, i,   // same storage group but different devices
//                    otherMeasurement,  // sensor name
//                    tsDataType,
//                    valueEncoding
//            );
//            session.executeNonQueryStatement(createOtherSql);
//        }

//    // this is to make all following inserts unseq chunks
//    if (timestamp_precision.toLowerCase().equals("ns")) {
//      session.insertRecord(
//          device,
//          1683616109697000000L, // ns
//          // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
//          Collections.singletonList(measurement),
//          Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
//          parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
//      if (cntOther > 0) {
//        for (int i = 1; i <= cntOther; i++) {
//          session.insertRecord(
//              otherDevice + i,
//              1683616109697000000L, // ns
//              // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
//              Collections.singletonList(otherMeasurement),
//              Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
//              parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
//        }
//      }
//    } else if (timestamp_precision.toLowerCase().equals("us")) {
//      session.insertRecord(
//          device,
//          1683616109697000L, // us
//          // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
//          Collections.singletonList(measurement),
//          Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
//          parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
//      if (cntOther > 0) {
//        for (int i = 1; i <= cntOther; i++) {
//          session.insertRecord(
//              otherDevice + i,
//              1683616109697000L, // us
//              // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
//              Collections.singletonList(otherMeasurement),
//              Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
//              parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
//        }
//      }
//    } else { // ms
//      session.insertRecord(
//          device,
//          1683616109697L, // ms
//          // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
//          Collections.singletonList(measurement),
//          Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
//          parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
//      if (cntOther > 0) {
//        for (int i = 1; i <= cntOther; i++) {
//          session.insertRecord(
//              otherDevice + i,
//              1683616109697L, // ms
//              // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
//              Collections.singletonList(otherMeasurement),
//              Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
//              parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
//        }
//      }
//    }
//    session.executeNonQueryStatement("flush");

        List<MeasurementSchema> schemaList = new ArrayList<>();
        if (precompute) {
//            schemaList.add(
//                    new MeasurementSchema(measurements[0], TSDataType.INT64, TSEncoding.valueOf("TS_2DIFF")));
//            schemaList.add(
//                    new MeasurementSchema(measurements[1], TSDataType.DOUBLE, TSEncoding.valueOf("GORILLA")));
            schemaList.add(
                    new MeasurementSchema(measurements[0], TSDataType.DOUBLE, TSEncoding.valueOf("PLAIN")));
            schemaList.add(
                    new MeasurementSchema(measurements[1], TSDataType.DOUBLE, TSEncoding.valueOf("PLAIN")));
        } else {
//            schemaList.add(
//                    new MeasurementSchema(measurements[2], TSDataType.DOUBLE, TSEncoding.valueOf("GORILLA")));
            schemaList.add(
                    new MeasurementSchema(measurements[2], TSDataType.DOUBLE, TSEncoding.valueOf("PLAIN")));
        }


//        List<MeasurementSchema> otherSchemaList = new ArrayList<>();
//        otherSchemaList.add(
//                new MeasurementSchema(otherMeasurement, tsDataType, TSEncoding.valueOf(valueEncoding)));

        Tablet tablet = new Tablet(device, schemaList, iotdb_chunk_point_size);
        long[] timestamps = tablet.timestamps;
        Object[] values = tablet.values;

//        Tablet otherTablet = new Tablet("tmp", otherSchemaList, iotdb_chunk_point_size);
//        long[] otherTimestamps = otherTablet.timestamps;
//        Object[] otherValues = otherTablet.values;

        File f = new File(filePath);
        String line = null;
        BufferedReader reader = new BufferedReader(new FileReader(f));
        if (hasHeader) {
            line = reader.readLine(); // has header id,x,y,z
        }
        long globalCnt = 0;
        while (((line = reader.readLine()) != null) && (maxPointWritten < 0
                || globalCnt < maxPointWritten)) {
            globalCnt++;

            String[] split = line.split(",");

            int row = tablet.rowSize++;

            if (precompute) {
                int idx = Integer.parseInt(split[0]);
                double pre_t = Double.parseDouble(split[1]);
                double pre_v = Double.parseDouble(split[2]);

                timestamps[row] = idx;

                // pre_t
                double[] long_sensor = (double[]) values[0];
                long_sensor[row] = pre_t;
                // pre_v
                double[] double_sensor = (double[]) values[1];
                double_sensor[row] = pre_v;
            } else {
                long time = (long) Double.parseDouble(split[0]);
                double value = Double.parseDouble(split[1]);

                timestamps[row] = time;

                double[] double_sensor = (double[]) values[0];
                double_sensor[row] = value;
            }

            if (tablet.rowSize == tablet.getMaxRowNumber()) { // chunk point size
                session.insertTablet(tablet, false);
                tablet.reset();
            }

//            if (cntOther > 0) {
//                row = otherTablet.rowSize++; // note ++
//                otherTimestamps[row] = globalTimestamp;
//                double[] other_double_sensor = (double[]) otherValues[0];
//                other_double_sensor[row] = double_value;
//                if (otherTablet.rowSize == otherTablet.getMaxRowNumber()) { // chunk point size
//                    for (int i = 1; i <= cntOther; i++) {
//                        otherTablet.deviceId = otherDevice + i;
//                        session.insertTablet(otherTablet, false);
//                    }
//                    otherTablet.reset();
//                }
//            }

        }

        // flush the last Tablet
        if (tablet.rowSize != 0) {
            session.insertTablet(tablet, false);
            tablet.reset();
        }
//        if (cntOther > 0) {
//            if (otherTablet.rowSize != 0) {
//                for (int i = 1; i <= cntOther; i++) {
//                    otherTablet.deviceId = otherDevice + i;
//                    session.insertTablet(otherTablet, false);
//                }
//                otherTablet.reset();
//            }
//        }
        session.executeNonQueryStatement("flush");
        session.close();
    }

    public static Object parseValue(String value, TSDataType tsDataType) throws IOException {
        if (tsDataType == TSDataType.DOUBLE) {
            return Double.parseDouble(value);
        } else {
            throw new IOException("data type wrong");
        }
    }
}
