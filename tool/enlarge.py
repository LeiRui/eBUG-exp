import pandas as pd
import os

# folder_path = 'D:\\datasets\\starLightCurve_enlarge'
folder_path = '/root/starLightCurve_enlarge' # copy starLightCurve.csv from UCRsets-single twice into this directory
print("make sure starLightCurve_enlarge directory only contain two copied starLightCurve csv!")

excel_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

combined_series = pd.DataFrame()
global_max_time = None

for file in excel_files:
    file_path = os.path.join(folder_path, file)
    print(f"{file}")

    df = pd.read_csv(file_path,header=None)

    timestamps = df.iloc[:,0]
    values = df.iloc[:,1]

    if global_max_time is not None:
        time_offset = global_max_time + 1 - timestamps.min()
        timestamps += time_offset

    global_max_time = timestamps.max()

    temp_df = pd.DataFrame({'timestamp': timestamps, 'value': values})
    combined_series = pd.concat([combined_series, temp_df], ignore_index=True)

# combined_series.sort_values('timestamp', inplace=True)

output_path = os.path.join(folder_path, 'StarLightCurves_enlarge.csv')
combined_series.to_csv(output_path, index=False, header=False)
print(f"saved to: {output_path}")

print("\n concatenated length: ") # 16867328
print(len(combined_series))