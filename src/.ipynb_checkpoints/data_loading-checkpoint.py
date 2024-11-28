
import os
import pandas as pd
import re

def load_stata_files(data_dir):
    """
    Load stata files from a specified directory.
    """
    stata_files = [f for f in os.listdir(data_dir) if f.endswith('.dta') and f.startswith('ZA6838_w')]
    data = {}
    for file in stata_files:
        try:
            wave = re.search(r'_w(\d+)', file).group(1)
            keyname = f"kp{wave}_840_c1"
            wave_date = pd.to_datetime(pd.read_stata(os.path.join(data_dir, file), convert_categoricals=False).field_start.iloc[0]).date()
            data[keyname] = wave_date
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return data

def load_csv_data(file_path, sep=',', encoding='utf-8'):
    """
    Load CSV data from a specified file path.
    """
    return pd.read_csv(file_path, sep=sep, encoding=encoding)
