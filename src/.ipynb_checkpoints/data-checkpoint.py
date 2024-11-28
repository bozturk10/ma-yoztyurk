import os
import json
import os
import json
import pandas as pd
import re

def read_stata_data(file):
    df=pd.read_stata(file, convert_categoricals=False) 
    wave_id = re.search(r'_w(.*?)_', file)
    if wave_id:
        wave_id = wave_id.group(1)
    #df['filename']=file
    df['wave_id']=wave_id

    return df