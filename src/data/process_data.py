import pandas as pd
import numpy as np
import os
from src.data.read_data import load_lookup_data, load_raw_survey_data
from src.paths import (
    CODING_DIR,
    PROJECT_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROMPT_DIR,
    GLES_DIR,
)
from sklearn.preprocessing import MultiLabelBinarizer


coding_list_dict = load_lookup_data(
    os.path.join(CODING_DIR, "first_most_imp_coding_list.json")
)
leaning_party_dict = load_lookup_data(
    os.path.join(CODING_DIR, "leaning_party_dict.json")
)
gender_dict = load_lookup_data(os.path.join(CODING_DIR, "gender_dict.json"))
ostwest_dict = load_lookup_data(os.path.join(CODING_DIR, "ostwest_dict.json"))
schulabschluss_dict = load_lookup_data(
    os.path.join(CODING_DIR, "schulabschluss_dict.json")
)
berufabschluss_lookup = pd.read_csv(
    os.path.join(CODING_DIR, "berufabschluss_lookup.csv")
)
berufabschluss_dict = load_lookup_data(
    os.path.join(CODING_DIR, "berufabschluss_dict.json")
)


df_2320 = pd.read_csv(os.path.join(CODING_DIR, "df_2320.csv"))
df_2330 = pd.read_csv(os.path.join(CODING_DIR, "df_2330.csv"))

edu_lookup = pd.read_csv(os.path.join(CODING_DIR, "education_lookup.csv"))
month_names_dict = load_lookup_data(os.path.join(CODING_DIR, "month_names_german.json"))


def process_2320_data(dfs_dict):
    """
    assigns education level to each respondent based on the education level variables in the dataframes
    """
    a2_2320 = dfs_dict["a2"].filter(regex="lfdn$|kp(.*?)_2320", axis=1)
    w21_2320 = dfs_dict["21"].filter(regex="lfdn$|kp(.*?)_2320", axis=1)
    w1to9_2320 = dfs_dict["1to9"].filter(regex="lfdn$|kp(.*?)_2320", axis=1)
    all_2320 = pd.merge(w1to9_2320, w21_2320, on="lfdn", how="outer").merge(
        a2_2320, on="lfdn", how="outer"
    )

    for col in all_2320.filter(regex="kp(.*?)_2320", axis=1).columns:
        all_2320.loc[all_2320[col] < 0, col] = np.nan

    all_2320.sort_values(
        by=["kp1_2320", "kpa1_2320", "kp9_2320", "kp21_2320", "kpa2_2320"]
    )

    all_2320["code_2320"] = (
        all_2320["kp1_2320"]
        .combine_first(all_2320["kpa1_2320"])
        .combine_first(all_2320["kp9_2320"])
        .combine_first(all_2320["kpa2_2320"])
        .combine_first(all_2320["kp21_2320"])
    )

    all_2320["source_combined"] = np.nan
    all_2320.loc[(all_2320["kp1_2320"].notna()), "source_combined"] = "kp1_2320"
    all_2320.loc[(all_2320["kpa1_2320"].notna()), "source_combined"] = "kpa1_2320"
    all_2320.loc[(all_2320["kp9_2320"].notna()), "source_combined"] = "kp9_2320"
    all_2320.loc[(all_2320["kpa2_2320"].notna()), "source_combined"] = "kpa2_2320"
    all_2320.loc[(all_2320["kp21_2320"].notna()), "source_combined"] = "kp21_2320"

    return all_2320


def process_2330_data(dfs_dict):
    """
    assigns education level (berufabsc) to each respondent based on the education level variables in the dataframes
    """
    dfa2 = dfs_dict["a2"]
    df1to9 = dfs_dict["1to9"]

    w1to9_2330 = df1to9.filter(regex="lfdn$|kp(.*?)_2330", axis=1).astype(int)
    wa2_2330 = dfa2.filter(regex="lfdn$|kp(.*?)_2330", axis=1).astype(int)
    all_2330 = pd.merge(w1to9_2330, wa2_2330, on="lfdn", how="outer")

    for col in all_2330.filter(regex="kp(.*?)_2330", axis=1).columns:
        all_2330.loc[all_2330[col] < 0, col] = np.nan

    all_2330["code_2330"] = (
        all_2330["kp1_2330"]
        .combine_first(all_2330["kpa1_2330"])
        .combine_first(all_2330["kpa2_2330"])
    )

    all_2330["source_combined"] = "source"
    all_2330.loc[all_2330["kp1_2330"].notna(), "source_combined"] = "kp1_2330"
    all_2330.loc[all_2330["kpa1_2330"].notna(), "source_combined"] = "kpa1_2330"
    all_2330.loc[all_2330["kpa2_2330"].notna(), "source_combined"] = "kpa2_2330"

    return all_2330


def get_education_lookup(dfs_dict):
    """
    returns a lookup table with the schulabschluss and berufabschluss of each respondent
    """

    all_2320 = process_2320_data(dfs_dict)
    all_2330 = process_2330_data(dfs_dict)
    edu_lookup_2320_2330 = pd.merge(
        all_2320[["lfdn", "code_2320"]], all_2330[["lfdn", "code_2330"]], on="lfdn"
    )

    return edu_lookup_2320_2330


def save_edu_lookup(dfs_dict):
    edu_lookup_2320_2330 = get_education_lookup(dfs_dict)
    edu_lookup_2320_2330.to_csv(
        os.path.join(CODING_DIR, "education_lookup.csv"), index=False
    )

def process_open_ended_new(wave_open_ended_df, df_coding_840s, wave_number):
    #oe_answer_col = f"kp{wave_number}_840s"
    #oe_class_col = f"kp{wave_number}_840_c1"

    i=wave_number
    regexstr=f"lfdn|kp{i}_840_c1|kp{i}_840_c2|kp{i}_840_c3|kp{i}_840s"
    wave_i_df=df_coding_840s.filter(regex=regexstr, axis=1).dropna().rename(columns=lambda x: x.replace(f"kp{i}_840", "kpx_840")).reset_index(drop=True)
    wave_i_df['wave']=i

    wave_coding_df=wave_i_df
    wave_coding_df = wave_coding_df[(wave_coding_df.kpx_840_c1.ge(0)) | (wave_coding_df.kpx_840_c1.isin([-99, -98]))]
    wave_coding_df.kpx_840_c2 = wave_coding_df.kpx_840_c2.mask(wave_coding_df.kpx_840_c2 < 0, 0).astype(int) 
    wave_coding_df.kpx_840_c3 = wave_coding_df.kpx_840_c3.mask(wave_coding_df.kpx_840_c3 < 0, 0).astype(int) 
    wave_coding_df.kpx_840_c1 = wave_coding_df.kpx_840_c1.astype(int) 

    df= pd.read_csv(os.path.join(CODING_DIR,'map.csv'))
    lookup= dict(zip(df.subclassid,df.upperclass_id))
    for col in wave_coding_df.filter(like='kpx_840_c').columns:
        wave_coding_df[col] = wave_coding_df[col].map(lookup)
    
    
    labels_list = wave_coding_df.filter(regex='kpx_840_c1|kpx_840_c2|kpx_840_c3').apply(lambda x: list(x[x.notna()].astype(int)), axis=1)
    classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).upperclass_id.unique())) }    

    wave_coding_df["highest_prob_label"]=wave_coding_df['kpx_840_c1']
    wave_coding_df['labels_list']= labels_list
    #convert labels to binarized format
    wave_coding_df = wave_coding_df.rename(columns={'kpx_840s':'text'}) #kpx_840s
    classes = list(classid2trainid.keys())
    mlb = MultiLabelBinarizer(classes=classes)
    sparse_matrix = mlb.fit_transform(labels_list).astype(float).tolist()   
    wave_coding_df['labels'] = sparse_matrix


    wave_open_ended_df_merged = pd.merge(
        wave_coding_df,
        wave_open_ended_df[["lfdn", f"kp{wave_number}_840s"]],
        left_on="lfdn_od", #user id in the original data
        right_on="lfdn",
        how="left",
    )
    wave_open_ended_df_merged= wave_open_ended_df_merged.dropna(subset=[f"kp{wave_number}_840s"])
    wave_open_ended_df_merged = wave_open_ended_df_merged.drop(['lfdn_x','lfdn_y','text'], axis=1)
    wave_open_ended_df_merged = wave_open_ended_df_merged.rename(
        {
         f"kp{wave_number}_840s": "text",
         'lfdn_od':'lfdn'}, axis=1
    )
    return wave_open_ended_df_merged

def process_open_ended(wave_open_ended_df, df_coding_840s, wave_number):
    oe_answer_col = f"kp{wave_number}_840s"
    oe_class_col = f"kp{wave_number}_840_c1"
    wave_open_ended_df = wave_open_ended_df[["lfdn", oe_answer_col]].rename(
        {oe_answer_col: "kpx_840_text"}, axis=1
    )
    wave_open_ended_df_merged = pd.merge(
        wave_open_ended_df[["lfdn", "kpx_840_text"]],
        df_coding_840s.filter(regex="lfdn_od|" + oe_class_col),
        left_on="lfdn",
        right_on="lfdn_od",
        how="inner",
    )
    wave_open_ended_df_merged["kpx_840_class1_name"]=wave_open_ended_df_merged[
        oe_class_col
    ].map(coding_list_dict)
    wave_open_ended_df_merged = wave_open_ended_df_merged.drop("lfdn_od", axis=1)
    wave_open_ended_df_merged = wave_open_ended_df_merged.rename(
        {
         oe_class_col: "kpx_840_cid"}, axis=1
    ) # rename columns to match format for different waves 

    return wave_open_ended_df_merged

def process_wave_data_old(wave_df, wave_open_ended_df_merged, wave_number):
    wave_df = pd.merge(
        wave_df, wave_open_ended_df_merged, left_on="lfdn", right_on="lfdn"
    )
    wave_df = wave_df.merge(edu_lookup, on="lfdn", how="left")
    wave_df = wave_df.merge(berufabschluss_lookup, on="code_2330", how="left")

    wave_df["leaning_party"] = wave_df[f"kp{wave_number}_2090a"].apply(
        lambda x: leaning_party_dict[x] if x in leaning_party_dict else x
    )
    wave_df["gender"] = wave_df["kpx_2280"].map(gender_dict)
    wave_df["age"] = pd.to_datetime(wave_df.field_start.iloc[0]).year - wave_df[
        "kpx_2290s"
    ].str.extract(r"(\d+)").astype(float)
    wave_df["age_group"] = pd.cut(
        wave_df["age"],
        bins=[18, 30, 45, 60, float("inf")],
        labels=["18-29 Years", "30-44 Years", "45-59 Years", "60 Years and Older"],
    )

    wave_df = wave_df[wave_df["age"].notna()]  # drop if age is nan
    wave_df.age = wave_df.age.astype(int)

    wave_df.ostwest = wave_df.ostwest.map(ostwest_dict)
    wave_df = wave_df[
        wave_df["ostwest"].str.contains("-") == False
    ]  # filter ostwest is empty
    wave_df = wave_df[
        wave_df["leaning_party"].str.contains("-") == False
    ]  # filter leaning_party is empty
    wave_df = wave_df[wave_df["code_2330"].notna()]
    wave_df = wave_df[wave_df["code_2320"].notna()]

    wave_df["schulabschluss_clause"] = wave_df["code_2320"].map(schulabschluss_dict)
    return wave_df

def process_wave_data(wave_df, wave_open_ended_df_merged, wave_number):
    wave_df = pd.merge(
        wave_df, wave_open_ended_df_merged, on="lfdn"
    )
    df_2320_lookup= df_2320[['lfdn', f'kp{wave_number}_2320']].rename({f'kp{wave_number}_2320':'code_2320'},axis=1) # schulabschluss values for wave_number
    df_2330_lookup= df_2330[['lfdn', f'kp{wave_number}_2330']].rename({f'kp{wave_number}_2330':'code_2330'},axis=1) # berufabschluss values for wave_number
    wave_df = wave_df.merge(df_2320_lookup, on="lfdn", how="left")
    wave_df = wave_df.merge(df_2330_lookup, on="lfdn", how="left")

    wave_df["leaning_party"] = wave_df[f"kp{wave_number}_2090a"].apply(
        lambda x: leaning_party_dict[x] if x in leaning_party_dict else x
    )
    wave_df["gender"] = wave_df["kpx_2280"].map(gender_dict)
    wave_df["age"] = pd.to_datetime(wave_df.field_start.iloc[0]).year - wave_df[
        "kpx_2290s"
    ].str.extract(r"(\d+)").astype(float)
    wave_df["age_group"] = pd.cut(
        wave_df["age"],
        bins=[18, 30, 45, 60, float("inf")],
        labels=["18-29 Years", "30-44 Years", "45-59 Years", "60 Years and Older"],
    )

    wave_df = wave_df[wave_df["age"].notna()]  # drop if age is nan
    wave_df.age = wave_df.age.astype(int)

    wave_df.ostwest = wave_df.ostwest.map(ostwest_dict)
    wave_df = wave_df[
        wave_df["ostwest"].str.contains("-") == False
    ]  # filter ostwest is empty
    wave_df = wave_df[
        wave_df["leaning_party"].str.contains("-") == False
    ]  # filter leaning_party is empty
    wave_df = wave_df[wave_df["code_2330"].notna()]
    wave_df = wave_df[wave_df["code_2320"].notna()]

    wave_df["schulabschluss_clause"] = wave_df["code_2320"].map(schulabschluss_dict)
    wave_df["berufabschluss_clause"] = wave_df["code_2330"].map(berufabschluss_dict)
    return wave_df


def save_2320_2330_lookup(df_2320,df_2330):
    df_2320.to_csv(os.path.join(CODING_DIR, 'df_2320.csv') ,index=False)
    df_2330.to_csv(os.path.join(CODING_DIR, 'df_2330.csv') ,index=False)

def get_2320_2330_lookups(dfs_dict): 
    cols_2330 = ['kp1_2330',
     'kp2_2330',
     'kp3_2330',
     'kp4_2330',
     'kpa1_2330',
     'kp5_2330',
     'kp6_2330',
     'kp7_2330',
     'kp8_2330',
     'kp9_2330',
     'kp10_2330',
     'kp11_2330',
     'kp12_2330',
     'kp13_2330',
     'kp14_2330',
     'kpa2_2330',
     'kp15_2330',
     'kp16_2330',
     'kp17_2330',
     'kp18_2330',
     'kp19_2330',
     'kp20_2330',
     'kp21_2330']

    cols_2320 = ['kp1_2320',
     'kp2_2320',
     'kp3_2320',
     'kp4_2320',
     'kpa1_2320', 
     'kp5_2320',
     'kp6_2320',
     'kp7_2320',
     'kp8_2320',
     'kp9_2320',
     'kp10_2320',
     'kp11_2320',
     'kp12_2320',
     'kp13_2320',
     'kp14_2320',
     'kpa2_2320',
     'kp15_2320',
     'kp16_2320',
     'kp17_2320',
     'kp18_2320',
     'kp19_2320',
     'kp20_2320',
     'kp21_2320']

    lfdn_list= list(set([lfdn_value for df in dfs_dict.values() for lfdn_value in df['lfdn'].values]))
    df_2320 =pd.DataFrame(index=lfdn_list,columns=cols_2320)
    df_2330 =pd.DataFrame(index=lfdn_list,columns=cols_2330)
    import numpy as np

    for key in dfs_dict.keys():
        cols_2320_key = dfs_dict[key].filter(regex='2320', axis=1).columns
        # Loop through each column and assign values with condition
        for col in cols_2320_key:
            print(key, col)
            values = np.where(dfs_dict[key][col].values < 0, np.nan, dfs_dict[key][col].values)
            df_2320.loc[dfs_dict[key]['lfdn'], col] = values

        cols_2330_key = dfs_dict[key].filter(regex='2330', axis=1).columns
        # Loop through each column and assign values with condition
        for col in cols_2330_key:
            print(key, col)
            values = np.where(dfs_dict[key][col].values < 0, np.nan, dfs_dict[key][col].values)
            df_2330.loc[dfs_dict[key]['lfdn'], col] = values
    df_2320 = df_2320.ffill(axis=1).infer_objects(copy=False)
    df_2330 = df_2330.ffill(axis=1).infer_objects(copy=False)
    df_2320.insert(0, 'lfdn', df_2320.index)
    df_2330.insert(0, 'lfdn', df_2330.index)
    return df_2320,df_2330


if __name__ == "__main__":

    wave_df_dict={}
    for wave_id in range(12, 22):
        wave_df, wave_open_ended_df, df_coding_840s = load_raw_survey_data(wave_id)
        wave_df_dict[wave_id]=wave_df
    
    df_2320,df_2330 = get_2320_2330_lookups(wave_df_dict)