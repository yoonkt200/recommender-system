import math
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def age_discretize(x):
    if x == np.nan:
        return '10'
    else:
        x = int(x)
        if x < 10:
            return '1'
        elif x < 20 and x >= 10:
            return '2'
        elif x < 30 and x >= 20:
            return '3'
        elif x < 40 and x >= 30:
            return '4'
        elif x < 50 and x >= 40:
            return '5'
        elif x < 60 and x >= 50:
            return '6'
        elif x < 70 and x >= 60:
            return '7'
        elif x < 80 and x >= 70:
            return '8'
        elif x < 90 and x >= 80:
            return '9'
        else:
            return '10'

def fare_discretize(x):
    if x < 10:
        return '1'
    elif x < 20 and x >= 10:
        return '2'
    elif x < 30 and x >= 20:
        return '3'
    elif x < 40 and x >= 30:
        return '4'
    elif x < 50 and x >= 40:
        return '5'
    elif x < 60 and x >= 50:
        return '6'
    elif x < 70 and x >= 60:
        return '7'
    elif x < 80 and x >= 70:
        return '8'
    elif x < 90 and x >= 80:
        return '9'
    else:
        return '10'

df1_path = "./dataset/titanic_dataset.csv"
df2_path = "./dataset/titanic_answer.csv"
df1 = pd.read_csv(df1_path)
df2 = pd.read_csv(df2_path)
df = df1.append(df2)

df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
df = df.dropna()

df['is_female'] = df['sex'].apply(lambda x: '1' if x == "female" else '0')
df['age'] = df['age'].apply(lambda x: age_discretize(x))
df['fare'] = df['fare'].apply(lambda x: fare_discretize(int(x)))

df['survived'] = df['survived'].astype('str')
df['pclass'] = df['pclass'].astype('str')
df['sibsp'] = df['sibsp'].astype('str')
df['parch'] = df['parch'].astype('str')

for col in df.columns:
    print("---------------")
    print(col)
    print(df[col].unique())


def make_encoding_label_dict(col_unique):
    encoded_dict = {}
    for idx, unique in enumerate(col_unique):
        encoded_dict[unique] = idx + 1

    return encoded_dict


encoded_pclass_dict = make_encoding_label_dict(df.pclass.value_counts().index.tolist())
encoded_age_dict = make_encoding_label_dict(df.age.value_counts().index.tolist())
encoded_sibsp_dict = make_encoding_label_dict(df.sibsp.value_counts().index.tolist())
encoded_parch_dict = make_encoding_label_dict(df.parch.value_counts().index.tolist())
encoded_fare_dict = make_encoding_label_dict(df.fare.value_counts().index.tolist())

def get_newcode(key, label_dict):
    com_len = len(label_dict)
    if key == np.nan:
        return
    else:
        if key in label_dict:
            return label_dict[key]
        else:
            return com_len + 1

df['pclass'] = df['pclass'].apply(lambda x: get_newcode(x, encoded_pclass_dict))
df['age'] = df['age'].apply(lambda x: get_newcode(x, encoded_age_dict))
df['sibsp'] = df['sibsp'].apply(lambda x: get_newcode(x, encoded_sibsp_dict))
df['parch'] = df['parch'].apply(lambda x: get_newcode(x, encoded_parch_dict))
df['fare'] = df['fare'].apply(lambda x: get_newcode(x, encoded_fare_dict))


df_x = df[['pclass', 'is_female', 'age', 'sibsp', 'parch', 'fare']]
df_y = df['survived']
df_x = df_x.reset_index(drop=True)
df_y = df_y.reset_index(drop=True)

col_len_dict = {'pclass': 3, 'sex': 1, 'age': 9, 'sibsp': 7, 'parch': 7, 'fare': 10}
col_accum_index_dict = {}
cumulative = 0
for key, value in col_len_dict.items():
    col_accum_index_dict[key] = cumulative
    cumulative = cumulative + value

out = np.zeros((df_x.shape[0], sum(col_len_dict.values())), dtype=float)

for idx, row in df_x.iterrows():
    for key, value in row.items():
        col_idx = 0
        out_val = 0
        if col_len_dict[key] == 1:
            col_idx = col_accum_index_dict[key]
            out_val = value
        else:
            col_idx = col_accum_index_dict[key] + (int(value) - 1)
            out_val = 1
        out[idx, col_idx] = out_val


columns = []
for key, value in col_len_dict.items():
    for i in range(1, value+1):
        columns.append(str(key) + "_" + str(i))


df_out = pd.DataFrame(out, columns=columns)
