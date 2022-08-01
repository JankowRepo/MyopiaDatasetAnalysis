import pandas as pd
import numpy as np

df = pd.read_excel('1. Data3.xlsx')

height=df.groupby(["gender", "birthday", "examination date"])["height"].mean()
weight=df.groupby(["gender", "birthday", "examination date"])["weight"].mean()
SRofRightEye=df.groupby(["gender", "birthday", "examination date"])["spherical refraction of right eye"].mean()
SRofLeftEye=df.groupby(["gender", "birthday", "examination date"])["spherical refraction of left eye"].mean()
CRofRightEye=df.groupby(["gender", "birthday", "examination date"])["cylinder refraction of right eye"].mean()
CRofLeftEye=df.groupby(["gender", "birthday", "examination date"])["cylinder refraction of left eye"].mean()
leftEyeAL = df.groupby(["gender", "birthday", "examination date"])["AL of left eye"].mean()
rightEyeAL = df.groupby(["gender", "birthday", "examination date"])["AL of right eye"].mean()

dateOfBi = [2006, 2007, 2008]
dateOfExamine = [2014, 2015, 2016, 2017]
gender = ["female", "male"]
z = 0

for k in gender:
    for i in dateOfBi:
        for j in dateOfExamine:
            df["height"].loc[
                (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)] = \
                df["height"].loc[
                    (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)].replace(np.nan, round(
                    height[z], 0))
            df["weight"].loc[
                (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)] = \
                df["weight"].loc[
                    (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)].replace(np.nan, round(
                    weight[z], 0))
            df["spherical refraction of right eye"].loc[
                (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)] = \
                df["spherical refraction of right eye"].loc[
                    (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)].replace(np.nan, round(
                    SRofRightEye[z], 0))
            df["spherical refraction of left eye"].loc[
                (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)] = \
                df["spherical refraction of left eye"].loc[
                    (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)].replace(np.nan, round(
                    SRofLeftEye[z], 0))
            df["cylinder refraction of right eye"].loc[
                (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)] = \
                df["cylinder refraction of right eye"].loc[
                    (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)].replace(np.nan, round(
                    CRofRightEye[z], 0))
            df["cylinder refraction of left eye"].loc[
                (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)] = \
                df["cylinder refraction of left eye"].loc[
                    (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)].replace(np.nan, round(
                    CRofLeftEye[z], 0))
            df["AL of right eye"].loc[
                (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)] = \
                df["AL of right eye"].loc[
                    (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)].replace(np.nan, round(
                    rightEyeAL[z], 2))
            df["AL of left eye"].loc[
                (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)] = df["AL of left eye"].loc[
                (df["gender"] == k) & (df["birthday"] == i) & (df["examination date"] == j)].replace(np.nan,
                                                                                                     round(leftEyeAL[z],
                                                                                                           2))


df["axis of right eye"]=df["axis of right eye"].replace("未知", 0.5)
df["axis of left eye"].fillna(0, inplace=True)
df["axis of right eye"].fillna(0, inplace=True)

# df.fillna(df.mean(), inplace=True)
df.to_excel(r'2. Filled_Raw_Data3.xlsx', index=False)
