import random
import copy
import pandas as pd
import pandas.core.frame
import numpy as np

def Prepare_additional_columns(df):
    max_ID = df["ID"].max()
    right_eye_0 = df.loc[:max_ID - 1]["spherical refraction of right eye"]
    left_eye_0 = df.loc[:max_ID - 1]["spherical refraction of left eye"]

    right_eye_1 = df.loc[max_ID:max_ID * 2 - 1]["spherical refraction of right eye"]
    left_eye_1 = df.loc[max_ID:max_ID * 2 - 1]["spherical refraction of left eye"]

    right_eye_2 = df.loc[max_ID * 2:max_ID * 3 - 1]["spherical refraction of right eye"]
    left_eye_2 = df.loc[max_ID * 2:max_ID * 3 - 1]["spherical refraction of left eye"]

    right_eye_3 = df.loc[max_ID * 3:max_ID * 4 + 2]["spherical refraction of right eye"]
    left_eye_3 = df.loc[max_ID * 3:max_ID * 4 + 2]["spherical refraction of left eye"]

    sphr_as_risk = -0.5

    One_year_later = np.zeros(len(df.index))
    Two_years_later = np.zeros(len(df.index))
    Three_years_later = np.zeros(len(df.index))
    Final_decision = np.zeros(len(df.index))
    for i in range(max_ID):
        One_year_later[i] = 1 if (right_eye_1.loc[i + max_ID] - right_eye_0.loc[i] <= sphr_as_risk) or (
                left_eye_1.loc[i + max_ID] - left_eye_0.loc[i] <= sphr_as_risk) and right_eye_1.loc[
                                     i + max_ID] <= -0.12 and \
                                 left_eye_1.loc[i + max_ID] <= -0.12 else 0

        Two_years_later[i] = 1 if (right_eye_2.loc[i + max_ID * 2] - right_eye_0.loc[i] <= sphr_as_risk) or (
                left_eye_2.loc[i + max_ID * 2] - left_eye_0.loc[i] <= sphr_as_risk) and right_eye_2.loc[
                                      i + max_ID * 2] <= -0.12 and left_eye_2.loc[i + max_ID * 2] <= -0.12 else 0

        Three_years_later[i] = 1 if (right_eye_3.loc[i + max_ID * 3] - right_eye_0.loc[i] <= sphr_as_risk) or (
                left_eye_3.loc[i + max_ID * 3] - left_eye_0.loc[i] <= sphr_as_risk) and right_eye_3.loc[
                                        i + max_ID * 3] <= -0.12 and left_eye_3.loc[i + max_ID * 3] <= -0.12 else 0

        Final_decision[i] = 1 if One_year_later[i] + Two_years_later[i] + Three_years_later[i] >= 1 else 0

    df["Degradation of sight one year later"] = One_year_later
    df["Degradation of sight two years later"] = Two_years_later
    df["Degradation of sight three years later"] = Three_years_later
    df["Degradation of sight final decision"] = Final_decision
    return df


def Normalize_data(df):
    df['gender'] = np.where(df['gender'] == "male", 1, 0)
    max_ID = df["ID"].max()
    columns_to_norm = df.iloc[:max_ID, 1:len(df.columns) - 4]
    columns_to_norm = columns_to_norm - columns_to_norm.min()
    columns_to_norm = (columns_to_norm - columns_to_norm.min()) / (columns_to_norm.max() + 0.0000000001)
    df.iloc[:max_ID, 1:len(df.columns) - 4] = columns_to_norm
    df = df[:max_ID]
    return df


class Patient:
    def __init__(self, info):
        self.info = info
        self.choosenFuturePatient = []
        self.choosenListOfPossibleFutures = pandas.core.frame.DataFrame


def PrepareDataset(listsOfPatients, counter):
    final_df = listsOfPatients[0].choosenFuturePatient[0].to_frame().T
    final_df = final_df[final_df["ID"] != listsOfPatients[0].choosenFuturePatient[0]["ID"]]

    for i in range(1, counter, 1):
        for j in range(4):
            listsOfPatients[i].choosenFuturePatient[j]["ID"] = i

    for j in range(4):
        for i in range(1, counter, 1):
            final_df = pd.concat([final_df, listsOfPatients[i].choosenFuturePatient[j].to_frame().T])

    return final_df


def FindSimilarPatients(patient):
    similar_patients = df.where((df["gender"] == patient.info["gender"])
                                & (df["birthday"] == patient.info["birthday"])
                                & (df["examination date"] == patient.info["examination date"] + 1)
                                & (df["height"] > patient.info["height"] + 1.5)
                                & (df["height"] < patient.info["height"] + 15)
                                & (df["weight"] > patient.info["weight"] - 3)
                                & (df["weight"] < patient.info["weight"] + 12)
                                & (df["spherical refraction of left eye"] <= patient.info[
        "spherical refraction of left eye"])
                                & (df["spherical refraction of left eye"] > patient.info[
        "spherical refraction of left eye"] - 1.7)
                                & (df["spherical refraction of right eye"] <= patient.info[
        "spherical refraction of right eye"])
                                & (df["spherical refraction of right eye"] > patient.info[
        "spherical refraction of right eye"] - 1.7)
                                & (df["AL of right eye"] >= patient.info["AL of right eye"])
                                & (df["AL of right eye"] < patient.info["AL of right eye"] + 0.7)
                                & (df["AL of left eye"] >= patient.info["AL of left eye"])
                                & (df["AL of left eye"] < patient.info["AL of left eye"] + 0.7)
                                ).dropna()
    patient.choosenListOfPossibleFutures = []
    patient.choosenListOfPossibleFutures.append(similar_patients)
    return patient


def DeleteUsedPatients(df, table):
    rowsToDrop = []
    for i in range(len(table.index)):
        if int(table.iloc[i]["ID"]) not in df["ID"].values:
            rowsToDrop.append(table.iloc[i]["ID"])
    for rowID in rowsToDrop:
        table = table[table["ID"] != rowID]
    return table


def AnotherMethod(df, patient):
    patient.choosenListOfPossibleFutures[0] = DeleteUsedPatients(df, patient.choosenListOfPossibleFutures[0])
    if len(patient.choosenListOfPossibleFutures[0].index) == 0:
        return df, patient
    choosen = patient.choosenListOfPossibleFutures[0].iloc[
        random.randrange(len(patient.choosenListOfPossibleFutures[0].index))]
    patient.choosenFuturePatient.append(choosen)
    patient.info = choosen
    patient = FindSimilarPatients(patient)
    df = df[df["ID"] != patient.info["ID"]]
    return df, patient


def FindAllPatientsHistory(df, mainCounter, fileToSave):
    listsOfPatients = []
    for i in range(513):
        newPatient = Patient(df.iloc[i])
        newPatient.choosenFuturePatient.append(df.iloc[i])
        newPatient = FindSimilarPatients(newPatient)
        if len(newPatient.choosenListOfPossibleFutures[0].index) != 0:
            pat=FindSimilarPatients(newPatient)
            listsOfPatients.append(pat)
    print(1," year. ",len(listsOfPatients)," patients left")

    for j in range(3):
        listsOfPatients.sort(key=lambda x: len(x.choosenListOfPossibleFutures[0].index))
        newListsOfPatients = []
        for i in range(len(listsOfPatients)):
            df, patient = AnotherMethod(df, listsOfPatients[i])
            if len(patient.choosenListOfPossibleFutures[0].index) > 0:
                newListsOfPatients.append(patient)
            if len(patient.choosenFuturePatient) == 4:
                newListsOfPatients.append(patient)
        listsOfPatients = newListsOfPatients
        print(j+2," year. ",len(listsOfPatients)," patients left")
    print("Return ",len(listsOfPatients)," patients")
    if len(listsOfPatients) > mainCounter:
        return len(listsOfPatients), PrepareDataset(listsOfPatients, len(listsOfPatients))
    return mainCounter, fileToSave


df = pd.read_excel(r"2. Filled_Raw_Data3.xlsx")
mainCounter = 0
fileToSave = 0
for i in range(5):
    mainCounter, df = FindAllPatientsHistory(df, mainCounter, fileToSave)

print(mainCounter)

df.to_excel(r'3. PatientsFound_Data3.xlsx', index=False)


df = Prepare_additional_columns(df)
df = Normalize_data(df)
df.to_excel(r'4. Normalized_Data3.xlsx', index=False)



