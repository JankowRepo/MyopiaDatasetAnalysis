import random
import pandas as pd

class Patient:
    def __init__(self, info):
        self.info = info
        self.choosenFuturePatient = []

def PrepareDataset(listsOfPatients,counter):
    final_df=listsOfPatients[0].choosenFuturePatient[0].to_frame().T
    final_df = final_df[final_df["ID"] != 1]

    for i in range(1,counter,1):
        for j in range(4):
            listsOfPatients[i].choosenFuturePatient[j]["ID"]=i

    for j in range(4):
        for i in range(1,counter,1):
            final_df=pd.concat([final_df, listsOfPatients[i].choosenFuturePatient[j].to_frame().T])

    return final_df

def FindSimilarPatients(df, patient):
    similar_patients = df.where((df["gender"] == patient.info["gender"])
                                & (df["birthday"] == patient.info["birthday"])
                                & (df["examination date"] == patient.info["examination date"] + 1)
                                & (df["height"] > patient.info["height"] + 1.5)
                                & (df["height"] < patient.info["height"] + 15)
                                & (df["weight"] > patient.info["weight"] - 6)
                                & (df["weight"] < patient.info["weight"] + 12)
                                & (df["cylinder refraction of left eye"] <= patient.info[
        "cylinder refraction of left eye"] +0.6)
                                & (df["cylinder refraction of left eye"] > patient.info[
        "cylinder refraction of left eye"] -1.5)
                                & (df["cylinder refraction of right eye"] <= patient.info[
        "cylinder refraction of right eye"] +0.6)
                                & (df["cylinder refraction of right eye"] > patient.info[
        "cylinder refraction of right eye"] -1.5)
                                & (df["AL of right eye"] >= patient.info["AL of right eye"]-0.2)
                                & (df["AL of right eye"] < patient.info["AL of right eye"] + 0.7)
                                & (df["AL of left eye"] >= patient.info["AL of left eye"]-0.2)
                                & (df["AL of left eye"] < patient.info["AL of left eye"] + 0.7)
                                ).dropna()
    if len(similar_patients)==0:
        return df, patient
    choosen=similar_patients.iloc[random.randrange(len(similar_patients))]
    patient.choosenFuturePatient.append(choosen)
    df = df[df["ID"] != choosen["ID"]]
    if len(patient.choosenFuturePatient)<=4:
        patient.info=choosen
        df, patient=FindSimilarPatients(df, patient)
    return df, patient


def FindAllPatientsHistory(df, mainCounter, fileToSave):
    listsOfPatients = []
    for i in range(513):
        newPatient = Patient(df.iloc[i])
        newPatient.choosenFuturePatient.append(df.iloc[i])
        df, patient=FindSimilarPatients(df, newPatient)
        if len(patient.choosenFuturePatient)==4:
            listsOfPatients.append(patient)

    if len(listsOfPatients)>mainCounter:
        return len(listsOfPatients), PrepareDataset(listsOfPatients, len(listsOfPatients))
    return mainCounter, fileToSave


df = pd.read_excel("Filled_Raw_Data3.xlsx.xlsx")
print(type(df))
mainCounter=0
fileToSave=0
for i in range(1):
    print(i, "  ", mainCounter)
    mainCounter, fileToSave = FindAllPatientsHistory(df, mainCounter, fileToSave)

print(mainCounter)
fileToSave.to_excel(r'Old_version_clean_Data3.xlsx', index = False)
