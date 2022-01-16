import numpy as np 
import librosa
import pandas as pd 
import os

class Patient:

    def __init__(self, diagnosis, filepath):
        self.diagnosis = diagnosis
        self.filepath = filepath
        self.sound_coeffs = np.zeros(0)

    def process_sound(self):
        sound, sample_rate = librosa.load(self.filepath)
        initial_coeff = librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=100)
        avg_initial_coeff = np.mean(initial_coeff, axis=1)
        self.sound_coeffs = avg_initial_coeff

    def return_X_data(self):
        return self.sound_coeffs

    def return_y_data(self):
        return self.diagnosis


diagnostics_df = pd.read_csv("Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv", header=None)
diagnostics_df.columns = ["patient_id", "diagnosis"]

folder_path = "Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"

print(diagnostics_df.head())

X_data = []
y_data = []

counter = 0

list_of_files = os.listdir(folder_path)
for filename in list_of_files:
    if filename.endswith(".wav"):
        filepath = folder_path+filename
        patient_id = filename[:3]
        # print(patient_id)
        # print(type(diagnostics_df["patient_id"].values[0]))
        rows = diagnostics_df[diagnostics_df["patient_id"]==int(patient_id)]
        diagnosis = rows["diagnosis"].values[0]
        # print(diagnosis)
        patient = Patient(diagnosis=diagnosis, filepath=filepath)
        patient.process_sound()
        X_data_item = patient.return_X_data()
        X_data.append(X_data_item)
        y_data.append(diagnosis) 

        if counter%10==0:
            print(counter) 

        counter += 1

X_data = np.array(X_data)

y_data_unique = list(set(y_data))

y_data = np.array(y_data)
y_data_one_hot = []

for y_sample in y_data:
    y_one_hot = np.zeros(len(y_data_unique))
    y_one_hot[y_data_unique.index(y_sample)] = 1
    y_data_one_hot.append(y_one_hot)

y_data_one_hot = np.array(y_data_one_hot)

# print(X_data.shape)
# print(y_data_one_hot.shape)
# print(y_data)

print("saving data...")
np.savetxt("data/X_data.csv", X_data, delimiter=",", fmt="%f")
np.savetxt("data/y_data.csv", y_data, delimiter=",", fmt="%s")
np.savetxt("data/y_data_one_hot.csv", y_data_one_hot, delimiter=",", fmt="%f")
