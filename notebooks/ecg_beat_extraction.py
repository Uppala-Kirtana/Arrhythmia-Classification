import wfdb
import numpy as np
import pandas as pd
import scipy as sp
from scipy import io as sio
from scipy import signal as sps
import matplotlib.pyplot as plt

import os
import sys
from os.path import join as osj
from bisect import bisect
from collections import defaultdict
import pickle
import json

# custom libraries
from progress_bar import print_progress

DATA_ROOT = osj("/content", "mit-bih-arrhythmia-database-1.0.0")
RECORDS = osj(DATA_ROOT, "RECORDS")

patient_ids = pd.read_csv(RECORDS, delimiter=",", header=None).to_numpy().reshape(-1)

def get_ecg_signals(patient_ids):
    lead0 = {}
    lead1 = {}
    for id_ in patient_ids:
        signals, info = wfdb.io.rdsamp(osj(DATA_ROOT, str(id_)))
        lead0[id_] = signals[:, 0]
        lead1[id_] = signals[:, 1]
    return lead0, lead1

def get_ecg_info(patient_ids):
    _, info = wfdb.io.rdsamp(osj(DATA_ROOT, str(patient_ids[0])))
    resolution = 2**11  # Number of possible signal values we can have.
    info["resolution"] = 2**11
    return info

lead0, lead1 = get_ecg_signals(patient_ids)

ecg_info = get_ecg_info(patient_ids)

def get_paced_patients(patient_ids):
    paced = []
    for id_ in patient_ids:
        annotation = wfdb.rdann(osj(DATA_ROOT, str(id_)), extension='atr')
        labels = np.unique(annotation.symbol)
        if ("/" in labels):
            paced.append(id_)
    return np.array(paced)

paced_patients = get_paced_patients(patient_ids)

def get_all_beat_labels(patient_ids):
    all_labels = []
    for id_ in patient_ids:
        annotation = wfdb.rdann(osj(DATA_ROOT, str(id_)), extension='atr')
        labels = np.unique(annotation.symbol)
        all_labels.extend(labels)
    return np.unique(all_labels)

all_beat_labels = get_all_beat_labels(patient_ids)

def get_rpeaks_and_labels(patient_ids):
    rpeaks = {}
    labels = {}
    for id_ in patient_ids:
        annotation = wfdb.rdann(osj(DATA_ROOT, str(id_)), extension='atr')
        rpeaks[id_] = annotation.sample
        labels[id_] = np.array(annotation.symbol)
    return rpeaks, labels

rpeaks, labels = get_rpeaks_and_labels(patient_ids)

def get_normal_beat_labels():
    """
    The MIT-BIH labels that are classified as healthy/normal. Check wfdb.Annotation documentation for description of labels.
    N: {N, L, R, e, j}. 
    """
    return np.array(["N", "L", "R", "e", "j"])

def get_abnormal_beat_labels():
    """
    The MIT-BIH labels that are classified as arrhythmia/abnormal. Check wfdb.Annotation documentation for description of labels.
    S: {S, A, J, a} - V: {V, E} - F: {F} - Q: {Q}
    """
    return np.array(["S", "A", "J", "a", "V", "E", "F", "Q"])

def get_beat_class(label):
    """
    A mapping from labels to classes, based on the rules described in get_normal_beat_labels() and get_abnormal_beat_labels().
    """
    if label in ["N", "L", "R", "e", "j"]:
        return "N"
    elif label in ["S", "A", "J", "a"]:
        return "S"
    elif label in ["V", "E"]:
        return "V"
    elif label == "F" or label == "Q":
        return label
    return None

def get_beats(patient_ids, signals, rpeaks, labels, beat_trio=False, centered=False, lr_offset=0.1, matlab=False):
    """
    For each patient:
    Converts its ECG signal to an array of valid beats, where each rpeak with a valid label is converted to a beat of length 128 by resampling (Fourier-Domain).
    Converts its labels to an array of valid labels, and a valid label is defined in the functions get_normal_beat_labels() and get_abnormal_beat_labels().
    Converts its valid labels to an array of classes, where each valid label is one of 5 classes, (N, S, V, F, Q).
    
    Parameters
    ----------
    beat_trio: bool, default=False
        If True, generate beats as trios.
        
    centered: bool, default=False
        Whether the generated beats have their peaks centered.
        
    lr_offset: float, default=0.1, range=[0, 1]
        A beat is extracted by finding the beats before and after it, and then offsetting by some samples. This parameter controls how many samples are
        offsetted. If the lower beat is L, and the current beat is C, then we offset by `lr_offset * abs(L - C)` samples.
        
    matlab: bool, default=False
        If True, dictionary keys become strings to be able to save the dictionary as a .mat file.
    """
    
    beat_length = 128
    get_key_name = lambda patient_id: f"patient_{patient_id}" if matlab else patient_id
    
    beat_data = {get_key_name(patient_id):{"beats":[], "class":[], "label":[]} for patient_id in patient_ids}
    
    for j, patient_id in enumerate(patient_ids):
        key_name = get_key_name(patient_id)
        
        # Filter out rpeaks that do not correspond to a valid label.
        valid_labels = np.concatenate((get_normal_beat_labels(), get_abnormal_beat_labels()))
        valid_idx = np.where(np.isin(labels[patient_id], valid_labels))[0]
        valid_rpeaks = rpeaks[patient_id][valid_idx]
        valid_labels = labels[patient_id][valid_idx]
        
        for i in range(1, len(valid_rpeaks) - 1):
            lpeak = valid_rpeaks[i - 1]
            cpeak = valid_rpeaks[i]
            upeak = valid_rpeaks[i + 1]
    
            if beat_trio:
                lpeak = int(lpeak - (lr_offset * abs(cpeak - lpeak)))
                upeak = int(upeak + (lr_offset * abs(cpeak - upeak)))
            else:
                lpeak = int(lpeak + (lr_offset * abs(cpeak - lpeak)))
                upeak = int(upeak - (lr_offset * abs(cpeak - upeak)))
            
            if centered:
                ldiff = abs(lpeak - cpeak)
                udiff = abs(upeak - cpeak)
                diff = min(ldiff, udiff)
                
                # Take same number of samples from the center.
                beat = signals[patient_id][cpeak - diff:cpeak + diff + 1]
            else:
                beat = signals[patient_id][lpeak:upeak]
            
            # Resampling in the frequency domain instead of in the time domain (resample_poly)
            # beat = sp.signal.resample_poly(beat, beat_length, len(beat))
            beat = sp.signal.resample(beat, beat_length)
    
            # detrend the beat and normalize it.
            beat = sps.detrend(beat)
            beat = beat / np.linalg.norm(beat, ord=2)
        
            label = valid_labels[i]
        
            beat_data[key_name]["beats"].append(beat)
            beat_data[key_name]["class"].append(get_beat_class(label))
            beat_data[key_name]["label"].append(label)
        beat_data[key_name]["beats"] = np.stack(beat_data[key_name]["beats"])
        beat_data[key_name]["class"] = np.stack(beat_data[key_name]["class"])
        beat_data[key_name]["label"] = np.stack(beat_data[key_name]["label"])
        
        print_progress(j + 1, len(patient_ids), opt=[f"{patient_id}"])
    return beat_data

beat_data = get_beats(patient_ids, lead0, rpeaks, labels, beat_trio=False, centered=False, lr_offset=0.1)

fivemin = ecg_info["fs"] * 60 * 5
fivemin_index = []
for patient_id in patient_ids:
    idx = bisect(rpeaks[patient_id], fivemin) - 1
    fivemin_index.append(idx)
fivemin_beat_data = {patient_id:{"beats":[], "class":[], "label":[]} for patient_id in patient_ids}
remaining_beat_data = {patient_id:{"beats":[], "class":[], "label":[]} for patient_id in patient_ids}

for i, patient_id in enumerate(patient_ids):
    normal_idx = np.where(beat_data[patient_id]["class"][0:fivemin_index[i]] == "N")[0]
    other_idx = np.setdiff1d(np.arange(0, len(beat_data[patient_id]["class"])), normal_idx)
    
    assert len(normal_idx) + len(other_idx) == len(beat_data[patient_id]["class"]), "Some beats are not taken into account!"
    
    fivemin_beat_data[patient_id]["beats"] = beat_data[patient_id]["beats"][normal_idx, :]
    fivemin_beat_data[patient_id]["class"] = beat_data[patient_id]["class"][normal_idx]
    fivemin_beat_data[patient_id]["label"] = beat_data[patient_id]["label"][normal_idx]
    
    assert np.count_nonzero(fivemin_beat_data[patient_id]["class"] != "N") == 0, "Abnormal beat misplaced!"
    
    remaining_beat_data[patient_id]["beats"] = beat_data[patient_id]["beats"][other_idx, :]
    remaining_beat_data[patient_id]["class"] = beat_data[patient_id]["class"][other_idx]
    remaining_beat_data[patient_id]["label"] = beat_data[patient_id]["label"][other_idx]

DATASET_PATH = osj("/content/dataset_beats")  # choose a path

DATASET_PATH1 = osj("/content/dataset_beats_trios")  # choose a path

with open(osj(DATASET_PATH, "5min_normal_beats.pkl"), "wb") as f:
    pickle.dump(fivemin_beat_data, f)
    
with open(osj(DATASET_PATH, "25min_beats.pkl"), "wb") as f:
    pickle.dump(remaining_beat_data, f)
    
with open(osj(DATASET_PATH, "30min_beats.pkl"), "wb") as f:
    pickle.dump(beat_data, f) 

with open(osj(DATASET_PATH1, "5min_normal_beats1.pkl"), "wb") as f:
    pickle.dump(fivemin_beat_data, f)
    
with open(osj(DATASET_PATH1, "25min_beats1.pkl"), "wb") as f:
    pickle.dump(remaining_beat_data, f)
    
with open(osj(DATASET_PATH1, "30min_beats1.pkl"), "wb") as f:
    pickle.dump(beat_data, f) 
