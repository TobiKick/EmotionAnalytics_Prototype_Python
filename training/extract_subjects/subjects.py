#!/usr/bin/env python
# coding: utf-8


import os
import json
import pickle

def updateCount(store, value):
    try:
        store[value] = store[value] + 1
    except KeyError as e:
        store[value] = 1
    return store
    
def get_subjects_from_folder(root_data_dir):
    subjects = {}
    frames_per_subject = {}
    videos_per_subject = {}
    
    for train_dir in os.listdir(root_data_dir):
        for subdir, dirs, files in os.walk(os.path.join(root_data_dir, train_dir)):
            for file in files:
                if file[-5:] == '.json':
                    with open(os.path.join(root_data_dir, train_dir, file)) as p:
                        data = json.load(p)
                    frames = data['frames']    
                    subject = data['actor']

                    for key, value in frames.items():
                        try:
                            sub_dict = subjects[subject]
                        except KeyError as e:
                            subjects.update({subject: {}})
                            sub_dict = subjects[subject]
                       
                        sub_dict.update({str(train_dir + '/' + key + '.png'): [value['valence'], value['arousal']]})
                        subjects.update({subject: sub_dict})
                        
                        frames_per_subject = updateCount(frames_per_subject, subject)
                    videos_per_subject = updateCount(videos_per_subject, subject)
    
    return videos_per_subject, frames_per_subject, subjects
    

path_to_folder = os.getcwd()               
videos_per_subject, frames_per_subject, subjects = get_subjects_from_folder(path_to_folder)

count = 0
for key, value in subjects.items():
    for k,v in value.items():
        count = count + 1
    print(key)
    print(count)
    count = 0   
        
print("Count: " + str(count))
# for k, v in frames_per_subject.items():
#    print("Key " + k + " has occurred "  + str(v) + " times")
    
#for k, v in videos_per_subject.items():
#    print("Key " + k + " has occurred "  + str(v) + " times")

with open('subjects.pkl', 'wb') as f:
    pickle.dump(subjects, f, pickle.HIGHEST_PROTOCOL)
