import os
import json
import numpy as np
from numpy import asarray
from collections import defaultdict
from itertools import chain
import statistics
import csv
from mtcnn import MTCNN
import cv2
from PIL import Image
import math

####################################

def constructing_data_list(root_data_dir, path_data_faces):
    subjects = {}
    d = defaultdict(list)
    videoClips = []
    filenames = []
    labels = []
    clips_count = 0
    frames_count = 0
    frames_min = 9999999
    frames_max = 0
    frames_list = []
    video_list = []

    for train_dir in os.listdir(root_data_dir):
        for subdir, dirs, files in os.walk(os.path.join(root_data_dir, train_dir)):
            for file in files:
                if file[-5:] == '.json':
                    video_list.append(train_dir)
                    with open(os.path.join(root_data_dir, train_dir, file)) as p:
                        data = json.load(p)
                    frames = data['frames']    
                    actor = data['actor']

                    frames_count = 0
                    for key, value in frames.items():
                        print(os.path.join(path_data_faces, str(train_dir), (str(key) + '.png')))
                        img = cv2.imread(os.path.join(path_data_faces, str(train_dir), (str(key) + '.png')))
                        if img is not None:
                            try:
                                sub_dict = subjects[actor]
                            except KeyError as e:
                                subjects.update({actor: {}})
                                sub_dict = subjects[actor]
                        
                            sub_dict.update({str(train_dir + '/' + key + '.png'): [value['valence'], value['arousal']]})
                            subjects.update({actor: sub_dict})
                            frames_count = frames_count + 1

                    if frames_count > 0:
                        d[actor].append(train_dir)
                        clips_count = clips_count + 1
                        frames_list.append(frames_count)
                        if frames_count < frames_min:
                            frames_min = frames_count
                        if frames_count > frames_max:
                            frames_max = frames_count

    print("STATISTICS for frames where a face was detected with MTCNN")
    print("# video clips: " + str(clips_count))
    print("# subjects: " + str(len(subjects)))
    print("# videos per subject: " + str(clips_count / len(subjects)))
    print("# min frames per video: " + str(frames_min))
    print("# max frames per video: " + str(frames_max))
    print("# avg frames per video: " + str(sum(frames_list)/600))
    print("# median frames per video: " + str(statistics.median(frames_list)))              


    ## split videos into 5 folds while keeping them subject independent
    # d contains the (key, value) as (subject, videos)
    # videoClips constains [subject, len(videos), -1]
    d_sorted = sorted(d.items())
    for elem in d_sorted:
        videoClips.append([str(elem[0]), int(len(elem[1])), -1])
    
    a = np.array(videoClips)
    a = a[a[:,1].astype(np.int).argsort()]
    videoClips = a[::-1]

    # drop the last X number of subjects so that each fold contains an equal number of video-clips
    # assign each subject in the videoClips array a fold
    # videoClips constains [subject, nr_videos, nr_fold] sorted in descending order of nr_videos
    fold_max = math.floor((clips_count/5))   # 5 folds
    nr_of_subjects_to_drop = clips_count - (fold_max * 5)
    videoClips = videoClips[:-nr_of_subjects_to_drop, :]

    fold_size = [fold_max, fold_max, fold_max, fold_max, fold_max]
    i = 0
    for elem in videoClips:
        assigned = False
        while assigned == False:
            if (fold_size[i] - int(elem[1])) >= 0:
                fold_size[i] = fold_size[i] - int(elem[1])
                elem[2] = i
                assigned = True

            if i == 4:
                i = 0
            else:
                i = i + 1
        print(fold_size)
    print(videoClips[0])

    with open('subject_videos_fold.csv', "w", newline='') as fp:   # "w"   if the file exists it clears it and starts writing from line 1
        wr = csv.writer(fp, delimiter=',')
        for elem in videoClips:
            wr.writerow([elem])


    ## go through each frame and assign it the proper fold
    ## output will be an array containing rows of [nr_fold, path_to_frame, valence, arousal]
    content = []
    videoClips = np.array(videoClips)
    for key, value in subjects.items():
        index = np.where(videoClips == str(key))     
        if index[0].size == 0:
            print("not found")
            print(str(key))
            print(index[0].size)
        else:
            fold = int(videoClips[index[0][0]][2]) 
            for k, v in value.items():
                content.append([fold, k, v[0], v[1]])

    content = np.array(content)
    content_folder = np.array([item[0:3] for item in content[:,1]])

    ## group the elements of the above created content array 
    ## in a way that shape is (5 folds, 119 video clips, number of frames)
    a = []
    for i in range(0, 5):
        # all indices of frames that belong to current video clip
        b = []
        for video in video_list:
            # all indices of frames that belong to the current fold & current video clip
            indices = np.where((content[:,0] == str(i)) & (content_folder == str(video)))[0]

            if indices.size > 0:
                c = []
                for index in indices:
                    c.append([video, content[index][1], content[index][2], content[index][3]])
                b.append(c)
        a.append(b)

    a = np.array(a)
    print(a.shape)
    return a

####################################

detector = MTCNN()
def detect_face(image):
    face = detector.detect_faces(image)

    if len(face) >= 1:
        return face
    elif len(face) > 1:
        return face[0]
    else:
        # print("No face detected")
        return []

####################################

def extract_face_from_image(image):
    face_image = Image.fromarray(image)
    face_image = face_image.resize((224, 224))
    image = asarray(face_image)

    face = detect_face(image)
    # print(face)

    if face == []: 
        print("No face detected")
        return []      # discard the image and its label from training
    else:
        # extract the bounding box from the requested face
        box = np.asarray(face[0]["box"])
        # print(box)
        box[box < 0] = 0
        x1, y1, width, height =  box
        x2, y2 = x1 + width, y1 + height

        face_boundary = image[y1:y2, x1:x2]
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize((224, 224))
        out = asarray(face_image) 
        return out

####################################

def sequence_data(path, data, SEQUENCE_LENGTH, pixels):
    faces = []
    labels = []
    print(data.shape)

    for videos in data:  ## runs 5 times: access to each fold containg 120 videos
        a3, b3 = [], []

        for frames in videos: ## runs 120 times: access to each video containing a XX sequence of frames
            a1, b1, a2, b2 = [], [], [], []

            for frame in frames:  ## runs XX times: access to each individual frame
                img = cv2.imread(os.path.join(path, frame[1]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # resize the image to 224x224x3
                    face_image = Image.fromarray(img)
                    face_image = face_image.resize((224, 224))
                    img = asarray(face_image)

                    a1.append(img)
                    b1.append([frame[2], frame[3]])

            if len(b1) > 0:
                ## sequencing
                if len(b1) >= SEQUENCE_LENGTH: # take a sample
                    step = len(b1) / SEQUENCE_LENGTH
                    count = step # includes the first element
                    t = 0
                    for i in range(0, len(b1)):
                        if count >= step:
                            a2.append(a1[i])
                            b2.append(b1[i])
                            count = count - step
                            t = t + 1
                        count = count + 1

                    for i in range (0, (SEQUENCE_LENGTH - t)):
                        a2.append(a1[len(b1)-1])
                        b2.append(b1[len(b1)-1])
                    
                else: # augment data
                    multiplicator = SEQUENCE_LENGTH / len(b1)
                    m = multiplicator
                    t = 0
                    for i in range(0, len(b1)):
                        while m >= 1:
                            a2.append(a1[i])
                            b2.append(b1[i])
                            m = m - 1
                            t = t + 1
                        m = m + multiplicator

                    # check if it matched
                    add = SEQUENCE_LENGTH - t
                    for i in range(0, add):
                        a2.append(a1[len(b1)-1])
                        b2.append(b1[len(b1)-1])

                print("Frames per video: " + str(len(b1)))
                if (SEQUENCE_LENGTH - len(b2)) != 0:
                    print("Difference: " + str(SEQUENCE_LENGTH - len(b2)))

                a3.append(a2)
                b3.append(b2)
            else:
                print("SOMETHING WENT WRONG")
                print(b1)
            
        faces.append(a3)
        labels.append(b3)
    
    faces = np.array(faces)
    labels = np.array(labels)
    print(faces.shape)
    print(labels.shape)
    labels =  labels.astype(np.float)
    print(labels.min())
    print(labels.max())
    labels = np.true_divide(labels, 10)
    print(labels.min())
    print(labels.max())
    return faces, labels

########################################################################################
########################################################################################

path = r"C:/Users/Tobias/Desktop/AFEW-VA"
path_face = r"C:/Users/Tobias/Desktop/AFEW-VA_JUST_FACE"
SEQUENCE_LENGTH = 45
pixels = 224

data = constructing_data_list(path, path_face)
np.save('subject_independent_folds.npy', data)

data = np.load('subject_independent_folds.npy', allow_pickle=True)  # just list with path to folder -> no actual image data is loaded
faces, labels = sequence_data(path_face, data, SEQUENCE_LENGTH, pixels)
np.save('faces_sequenced_45.npy', faces)
np.save('labels_sequenced_45.npy', labels)