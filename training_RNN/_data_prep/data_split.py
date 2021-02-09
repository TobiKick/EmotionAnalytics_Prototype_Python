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

####################################

def constructing_data_list(root_data_dir):
    subjects = {}
    d = defaultdict(list)
    videoClips = []
    filenames = []
    labels = []
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
                    d[actor].append(train_dir)

                    frames_count = 0
                    for key, value in frames.items():
                        try:
                            sub_dict = subjects[actor]
                        except KeyError as e:
                            subjects.update({actor: {}})
                            sub_dict = subjects[actor]
                       
                        sub_dict.update({str(train_dir + '/' + key + '.png'): [value['valence'], value['arousal']]})
                        subjects.update({actor: sub_dict})
                        frames_count = frames_count + 1
                    
                    frames_list.append(frames_count)
                    if frames_count < frames_min:
                        frames_min = frames_count
                    if frames_count > frames_max:
                        frames_max = frames_count

    print("# video clips: 600")
    print("# subjects: 240")
    print("# videos per subject: 2.5")
    print("# min frames per video: " + str(frames_min))
    print("# max frames per video: " + str(frames_max))
    print("# avg frames per video: " + str(sum(frames_list)/600))
    print("# median frames per video: " + str(statistics.median(frames_list)))              


    ## split videos into 5 folds while keeping them subject independent
    d_sorted = sorted(d.items())
    for elem in d_sorted:
        videoClips.append([str(elem[0]), int(len(elem[1])), -1])

    a = np.array(videoClips)
    a = a[a[:,1].astype(np.int).argsort()]
    videoClips = a[::-1]

    fold_size = [120, 120, 120, 120, 120]
    i = 0
    for elem in videoClips:
        check = False
        while check == False:
            if (fold_size[i] - int(elem[1])) >= 0:
                fold_size[i] = fold_size[i] - int(elem[1])
                elem[2] = i
                check = True

            if i == 4:
                i = 0
            else:
                i = i + 1
    # print(videoClips[0])

    with open('subject_videos_fold.csv', "w", newline='') as fp:   # "w"   if the file exists it clears it and starts writing from line 1
        wr = csv.writer(fp, delimiter=',')
        for elem in videoClips:
            wr.writerow([elem])


    ## determine and assign the fold in which the frame needs to be put
    content = []
    videoClips = np.array(videoClips)
    for key, value in subjects.items():
        index = np.where(videoClips == str(key))
        fold = int(videoClips[index[0][0]][2]) 

        for k, v in value.items():
            content.append([fold, k, v[0], v[1]])

    content = np.array(content)
    content_folder = np.array([item[0:3] for item in content[:,1]])

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

def extract_face_from_image(image, pixels):
    face_image = Image.fromarray(image)
    face_image = face_image.resize((pixels, pixels))
    image = asarray(face_image)

    face = detect_face(image)
    # print(face)

    if face == []: 
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
        face_image = face_image.resize((pixels, pixels))
        out = asarray(face_image) 
        return out

####################################

def sequence_data(path, data, SEQUENCE_LENGTH, pixels):
    faces = []
    valence = []
    arousal = []
    print(data.shape)

    for videos in data:  ## runs 5 times: access to each fold containg 120 videos
        a3, b3, c3 = [], [], []

        for frames in videos: ## runs 120 times: access to each video containing a XX sequence of frames
            a1, b1, c1, a2, b2, c2 = [], [], [], [], [], []

            for frame in frames:  ## runs XX times: access to each individual frame
                img = cv2.imread(os.path.join(path, frame[1]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                    face = extract_face_from_image(img, pixels)

                    if len(face) > 0:
                        a1.append(face)
                    else:
                        img = Image.fromarray(img)
                        img = img.resize((pixels, pixels))
                        img = asarray(img) 
                        a1.append(img)

                    b1.append(frame[2])
                    c1.append(frame[3])

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
                            c2.append(c1[i])
                            count = count - step
                            t = t + 1
                        count = count + 1

                    for i in range (0, (SEQUENCE_LENGTH - t)):
                        a2.append(a1[len(b1)-1])
                        b2.append(b1[len(b1)-1])
                        c2.append(c1[len(b1)-1])
                    
                else: # augment data
                    multiplicator = SEQUENCE_LENGTH / len(b1)
                    m = multiplicator
                    t = 0
                    for i in range(0, len(b1)):
                        while m >= 1:
                            a2.append(a1[i])
                            b2.append(b1[i])
                            c2.append(c1[i])
                            m = m - 1
                            t = t + 1
                        m = m + multiplicator

                    # check if it matched
                    add = SEQUENCE_LENGTH - t
                    for i in range(0, add):
                        a2.append(a1[len(b1)-1])
                        b2.append(b1[len(b1)-1])
                        c2.append(c1[len(b1)-1])

                print("Frames per video: " + str(len(b1)))
                if (SEQUENCE_LENGTH - len(b2)) != 0:
                    print("Difference: " + str(SEQUENCE_LENGTH - len(b2)))

                a3.append(a2)
                b3.append(b2)
                c3.append(c2)
            else:
                print("SOMETHING WENT WRONG")
                print(b1)
            
        faces.append(a3)
        valence.append(b3)
        arousal.append(c3)
    
    faces = np.array(faces)
    valence = np.array(valence)
    arousal = np.array(arousal)
    print(faces.shape)
    print(valence.shape)
    return faces, valence, arousal

########################################################################################
########################################################################################

path = r"C:/Users/Tobias/Desktop/AFEW-VA"
SEQUENCE_LENGTH = 45
pixels = 224

data = constructing_data_list(path)
np.save('subject_independent_folds.npy', data)

data = np.load('subject_independent_folds.npy', allow_pickle=True)
f, v, a = sequence_data(path, data, SEQUENCE_LENGTH, pixels)
np.save('faces_sequenced_45.npy', f)
np.save('valence_sequenced_45.npy', v)
np.save('arousal_sequenced_45.npy', a)