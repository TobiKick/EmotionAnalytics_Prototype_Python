import numpy as np
from numpy import asarray

from PIL import Image
import os
import json
import cv2
import csv
import matplotlib.pyplot as plt

from mtcnn import MTCNN
detector = MTCNN()




def extract_face_from_image(image, discard_undetected_faces):
	"""
		Args:
		image : the read-in image data
		discard_undetected_faces: a boolean value that indicates whether images should be discarded where the module could not detect a face
		Returns:
		A numpy array that is 1) filled with the detected face cropped along the bounding box or 2) empty when face was not detected or 3) the original images when the face was not detected but images are not discarded
	"""
	image = asarray(image)
	face = detect_face(image)
	print(face)

	if face == []: 
		if discard_undetected_faces == True:
			return []      # discard the image and its label from training
		else:
			return image   # the original image
	else:
		# extract the bounding box from the requested face
		box = np.asarray(face[0]["box"])
		print(box)
		box[box < 0] = 0
		x1, y1, width, height =  box
		x2, y2 = x1 + width, y1 + height

		face_boundary = image[y1:y2, x1:x2]
		face_image = Image.fromarray(face_boundary)
		out = asarray(face_image)
		return out


def detect_face(image):
	"""
	Args: 
		image: the input image in numpy
	Returns: 
		A numpy array containing the face's bounding box and facial landmark points. If a face could not be detected an empty array will be returned.
	"""
	face = detector.detect_faces(image)

	if len(face) >= 1:
		return face
	elif len(face) > 1:
		return face[0]
	else:
		print("No face detected")
		print(face)
		return []


def get_labels_from_file(root_data_dir, train_dir, f):
    filenames = []
    labels = []

    path_to_file = os.path.join(root_data_dir, train_dir, f)
    with open(path_to_file) as p:
        data = json.load(p)
        
    if not 'frames' in data or len(data['frames']) == 0:     
        exit(0)
    else:
        frames = data['frames']
        for key, value in frames.items():
            filenames.append([str(train_dir), str(key + '.png')])
            labels.append([value['valence'], value['arousal']])
    
    return filenames, labels


def get_filenames_labels(root_data_dir):
    file_path = []
    labels = []

    for train_dir in os.listdir(root_data_dir):
        for subdir, dirs, files in os.walk(os.path.join(root_data_dir, train_dir)):
            for f in files:
                if f[-5:] == '.json':
                    f, l = get_labels_from_file(root_data_dir, train_dir, f)
                    file_path.extend(f)
                    labels.extend(l)

    x = np.array(file_path)
    y = np.array(labels)
    return x, y




path_in = r"C:/Users/Tobias/Desktop/AFEW-VA"
path_out = r"C:/Users/Tobias/Desktop/AFEW-VA_JUST_FACE"
x, y = get_filenames_labels(path_in)

print(len(x))
for i in range(0, (len(x) -1)):
    print(os.path.join(path_in, str(x[i][0]), str(x[i][1])))
    img = cv2.imread(os.path.join(path_in, str(x[i][0]), str(x[i][1])))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = extract_face_from_image(img, True)
        if face != []:
            plt.imsave(os.path.join(path_out, str(x[i][0]), str(x[i][1])), face)
            with open(os.path.join(path_out, str(x[i][0]), 'labels.csv'), "a", newline='') as fp:  # a for append
                wr = csv.writer(fp, delimiter=',')
                wr.writerow([os.path.join(path_out, str(x[i][0]), str(x[i][1])), str(y[i][0]), str(y[i][1])])


