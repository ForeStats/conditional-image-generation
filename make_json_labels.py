# Script to create 'dataset.json' file from the images split 
# into folders for each class

import os
import random
import json

data_dict = {}
data_dict['labels'] = []
label_counter = 0

input_folder = 'data/'

with open(os.path.join(input_folder, 'dataset.json'), 'w') as outfile:

    for root, subdirs, files in os.walk(input_folder):
        if len(subdirs) > 0:
            base_dir = root	
            continue

        current_subdir = os.path.split(root)[1]

        for filename in files:
            file_path = os.path.join(current_subdir, filename)
            #print('\t- file %s (full path: %s)' % (filename, file_path))
            data_dict['labels'].append([file_path, label_counter])

        label_counter += 1

    json.dump(data_dict, outfile)
