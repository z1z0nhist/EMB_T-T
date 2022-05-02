import os
import glob
import pandas as pd
"""
  path : image folder path
  result : dict { PATH : image folder full path name , labels : image label
"""
def make_csv_file(path):
    labels = []
    img_path = []
    for label in os.listdir(path):
        img_path.extend(glob.glob(os.path.join(path,label)+ '/*.jpg'))
        labels.extend([label]*len(glob.glob(os.path.join(path,label)+ '/*.jpg')))
    csv = pd.DataFrame({'PATH' : img_path,
                      'labels' : labels})
    return csv

###############   make csv in aws ################
def aws_make_csv_file(path, conn, bucket, subfolder):
    contents = conn.list_objects(Bucket=bucket, Prefix=subfolder+path)['Contents']

    labels = []
    img_path = []
    for label in contents:
        img_path.extend([f's3://'+bucket+'/'+label['Key']])
        labels.extend([label['Key'][len(subfolder+path)+1:].split('/')[0]])
    print(img_path[0])
    print(labels[0])
    csv = pd.DataFrame({'PATH' : img_path, 'labels' : labels})
        
    return csv
