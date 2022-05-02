import os
import glob
import pandas as pd
def make_csv_file(path):
  """
  path : image folder path
  result : dict { PATH : image folder full path name , labels : image label
  """
  labels = []
  img_path = []
  for label in os.listdir(path):
    img_path.extend(glob.glob(os.path.join(path,label)+ '/*.jpg'))
    labels.extend([label]*len(glob.glob(os.path.join(path,label)+ '/*.jpg')))
  csv = pd.DataFrame({'PATH' : img_path,
                      'labels' : labels})
  return csv