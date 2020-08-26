import pandas as pd
import os
df = pd.read_csv('ophthalmoscope_v3.csv')

mode = 'train'
quality = 'good'

paths = mode
paths = os.path.join(paths,'clahe')
paths = [os.path.join(paths, class_name) for class_name in os.listdir(paths)]
paths = [os.path.join(path, quality) for path in paths]
image_paths = [os.path.join(path, file_name) for path in paths for file_name in os.listdir(path)] 

names = [image_path.split('/')[-1][:-4] for image_path in image_paths]

mode_age = df[df['path'].isin(names)]['age']
hist = mode_age.plot.hist(bins=10, alpha=0.5,figsize = (10,10), title='Train data')
fig = hist.get_figure()
fig.savefig(f'{mode}_{quality}.jpg')