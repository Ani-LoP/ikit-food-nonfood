import os
import splitfolders
import shutil

path_from = 'data//data'
path_to = 'data//prepared_data'

try :
    shutil.rmtree(path_to)
except :
    print(path_to + ' уже удалена')


os.mkdir(path_to)
splitfolders.ratio(path_from, path_to, ratio=(0.7, 0.15, 0.15), seed=13, group_prefix=None)