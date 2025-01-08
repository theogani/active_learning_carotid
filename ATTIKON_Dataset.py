import SimpleITK as sitk
import os
from scipy.io import loadmat
from glob import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
from global_vars import *

'''
sys_diat_mat file has info in absolute reference (frames dim).

Mat files have info in the part for which we have motion info (frames dim)
'''


def read(x):
    reader = sitk.ImageFileReader()
    reader.SetFileName(x)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)[..., 0]


def write(x, path):
    if not os.path.isdir(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(sitk.GetImageFromArray(x.astype(np.uint8)))
    return


class Case:

    def __init__(self, name, root_path=DATASET_PATH):
        self.name = name
        self.video = read(os.path.join(root_path, self.name))
        self.patientID, self.videoID = int(name.split('_')[0][2:]), int(name.split('_')[1])
        df = pd.read_csv(IMAGES_CSV_PATH)
        if self.videoID in df.VideoID.values:
            self.diagnosis = df[df.VideoID == self.videoID].Diagnosis.squeeze()
            self.stenosis = df[df.VideoID == self.videoID].Stenosis.squeeze()
            self.risk = df[df.VideoID == self.videoID].Risk.squeeze()
        else:
            self.diagnosis = None
            self.stenosis = None
            self.risk = None
        mat_file = loadmat(os.path.join(MAT_FILES_PATH, '{}.mat'.format(name[2:-2])), squeeze_me=True)
        for i in mat_file.keys():
            if i[:2] == '__':
                continue
            else:
                setattr(self, i, mat_file[i])
        sys_dia_mat_file = loadmat(SYS_DIA_MAT_PATH)['sys_and_diast_upd']
        for plaque_num in range(1, mat_file['Nplaques'] + 1):
            temp = [(i[1], i[2]) for i in sys_dia_mat_file if name[2:-1] + str(plaque_num) in i[0][0]]
            if len(temp) > 1:
                raise ValueError('Something went wrong.')
            sys, dia = temp[0]
            setattr(self, f'plaque{plaque_num}_sys', np.squeeze(sys))
            setattr(self, f'plaque{plaque_num}_dia', np.squeeze(dia))

    def create_mask(self, plaque=None, sys=None, dia=None):
        from scipy.ndimage import binary_fill_holes, label
        from scipy.ndimage import binary_closing
        # cols are for different points, rows are for different momments

        if plaque:
            msk = np.zeros(self.video.shape[1:])
            msk[np.rint(self.x_wp[0]).astype(int) - 1, np.rint(self.y_wp[0]).astype(int) - 1] += 1
            msk = binary_closing(msk, np.ones((4, 4)))
            msk, num_feat = label(msk, np.ones((3, 3)))
            # find which columns of mat file are for the current plaque
            columns = []
            
            plaque_label = np.unique(msk[getattr(self, f'plaque{plaque}')[1]-1, getattr(self, f'plaque{plaque}')[0]-1])
            if len(plaque_label)>1:
                raise ValueError('Something strange.')
            else:
                plaque_label = plaque_label[0]
            
            for id, (x_, y_) in enumerate(
                    zip(self.x_wp[0].astype(int) - 1,
                        self.y_wp[0].astype(int) - 1)):
                if msk[x_, y_] == plaque_label:
                    columns.append(id)
            
            if sys is not None:
                rows = [getattr(self, f'plaque{plaque}_sys')[sys] - self.frame1]  # sys/dia have absolute reference.
            
            elif dia is not None:
                rows = [getattr(self, f'plaque{plaque}_dia')[dia] - self.frame1]  # sys/dia have absolute reference.
            
            else:
                rows = range(self.x_wp.shape[0])
            
            msk = np.zeros(((len(rows),) + self.video.shape[1:]))
        else:
            
            if (sys is not None) or (dia is not None):
                raise KeyError('Does not make sense to ask for a systole or diastole of an undifined plaque.')
            
            rows = range(self.x_wp.shape[0])
            columns = list(range(self.x_wp.shape[1]))
            msk = np.zeros(((len(rows),) + self.video.shape[1:]))

        for i, j in enumerate(rows):
            msk[i,
                np.rint(self.x_wp[j][columns]).astype(int) - 1,
                np.rint(self.y_wp[j][columns]).astype(int) - 1] += 1
            k = 10
            msk[i] = binary_closing(msk[i], np.ones((k, k)))
            msk[i] = binary_fill_holes(msk[i]).astype(int)
            msk[i], num_feat = label(msk[i], np.ones((3, 3)))

        return np.squeeze(msk).astype(bool)

    def crop_plaque(self, background='white', **kwargs):
        msk = self.create_mask(**kwargs)

        if 'sys' in kwargs.keys() and kwargs["sys"] is not None:
            if 'plaque' not in kwargs.keys():
                raise KeyError('Does not make sense to ask for a systole or diastole of an undifined plaque.')
            return msk * self.video[getattr(self, f'plaque{kwargs["plaque"]}_sys')[kwargs["sys"]]] + (
                255 if background == 'white' else 0) * np.logical_not(msk)

        elif 'dia' in kwargs.keys() and kwargs["dia"] is not None:
            if 'plaque' not in kwargs.keys():
                raise KeyError('Does not make sense to ask for a systole or diastole of an undifined plaque.')
            return msk * self.video[getattr(self, f'plaque{kwargs["plaque"]}_dia')[kwargs["dia"]]] + (
                255 if background == 'white' else 0) * np.logical_not(msk)

        else:
            return msk * self.video[self.frame1-1: self.lastframe] + (255 if background == 'white' else 0) * np.logical_not(msk)


def imadjust(src, vin=(0, 255), vout=(0, 190)):
    im = src.copy()
    im[im < vin[0]] = vin[0]
    im[im > vin[1]] = vin[1]
    im = (im - vin[0])/(vin[1] - vin[0])
    im = im * vout[1] + vout[0]
    return im.astype(np.uint8)


def read_dataset(root_path=DATASET_PATH):
    return [Case(os.path.basename(i), root_path) for i in tqdm(glob(os.path.join(root_path, 'ID*_*_B*')))]


def read_dataset_as_df():
    carotid_list = read_dataset()

    for i in carotid_list:
        i.x_wp = i.x_wp[:i.lastframe - i.frame1 + 1]
        i.y_wp = i.y_wp[:i.lastframe - i.frame1 + 1]

    carotid_df = pd.DataFrame.from_records([i.__dict__ for i in carotid_list])
    print('Initial Dataframe:')
    print(carotid_df.risk.value_counts(dropna=False))

    carotid_df = carotid_df[~carotid_df.risk.isna()]
    print('\nAfter droping unlabeled:')
    print(carotid_df.risk.value_counts(dropna=False))
    return carotid_df