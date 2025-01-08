import os
from global_vars import CUBS_DATASET_PATH
from PIL import Image
import numpy as np
import pandas as pd

numeric_features_with_commas = ['BMI', 'Glucose', 'Tchol', 'HDL', 'LDL', 'Trigl', 'Creat', 'Apoa1', 'ApoB']

def read_images(x, image='BOTH'):
    path = os.path.join(CUBS_DATASET_PATH, 'IMAGES', x['Patient ID'])
    if image.lower()=='r':
        return np.array(Image.open(path + '_R.tiff'))
    elif image.lower()=='l':
        return np.array(Image.open(path + '_L.tiff'))
    elif image.lower()=='both':
        return np.array([np.array(Image.open(path + '_R.tiff')), np.array(Image.open(path + '_L.tiff'))])
    else:
        raise(ValueError)


def read_dataset(years_to_high_risk=3):
    cubs = pd.read_csv(CUBS_DATASET_PATH / 'ClinicalDatabase-CUBS.csv', sep=';')
    cubs.drop('Sex', axis=1, inplace=True)
    cubs.rename({'Unnamed: 0': 'hospital', 'Sex.1': 'Sex'}, axis=1, inplace=True)
    cubs.hospital = cubs.hospital.ffill()
    cubs[numeric_features_with_commas] = cubs[numeric_features_with_commas].map(
        lambda x: float(x.replace(',', '.')) if type(x) == str else x, na_action='ignore')

    cubs = cubs[cubs['Base CVE'].notna() & cubs['FUPEvents'].notna()]
    cubs['risk'] = ((cubs['Base CVE'] == 1) | (cubs['FUPEvents'] == 1)) * 1
    cubs.loc[(cubs['Base CVE'] == 0) & (cubs['FUPEvents'] == 1), 'risk'] = (cubs[(cubs['Base CVE'] == 0) & (
                cubs['FUPEvents'] == 1)].TimetoEvent <= years_to_high_risk) * 1
    cubs['image'] = cubs.progress_apply(read_images, axis=1)
    return cubs