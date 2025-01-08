from tqdm import tqdm
from pathlib import Path
import pandas as pd
import cv2
from typing import Union, Tuple
from global_vars import OOD_DATASET_PATH

tqdm.pandas()


def read_dataset(directory: Union[str, Path]=OOD_DATASET_PATH, extensions=None,
                             target_size: Tuple[int, int] = (434, 532)) -> pd.DataFrame:
    if extensions is None:
        extensions = ['png']

    def load_image(path):
        try:
            image = cv2.imread(str(path))
            image = cv2.resize(image, target_size)
            return image
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    directory = Path(directory)
    data = []

    # Collect image paths and classes
    for class_dir in directory.iterdir():
        if class_dir.is_dir():
            for ext in extensions:
                for image_path in class_dir.glob(f'*.{ext}'):
                    data.append({'file_path': str(image_path), 'class': class_dir.name})
    df = pd.DataFrame(data)
    df['image'] = df.file_path.progress_apply(load_image)
    return df
