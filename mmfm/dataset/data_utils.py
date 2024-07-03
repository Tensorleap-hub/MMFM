from mmfm.gcs_utils import _download
from os.path import join
from mmfm.config import cnf
from os import makedirs
from PIL import ImageOps


def init_dataset():
    cloud_dict = 'iconqa'
    dict_path = _download(cloud_file_path=f'{cloud_dict}/dictionary.pkl')
    data_path = join(cnf.local_data_path, "iconqa_data")
    makedirs(data_path, exist_ok=True)
    json_names = ['problems', 'pid_splits', 'pid2skills']
    for json_name in json_names:
        _download(cloud_file_path=f'{cloud_dict}/{json_name}.json',
                  local_file_path=f'{data_path}/{json_name}.json')
    return dict_path


def crop_and_padding(img, padding=3):
    # Crop the image
    bbox = img.getbbox()  # [left, top, right, bottom]
    bbox = (0, 0, *cnf.img_size)
    img = img.crop(bbox)

    # Add padding spaces to the 4 sides of an image
    desired_size = max(img.size) + padding * 2
    if img.size[0] < desired_size or img.size[1] < desired_size:
        delta_w = desired_size - img.size[0]
        delta_h = desired_size - img.size[1]
        padding = (padding, padding, delta_w - padding, delta_h - padding)
        img = ImageOps.expand(img, padding, (255, 255, 255))
    return img
