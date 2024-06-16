from typing import List, Union

import numpy as np

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse 
from code_loader.contract.enums import Metric, DatasetMetadataType
from code_loader.contract.visualizer_classes import LeapHorizontalBar
from mmfm.dataset.dataset import Dictionary, IconQAFeatureDataset
from mmfm.gcs_utils import _download
from mmfm.config import cnf
from mmfm.dataset.data_utils import init_dataset, crop_and_padding
from PIL import Image

# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    dict_path = init_dataset()
    dictionary = Dictionary.load_from_file(dict_path)
    eval_dset = IconQAFeatureDataset('test', 'choose_txt', 'resnet101_pool5_79_icon',
                                     cnf.local_data_path, dictionary, 'bert-small', 34) # generate test data
    dataset = PreprocessResponse(length=len(eval_dset), data={'dataset': eval_dset})
    res = [dataset, dataset]
    return res


# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image. 
def img_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    img_path = preprocess.data['dataset'][idx]['img_path']
    fp = _download(f'{cnf.cloud_dict}/{img_path}')
    img = Image.open(fp)
    const_img_size = img.resize(cnf.img_size)
    padded_img = crop_and_padding(const_img_size)
    return np.array(padded_img)


def question_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['dataset'][idx]['question_token']


def choice_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['dataset'][idx]['choice_token'].swapaxes(0, 1)

# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return preprocessing.data['labels'][idx].astype('float32')

# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
def metadata_label(idx: int, preprocess: PreprocessResponse) -> int:
    one_hot_digit = gt_encoder(idx, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)
    return digit_int


def bar_visualizer(data: np.ndarray) -> LeapHorizontalBar:
    return LeapHorizontalBar(data, LABELS)


LABELS = ['0','1','2','3','4','5','6','7','8','9']
# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_input(function=img_encoder, name='image')
leap_binder.set_ground_truth(function=gt_encoder, name='classes')
# leap_binder.set_metadata(function=metadata_label, metadata_type=DatasetMetadataType.int, name='label')
# leap_binder.add_prediction(name='classes', labels=LABELS)
# leap_binder.set_visualizer(name='horizontal_bar_classes', function=bar_visualizer, visualizer_type=LeapHorizontalBar.type)
