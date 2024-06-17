import platform
from typing import List, Union, Dict

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
from code_loader.contract.visualizer_classes import LeapImage, LeapText
from code_loader.contract.enums import LeapDataType
import tensorflow as tf

# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    dict_path = init_dataset()
    dictionary = Dictionary.load_from_file(dict_path)
    eval_dset = IconQAFeatureDataset('test', 'choose_txt', 'resnet101_pool5_79_icon',
                                     cnf.local_data_path, dictionary, 'bert-small', 34) # generate test data
    dataset = PreprocessResponse(length=min(len(eval_dset), cnf.max_imgs), data={'dataset': eval_dset})
    fake_dataset = PreprocessResponse(length=20, data={'dataset': eval_dset})
    leap_binder.cache_container["tokenizer"] = eval_dset.tokenizer
    leap_binder.cache_container["tokenizer_choice"] = eval_dset.dictionary
    res = [dataset, fake_dataset]
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


def image_visualizer(data: np.ndarray):
    return LeapImage(data[:cnf.img_size[1], :cnf.img_size[0]].astype(np.uint8))


def heatmap_image_visualizer(data: np.ndarray):
    return data[:cnf.img_size[1], :cnf.img_size[0]]


def question_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['dataset'][idx]['question_token']


def choice_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['dataset'][idx]['choice_token'].swapaxes(0, 1)[..., None]


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return preprocessing.data['dataset'][idx]['gt']


def question_visualizer(tokens: np.ndarray) -> LeapText:
    decoded_text = leap_binder.cache_container['tokenizer'].convert_ids_to_tokens(tokens)
    decoded_text = [token.replace(chr(9601), '').replace("##", "").replace("[PAD]", "").
                   replace("[CLS]", "").replace("[SEP]", "") for token in decoded_text]
    return LeapText(decoded_text)


def choice_visualizer(data: np.ndarray) -> LeapText:
    print("choice vis")
    print(data.shape)
    data = data[..., 0]
    print(data.shape)
    idx2word = leap_binder.cache_container['tokenizer_choice'].idx2word
    c_num = data.shape[1]
    max_tokens = data.shape[0]
    text_list = []
    for i in range(c_num):
        if data[:, i].sum() == 0: #padding
            text_list += ['']*max_tokens
        else:
            for j in range(max_tokens):
                token_id = data[j, i]
                if token_id >= len(idx2word):
                    curr_word = ''
                else:
                    curr_word = idx2word[int(data[j, i])]
                text_list.append(curr_word)
        # text_list.append('[SEP]')
    return LeapText(text_list)


def choice_visualizer_heatmap(data: np.ndarray) -> np.ndarray:
    print("choice vis heatmap")
    tf.print("tf - choice vis heatmap")
    print(type(data))
    print(data.shape)
    return data.swapaxes((0, 1)).reshape(-1)


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).


def metadata_label(idx: int, preprocess: PreprocessResponse) -> int:
    one_hot_digit = gt_encoder(idx, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)
    return digit_int


def bar_visualizer(data: np.ndarray) -> LeapHorizontalBar:
    return LeapHorizontalBar(data, LABELS)


def get_metadata(idx: int, preprocess: PreprocessResponse) -> Dict[str, Union[str, int, float]]:
    question_length = (preprocess.data['dataset'][idx]['question_token'] == 0).argmax()
    if question_length == 0:
        question_length = 34
    return {'question_id': int(preprocess.data['dataset'][idx]['question_id']),
            'question_length': int(question_length),
            'num_choices': int(sum(preprocess.data['dataset'][idx]['choice_token'].sum(axis=-1) > 0)),
            }


LABELS = ['1', '2', '3', '4', '5']
# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_input(function=img_encoder, name='image')
leap_binder.set_input(function=question_encoder, name='question')
leap_binder.set_input(function=choice_encoder, name='choices')
leap_binder.set_visualizer(function=image_visualizer,
                           name="image_vis",
                           visualizer_type=LeapDataType.Image,
                           heatmap_visualizer=heatmap_image_visualizer)
leap_binder.set_visualizer(function=question_visualizer,
                           name="question_vis",
                           visualizer_type=LeapDataType.Text)
leap_binder.set_visualizer(function=choice_visualizer,
                           name="choice_vis",
                           visualizer_type=LeapDataType.Text,
                           heatmap_visualizer=choice_visualizer_heatmap)
leap_binder.set_ground_truth(function=gt_encoder, name='options')
leap_binder.add_prediction(name='pred-options', labels=LABELS)
leap_binder.set_metadata(function=metadata_label, name='label')
leap_binder.set_metadata(function=get_metadata, name='')
