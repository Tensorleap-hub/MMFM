# from mmfm.packages_helpers import install_all_packages

# install_all_packages()
from readability import Readability

import platform
from typing import List, Union, Dict
import tensorflow as tf

import numpy as np
# import textstat

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

import spacy


# nlp = spacy.load("en_core_web_sm")


# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    dict_path = init_dataset()
    dictionary = Dictionary.load_from_file(dict_path)
    train_dset = IconQAFeatureDataset('train', 'choose_txt', 'resnet101_pool5_79_icon',
                                      cnf.local_data_path, dictionary, 'bert-small', 34)  # generate test data
    val_dset = IconQAFeatureDataset('val', 'choose_txt', 'resnet101_pool5_79_icon',
                                    cnf.local_data_path, dictionary, 'bert-small', 34)  # generate test data
    test_dset = IconQAFeatureDataset('test', 'choose_txt', 'resnet101_pool5_79_icon',
                                     cnf.local_data_path, dictionary, 'bert-small', 34)  # generate test data
    train_dataset = PreprocessResponse(length=min(len(train_dset), cnf.max_imgs), data={'dataset': train_dset})
    val_dataset = PreprocessResponse(length=min(len(val_dset), cnf.max_imgs), data={'dataset': val_dset})
    test_dataset = PreprocessResponse(length=min(len(test_dset), cnf.max_imgs), data={'dataset': test_dset})
    leap_binder.cache_container["tokenizer"] = train_dset.tokenizer
    leap_binder.cache_container["tokenizer_choice"] = train_dset.dictionary
    res = [train_dataset, val_dataset, test_dataset]
    return res


# def preprocess_func() -> List[PreprocessResponse]:
#     dict_path = init_dataset()
#     dictionary = Dictionary.load_from_file(dict_path)
#     eval_dset = IconQAFeatureDataset('test', 'choose_txt', 'resnet101_pool5_79_icon',
#                                      cnf.local_data_path, dictionary, 'bert-small', 34)  # generate test data
#     dataset = PreprocessResponse(length=min(len(eval_dset), cnf.max_imgs), data={'dataset': eval_dset})
#     fake_dataset = PreprocessResponse(length=20, data={'dataset': eval_dset})
#     leap_binder.cache_container["tokenizer"] = eval_dset.tokenizer
#     leap_binder.cache_container["tokenizer_choice"] = eval_dset.dictionary
#     res = [dataset, fake_dataset]
#     return res


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
        if data[:, i].sum() == 0:  # padding
            text_list += [''] * max_tokens
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


def calculate_syntactic_complexity(question):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)
    return float(max([len(list(token.ancestors)) for token in doc]))


def get_pos_tags(question):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)
    return [(token.text, token.pos_) for token in doc]


# def recognize_named_entities(question):
#     question = ' '.join(filter(None, question))
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(question)
#     return [(ent.text, ent.label_) for ent in doc.ents]


# def calculate_readability_score(question):
#     question = ' '.join(filter(None, question))
#     return float(textstat.flesch_reading_ease(question))


def classify_question_type(question):
    question_types = {"what", "where", "when", "why", "how", "who", "which", "whom", "whose"}
    for word in question:
        word = word.lower()
        if word in question_types:
            return str(word)
    return "other"


def get_analyzer(text):
    try:
        analyzer = Readability(text)
    except:
        analyzer = None
    return analyzer


def get_statistics(key: str, question) -> float:
    question = ' '.join(filter(None, question))
    analyzer = get_analyzer(question)
    if analyzer is not None:
        return float(analyzer.statistics()[str(key)])
    else:
        return -1


def get_readibility_score(key: str, question) -> float:
    question = ' '.join(filter(None, question))
    analyzer = get_analyzer(question)
    if analyzer is not None:
        return float(np.round(analyzer.__getattribute__(str(key)), 3))
    else:
        return -1


def get_num_letters(question):
    question = [word for word in question if '?' not in word]
    question = ''.join(filter(None, question))
    return float(len(question))


def get_num_words(question):
    question = [word for word in question if '?' not in word]
    return float(len(list(filter(None, question))))


def question_metadata(idx: int, preprocess: PreprocessResponse) -> Dict[str, Union[str, int, float]]:
    question_token = preprocess.data['dataset'][idx]['question_token']
    decoded_question = leap_binder.cache_container['tokenizer'].convert_ids_to_tokens(question_token)
    decoded_question = [token.replace(chr(9601), '').replace("##", "").replace("[PAD]", "").
                        replace("[CLS]", "").replace("[SEP]", "") for token in decoded_question]

    skills = preprocess.data['dataset'].entries[idx]['skills']
    skills = [word.lower() for word in skills]
    skills.sort()

    question_metadata_functions = {
        "Num_letters": get_num_letters(decoded_question),
        "Num_words": get_num_words(decoded_question),
        "Question_word": classify_question_type(decoded_question),
        "Label": preprocess.data['dataset'].entries[idx]['label'],
        "Grade": preprocess.data['dataset'].entries[idx]['grade'],
        "Skills": '_'.join(skills),
        "Skills_number": float(len(skills))
        # "ARI": get_readibility_score('ari', decoded_question),
        # "Dale_chall": get_readibility_score('dale_chall', decoded_question),
        # "Flesch": get_readibility_score('flesch', decoded_question),
        # "Flesch_kincaid": get_readibility_score('flesch_kincaid', decoded_question),
        # "Syntactic Complexity": calculate_syntactic_complexity(question),
        # "Readability Score": calculate_readability_score(question),
    }

    return question_metadata_functions


def skills_metadata(idx: int, preprocess: PreprocessResponse) -> Dict[str, Union[str, int, float]]:
    skills = preprocess.data['dataset'].entries[idx]['skills']
    skills = [word.lower() for word in skills]
    skills.sort()
    skills_dict = {skill: 0 for skill in cnf.skills}
    for key in skills:
        if key in skills_dict:
            skills_dict[key] = 1

    return skills_dict



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
leap_binder.set_metadata(function=question_metadata, name='question')
leap_binder.set_metadata(function=skills_metadata, name='skills')

if __name__ == '__main__':
    leap_binder.check()
