import os
from leap_binder import preprocess_func, img_encoder, question_encoder, choice_encoder
import tensorflow as tf
# import numpy as np
#
# from leap_binder import preprocess_load, get_input_func, gt_encoder, metadata_is_truncated, metadata_length,\
#     metadata_dict, tokens_decoder_leap, metadata_deleted_columns_names, metadata_deleted_column, \
#     metadata_original_index, metadata_augment_index
#
# from bert_classification.loss import CE_loss
# from bert_classification.utils.model_utils import OnnxSqrt, OnnxErf
# from bert_classification.utils.utils import load_secret_to_env
# from bert_classification.config import CONFIG
# from download_models import download_models
# from json import dumps, load
# import onnxruntime as rt


def check_custom_integration():
    print("started custom tests")
    H5_MODEL_PATH = "model/end2end.h5"
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(dir_path, H5_MODEL_PATH)
    model = tf.keras.models.load_model(model_path)
    x = preprocess_func()
    img = img_encoder(0, x[0])[None, ...]
    question = question_encoder(0, x[0])[None, ...]
    choices = choice_encoder(0, x[0])[None, ...]
    res = model([question, img, choices])


if __name__ == '__main__':
    check_custom_integration()