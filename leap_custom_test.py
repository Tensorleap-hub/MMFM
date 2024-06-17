import os
from leap_binder import preprocess_func, img_encoder, question_encoder, choice_encoder,\
    gt_encoder, image_visualizer
import tensorflow as tf
import matplotlib.pyplot as plt
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
    x = preprocess_func()
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    img = img_encoder(0, x[0])[None, ...]
    vis_image = image_visualizer(img[0])
    plt.imshow(vis_image.data)
    question = question_encoder(0, x[0])[None, ...]
    choices = choice_encoder(0, x[0])[None, ...]
    gt = gt_encoder(0, x[0])[None, ...]
    model = tf.keras.models.load_model(model_path)
    res = model([question, img, choices])
    ls = loss(tf.nn.softmax(res), gt)
    print(1)


if __name__ == '__main__':
    check_custom_integration()