import os
from leap_binder import preprocess_func, img_encoder, question_encoder, choice_encoder, \
    gt_encoder, image_visualizer, question_visualizer, choice_visualizer, get_metadata, question_metadata, \
    skills_metadata, choice_gt_vis, heatmap_image_visualizer, choice_visualizer_heatmap
import tensorflow as tf
import matplotlib.pyplot as plt

from leap_binder import leap_binder
from code_loader.helpers import visualize
import numpy as np
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

    plot_vis = True
    check_generic = True

    if check_generic:
        leap_binder.check()


    H5_MODEL_PATH = "model/end2end.h5"
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(dir_path, H5_MODEL_PATH)
    x = preprocess_func()
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model = tf.keras.models.load_model(model_path)
    for i in range(20):
        # import input and gt
        img = img_encoder(i, x[1])[None, ...]
        question = question_encoder(i, x[1])[None, ...]
        choices = choice_encoder(i, x[1])[None, ...]
        gt = gt_encoder(i, x[1])[None, ...]
        print(img.shape)
        print(question.shape)
        print(choices.shape)

        # metrics
        y_pred = model([question, img, choices[..., 0]])
        ls = loss(tf.nn.softmax(y_pred), gt)

        #import vis
        vis_image = image_visualizer(img)
        # heatmap_image = heatmap_image_visualizer(np.zeros((332, 332, 1), dtype=np.uint8))
        decoded_text = question_visualizer(question)
        decoded_choice = choice_visualizer(choices)
        # heatmap_choice = choice_visualizer_heatmap(choices)
        choice_gt_vis_ = choice_gt_vis(choices, gt)

        if plot_vis:
            visualize(vis_image)
            visualize(decoded_text)
            visualize(decoded_choice)
            visualize(choice_gt_vis_)


        # metadata
        metadata_dict = get_metadata(i, x[1])
        metadata_q = question_metadata(i, x[1])
        metadata_skills = skills_metadata(i, x[1])


if __name__ == '__main__':
    check_custom_integration()
