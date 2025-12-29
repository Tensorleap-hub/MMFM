import os

from code_loader.contract.datasetclasses import PreprocessResponse, PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from code_loader.plot_functions.visualize import visualize

from leap_binder import img_encoder, question_encoder, choice_encoder, gt_encoder, categorical_crossentropy_loss, \
    image_visualizer, question_visualizer, choice_visualizer, choice_gt_vis, get_metadata, question_metadata, \
    skills_metadata, preprocess_func



prediction_type1 = PredictionTypeHandler('pred-options', ['1', '2', '3', '4', '5'],channel_dim=-1)


@tensorleap_load_model([prediction_type1])
def load_model():
    H5_MODEL_PATH = "model/end2end.h5"
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(dir_path, H5_MODEL_PATH)
    import tensorflow as tf
    return tf.keras.models.load_model(os.path.join(dir_path, model_path))


@tensorleap_integration_test()
def check_custom_integration(idx, preprocess_response: PreprocessResponse):
    print("started custom tests")


    model = load_model()


    # import input and gt
    img = img_encoder(idx, preprocess_response)
    question = question_encoder(idx, preprocess_response)
    choices = choice_encoder(idx, preprocess_response)
    gt = gt_encoder(idx, preprocess_response)

    # metrics
    y_pred = model([question, img, choices])
    ls = categorical_crossentropy_loss(gt, y_pred)

    #import vis
    vis_image = image_visualizer(img)
    decoded_text = question_visualizer(question)
    decoded_choice = choice_visualizer(choices)
    choice_gt_vis_ = choice_gt_vis(choices, gt)


    visualize(vis_image)
    visualize(decoded_text)
    visualize(decoded_choice)
    visualize(choice_gt_vis_)


    # metadata
    metadata_dict = get_metadata(idx, preprocess_response)
    metadata_q = question_metadata(idx, preprocess_response)
    metadata_skills = skills_metadata(idx, preprocess_response)


if __name__ == '__main__':
    responses = preprocess_func()
    train = responses[0]
    check_custom_integration(0, train)
