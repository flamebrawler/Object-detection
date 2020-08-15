import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


class ObjectDetectionModel:

    def __init__(self, base_path='ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/'):
        config = load_config(base_path+'pipeline.config')
        self.model = model_builder.build(model_config=config, is_training=False)
        restore_checkpoint(self.model, base_path+'checkpoint/ckpt-0')
        self.index, _ = load_index()

    def __call__(self, image):
        image = np.array(image)
        tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        return self.detect(tensor)

    @tf.function
    def detect(self, tensor):
        preprocessed_image, shapes = self.model.preprocess(tensor)
        prediction_dict = self.model.predict(preprocessed_image, shapes)
        return self.model.postprocess(prediction_dict, shapes)


def load_index(label_map_path='models/research/object_detection/data/mscoco_label_map.pbtxt'):
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
    return category_index, label_map_dict


def load_config(pipeline_path):
    configs = config_util.get_configs_from_pipeline_file(pipeline_path)
    return configs['model']


def restore_checkpoint(model, checkpoint_path):
    ckpt = tf.compat.v2.train.Checkpoint(model=model)
    ckpt.restore(checkpoint_path).expect_partial()


def generate_image(image, detections, index=None, minscore=.3):
    image_detection = image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_detection,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.uint32) + 1,
        detections['detection_scores'][0].numpy(),
        index,
        use_normalized_coordinates=True,
        min_score_thresh=minscore)

    return image_detection



