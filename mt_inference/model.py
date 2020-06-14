from abc import ABC, abstractmethod
from PIL import Image
import tensorflow as tf
import numpy as np


class Model(ABC):
    def __init__(self, frozen_graph_path: str):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        with tf.io.gfile.GFile(frozen_graph_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")

        self.session = tf.compat.v1.Session(graph=self.graph)

    @abstractmethod
    def run(self, input):
        pass


class DeepLabModel(Model):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = "ImageTensor:0"
    OUTPUT_TENSOR_NAME = "SemanticProbabilities:0"  # SemanticPredictions:0
    INPUT_SIZE = 623

    def run(self, image: Image) -> np.array:
        """
        Runs inference on single image.

        Args:
        image: input image in RGB format.

        Returns:
        Probabilities map with dimensions of image and probabilities for class 1 (Lupine)
        """
        # TODO assert image.size < self.INPUT_SIZE
        # TODO assert format is RGB or convert

        prob_map = self.session.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)]},
        )

        # output tensor returns two probabilities for every pixel
        # [prop_background, prob_lupine], which sum up to 1
        # so it is enough to only return one of those two probabilities
        return prob_map[0][..., 1]

        """
        # this converts probabilities to predict
        # argmax returns index of max value ^= predicition
        return np.argmax(batch_seg_map[0], axis=2)
        """
