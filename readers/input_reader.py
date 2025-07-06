import numpy
import onnxruntime
import os
import onnx
from onnxruntime.quantization import CalibrationDataReader
from processors.model_preprocessor import ModelPreprocessor
from PIL import Image

import numpy as np


def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0, shape=(1, 3, 224, 224), model_name="", model_library=None):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        # print(shape)
        preprocessing_data = {
            
            "input": shape,#np.transpose(shape, (0, 3, 1, 2)),
            "input_dimension": (width, height),
            "library": model_library
        }
        # print (preprocessing_data)
        # Set library to None, so that the name is utilized for preprocessing selection.
        model_preprocessor = ModelPreprocessor(preprocessing_data)
        pillow_img = model_preprocessor.preprocess(model_name, pillow_img)
        # Preprocess image using preprocessor.
        input_data = numpy.float32(pillow_img)
        # print(input_data.shape)
        if (shape[1] == input_data.shape[3] and shape[3] == input_data.shape[1]):
            nchw_data = np.transpose(input_data, (0, 2, 3, 1))  # ONNX Runtime standard
        else:
            nchw_data = input_data
        
        # if (input_data.shape[1] == shape[1] and input_data.shape[3] == shape[3]):
        #     nchw_data = input_data
        # else:

        #     else:
        #         nchw_data = input_data
            
        unconcatenated_batch_data.append(nchw_data)
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    print (batch_data.shape)
    return batch_data


class InputReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str, model_name: str, model_library: str):
        self.enum_data = None
        self.model_name = model_name
        self.model_library = model_library

        # Use inference session to get input shape.
        onnx_model = onnx.load(model_path)
        session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=['CPUExecutionProvider']) # , provider_options=[{"device_type": "GPU_FP32"}]
        # TODO: Add condition for dimensions
        shape = session.get_inputs()[0].shape

        if shape[-1] is None:
            height = 224
            width = 224
            shape = (1, 3, 224, 224)
        elif shape[-1] < shape[-2]:
            (_, height, width, _) = shape
        else:
            (_, _, height, width) = shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, height, width, size_limit=0, shape=shape, model_name=self.model_name, model_library=model_library
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)
        # print(self.datasize)

    def get_next(self):
        if self.enum_data is None:
            # , "image_shape": [[224, 224]] # OD
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
            
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None