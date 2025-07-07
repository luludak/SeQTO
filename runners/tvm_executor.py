import math

import os

os.environ["TVM_LOG_DEBUG"] = "1"
os.environ["TVM_BACKTRACE"] = "1"

import tvm

import json
import traceback
import tvm.runtime as runtime
import onnxruntime as ort
import numpy as np
from tvm.contrib import graph_executor
from tvm.contrib import utils, graph_runtime
from tvm import rpc
from PIL import Image
from helpers.time_helper import TimeHelper
from scipy.special import softmax
from os import listdir
from os.path import isfile, join, split

from processors.model_preprocessor import ModelPreprocessor

from tvm.contrib.debugger.debug_executor import GraphModuleDebug

from tensorflow import keras
from keras import backend as K
from keras.models import load_model
import tensorflow.compat.v1 as tf
import torch
import torchvision.models as torchvision_models

from tensorflow.python.platform import gfile

class Executor:
    def __init__(self, models_data, images_data, connection_data):
        self.model_name = models_data["model_name"]
        self.model_path = models_data["model_path"]
        self.graph_path = models_data["graph_path"]
        self.params_path = models_data["params_path"]
        self.input_dimension = models_data["input_dimension"]
        self.input = models_data["input"]
        self.input_name = models_data["input_name"]
        self.output = models_data["output"]
        self.library = models_data["library"] if "library" in models_data else None
        self.connection_id = "local"
        self.connection_data = connection_data
        self.input_images_folders = images_data["input_images_folders"]
        self.error_base_folder = split(self.model_path)[0] + "/errors"
        self.module = None
        self.device = None
        self.loaded_module = None
        self.time = TimeHelper()
        self.module_debug = None

    def prepare(self, build):
        key = build["key"]
        tracker = rpc.connect_tracker(str(build["address"]), int(build["port"]))
        remote = tracker.request(key, priority=0, session_timeout=99999999)

        device_id = self.connection_data["id"] if "id" in self.connection_data else 0
        host_type = self.connection_data["host_type"] or None
        if(host_type == "local_no_rpc" or host_type == "local" or remote == None):
            self.loaded_module = runtime.module.load_module(self.model_path)
            target_framework = self.connection_data["target_framework"]
            self.device = tvm.device(str(target_framework), device_id)
            self.module = graph_executor.GraphModule(self.loaded_module["default"](self.device))

        else:
            remote.upload(self.model_path)
            graph = open(self.graph_path).read()
            params = bytearray(open(self.params_path, "rb").read())
            # TODO: Load Graph and Params.
            dir_path, lib_name = split(self.model_path)
            target_framework = self.connection_data["target_framework"]
            self.loaded_module = remote.load_module(lib_name)

            self.device = remote.device(target_framework, device_id)

            self.module = graph_executor.GraphModule(self.loaded_module["default"](self.device))

        return self

    def execute(self):
        if(self.module == None or self.device == None):
            raise Exception("Error: Device not initialized.")
        return "ts_0"

    def extract_image_names(self, folder_path):
        return [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    def process_images(self):
        self.process_images_with_output(module, self.input_images_folder, self.output_images_folder)
        return self

    def get_last_folder(self, folder_path):
        return os.path.basename(os.path.normpath(folder_path))

    def process_images_with_io(self, input_folder, output_folder=None, model_name=None, variant=None, specific_images=None, include_certainties = False, should_write_to_file=False):

        if (model_name == None):
            model_name = self.model_name

        if should_write_to_file:
            os.makedirs(output_folder, exist_ok=True)

        print("Processing Dataset images in folder: " + input_folder)

        if specific_images is None:
            image_names = self.extract_image_names(input_folder)
        else:
            print("(Image subset considered.)")
            image_names = specific_images

        image_names_length = len(image_names)
        image_length = (image_names_length - 1) if (image_names_length > 1) else image_names_length
        step = (image_length // 4) if image_length > 4 else image_length
        
        count = 0
        same_ranks = 0
        prev_ranks = None

        input_shape_inferred = self.module.get_input(0).shape

        data = {
            "input": input_shape_inferred,
            "input_dimension": self.input_dimension,
            "library": self.library
        }
        model_preprocessor = ModelPreprocessor(data)
        error_occured = False
        errors_no = 0
        output_object = {}
        exec_times = {}

        for image_name in image_names:

            if(count % step == 0):
                print("Complete: " + str((count // step) * 25) + "%")

            count += 1
            if (image_name.startswith(".")):
                print("Skipping " + image_name)
                continue

            img_path = input_folder + "/" + image_name
           
            img = Image.open(img_path)
            if(img.mode == "L"):
                img = img.convert("RGB")

            img = img.resize(self.input_dimension)
            img = model_preprocessor.preprocess(self.model_name, img)

            image_name_extracted = image_name.rsplit('.', 1)[0]

            if should_write_to_file:
                output_file_path = output_folder + "/" + image_name_extracted.replace(".", "_") + ".txt"
            
            try:
                start_timestamp = self.time.get_epoch_timestamp(False)

                self.module.set_input(self.input_name, img)
                self.module.run()
                out = tvm.nd.empty(self.output, device=self.device)
                tvm_output = self.module.get_output(0, out).numpy()
                end_timestamp = self.time.get_epoch_timestamp(False)

                scores = softmax(tvm_output)
                if(len(scores) > 2):
                    scores = np.squeeze(scores, [0, 2])

                scores = np.squeeze(scores)
                ranks = np.argsort(scores, kind='stable')[::-1]
                extracted_ranks = ranks[0:5]

                if(prev_ranks is not None and np.array_equal(prev_ranks, extracted_ranks)):
                    same_ranks += 1
                else:
                    prev_ranks = extracted_ranks
                    same_ranks = 0

                if(same_ranks == 100):
                    print ("Warning: Execution produced same ranks for 100 consecutive inputs. Stopped.")
                    return self

                if (should_write_to_file):
                    with open(output_file_path, 'w') as output_file:

                        for rank in extracted_ranks:
                            print("%s, %f" % (rank, scores[rank]), file = output_file)
                        
                        print("\n%f" % (end_timestamp - start_timestamp), file = output_file)
                        output_file.close()
                
                if include_certainties:
                    output_object[image_name_extracted] = [(rank, str(scores[rank])) for rank in extracted_ranks.tolist()]
                else:
                    output_object[image_name_extracted] = extracted_ranks.tolist()
                exec_times[image_name_extracted] = end_timestamp - start_timestamp

            except Exception as e:
                ts_floor = str(math.floor(self.time.get_epoch_timestamp(False)))
                errors_no += 1
                if(not error_occured):
                        error_occured = True
                        print("One or more image errors have occured. See execution log of the model for details (ts: " + ts_floor + ").")
                
                if(errors_no == 100):
                    print ("100 Errors have occured. Stopping execution.")
                    return self

                folder_to_write = join(self.error_base_folder, ts_floor)
                os.makedirs(folder_to_write, exist_ok=True)
                f = open(join(folder_to_write, 'error_log.txt'), 'a+')
                f.write("Model: " + model_name + "\nOutput folder:" + output_folder + "\nImage: "+ img_path + "\n")
                traceback.print_exc(file=f)
                f.close()

        
        return {
            "output": output_object,
            "times": exec_times
        }

