import sys
sys.path.append("..")

from .tvm_executor import Executor

import os

class ObjectsExecutor(Executor):
    def __init__(self, models_data, images_data, connection_data):
        Executor.__init__(self, models_data, images_data, connection_data)

    def execute(self, build, specific_images=None):
        device_id = self.connection_data["id"] if "id" in self.connection_data else 0
        timestamp_str = str(self.time.get_epoch_timestamp()) + "_device" + str(device_id)
        
        self.prepare(build)
        print("Executing model " + self.model_name + ", execution timestamp: " + timestamp_str)
    
        return self.process_images_with_io(self.input_images_folders[0], "", self.model_name, "", specific_images, should_write_to_file=False)
