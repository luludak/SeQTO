
from runners.tvm_objects import ObjectsExecutor

class TVMRunner:

    def __init__(self, build_config):
        self.build = build_config

    def execute_tvm(self, models_data, images_data, specific_images):
        objectsExecutor = ObjectsExecutor(models_data, images_data, self.build)
        return objectsExecutor.execute(self.build, specific_images)