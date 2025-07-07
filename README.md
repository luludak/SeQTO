# SeQTO: A Selective Quantization Tuner for ONNX Models
SeQTO is a tool for deep neural network (DNN) quantization analysis, enabling step-by-step selective quantization and profiling to identify the optimal quantized model. It supports the generation of selectively quantized models, deployment on CPUs (via ONNX Runtime) and GPUs (via Apache TVM), and evaluation across datasets. SeQTO also applies multi-objective Pareto Front analysis to guide the selection of the best quantization candidate.

## Installation
Install necessary packages using `pip`, by doing `pip install -r requirements.txt`.
If you want to run your model in TVM, you will need to compile TVM passing the necessary flags for the backend required (e.g., OpenCL). More instructions can be found [here](https://tvm.apache.org/docs/install/from_source.html).
In addition, if you want to run on Android devices, follow the [TVM tutorial](https://tvm.apache.org/docs//v0.15.0/how_to/deploy_models/deploy_model_on_android.html) for that purpose.

## Usage
Run by doing `python main.py`. This will run the system without visualization features.
You can also setup the system in the `config.json` file.

To visualize the results, pass the following arguments.
`-v / --visualize <runs_folder_path>`: Visualize data from file.
    
`-s/-save_only`: Visualize and save figures (no show).

`-s/-show_only`: Visualize and show figures (no save).

## Configuration
The tool can be configured by updating the values in `config.json` file. By setting `run_type` to ONNX the models will be deployed on CPU, while setting `tvm` will attempt compilation and GPU deployment using Apache TVM
and RPC. The TVM settings are set in the `tvm` object.

In addition, SeQTO supports the fetching of multiple models via ONNX Hub. You can filter these models accordingly via the `onnx_hub` object.

The `images` object allows the definition of the dataset folder, along with the calibration datasets folders. It also allows the selection of random `K` samples (e.g., in cases of large models), but also chunking each run
in case the run is unstable (e.g., TVM RPC timeouts to a mobile device).

## Results
The results of our experiments can be found in the `results` folder, both in raw `data` and `visual` formats - each on their respective subfolders.
