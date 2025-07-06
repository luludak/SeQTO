# A SeQTO: Selective Quantization Tuner for ONNX Models
SeQTO is a tool that allows DNN model quantization analysis, step-by-step selective quantization, and profiling.

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


## Results
The results of our experiments can be found in the `results` folder, both in raw `data` and `visual` formats - each on their respective subfolders.
