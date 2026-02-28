
# SeQTO: A Selective Quantization Tuner for ONNX Models
SeQTO is a tool for deep neural network (DNN) quantization analysis, enabling step-by-step selective quantization and profiling to identify the optimal quantized model. It supports the generation of selectively quantized models, deployment on CPUs (via `ONNX Runtime`) and GPUs (via `Apache TVM`), and evaluation across datasets. SeQTO also applies multi-objective Pareto Front analysis to guide the selection of the best quantization candidate.
SeQTO allows the fetching of models via the ONNX Model hub of via local file. In addition, it supports the run of both static and dynamic quantization in the ONNX quantizer.

![SeQTO](https://github.com/luludak/SeQTO/blob/main/SeQTO.png)

## Purpose

The primary purpose of SeQTO is to support multi objective task minimization by enabling selective quantization of ONNX models for both CPU and GPU targets. SeQTO analyzes model layers using error metrics to identify those most likely to negatively impact outcomes such as accuracy degradation. Based on this ranking, it applies selective quantization incrementally, quantizing one layer per cycle while accumulating prior changes, and profiles the effect of each cycle on metrics such as accuracy and model size. This process helps users identify which layers contribute most effectively to quantization while minimizing undesired side effects, particularly accuracy loss. In addition, SeQTO is able to visualize the results, highlighting objective minimization via Pareto Front. For Example (EfficientNet-Lite4, deployed using TVM on a Mali GPU, using Static Quantization):

![Visual](https://github.com/luludak/SeQTO/blob/main/results/visual/d111689907c06eea7c82e4833ddef758da6453b9d4cf60b7e99ca05c7cbd9c12_efficientnet-lite4-11_upd_tvm_static_out_new.jpg)

## Features
- Support of ONNX models being deployed on CPUs (via ONNX Runtime) and GPUs (via Apache TVM) - including mobile devices.
- Support of both static and dynamic model quantization.
- Automatic model fetching from the official ONNX Model Hub.
- Automatic model analysis to determine potentially error-prone layers upon model quantization.
- Support of chunking runs, allowing the completion of demanding results upon unstable setups (e.g., RPC communication with a mobile GPU).
- Automatic visual profiling of the generated results.
- Multi-objective Pareto Front profiling, currently supporting accuracy, model size and execution times.

## Tool Adaptability & Generalizability
SeQTO relies on ONNX and the ONNX Quantizer to load and quantize models, which makes it inherently extensible beyond the sample models used in the paper. For deployment, it uses ONNX Runtime for CPU execution and supports GPU deployment through TVM, a general-purpose compiler framework. Overall, this design makes SeQTO broadly generalizable to other model architectures and families, provided they are supported by ONNX and/or TVM.

To apply SeQTO to new model types, two requirements must be met. First, since SeQTO performs differential testing, appropriate comparators must be implemented to evaluate model outputs. These can be defined in the evaluators folder and adapted depending on the task (e.g., classification, detection, text generation). Second, a suitable test dataset must be provided. For image-based models such as object detectors, users can simply supply a different image folder. For other modalities, users can integrate alternative datasets, for example via libraries such as datasets.

This extension is entirely feasible. In another project we implemented, [DiTOX](https://github.com/luludak/DiTOX), which also relies on ONNX and ONNX Runtime for differential testing of ONNX models, we implemented comparators and datasets for object detection and text generation tasks. These comparators can be reused or adapted to suit different use cases.

In summary, SeQTO is generalizable to a wide range of model types and families, as long as (1) suitable comparators are implemented and (2) an appropriate evaluation dataset is defined.

## Installation
*Important Note: the project dependencies have been recently updated and tested using `venv` in Ubuntu 22.04 and Mac OS Monterey, with the dependencies being cross-checked there.
We expect the system to work in newer versions without problems, however troubleshooting might be required e.g., for package updates. In all cases, the system is provided as-is, with no guarantees.
Also, make sure the right permissions are set to your folder as you install the necessary packages.*

We recommend that you use a virtual environment tool, such as `venv`, which we also recommend.
Using  `venv`, create a new environment by doing `python -m venv .venv`, and then activate the environment by doing `. .venv/bin/activate`.

Inside the new environment, nstall necessary packages using `pip`, by doing `pip install -r requirements.txt`. Important: the system is tested using `Python 3.10`, which is the recommended version.
If you want to run your model in TVM, you will need to compile TVM passing the necessary flags for the backend required (e.g., OpenCL). More instructions can be found [here](https://tvm.apache.org/docs/install/from_source.html).

SeQTO can work using ONNX and ONNXRuntime without having TVM installed `https://tvm.apache.org/`.
However, if you want to run on Android devices, install TVM and follow the [TVM tutorial](https://tvm.apache.org/docs//v0.15.0/how_to/deploy_models/deploy_model_on_android.html) for that purpose.


## Usage
Run by doing `python main.py`. This will run the system without visualization features.
You can also setup the system in the `config.json` file.

To visualize the results, pass the following arguments.
`-v/--visualize <runs_folder_path>`: Visualize data from file.
    
`-s/-save_only`: Visualize and save figures (no show).

`-s/-show_only`: Visualize and show figures (no save).

After determining the selective quantization order by analyzing error-prone layers, SeQTO fully quantizes the model and performs differential testing against the original non-quantized model using metrics such as accuracy and model size. It then enters an iterative process in which it progressively excludes layers from quantization, one cycle at a time, while automatically recording the selected metrics relative to the original model.

The process completes when all layers have been excluded or when the accuracy difference reaches zero compared to the original model. Upon completion, SeQTO generates a JSON report for result visualization, while also compiuting and visualizing the top-3 Pareto-optimal solutions.

## Configuration
The tool can be configured by updating the values in `config.json` file. By setting `run_type` to ONNX the models will be deployed on CPU, while setting `tvm` will attempt compilation and GPU deployment using Apache TVM and RPC. If installed and used, the TVM settings are set in the `tvm` object.

In addition, SeQTO supports the fetching of multiple models via ONNX Hub. You can filter these models accordingly via the `onnx_hub` object. Alternatively, you can disable this setting, create a folder named  `local_models` and put your ONNX models there for your runs.

The `images` object allows the definition of the dataset folder, along with the calibration datasets folders. It also allows the selection of random `K` samples (e.g., in cases of large models), but also chunking each run in case the run is unstable (e.g., TVM RPC timeouts to a mobile device). The user can define any dataset by adding his/her own images and specifying the folder in this configuration element.

You can also use the `onnx` object to configure (1) the desired accuracy `threshold` for the experiments to stop, (2) if additional ONNX `preprocessing` will be applied to the models (apart for the per-input pre-processing), and (3) whether the model should be statically quantized by setting the `quantize_static` property to `true`, or dynamically quantized by setting it to `false`.

## Repository Information

```
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                          17            618            279           1948
JSON                             1              1              0             51
Markdown                         1             22              0             51
Text                             1              4              0             31
-------------------------------------------------------------------------------
SUM:                            20            645            279           2081
-------------------------------------------------------------------------------
```

## Results
The results of our experiments can be found in the `results` folder, both in raw `data` and `visual` formats - each on their respective subfolders. We also provide a summary file for all the runs.


## Notes
Any sample dataset images provided in the repository are sourced from the internet, and their copyright(s) belong to their owner(s). The only reason of provision is purely for non-commercial, demonstration purposes of DiTOX (to showcase how it works), with respect to the copyright(s) of their owner(s) and no intention to infringe them.

