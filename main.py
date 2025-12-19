import numpy as np
np.float_ = np.float64

import os
import argparse
import onnx
import onnxruntime
import json
import time
import hashlib
import subprocess
from os import listdir
from os.path import isfile, join, split #, exists, normpath, basename
from helpers.model_helper import load_config
from benchmarking.model_benchmarking import benchmark
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantFormat, QuantType
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations, compute_activation_error, compute_weight_error,
    create_activation_matching, create_weight_matching,
    modify_model_output_intermediate_tensors)
from helpers.model_helper import get_size
from helpers.value_helper import ValueHelper
from onnx import version_converter
from onnx import hub
import random

from readers import input_reader
from builders.tvm_builder import TVMBuilder
from runners.tvm_runner import TVMRunner
from runners.onnx_runner import ONNXRunner

import matplotlib.pyplot as plt

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

from argparse import ArgumentParser

def _generate_aug_model_path(model_path: str) -> str:
    aug_model_path = (
        model_path[: -len(".onnx")] if model_path.endswith(".onnx") else model_path
    )
    return aug_model_path + ".save_tensors.onnx"

def normalize_value(value, max_value, min_value):
    return (value - min_value) / (max_value - min_value)

def get_model_path(script_dir, model_obj):
    return script_dir + "/models_cache/" + model_obj.model_path.replace("/model/", "/model/" + \
            model_obj.metadata["model_sha"] + "_").replace("/models/", "/models/" + \
            model_obj.metadata["model_sha"] + "_").replace("/preproc/", "/preproc/" + \
            model_obj.metadata["model_sha"] + "_")

def plot_by_non_domination_rank(X, orig_X, labels):
    X = np.array(X)
    orig_X = np.array(orig_X)
    nums = [i for i in range(len(X))]

    palette = [
        (0.172, 0.627, 0.172, 0.8),  # green
        (1.000, 0.498, 0.054, 0.8),  # orange
        (0.121, 0.466, 0.705, 0.8),  # blue
        (0.839, 0.153, 0.157, 0.8),  # red
        (0.580, 0.404, 0.741, 0.8),  # purple
        (0.549, 0.337, 0.294, 0.8),  # brown
        (0.890, 0.467, 0.761, 0.8),  # pink
        (0.498, 0.498, 0.498, 0.8),  # gray
        (0.737, 0.741, 0.133, 0.8),  # olive
        (0.090, 0.745, 0.811, 0.8),  # cyan
    ]

    for num_index in range(len(labels)):
        plt.plot(nums, X[:, num_index], color=palette[num_index], linewidth=3, label=labels[num_index])

    plt.xlabel("Index", fontsize=14)
    plt.ylabel("Normalized Range (%)", fontsize=14)

    nds = NonDominatedSorting()
    # TODO: Add weights in parameters for pareto.
    
    front = nds.do(X, only_non_dominated_front=False)
    ideal = np.min(X, axis=0)  # Best values in each objective
    distances = np.linalg.norm(X - ideal, axis=1)

    # Rank from best (closest) to worst.
    # Utilize point distance (Euclidean) from optimal point.
    ranking = np.argsort(distances)
    ranking_in_front = [e for e in ranking if e in front[0]]

    threshold = 3
    colors = [(0, 0, 0.5, alpha) for alpha in np.linspace(1, 0.3, threshold)]

    pareto_fronts = []
    for i in range(len(ranking_in_front)):
        if i >= threshold:
            break

        plt.axvline(x=ranking_in_front[i], color=colors[i], linestyle="--", label=f"Pareto #" + str(i + 1))
        print("Pareto " + str(i + 1))
        print("Version Index: " + str(ranking_in_front[i]))
        pareto_front = {
            "Pareto ID": str(i + 1),
            "Version Index": str(ranking_in_front[i])
        }
        for lb_i in range(len(labels)):
            print(labels[lb_i] + ": " + str(orig_X[ranking_in_front[i], lb_i]))
            pareto_front[labels[lb_i]] = str(orig_X[ranking_in_front[i], lb_i])

        pareto_fronts.append(pareto_front)


    plt.legend()
    plt.grid()
    plt.plot()
    
    return pareto_fronts

def min_max_normalize_value(array, value, maximize=True, append_value_to_array=False):
    """
    Normalize a single value based on min-max normalization using a reference array.

    Parameters:
    - array: list or 1D array of reference values
    - value: the value to normalize
    - maximize: True if the objective is to be maximized, False if minimized
    - append_value_to_array: Whether or not the value should be appended to the array.

    Returns:
    - normalized_value: float in [0, 1]
    """
    if append_value_to_array:
        array.append(value)

    min_val = min(array)
    max_val = max(array)

    if max_val == min_val:
        return 50  # or another default when no variation

    if maximize:
        return ((value - min_val) / (max_val - min_val)) * 100
    else:
        return ((max_val - value) / (max_val - min_val)) * 100

def normalize_objectives(data, maximize=[True, True, True]):
    """
    Min-max normalize each column of a 2D array based on optimization direction.

    Parameters:
    - data: np.ndarray of shape (n_samples, n_objectives)
    - maximize: list of booleans, where True means the objective is to be maximized,
                and False means it is to be minimized.

    Returns:
    - normalized_data: np.ndarray of shape (n_samples, n_objectives), values in [0, 1]
    """
    data = np.array(data, dtype=float)
    normalized = np.zeros_like(data)

    for i in range(data.shape[1]):
        col = data[:, i]
        min_val = np.min(col)
        max_val = np.max(col)

        if max_val == min_val:
            # Avoid division by zero; set to 0.5 (or whatever constant you'd prefer)
            normalized[:, i] = 50
        elif maximize[i]:
            normalized[:, i] = ((col - min_val) / (max_val - min_val)) * 100
        else:
            normalized[:, i] = ((max_val - col) / (max_val - min_val)) * 100

    return normalized

def main():

    # Generic config.
    config = load_config('./config.json')
    parser = ArgumentParser()

    parser.add_argument("-v", "--visualize", dest="visualize",
                    help="Visualize data from file.")
    
    parser.add_argument("-s", "--save_only", action="store_true",
        help="Visualize and show figures.")
    
    parser.add_argument("-so", "--show_only", action="store_true",
        help="Visualize and show figures (no save).")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # TODO: Refactor - Move visualizer into separate class.
    # This will be completed before tool publication.
    if args.visualize:
        labels = config["visualizer"]["labels"]
        active_labels = config["visualizer"]["active_labels"]

        visualize_path = script_dir + "/" + args.visualize
        file_name = os.path.basename(visualize_path)
        plt.legend(loc='lower left') 
        if os.path.isfile(visualize_path):
        
            json_data = open(visualize_path, "r")
            data = json.load(json_data)
            dissimilarities = [float(d) for d in data["dissimilarities"]]
            quant_size = [float(s.replace(" MB", "")) for s in data["size"]["quantized"]]

            exec_times = [float(t) for t in data["benchmarks"]["quantized"]]
            base_exec_time = float(data["benchmarks"]["original"])
            base_exec_time_normalized = min_max_normalize_value(exec_times, base_exec_time)

            data = (dissimilarities, quant_size, exec_times)
            orig_X = [list(d) for d, active in zip(data, active_labels) if active]
            if len(orig_X) == 0:
                print("No active metrics defined to visualize. Exiting...")
                return
            X = normalize_objectives(orig_X)
            title = file_name.split("_", 1)[1].replace(".json", "").\
                         replace("_", " ").replace("out", "")
            fig = plt.figure()
            fig.suptitle(title, fontsize=16)
            
            if (active_labels[2]):
                plt.axhline(y=base_exec_time_normalized, color="black", linestyle="--", label=f"Base Exec. Time")
            
            plot_by_non_domination_rank(X, orig_X, [l for i, l in enumerate(labels) if active_labels[i]])
            file_to_save = visualize_path.replace(".json", ".jpg")
            
            if not args.show_only:
                plt.gcf().set_size_inches(14, 8)
                plt.savefig(file_to_save)
                print("Figures generated in the same directory as the input file.")

            if not args.save_only:
                plt.show()


        else:
            #  and f.endswith(".json")
            json_files = [f for f in listdir(visualize_path) if isfile(join(visualize_path, f))]
            fig_count = 0
            all_pareto_fronts = {}
            for json_file in json_files:
                if not json_file.endswith(".json") or json_file.endswith("summary.json"):
                    continue

                json_file_path = join(visualize_path, json_file)
                json_data = open(json_file_path, "r")
                data = json.load(json_data)
                dissimilarities = [float(d) for d in data["dissimilarities"]]
                quant_size = [float(s.replace(" MB", "")) for s in data["size"]["quantized"]]

                exec_times = [float(t) for t in data["benchmarks"]["quantized"]]
                
                base_exec_time = float(data["benchmarks"]["original"])
                base_exec_time_normalized = min_max_normalize_value(exec_times, base_exec_time)

                data = (dissimilarities, quant_size, exec_times)
                orig_X = [list(d) for d, active in zip(data, active_labels) if active]
                orig_X = list(zip(*orig_X))
                print (orig_X)

                if len(orig_X) == 0:
                    print("No active metrics defined to visualize. Exiting...")
                    return
                X = normalize_objectives(orig_X)

                fig = plt.figure(fig_count)
                if (active_labels[2]):
                    plt.axhline(y=base_exec_time_normalized, color="black", linestyle="--", label=f"Base Exec. Time")
                fig_count += 1
                title = json_file.split("_", 1)[1].\
                             replace(".json", "").replace("_", " ").replace("out", "")
                print(title)
                # fig.suptitle(title, fontsize=16)
                # plot_by_non_domination_rank(X, orig_X, labels)
                pareto_fronts = plot_by_non_domination_rank(X, orig_X, [l for i, l in enumerate(labels) if active_labels[i]])
                all_pareto_fronts[title] = {
                    "all_values": orig_X,
                    "fronts": pareto_fronts
                }
                file_to_save = json_file_path.replace(".json", ".jpg")
                # print(file_to_save)
                
                plt.legend(loc='lower left', prop={'size': 14}) 
                if not args.show_only:
                    plt.gcf().set_size_inches(14, 8)
                    plt.savefig(file_to_save, bbox_inches='tight')
            out_json = json.dumps(all_pareto_fronts, indent=2)
            with open(visualize_path + "/summary.json", "w") as outfile:
                outfile.write(out_json)
            if not args.show_only:
                print("Figures generated in the same directory as the input file(s).")
            
            if not args.save_only:
                plt.show()
                
        return

    images_config = config["images"]
    calibr_images_folder = script_dir + images_config["calibration_images_rel_path"] # 
    images_folder = script_dir + images_config["images_folder_rel_path"] # '/images/ten/'
    small_calibr_images_folder = script_dir + images_config["small_calibr_images_folder_rel_path"] #'/images/ten'
    should_quantize_static = config["onnx"]["quantize_static"]

    run_type = config["run_type"]

    print ("Run Type: " + run_type)

    # Common to all models configuration
    device_name = config["tvm"]["devices"]["selected"]
    build = config["tvm"]["devices"][device_name]

    # RPC Setup

    tvm_runner = TVMRunner(build)
    onnx_runner = ONNXRunner({})

    value_helper = ValueHelper()

    hub.set_dir(script_dir + "/models_cache")

    models_to_run = []
    models_config = []

    hub_config = config["onnx_hub"]

    if hub_config["activate"]:
        start = hub_config["start"]
        end = hub_config["end"]
        all_models = hub.list_models()
        for i, model in enumerate(all_models):

            if i < start or i > end:
                continue
            
            # For Example: models to run from ONNX Model Repo:
            #  Opset 10: MobileNetV2, ShuffleNet-v2,
            #  Opset 11: EfficientNet-Lite4,
            #  Opset12: GoogleNet, Inception-1, MNIST, ResNet50-fp32
            if hub_config["filter"]["enabled"]:
                if hub_config["filter"]["name"] not in model.model or \
                    ("opset" in hub_config["filter"] and \
                    model.opset != int(hub_config["filter"]["opset"])):
                    continue

            models_to_run.append(get_model_path(script_dir, model))
            print("MODEL: " + str(model.model) + " OPSET: " + str(model.opset))
            print(model.metadata)
            io_ports = model.metadata["io_ports"] if "io_ports" in model.metadata else {
                "inputs": [{"name": "input", "shape": [1, 3, 224, 224]},\
                           {"name": "image_shape", "shape": [1, 3, 224, 224]}],
                "outputs": [{"name": "output", "shape": [1, 1000]}]
            }
            input_shape = io_ports["inputs"][0]["shape"]
            print(input_shape)
            if not isinstance(input_shape[0], (int, float)):
                input_shape[0] = 1
            input_has_numbers = all(isinstance(item, (int, float)) for item in input_shape) 

            model_config_to_append = {
                "model_name": model.model,
                "opset": model.opset,
                "input": input_shape,
                "input_name": io_ports["inputs"][0]["name"],
                "output": [224 if e == None else (e if value_helper.is_int(e) else 1) for e in io_ports["outputs"][0]["shape"]],
            }

            if input_has_numbers:
                if input_shape[2] > 224 and input_shape[3] > 224:
                    model_config_to_append["input_dimension"] = [input_shape[2], input_shape[3]]
                else:
                    model_config_to_append["input_dimension"] = [max(input_shape), max(input_shape)]
            else:
                model_config_to_append["input_dimension"] = [224, 224]

            models_config.append(model_config_to_append)
    else:
        models_to_run = [script_dir + "/local_models/" + m for m in listdir(script_dir + "/local_models") if m.endswith(".onnx") and "_quant" not in m]
        models_config = [load_config(m) for m in models_to_run]
    for i, model_to_run in enumerate(models_to_run):
        
        tvm_builder = TVMBuilder({"build": build})
        float_model_path = model_to_run

        # Load model so that it can be found locally.
        # Note: normally, the hub API is checking for the file,
        # But also performs checksum check. Given that we upgrade opset to 11+,
        # We need to check manually.

        # TODO: Refactor.
        qdq_model_path = float_model_path.replace(".onnx", "_quant.onnx")
        model_config = models_config[i]

        # print(not os.path.exists(float_model_path))
        if not os.path.exists(float_model_path):
            hub.load(model_config["model_name"], opset=model_config["opset"])

        model = onnx.load(float_model_path)
        graph = model.graph
        nodes = graph.node

        float_named_model_path = float_model_path.replace(".onnx", "_upd.onnx")

        count = 0
        # Process models, so that, if nodes have no names, they are assigned with one.
        for node in nodes:
            if not node.name:
                node.name = "rand_node_name_" + str(count)
                count = count + 1
        onnx.save(model, float_named_model_path)

        float_model_path = float_named_model_path

        if model_config["opset"] < 11:
            print("Model opset found below 11. Upgrading model opset to 11...")
            float_model_to_upgrade = onnx.load(float_model_path)
            converted_model = version_converter.convert_version(float_model_to_upgrade, 11)
            onnx.save(converted_model, float_model_path)

        # Explicit setting for preprocessing:
        model_config["library"] = config["options"]["preprocessing_setting"]

        shape = model_config["input"]
        input_dimension = model_config["input_dimension"]
        
        float_tvm_path = None
        images_paths = None

        if run_type == "tvm":
            images_paths = [f for f in listdir(images_folder) \
                        if isfile(join(images_folder, f))]

        else:
            images_paths = [join(images_folder, f) for f in listdir(images_folder) \
                        if isfile(join(images_folder, f))]

        
        calibration_dataset_path = calibr_images_folder #args.calibrate_dataset
        small_calibration_dataset_path = small_calibr_images_folder

        images_chunk = images_config["chunk"]
        start = 0
        end = start + images_chunk
        total_end = len(images_paths)

        base_model_out = {}
        base_model_times = {}
        base_times_list = []

        postfix = run_type + "_" + ("static" if should_quantize_static else "dynamic")

        base_run_file = float_model_path.replace(".onnx", "_" + postfix + "_run.json")
        float_pre_model_path = float_model_path.replace(".onnx", "_pre.onnx")
        result_object_file = float_model_path.replace(".onnx", "_" +  postfix + "_out.json")

        if config["onnx"]["preprocess_model"]:
            try:
                p = subprocess.Popen(['python3 -m onnxruntime.quantization.preprocess --input ' + float_model_path + " --output " + float_pre_model_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
                (output, err) = p.communicate()  
                p.wait()
                if p.returncode != 0:
                    print(output)
                print ("Preprocessed model for quantization.")
                # io_ports["inputs"].append({"name": "OC2_DUMMY_1", "shape": [1, 1]})

            except subprocess.CalledProcessError as e:
                print('Fatal error: code={}, out="{}"'.format(e.returncode, e.output))

        if os.path.exists(base_run_file):
            print("Loading cached data for base model...")
            base_file_data = open(base_run_file, "r")
            json_data = json.load(base_file_data)
            base_model_out = json_data["base_model_out"]
            base_model_times = json_data["base_model_times"]
            base_times_list = json_data["base_times_list"]
            float_pre_model_path = float_model_path

        else:
            # Preprocess, then use this version.

            float_pre_model_path = float_model_path
            if run_type == "tvm":
                tvm_path = os.path.dirname(float_pre_model_path)
                (float_tvm_path, graph_path, params_path) = tvm_builder.build_tvm(float_pre_model_path, tvm_path, shape)
                
            while start < total_end:
                images_paths_chunk = images_paths[start:end]
                
                print ("Running original model from " + str(start) + " to " + str(end))

                if run_type == "onnx":
                    result = onnx_runner.execute_onnx_model(onnx.load(float_pre_model_path), images_paths_chunk, config={
                        "input_name": model_config["input_name"],
                        "input_shape": shape,
                        "input_dimension": input_dimension,
                        "model_name": model_config["model_name"]
                    })
                    base_model_out.update(result["output"])
                    base_model_times.update(result["times"])
                    base_times_list.extend(list(result["times"].values())[1:])
                else:
                    model_config["model_path"] = float_tvm_path
                    model_config["graph_path"] = graph_path
                    model_config["params_path"] = params_path

                    tvm_images_data = {
                        "input_images_folders" : [images_folder]
                    }

                    tvm_runner = TVMRunner(build)
                    result = tvm_runner.execute_tvm(model_config, tvm_images_data, images_paths_chunk)
                    base_model_out.update(result["output"])
                    base_model_times.update(result["times"])
                    base_times_list.extend(list(result["times"].values())[1:])

                start = start + images_chunk
                end = end + images_chunk

            out_json = json.dumps({
                "base_model_out": base_model_out,
                "base_model_times": base_model_times,
                "base_times_list": base_times_list
            }, indent=2)
            with open(base_run_file, "w") as outfile:
                outfile.write(out_json)

        total_dissimilar_percentage = 100
        threshold = config["onnx"]["threshold"]
        new_quant_model_path = qdq_model_path
        prev_quant_model_path = None
        
        all_dissimilarities = []
        quantized_benchmark = []
        quantized_sizes = []
        wlist = None
        actlist = None
        nodes_to_exclude = None
        skipped_nodes = None

        input_data_reader = input_reader.InputReader(
            calibration_dataset_path, float_pre_model_path, model_config["model_name"], model_config["library"]
        )

        small_input_data_reader = input_reader.InputReader(
            small_calibration_dataset_path, float_pre_model_path, model_config["model_name"], model_config["library"]
        )

        if not os.path.exists(result_object_file):

            # Config nodes considered only if no previous run is set.
            nodes_to_exclude = config["options"]["nodes_to_exclude"]
            skipped_nodes = config["options"]["skipped_nodes"]

            quantize_static(
                float_pre_model_path,
                qdq_model_path,
                input_data_reader,
                quant_format=QuantFormat.QDQ,
                per_channel=False,
                weight_type=QuantType.QInt8,
                nodes_to_exclude=nodes_to_exclude
            )

            # Perform activation comparison.
            aug_float_model_path = _generate_aug_model_path(float_model_path)
            modify_model_output_intermediate_tensors(float_model_path, aug_float_model_path)
            small_input_data_reader.rewind()
            float_activations = collect_activations(aug_float_model_path, small_input_data_reader)

            aug_qdq_model_path = _generate_aug_model_path(qdq_model_path)
            modify_model_output_intermediate_tensors(qdq_model_path, aug_qdq_model_path)
            small_input_data_reader.rewind()
            qdq_activations = collect_activations(aug_qdq_model_path, small_input_data_reader)

            act_matching = create_activation_matching(qdq_activations, float_activations)
            act_error = compute_activation_error(act_matching)

            xerr_list = [x[1]['xmodel_err'] for x in act_error.items()]
            xmodel_max = np.max(xerr_list)
            xmodel_min = np.min(xerr_list)

            qdq_err_list = [x[1]['qdq_err'] for x in act_error.items()]
            qdq_model_max = np.max(qdq_err_list)
            qdq_model_min = np.min(qdq_err_list)  

            act_error_new = {}
            w_error_new = {}

            for key in act_error:
                elem = act_error[key]
                # print(elem)
                norm_xerr = normalize_value(elem['xmodel_err'], xmodel_max, xmodel_min)
                norm_qdq_err = normalize_value(elem['qdq_err'], qdq_model_max, qdq_model_min)
                act_error_new[key] = (0.5*norm_xerr + 0.5*norm_qdq_err)

            actlist = sorted(act_error_new.items(), key = lambda x: x[1],reverse=True)

            matched_weights = create_weight_matching(float_model_path, new_quant_model_path)
            weights_error = compute_weight_error(matched_weights)
            
            w_list = [x[1] for x in weights_error.items()]
            max_w_err = np.max(w_list)
            min_w_err = np.min(w_list)

            for key in weights_error:
                elem = weights_error[key]
                w_error_new[key] = normalize_value(elem, max_w_err, min_w_err)

            wlist = sorted(w_error_new.items(), key = lambda x: x[1], reverse=True)

        else:
            print("Loading data from previous execution...")
            file_data = open(result_object_file, "r")
            result_object = json.load(file_data)
            actlist = result_object["activations"]
            wlist = result_object["weights"]
            nodes_to_exclude = result_object["excluded_nodes"]
            skipped_nodes = result_object["skipped_nodes"]
            all_dissimilarities = result_object["dissimilarities"]
            quantized_benchmark = result_object["benchmarks"]["quantized"]
            quantized_sizes = result_object["size"]["quantized"]
        
        float_model = onnx.load(float_pre_model_path)

        result_object = {
            "activations": actlist,
            "weights": wlist,
            "excluded_nodes": nodes_to_exclude,
            "skipped_nodes": skipped_nodes,
            "dissimilarities": all_dissimilarities,
            "benchmarks": {
                "original": sum(base_times_list)/len(base_times_list),
                "quantized": quantized_benchmark
            },
            "size": {
                "original": get_size(float_pre_model_path),
                "quantized": quantized_sizes
            }
        }

        actlist = actlist + wlist
        print(actlist)

        while True:

            new_node_found = None
            if os.path.exists(result_object_file):
                new_node_found = append_node_to_exclude(onnx_runner, float_model, actlist, nodes_to_exclude)

                if new_node_found == None:
                    print("No new node found. Completing process...")
                    break

            print("Building Quantized Model: " + new_quant_model_path)
            print(nodes_to_exclude)

            if not os.path.exists(float_pre_model_path):
                try:
                    p = subprocess.Popen(['python3 -m onnxruntime.quantization.preprocess --input ' + float_model_path + " --output " + float_pre_model_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
                    (output, err) = p.communicate()  
                    p.wait()
                    if p.returncode != 0:
                        print(output)
                    print ("Preprocessed Model for new quantization.")

                except subprocess.CalledProcessError as e:
                    print('Fatal error: code={}, out="{}"'.format(e.returncode, e.output))

            if should_quantize_static:
                input_data_reader.rewind()
                quantize_static(
                    float_pre_model_path,
                    new_quant_model_path,
                    input_data_reader,
                    quant_format=QuantFormat.QDQ,
                    per_channel=False,
                    weight_type=QuantType.QInt8,
                    nodes_to_exclude=nodes_to_exclude
                )
            else:
                quantize_dynamic(
                    float_pre_model_path,
                    new_quant_model_path,
                    per_channel=False,
                    weight_type=QuantType.QUInt8,
                    nodes_to_exclude=nodes_to_exclude
                )
            skip_execution = False
            if prev_quant_model_path is not None:
                prev_quant_model_hash = hashlib.md5(open(prev_quant_model_path,'rb').read()).hexdigest()
                new_quant_model_hash = hashlib.md5(open(new_quant_model_path,'rb').read()).hexdigest()

                if prev_quant_model_hash == new_quant_model_hash:
                    print("Model " + new_quant_model_path + " has the same hash has as the previous model. Skipping...")
                    skipped_nodes.append(new_node_found)
                    skip_execution = True

            if not skip_execution:
                
                # Rebuild new TVM instance.
                if run_type == "tvm":
                    tvm_path = os.path.dirname(new_quant_model_path)
                    (new_tvm_model_path, new_graph_path, new_params_path) = tvm_builder.build_tvm(new_quant_model_path, tvm_path, shape)
                
                start = 0
                end = start + images_chunk
                quant_model_out = {}
                quant_model_times = {}
                quant_times_list = []

                opt_images_paths = images_paths

                if images_config["random_enabled"]:
                    print ("Random sample enabled. Selecting " + str(images_config["random_k"]) + " images.")
                    opt_images_paths = random.choices(images_paths, k=images_config["random_k"])
                    total_end = images_config["random_k"]


                while start < total_end:
                    if end > total_end:
                        end = total_end
                    images_paths_chunk = opt_images_paths[start:end]
                    print ("Running quantized model from " + str(start) + " to " + str(end))
                    if run_type == "onnx":
                        result = onnx_runner.execute_onnx_model(onnx.load(new_quant_model_path), images_paths_chunk, config={
                            "input_shape": shape,
                            "input_dimension": input_dimension,
                            "model_name": model_config["model_name"]
                        })
                        quant_model_out.update(result["output"])
                        quant_model_times.update(result["times"])
                        quant_times_list.extend(list(result["times"].values())[1:])
                    else:
                        model_config["model_path"] = new_tvm_model_path
                        model_config["graph_path"] = new_graph_path
                        model_config["params_path"] = new_params_path
                        tvm_images_data = {
                            "input_images_folders" : [images_folder],
                        }

                        tvm_runner = TVMRunner(build)
                        result = tvm_runner.execute_tvm(model_config, tvm_images_data, images_paths_chunk)
                        
                        # Update Quant data lists with output and times.
                        quant_model_out.update(result["output"])
                        quant_model_times.update(result["times"])
                        quant_times_list.extend(list(result["times"].values())[1:])
                    start = start + images_chunk
                    end = end + images_chunk


                evaluation = onnx_runner.evaluate(base_model_out, quant_model_out)

                dissimilar_percentage = evaluation["percentage_dissimilar"]

                all_dissimilarities.append(dissimilar_percentage)

                # Delete this once added part with execution times from runs.
                #quantized_benchmark.append(benchmark(new_quant_model_path, model_config))
                quantized_benchmark.append(sum(quant_times_list)/len(quant_times_list))
                quantized_sizes.append(get_size(new_quant_model_path))

                print("Dissimilarity: " + str(dissimilar_percentage))

            
            prev_quant_model_path = new_quant_model_path.replace(".onnx", "_old.onnx")
            os.rename(new_quant_model_path, prev_quant_model_path)
            

            out_json = json.dumps(result_object, indent=2)
            with open(result_object_file, "w") as outfile:
                outfile.write(out_json)
            

            if (dissimilar_percentage <= total_dissimilar_percentage):
                total_dissimilar_percentage = dissimilar_percentage

                if (dissimilar_percentage <= threshold):
                    print("Threshold reached.")
                    break


def append_node_to_exclude(onnx_runner, float_model, actlist, nodes_to_exclude):
    for item in actlist:
        node_name = item[0]
        nodes = onnx_runner.get_nodes_containing_input(float_model, node_name)

        if node_name not in nodes_to_exclude:
            nodes_to_exclude.append(node_name)
            print("New node found: " + node_name)
            return node_name
        else:
            for node in nodes:
                if node.name not in nodes_to_exclude:

                    print("New node found: " + node.name)
                    nodes_to_exclude.append(node.name)
                    return node.name
                
    return None

if __name__ == "__main__":
    main()
