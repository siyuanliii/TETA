""" evaluate.py

Run example:
evaluate.py --USE_PARALLEL False --METRICS TETA --TRACKERS_TO_EVAL qdtrack

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,  # Raises exception and exits with error
        'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
        'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
        'OUTPUT_TEM_RAW_DATA': True, # Output detailed statistics for each class
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/gt/tao/tao_training'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/tao/tao_training'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': None,  # Classes to eval (if None, all classes)
        'SPLIT_TO_EVAL': 'training',  # Valid: 'training', 'val'
        'PRINT_CONFIG': True,  # Whether to print current config
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        'MAX_DETECTIONS': 300,  # Number of maximal allowed detections per image (0 for unlimited)
    Metric arguments:
        'METRICS': ['TETA']
"""

import sys
import os
import argparse
import pickle
import numpy as np
import json
from multiprocessing import freeze_support

from teta.config import parse_configs
from teta.datasets import TAO
from teta.eval import Evaluator
from teta.metrics import TETA


def compute_teta_on_ovsetup(teta_res, base_class_names, novel_class_names):
    if "COMBINED_SEQ" in teta_res:
        teta_res = teta_res["COMBINED_SEQ"]

    frequent_teta = []
    rare_teta = []
    for key in teta_res:
        if key in base_class_names:
            frequent_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))
        elif key in novel_class_names:
            rare_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))

    print("Base and Novel classes performance")

    # print the header
    print(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "TETA50:",
            "TETA",
            "LocA",
            "AssocA",
            "ClsA",
            "LocRe",
            "LocPr",
            "AssocRe",
            "AssocPr",
            "ClsRe",
            "ClsPr",
        )
    )

    if frequent_teta:
        freq_teta_mean = np.mean(np.stack(frequent_teta), axis=0)

        # print the frequent teta mean
        print("{:<10} ".format("Base"), end="")
        print(*["{:<10.3f}".format(num) for num in freq_teta_mean])

    else:
        print("No Base classes to evaluate!")
        freq_teta_mean = None
    if rare_teta:
        rare_teta_mean = np.mean(np.stack(rare_teta), axis=0)

        # print the rare teta mean
        print("{:<10} ".format("Novel"), end="")
        print(*["{:<10.3f}".format(num) for num in rare_teta_mean])
    else:
        print("No Novel classes to evaluate!")
        rare_teta_mean = None

    return freq_teta_mean, rare_teta_mean


def evaluate():
    """Evaluate with TETA."""
    eval_config, dataset_config, metrics_config = parse_configs()
    evaluator = Evaluator(eval_config)
    dataset_list = [TAO(dataset_config)]
    metrics_list = []
    metric = TETA(exhaustive=False)
    if metric.get_name() in metrics_config["METRICS"]:
        metrics_list.append(metric)
    if len(metrics_list) == 0:
        raise Exception("No metrics selected for evaluation")

    tracker_name = dataset_config["TRACKERS_TO_EVAL"][0]
    resfile_path = dataset_config["TRACKERS_FOLDER"]
    dataset_gt = json.load(open(dataset_config["GT_FOLDER"]))
    eval_results, _ = evaluator.evaluate(dataset_list, metrics_list)

    eval_results_path = os.path.join(
        resfile_path, tracker_name, "teta_summary_results.pth"
    )
    eval_res = pickle.load(open(eval_results_path, "rb"))

    base_class_synset = set(
        [
            c["name"]
            for c in dataset_gt["categories"]
            if c["frequency"] != "r"
        ]
    )
    novel_class_synset = set(
        [
            c["name"]
            for c in dataset_gt["categories"]
            if c["frequency"] == "r"
        ]
    )

    compute_teta_on_ovsetup(
        eval_res, base_class_synset, novel_class_synset
    )


if __name__ == "__main__":
    freeze_support()
    evaluate()
