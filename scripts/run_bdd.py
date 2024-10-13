import logging
import os
import time
from pathlib import Path

from scalabel.eval.box_track import BoxTrackResult, bdd100k_to_scalabel
from scalabel.eval.hota import HOTAResult, evaluate_track_hota
from scalabel.eval.hotas import evaluate_seg_track_hota
from scalabel.eval.mot import TrackResult, acc_single_video_mot, evaluate_track
from scalabel.eval.mots import acc_single_video_mots, evaluate_seg_track
from scalabel.eval.teta import TETAResult, evaluate_track_teta
from scalabel.eval.tetas import evaluate_seg_track_teta
from scalabel.label.io import group_and_sort, load, load_label_config

MOT_CFG_FILE = os.path.join(
    str(Path(__file__).parent.absolute()), "dataset_configs/box_track.toml"
)
MOTS_CFG_FILE = os.path.join(
    str(Path(__file__).parent.absolute()), "dataset_configs/seg_track.toml"
)

import argparse

class TETA_BDD100K_Evaluator:
    def __init__(self, scalabel_gt, resfile_path, metrics, with_mask, logger, nproc):
        self.scalabel_gt = scalabel_gt
        self.resfile_path = resfile_path
        self.metrics = metrics
        self.with_mask = with_mask
        self.logger = logger
        self.nproc = nproc

    def evaluate(self):
        """Evaluate with TETA, HOTA, ClearMOT on BDD100K."""

        eval_results = dict()

        bdd100k_config = load_label_config(MOT_CFG_FILE)
        print("Start loading.")

        gts = group_and_sort(load(self.scalabel_gt).frames)
        results = group_and_sort(load(self.resfile_path).frames)
        print("gt_len", len(gts), "results", len(results))
        print("Finish loading")
        print("Start evaluation")
        print("Ignore unknown cats")

        self.logger.info("Tracking evaluation.")
        t = time.time()
        gts = [bdd100k_to_scalabel(gt, bdd100k_config) for gt in gts]
        results = [bdd100k_to_scalabel(result, bdd100k_config) for result in results]

        if "CLEAR" in self.metrics:
            if self.with_mask:
                mot_result = evaluate_seg_track(
                    acc_single_video_mots,
                    gts,
                    results,
                    bdd100k_config,
                    ignore_unknown_cats=True,
                        nproc=self.nproc,
                )
            else:
                mot_result = evaluate_track(
                    acc_single_video_mot,
                    gts,
                    results,
                    bdd100k_config,
                    ignore_unknown_cats=True,
                        nproc=self.nproc,
                )
            print("CLEAR and IDF1 results :")
            print(mot_result)
            print(mot_result.summary())

        if "HOTA" in self.metrics:
            if self.with_mask:
                hota_result = evaluate_seg_track_hota(
                        gts, results, bdd100k_config, self.nproc
                )
            else:
                    hota_result = evaluate_track_hota(gts, results, bdd100k_config, self.nproc)
            print("HOTA results :")
            print(hota_result)
            print(hota_result.summary())

        if "TETA" in self.metrics:
            if self.with_mask:
                teta_result = evaluate_seg_track_teta(
                        gts, results, bdd100k_config, self.nproc
                )
            else:
                    teta_result = evaluate_track_teta(gts, results, bdd100k_config, self.nproc)

            print("TETA results :")
            print(teta_result)
            print(teta_result.summary())

        if (
            "CLEAR" in self.metrics
            and "HOTA" in self.metrics
            and "TETA" in self.metrics
        ):
            print("Aggregated results: ")
            combined_result = BoxTrackResult(
                **{**mot_result.dict(), **hota_result.dict(), **teta_result.dict()}
            )
            print(combined_result)
            print(combined_result.summary())

        t = time.time() - t
        self.logger.info("evaluation finishes with %.1f s.", t)

        print("Completed evaluation")
        return eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tracking performance on BDD100K.")
    parser.add_argument('--scalabel_gt', required=True, help='Path to the ground truth file')
    parser.add_argument('--resfile_path', required=True, help='Path to the result file')
    parser.add_argument('--metrics', nargs='+', default=['TETA', 'HOTA', 'CLEAR'], help='List of metrics to evaluate')
    parser.add_argument('--with_mask', action='store_true', help='Whether to evaluate with mask')
    parser.add_argument('--nproc', type=int, default=8, help='Number of processes to use')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    evaluator = TETA_BDD100K_Evaluator(args.scalabel_gt, args.resfile_path, args.metrics, args.with_mask, logger, args.nproc)
    evaluator.evaluate()
