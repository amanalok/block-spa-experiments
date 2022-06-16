import logging
import sys

from mlops.common.config_loading import load_yaml_config

# from mlops.ml_tracking.tracker import Tracker
# from mlops.ml_tracking.tracker_composition import TrackerComposition
from mlops.provisioning.experiment_context import ExperimentContext
from mlops.provisioning.ml_training_provisioner import RunModes
from block_movement_pruning.masked_run_squad import main_single, create_parser

run_logger = logging.getLogger(__name__)

__all__ = ["run"]


def run(context: ExperimentContext, run_mode=RunModes.TRAIN, **kwargs):
    # configuration = load_yaml_config(
    #     config_file=context.config_file_path,
    #     model_dir=context.model_dir,
    #     is_chief=context.is_chief,
    #     barrier=context.barrier,
    # )
    # trackers = Tracker.trackers_from_configuration(configuration)

    sys.argv += context.get_train_args()
    args, uk = create_parser().parse_known_args()
    print("known args   : %s" % args)
    print("unknown args : %s" % uk)
    args.output_dir = context.model_dir
    args.data_dir = context.get_inputs_paths()["train"]
    if run_mode == RunModes.TRAIN:
        print(f"args : {args}")
        main_single(args)
        # with TrackerComposition(trackers=trackers) as tracker:
        #     main_single(args, tracker=tracker)
        # inputs_paths.get('train', ''), context.model_dir, tracker, args=args)

    elif run_mode == RunModes.PREDICT:
        raise NotImplementedError()


# This main body will not be called by the local provisioner

if __name__ == "__main__":
    import os

    os.system("nvidia-smi")
    exp_context = ExperimentContext.create()
    run(
        context=exp_context,
    )
