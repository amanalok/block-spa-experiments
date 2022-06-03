import argparse
import logging

from mlops.common.config_loading import load_yaml_config
# from mlops.ml_tracking.tracker import Tracker
# from mlops.ml_tracking.tracker_composition import TrackerComposition
from mlops.provisioning.experiment_context import ExperimentContext
from mlops.provisioning.ml_training_provisioner import RunModes
from ..block_movement_pruning.masked_run_squad import main_single, create_parser

run_logger = logging.getLogger(__name__)

__all__ = ["run"]


def run(
        context:ExperimentContext,
        run_mode=RunModes.TRAIN,
        **kwargs
):
    configuration = load_yaml_config(config_file=context.config_file_path,
                                     model_dir=context.model_dir,
                                     is_chief=context.is_chief,
                                     barrier=context.barrier)
    # trackers = Tracker.trackers_from_configuration(configuration)

    args, _ = create_parser().parse_known_args()
    args.output_dir = context.model_dir
    args.data_dir = context.get_inputs_paths()['train']
    if run_mode == RunModes.TRAIN:

        # with TrackerComposition(trackers=trackers) as tracker:
            # in case we are running a hyperparameter tuning job, we will just update args with the current settings to try out.
            # Will be an empty dict if we are not doing a hyper parameter tuning
        args = vars(args)
        hparams = context.get_sweeper_parameters()
        print(f"hparams : {hparams}")
        args.update(hparams)
        args = argparse.Namespace(**args)
        print(f"args : {args}")
        main_single(args)
            # inputs_paths.get('train', ''), context.model_dir, tracker, args=args)

    elif run_mode == RunModes.PREDICT:
         raise NotImplementedError()


# This main body will not be called by the local provisioner

if __name__ == '__main__':
    exp_context = ExperimentContext.create()
    run(context=exp_context,)
