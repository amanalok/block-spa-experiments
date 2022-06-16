from mlops.provisioning.ml_training_provisioner import MLTrainingProvisioner
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", default="block_movement_pruning/configs/remote_config.yaml"
    )
    args, uk = parser.parse_known_args()
    provisioner = MLTrainingProvisioner.from_config(args.config_file)
    if len(uk) != 0:
        print("Sending unknown args to the provisioner %s" % uk)
        provisioner.set_train_args(uk)
    provisioner.train()


if __name__ == "__main__":
    main()
