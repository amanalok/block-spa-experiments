import sys
from mlops.provisioning.ml_training_provisioner import MLTrainingProvisioner


def main(config_file):
    provisioner = MLTrainingProvisioner.from_config(config_file)
    provisioner.set_train_args(sys.argv)
    provisioner.train()


if __name__ == '__main__':
    config_file = '/Users/aman.alok/CactusCodeRepos/block_movement_pruning/entrypoints/remote_config.yaml'
    main(config_file)