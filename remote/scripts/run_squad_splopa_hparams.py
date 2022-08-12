import os
import subprocess
from hparams.parse_hparams import parse_excel_hparams
import math
from pathlib import Path


def flatten_list(original_list):
    return [element for sublist in original_list for element in sublist]


def try_int(val):
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val


if __name__ == "__main__":
    df = parse_excel_hparams(sheet_name="Details - SQuAD")

    logs_dir = Path("remote/scripts/logs/baselines")
    logs_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    for i, row in df.iterrows():
        identifier = f"{row['EXP ID']}_{row['Effective encoder remain weights %']:.2f}%"

        if identifier not in {
            "magnitude_1.0_*_1_2_null_0._3e-5_0._magnitude_null_0._10_epochs_60.10%"
        }:
            continue

        for adapter_lr, rank, protos, shared, block_size in [
            (3e-4, 2, 64, True, 32),
            (1e-4, 1, 64, True, 32),
            (3e-5, 1, 64, True, 32),
            # (0.005, 1, 64, True, 16),
            # (0.005, 1, 64, False, 32),
            # (0.005, 2, 64, True, 32),
            # (0.005, 1, 256, True, 32),
            # (0.005, 2, 256, True, 32),
            # (0.005, 2, 256),
            # (0.05, 1, 64),
            # (0.005, 1, 64),
            # (0.01, 2, 64),
            # (0.01, 1, 256),
        ]:
            id = f"splora_adapterlr={adapter_lr}_rank={rank}_protos={protos}_init=1e-4_shared={shared}_blocksize={block_size}_{identifier}"
            print(f">> Running {id}")
            command = [
                "python",
                # "block_movement_pruning/masked_run_squad.py",
                "remote/scripts/run_squad.py",
                "--identifier",
                id,
                "--overwrite_output_dir",
                "--output_dir",
                "runs/squad-bert-base-uncased-finetuned",
                "--data_dir",
                "squad_data",
                "--train_file",
                "train-v1.1.json",
                "--predict_file",
                "dev-v1.1.json",
                "--do_train",
                "--do_eval",
                "--do_lower_case",
                "--model_type",
                "masked_bert",
                "--model_name_or_path",
                "bert-base-uncased",
                "--mask_block_rows",
                str(block_size),
                "--mask_block_cols",
                str(block_size),
                "--adapter_learning_rate",
                str(adapter_lr),
                "--splopa_prototype_rank",
                str(rank),
                "--num_splopa_prototypes",
                str(protos),
                *flatten_list(
                    [
                        (f"--{k}", str(try_int(v)))
                        for k, v in row[5:].to_dict().items()
                        if isinstance(v, str) or not math.isnan(v)
                    ]
                ),
            ]
            if not shared:
                command.append("--splopa_prototypes_not_shared")

            fpath = logs_dir / f"{id}.txt"
            if fpath.exists():
                os.remove(fpath)

            f = open(str(fpath), "a")
            f.write("******* Command: *******\n")
            f.write(" ".join(command))
            f.write("\n************************\n")

            processes.append((subprocess.Popen(command, stdout=f), f))

    for (p, f) in processes:
        p.wait()
        f.close()
