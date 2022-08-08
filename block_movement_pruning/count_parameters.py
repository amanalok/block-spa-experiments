# Copyright 2020-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Count remaining (non-zero) weights in the encoder (i.e. the transformer layers).
Sparsity and remaining weights levels are equivalent: sparsity % = 100 - remaining weights %.
"""
import argparse
import os

import torch
from emmental.modules import MaskedSPLoPALinear


def expand_mask(mask, args):
    mask_block_rows = args.mask_block_rows
    mask_block_cols = args.mask_block_cols
    mask = torch.repeat_interleave(mask, mask_block_rows, dim=0)
    mask = torch.repeat_interleave(mask, mask_block_cols, dim=1)
    return mask


def count_parameters(
    state_dict,
    pruning_method,
    threshold,
    mask_block_rows=32,
    mask_block_cols=32,
    ampere_pruning_method="disabled",
):
    learned_count = 0  # Number of learned params in the encoder
    remaining_count = (
        0  # Number of remaining after pruning and adaptation in the encoder
    )
    encoder_count = 0  # Number of params in the encoder
    prototype_shapes_seen = []

    print(
        "name".ljust(60, " "),
        "Remaining Weights %".ljust(20, " "),
        "Remaining Weights".ljust(20, " "),
        "Learned Weights %".ljust(20, " "),
        "Learned Weights".ljust(20, " "),
    )
    for name, param in state_dict.items():
        if "encoder" not in name:
            continue

        if name.endswith(".weight"):
            (
                masked_weights,
                adapter_masked_pos_weights,
                adapter_proto_cols,
                adapter_proto_rows,
            ) = MaskedSPLoPALinear.masked_weights_from_state_dict(
                state_dict,
                name,
                pruning_method,
                threshold,
                ampere_pruning_method,
                mask_block_rows,
                mask_block_cols,
            )
            # Update total encoder weights
            encoder_count += param.numel()

            # Update encoder weights remaining after pruning and adaptation
            mask_weight_ones = (masked_weights != 0).sum().item()
            remaining_count += mask_weight_ones

            # Update number of learned params
            if adapter_proto_cols is not None:
                prototype_shape = (
                    adapter_proto_cols.shape[1],
                    adapter_proto_rows.shape[2],
                )
                if (
                    prototype_shape not in prototype_shapes_seen
                ):  # NB: Protypes are shared
                    prototype_shapes_seen.append(prototype_shapes_seen)
                    learned_count += adapter_proto_cols.numel()
                    learned_count += adapter_proto_rows.numel()

                mask_pos_ones = (adapter_masked_pos_weights != 0).sum().item()
                learned_count += mask_pos_ones

            print(
                name.ljust(60, " "),
                str(round(100 * mask_weight_ones / param.numel(), 3)).ljust(20, " "),
                str(mask_weight_ones).ljust(20, " "),
                str(round(100 * mask_pos_ones / param.numel(), 3)).ljust(20, " "),
                str(mask_pos_ones).ljust(20, " "),
            )
        elif MaskedSPLoPALinear.check_name(name):
            pass
        else:
            encoder_count += param.numel()
            if (
                name.endswith(".weight")
                and ".".join(name.split(".")[:-1] + ["mask_scores"]) in st
            ):
                pass
            else:
                remaining_count += param.numel()

    print("")
    print(f"Encoder Weights (global)  : {encoder_count}")
    print(
        f"Remaining Weights (global): {remaining_count} ({100 * remaining_count / encoder_count:.3f} %)"
    )
    print(
        f"Learned Weights (global)  : {learned_count} ({100 * learned_count / encoder_count:.3f} %)"
    )
    return remaining_count, encoder_count, learned_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pruning_method",
        choices=["l0", "topK", "sigmoied_threshold"],
        type=str,
        default="sigmoied_threshold",
        help="Pruning Method (l0 = L0 regularization, topK = Movement pruning, sigmoied_threshold = Soft movement pruning)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        help="For `topK`, it is the level of remaining weights (in %) in the fine-pruned model."
        "For `sigmoied_threshold`, it is the threshold \tau against which the (sigmoied) scores are compared."
        "Not needed for `l0`",
    )
    parser.add_argument(
        "--serialization_dir",
        type=str,
        default="bert-base-uncased",
        help="Folder containing the model that was previously fine-pruned",
    )
    parser.add_argument(
        "--mask_block_rows",
        default=1,
        type=int,
        help="Block row size for masks. Default is 1 -> general sparsity, not block sparsity.",
    )

    parser.add_argument(
        "--mask_block_cols",
        default=1,
        type=int,
        help="Block row size for masks. Default is 1 -> general sparsity, not block sparsity.",
    )
    args = parser.parse_args()

    st = torch.load(
        os.path.join(args.serialization_dir, "pytorch_model.bin"),
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    count_parameters(
        st,
        args.pruning_method,
        args.threshold,
        args.mask_block_rows,
        args.mask_block_cols,
    )
