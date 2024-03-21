# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""
from transformers import AutoConfig

import torch
import math
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.arguments import core_transformer_config_from_args
# from megatron.model.rotary_pos_embedding import apply_rotary_pos_emb, RotaryEmbedding
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
# from puzzle.utils.config import core_transformer_config_from_hf_config
from megatron.arguments import core_transformer_config_from_hf_config

from megatron.model import LlamaModel, LlamaForCausalLM
import wandb
import os
import subprocess

from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model


# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from typing import Union
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
# from megatron.arguments import core_transformer_config_from_args 
from megatron.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


os.environ["WANDB_API_KEY"] = "2ed6f8544ac3e30d5c08879166cc10d9c6232448"
os.environ["WANDB_MODE"] = "offline"

def to_peft(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    return model

def model_provider(pre_process=True, post_process=True, reward_base_model=False):
    """Build the model."""

    print_rank_0('building Llama model ...')
    args = get_args()
    # if not reward_base_model:
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    core_config = core_transformer_config_from_hf_config(model_config)
    model_class = LlamaForCausalLM
    # else:
    #     model_config = AutoConfig.from_pretrained(args.critic_model_name_or_path)
    #     core_config = core_transformer_config_from_hf_config(model_config)
    #     model_class = LlamaModel

    model = model_class(
            config=core_config,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=False
            )
    for param in model.parameters():
       if not 8 in param.shape:
          param.requires_grad_(False)
    # model_config = getattr(model, "config", {"model_type": "custom"})
    # print("hahahhah",model_config)
        
    # print(model)
    # model = to_peft(model)
    # print(model)
    # exit()

    # print(args)    
#    if torch.distributed.is_initialized():
#        if torch.distributed.get_rank() == 0:
#            name = f"{args.world_size=},{args.micro_batch_size=},{args.seq_length=},{args.tensor_model_parallel_size=},{args.pipeline_model_parallel_size},{args.world_size=}" 
#            wandb.init(
#                project="megatron baseline llama2 70b",
#                name=name,
#                config={"command": vars(args)},
#                )
#    else:
#        name = f"{args.world_size=},{args.micro_batch_size=},{args.seq_length=},{args.tensor_model_parallel_size=},{args.pipeline_model_parallel_size},{args.world_size=}"
#        wandb.init(
#            project="megatron baseline llama2 70b",
#            name=name,
#            config={"command": vars(args)},
#        )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        2, # tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}

def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()
    #print(f"before {torch.cuda.memory_allocated()}")
    output_tensor = model(tokens, position_ids, attention_mask,
                                        labels=labels)
    #print(f"after {torch.cuda.memory_allocated()}")

    # Output_tensor stores the standard loss, loos_func calculates the total loss.
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for llama ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        # data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(True))
    print_rank_0("> finished creating llama datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
 #   args=get_args()
#    print(args)
#    if torch.distributed.is_initialized():
#        if torch.distributed.get_rank() == 0:
#            name = f"{args.gpu=},{args.batch=},{args.seq_len=}"
#            wandb.init(
#                project=f"{args.wb_project_prefix}_{args.seed}",
#                name=name,
#                config={"command": sys.argv, **vars(args)},
#                )
#    else:
#        name = f"{args.gpu=},{args.batch=},{args.seq_len=}"
#        wandb.init(
#            project=f"{args.wb_project_prefix}_{args.seed}",
#            name=name,
#            config={"command": sys.argv, **vars(args)},
#        )
       pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
            args_defaults={'tokenizer_type': "SpmTokenizer"}
            )
            # args_defaults={'tokenizer_type': 'SpmTokenizer', 'log_timers_to_tensorboard','log_world_size_to_tensorboard', 'log_memory_to_tensorboard', 'log_timers_to_tensorboard', 'log_learning_rate_to_tensorboard','log_memory_to_tensorboard','log_world_size_to_tensorboard','log_batch_size_to_tensorboard','wandb_project': 'megatron','wandb_exp_name': 'llama_7b_4gpu_tp2pp2','wandb_save_dir': '/mnt/octave/data/siqizhu/mgt_temp/wandb','tensorboard_dir': '/mnt/octave/data/siqizhu/mgt_temp/tensorboard_log/7b_4gpu_tp2pp2'})
