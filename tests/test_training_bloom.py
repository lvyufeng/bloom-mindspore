import pytest
import numpy as np
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.dataset import GeneratorDataset

from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, \
    RunnerConfig
from mindformers.core.lr import WarmUpDecayLR
from mindformers import MindFormerBook, AutoModel, AutoConfig
from mindformers.tools import logger
from mindformers.models import BaseModel
from mindformers.core.optim import FusedAdamWeightDecay
from mindformers.pipeline import pipeline
from src.modeling_bloom import BloomLMHeadModel, BloomConfig

import mindspore
mindspore.set_context(mode=mindspore.GRAPH_MODE)

def generator():
    """dataset generator"""
    seq_len = 21
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)
    label_ids = input_ids
    train_data = (input_ids, input_mask, label_ids)
    for _ in range(512):
        yield (train_data[0],)

def test_gpt_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    # Config definition
    runner_config = RunnerConfig(epochs=1, batch_size=8, sink_mode=True, per_epoch_size=2)
    config = ConfigArguments(seed=2022, runner_config=runner_config)

    ms_config = BloomConfig(hidden_size=64, num_heads=8, num_layers=2, seq_length=20)

    # Model
    gpt_model = BloomLMHeadModel(ms_config)

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids"])
    dataset = dataset.batch(batch_size=8)

    # optimizer
    lr_schedule = WarmUpDecayLR(learning_rate=0.0001, end_learning_rate=0.00001, warmup_steps=0, decay_steps=512)
    optimizer = FusedAdamWeightDecay(beta1=0.009, beta2=0.999,
                                     learning_rate=lr_schedule,
                                     params=gpt_model.trainable_params())

    # callback
    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    callbacks = [loss_cb, time_cb]

    lm_trainer = Trainer(model=gpt_model,
                         config=config,
                         optimizers=optimizer,
                         train_dataset=dataset,
                         callbacks=callbacks)

    lm_trainer.train(dataset_sink_mode=False)
