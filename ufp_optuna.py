import time
import numpy as np
import math
import json
import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback


def problem_grouping(clients, time_slot, quad_constr, util_rate=1.5,
                     time_len=6, num_of_util_groups=15):  ##grouping by utility,then by time_slot_length
    num_of_dims = time_slot.shape[0]
    time_slot_min = time_slot.min(axis=1)
    time_slot_av = time_slot.mean(axis=1)
    time_slot_max = time_slot.max(axis=1)
    max_cap = time_slot.max()
    min_av_max_caps = np.concatenate((time_slot_min, time_slot_av, time_slot_max))
    # normalizing time capacities
    min_av_max_caps /= max_cap
    quad_constr /= max_cap
    av_cap_bytime = time_slot.mean(axis=0) / max_cap
    # Normalizing demands in each dimension by min capacity
    for dim in range(num_of_dims):
        clients[:, dim] /= time_slot_min[dim]
    # sum of demands of the first dimension normalized by min capacity of the first dimension
    total_demand = clients[:, 0].sum()
    ##Grouping by utilities
    max_util = clients[:, -2].max()
    min_util = clients[:, -2].min()
    av_util = clients[:, -2].mean()
    if 100 < max_util <= 1200:
        clients[:, -2] /= 10
    elif max_util > 1200:
        clients[:, -2] /= 300
    util_groups = []
    steps = np.zeros(num_of_util_groups)
    steps[0] = util_rate
    for i in range(1, steps.size):
        steps[i] = steps[i - 1] * util_rate
    bool_idx = (clients[:, -2] <= steps[0])
    util_groups.append(clients[np.where(bool_idx)[0]])
    for i in range(1, num_of_util_groups):
        bool_idx = (clients[:, -2] <= steps[i]) & (
                clients[:, -2] > steps[i - 1])
        util_groups.append(clients[np.where(bool_idx)[0]])
    # Normalizing by max_util
    for group in util_groups:
        group[:, -2] /= max_util

    ##Grouping by time_length
    final_groups = []
    gr_time = int(time_slot.shape[1] / time_len)
    for i in range(num_of_util_groups):
        if (util_groups[i].size > 0):
            group_time_lengths = util_groups[i][:, num_of_dims + 1] - util_groups[i][:, num_of_dims]
            min_time_len = np.min(group_time_lengths)
            max_time_len = np.max(group_time_lengths)
            steps = np.linspace(min_time_len, max_time_len, gr_time + 1)
            for j in range(gr_time - 1):
                bool_idx = (group_time_lengths >= steps[j]) & (group_time_lengths < steps[j + 1])
                final_groups.append(util_groups[i][np.where(bool_idx)[0]])
            bool_idx = (group_time_lengths >= steps[-2]) & (group_time_lengths <= steps[-1])
            final_groups.append(util_groups[i][np.where(bool_idx)[0]])
        else:
            for _ in range(gr_time):
                final_groups.append(util_groups[i])
    return final_groups, total_demand, min_av_max_caps, av_cap_bytime, quad_constr, min_util, av_util


def data_preprocess(groups, total_demand, quad_cap, min_av_max_caps, av_cap_bytime,
                    min_util, av_util, num_of_dims=3, num_of_clients=1500, time_length=96):
    length = len(groups)
    inputs = np.zeros((length + 1, 3 * num_of_dims + 1 + time_length + 2))
    labels = np.zeros((length + 1, 2))
    for i, group in enumerate(groups):
        if group.size > 0:
            inputs[i][:num_of_dims] = np.min(group[:, :num_of_dims], axis=0)  # minimum demand per dimension
            inputs[i][num_of_dims:2 * num_of_dims] = np.mean(group[:, :num_of_dims],
                                                             axis=0)  # average demand per dimension
            inputs[i][2 * num_of_dims:3 * num_of_dims] = np.max(group[:, :num_of_dims],
                                                                axis=0)  # maximum demand per dimension
            quadr_demands = ((group[:, :num_of_dims] ** 2).sum(axis=1) / 30000).mean()
            inputs[i][3 * num_of_dims] = quadr_demands
            inputs[i][3 * num_of_dims + 1:3 * num_of_dims + time_length + 1] = np.sum(
                group[:, num_of_dims + 2:num_of_dims + 2 + time_length],
                axis=0) / num_of_clients  # average time_occupancy

            inputs[i][-2] = np.mean(group[:, -2])  # average utility
            inputs[i][-1] = group.shape[0] / num_of_clients  # what part of clients is in this group
            labels[i][0] = np.sum(group[:, -1]) / num_of_clients  # what part is selected in the final answer
            if group[np.where(group[:, -1] == 1)].size > 0:
                labels[i][1] = group[np.where(group[:, -1] == 1)][:,
                               0].sum() / total_demand  # total demand of selected clients/total demand of the first dimension
    inputs[-1][:3 * num_of_dims] = min_av_max_caps
    inputs[-1][3 * num_of_dims] = quad_cap
    inputs[-1][3 * num_of_dims + 1:3 * num_of_dims + time_length + 1] = av_cap_bytime
    inputs[-1][-2] = min_util
    inputs[-1][-1] = av_util
    labels[-1][0] = 1 - labels[:, 0].sum()
    labels[-1][1] = 1 - labels[:, 1].sum()
    return inputs, labels


class DataSetMaker(Dataset):
    def __init__(self):
        self.states = list(glob.glob('states/*.npz'))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = np.load(self.states[idx])
        clients_list = state['cl']
        time_slot_capacity = state['tslot']
        quad_constr = state['quad_constr']
        f_groups, tot_dem, min_av_max_caps, av_cap_bytime, min_util, av_util, q_c = problem_grouping(clients_list,
                                                                                                     time_slot_capacity,
                                                                                                     quad_constr)
        inputs, labels = data_preprocess(f_groups, tot_dem, q_c, min_av_max_caps, av_cap_bytime,
                                         min_util, av_util)
        return np.float32(inputs), np.float32(labels)


dataset = DataSetMaker()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

train_set, valid_set, test_set = random_split(dataset, [8000, 739, 700])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(valid_set, batch_size=1)
test_loader = DataLoader(test_set, batch_size=1)


def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention part
        attn_out = self.self_attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TransformerPredictor(pl.LightningModule):

    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, dropout=0.0,
                 input_dropout=0.0):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers,
                                              input_dim=self.hparams.model_dim,
                                              dim_feedforward=2 * self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)
        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, add_positional_encoding=False):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.output_net(x)
        x = F.softmax(x, dim=-2)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        return optimizer

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_after_backward(self):
        raise NotImplementedError


class KnapsackPredictor(TransformerPredictor):

    def _calculate_loss(self, batch, mode="train"):
        inp_data, labels = batch
        preds = self.forward(inp_data)
        loss1 = F.l1_loss(preds[:, :, 0], labels[:, :, 0])
        loss2 = F.l1_loss(preds[:, :, 1], labels[:, :, 1])
        loss = loss1 + loss2
        acc1 = ((preds[:, :, 0] - labels[:, :, 0]) / preds[:, :, 0]).abs().max()
        acc2 = ((preds[:, :, 1] - labels[:, :, 1]) / preds[:, :, 1]).abs().max()
        acc3 = ((preds[:, :, 0] - labels[:, :, 0]) / preds[:, :, 0]).abs().mean()
        acc4 = ((preds[:, :, 1] - labels[:, :, 1]) / preds[:, :, 1]).abs().mean()

        # Logging
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_loss1", loss1)
        self.log(f"{mode}_loss2", loss2)
        self.log(f"{mode}_max_accuracy1", acc1)
        self.log(f"{mode}_max_accuracy2", acc2)
        self.log(f"{mode}_mean_accuracy1", acc3)
        self.log(f"{mode}_mean_accuracy2", acc4)
        return loss, (acc1, acc2)

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")

    def on_after_backward(self):
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.logger.experiment.add_scalar(f"grad_norm/{name}", param.grad.norm(), self.global_step)


CHECKPOINT_PATH = "C:\\Users\\zhira\\CSIE_PYTHON_PROJECTS\\UFP_FINAL"


def get_current_time_version():
    return time.strftime("%Y%m%d-%H%M%S")


class WeightLoggingCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            # Calculate statistics
            mean_val = param.mean()
            std_val = param.std()
            max_val = param.max()
            min_val = param.min()

            # Log the statistics
            trainer.logger.experiment.add_scalar(f'{name}_mean', mean_val, trainer.current_epoch)
            trainer.logger.experiment.add_scalar(f'{name}_std', std_val, trainer.current_epoch)
            trainer.logger.experiment.add_scalar(f'{name}_max', max_val, trainer.current_epoch)
            trainer.logger.experiment.add_scalar(f'{name}_min', min_val, trainer.current_epoch)


def train_knapsack(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "UfpCheckPoint")
    logger = TensorBoardLogger(
        save_dir="logs",  # Directory where logs will be saved
        name="MyExperiment",  # Name of the experiment, it will be the main folder
        version=get_current_time_version()  # Version of the current run (sub-folder)
    )
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(callbacks=[
        ModelCheckpoint(dirpath=root_dir, filename='UfpCheckPoint', save_weights_only=True, mode="min",
                        monitor="val_loss"),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(monitor="val_loss", min_delta=1e-7, patience=20, verbose=False, mode="min"),
        WeightLoggingCallback()],
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        logger=logger,
        devices=1,
        max_epochs=150,
        log_every_n_steps=10)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(root_dir, "UfpCheckPoint.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = KnapsackPredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = KnapsackPredictor(**kwargs)
        trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=1)
    test_result = trainer.test(model, test_loader, verbose=1)
    result = {"test_acc": test_result, "val_acc": val_result}

    model = model.to(device)
    return model, result


def objective(trial: optuna.trial.Trial):
    # Define the hyperparameters to optimize
    model_dim = trial.suggest_int('model_dim', 108, 200, 12)
    num_heads = trial.suggest_int('num_heads', 2, 6, 2)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    dropout = trial.suggest_float('dropout', 0, 0.2, step=0.1)


    # Create a model instance with the suggested hyperparameters
    model = KnapsackPredictor(108, model_dim, 2, num_heads, num_layers, lr, dropout, 0.0)

    # Create a PyTorch Lightning trainer with the Optuna Pruner
    trainer = pl.Trainer(
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_loss')],
        max_epochs=10,
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1
    )
    trainer.fit(model, train_loader, val_loader)

    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss


if __name__ == '__main__':
    knapsack_model = train_knapsack(input_dim=108, model_dim=156, num_heads=2, num_classes=2,
                                    num_layers=2, dropout=0.0, lr=28e-4)
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=1+0)
    #
    # # Print the best hyperparameters
    # best_trial = study.best_trial
    # print(f"Best trial: {best_trial.params}")
    #
    # best_trial_dict = best_trial.params
    # best_trial_dict["value"] = best_trial.value
    #
    # with open('best_trial.json', 'w') as outfile:
    #     json.dump(best_trial_dict, outfile, indent=4)

    # with open('best_trial.json', 'r') as infile:
    #     loaded_trial_data = json.load(infile)
    #
    # # Use the loaded data
    # print(loaded_trial_data)

##
