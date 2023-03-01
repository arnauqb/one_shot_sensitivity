import random
from re import X
import numpy as np
import torch
import os
import yaml
import torch.nn as nn
import math
import time
import pandas as pd
from pathlib import Path
from torch.autograd import Variable
from copy import copy
import matplotlib.pyplot as plt
import pdb

from one_shot_sensitivity.paths import default_data_path
from one_shot_sensitivity.abm_model import GradABM
from one_shot_sensitivity.data_utils import WEEKS_AHEAD, states, counties
from one_shot_sensitivity.model_utils import (
    EmbedAttenSeq,
    fetch_county_data_covid,
    fetch_county_data_flu,
    DecodeSeq,
    SEIRM,
    SIRS,
)
from one_shot_sensitivity.june import DistrictData, June

BENCHMARK_TRAIN = False
NUM_EPOCHS_DIFF = 1000
print("---- MAIN IMPORTS SUCCESSFUL -----")
epsilon = 1e-6

MIN_VAL_PARAMS = {
    "abm-covid": [
        1.0,
        0.001,
        0.01,
    ],  # r0, mortality rate, initial_infections_percentage
    "abm-flu": [1.05, 0.1],  # r0, initial_infections_percentage
    "seirm": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.01,
    ],  # beta, alpha, gamma, mu, initial_infections_percentage
    "sirs": [0.0, 0.1],  # beta, initial_infections_percentage
}
MAX_VAL_PARAMS = {
    "abm-covid": [8.0, 0.02, 1.0],
    "abm-flu": [2.6, 5.0],
    "seirm": [1.0, 1.0, 1.0, 1.0, 1.0],
    "sirs": [1.0, 5.0],
}

DAYS_HEAD = 4 * 7  # 4 weeks ahead

pi = torch.FloatTensor([math.pi])

SAVE_MODEL_PATH = Path("./Models/")

# neural network predicting parameters of the ABM


class CalibNNJune(nn.Module):
    def __init__(
        self,
        metas_train_dim,
        X_train_dim,
        device,
        training_weeks,
        n_districts,
        parameters_to_calibrate=None,
        hidden_dim=32,
        n_layers=2,
        bidirectional=True,
    ):
        super().__init__()

        self.device = device

        self.training_weeks = training_weeks

        """ tune """
        hidden_dim = 64
        out_layer_dim = 32

        self.emb_model = EmbedAttenSeq(
            dim_seq_in=X_train_dim,
            dim_metadata=metas_train_dim,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.decoder = DecodeSeq(
            dim_seq_in=1,
            rnn_out=hidden_dim,  # divides by 2 if bidirectional
            dim_out=out_layer_dim,
            n_layers=1,
            bidirectional=True,
        )
        out_layer_width = out_layer_dim * n_districts
        out_dim = len(parameters_to_calibrate)
        self.out_layer = [
            nn.Linear(in_features=out_layer_width, out_features=out_layer_width // 2),
            nn.ReLU(),
            nn.Linear(in_features=out_layer_width // 2, out_features=out_dim),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(1.0)

        self.out_layer.apply(init_weights)
        self.min_values = torch.tensor(
            [
                parameters_to_calibrate[param]["min_value"]
                for param in parameters_to_calibrate
            ],
            device=device,
        )
        self.max_values = torch.tensor(
            [
                parameters_to_calibrate[param]["max_value"]
                for param in parameters_to_calibrate
            ],
            device=device,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, meta):
        x_embeds, encoder_hidden = self.emb_model.forward(x.transpose(1, 0), meta)
        # create input that will tell the neural network which week it is predicting
        # thus, we have one element in the sequence per each week of R0
        time_seq = (
            # torch.arange(1, self.training_weeks + WEEKS_AHEAD + 1)
            torch.tensor([1])
            .repeat(x_embeds.shape[0], 1)
            .unsqueeze(2)
        )
        Hi_data = time_seq.float().to(self.device)
        # Hi_data = ((time_seq - time_seq.min()) / (time_seq.max() - time_seq.min())).to(
        #    self.device
        # )
        emb = self.decoder(Hi_data, encoder_hidden, x_embeds)
        emb = emb.reshape(1, 1, -1)
        out = self.out_layer(emb)
        out = self.min_values + (self.max_values - self.min_values) * self.sigmoid(out)
        return out


class CalibNN(nn.Module):
    def __init__(
        self,
        metas_train_dim,
        X_train_dim,
        device,
        training_weeks,
        hidden_dim=32,
        out_dim=1,
        n_layers=2,
        scale_output="abm-covid",
        bidirectional=True,
    ):
        super().__init__()

        self.device = device

        self.training_weeks = training_weeks

        """ tune """
        hidden_dim = 64
        out_layer_dim = 32

        self.emb_model = EmbedAttenSeq(
            dim_seq_in=X_train_dim,
            dim_metadata=metas_train_dim,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.decoder = DecodeSeq(
            dim_seq_in=1,
            rnn_out=hidden_dim,  # divides by 2 if bidirectional
            dim_out=out_layer_dim,
            n_layers=1,
            bidirectional=True,
        )

        out_layer_width = out_layer_dim
        self.out_layer = [
            nn.Linear(in_features=out_layer_width, out_features=out_layer_width // 2),
            nn.ReLU(),
            nn.Linear(in_features=out_layer_width // 2, out_features=out_dim),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.out_layer.apply(init_weights)
        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output], device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output], device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, meta):
        x_embeds, encoder_hidden = self.emb_model.forward(x.transpose(1, 0), meta)
        # create input that will tell the neural network which week it is predicting
        # thus, we have one element in the sequence per each week of R0
        time_seq = (
            torch.arange(1, self.training_weeks + WEEKS_AHEAD + 1)
            .repeat(x_embeds.shape[0], 1)
            .unsqueeze(2)
        )
        Hi_data = ((time_seq - time_seq.min()) / (time_seq.max() - time_seq.min())).to(
            self.device
        )
        emb = self.decoder(Hi_data, encoder_hidden, x_embeds)
        out = self.out_layer(emb)
        out = self.min_values + (self.max_values - self.min_values) * self.sigmoid(out)
        return out


class ParamModel(nn.Module):
    def __init__(
        self,
        metas_train_dim,
        X_train_dim,
        device,
        hidden_dim=50,
        n_layers=2,
        out_dim=1,
        scale_output="abm-covid",
        bidirectional=True,
        CUSTOM_INIT=True,
    ):
        super().__init__()

        self.device = device
        self.emb_model = EmbedAttenSeq(
            dim_seq_in=X_train_dim,
            dim_metadata=metas_train_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.layer1 = nn.Linear(in_features=hidden_dim, out_features=20)
        # used to bypass the RNN - we want to check what's happening with gradients
        self.layer_bypass = nn.Linear(in_features=metas_train_dim, out_features=20)
        self.meanfc = nn.Linear(in_features=20, out_features=out_dim, bias=True)
        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output], device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output], device=self.device)
        self.sigmoid = nn.Sigmoid()
        if CUSTOM_INIT:
            self.meanfc.bias = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x, meta):
        x_embeds = self.emb_model.forward(x.transpose(1, 0), meta)
        # use embedding for predicting: i) R0 and ii) Cases {for support counties} [FOR LATER]
        ro_feats = self.layer1(x_embeds)
        ro_feats = nn.ReLU()(ro_feats)
        out = self.meanfc(ro_feats)
        # else:
        """ bound output """
        out = self.min_values + (self.max_values - self.min_values) * self.sigmoid(out)

        return out


class LearnableParams(nn.Module):
    """doesn't use data signals"""

    def __init__(self, num_params, device, scale_output="abm-covid"):
        super().__init__()
        self.device = device
        self.learnable_params = nn.Parameter(torch.rand(num_params, device=self.device))
        self.min_values = torch.tensor(MIN_VAL_PARAMS[scale_output], device=self.device)
        self.max_values = torch.tensor(MAX_VAL_PARAMS[scale_output], device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        out = self.learnable_params
        """ bound output """
        out = self.min_values + (self.max_values - self.min_values) * self.sigmoid(out)
        return out


def normal(x, mu, sigma_sq):
    a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


def save_model(model, file_name, disease, region, week):
    path = SAVE_MODEL_PATH / disease / region
    path.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), path / (file_name + " " + week + ".pth"))


def load_model(model, file_name, disease, region, week, device):
    path = SAVE_MODEL_PATH / disease / region
    model.load_state_dict(
        torch.load(path / (file_name + " " + week + ".pth"), map_location=device)
    )
    return model


def param_model_forward(param_model, params, x, meta):
    # get R0 from county network
    if params["model_name"].startswith("GradABM-time-varying"):
        action_value = param_model.forward(x, meta)  # time-varying
    elif params["model_name"].startswith("june"):
        action_value = param_model.forward(x, meta)  # time-varying
    elif params["model_name"] == "ABM-expert":
        if params["disease"] == "COVID":
            action_value = torch.tensor(
                [2.5, 0.02, 0.5]
            )  # from CDC, for COVID -- previous I0 was 0.01
        if params["disease"] == "Flu":
            action_value = torch.tensor([1.3, 1.0])  # from CDC, for COVID
        action_value = action_value.repeat((meta.shape[0], 1))
    elif "ABM-pred-correction" in params["model_name"]:  # same as SEIRM-static, but get
        action_value = param_model.forward()
        if params["disease"] == "COVID":
            # NOTE: to fix, beta/gamma is for SIR, maybe not the same for SEIRM
            beta = action_value[0]
            gamma = action_value[2]
            mu = action_value[3]  # mortality rate
            initial_infections_percentage = action_value[4]
            action_value = torch.stack(
                [beta / (gamma + mu), mu, initial_infections_percentage]
            )
        elif params["disease"] == "Flu":
            beta = action_value[0]
            # D = action_value[:,1]
            D = 3.5
            initial_infections_percentage = action_value[1]
            action_value = torch.stack([beta * D, initial_infections_percentage])
        action_value = action_value.reshape(1, -1)  # make sure it's 2d
        print("R0 ABM-pred-correction", action_value)
    elif "GradABM-learnable-params" in params["model_name"]:
        action_value = param_model.forward()
        action_value = action_value.repeat((meta.shape[0], 1))
    else:
        raise ValueError("model name not valid")
    return action_value


def build_param_model(
    params,
    metas_train_dim,
    X_train_dim,
    device,
    CUSTOM_INIT=True,
    parameters_to_calibrate=None,
    n_districts=None,
):

    # get param dimension for ODE
    if params["disease"] == "COVID":
        ode_param_dim = 5
        abm_param_dim = 3
        scale_output_ode = "seirm"
        scale_output_abm = "abm-covid"
    elif params["disease"] == "Flu":
        ode_param_dim = 2
        abm_param_dim = 2
        scale_output_ode = "sirs"
        scale_output_abm = "abm-flu"
    training_weeks = params["num_steps"] / 7  # only needed for time-varying
    assert training_weeks == int(training_weeks)

    """ call constructor of param model depending on the model we want to run"""
    if params["model_name"].startswith("GradABM-time-varying") or params[
        "model_name"
    ].startswith("june"):
        param_model = CalibNNJune(
            metas_train_dim,
            X_train_dim,
            device,
            training_weeks,
            n_districts=n_districts,
            parameters_to_calibrate=parameters_to_calibrate,
        ).to(device)
    elif params["model_name"] == "ABM-expert":
        param_model = None
    elif "ABM-pred-correction" in params["model_name"]:
        # load the param model from ODE
        # NOTE: currently it uses only R0
        param_model = LearnableParams(ode_param_dim, device, scale_output_ode).to(
            device
        )
    elif "GradABM-learnable-params" in params["model_name"]:
        param_model = LearnableParams(abm_param_dim, device, scale_output_abm).to(
            device
        )
    else:
        raise ValueError("model name not valid")
    return param_model


def build_simulator(params, devices, counties):
    """build simulator: ABM or ODE"""

    if "ABM" in params["model_name"]:
        if params["joint"]:
            abm = {}
            # abm devices are different from the ones for the params model
            if len(devices) > 1:
                abm_devices = devices[1:]
            else:
                abm_devices = devices
            num_counties = len(counties)
            for c in range(num_counties):
                c_params = copy(params)
                c_params["county_id"] = counties[c]
                abm[counties[c]] = GradABM(c_params, abm_devices[c % len(abm_devices)])
        else:
            if len(devices) > 1:
                abm_device = devices[1]
            else:
                abm_device = devices[0]
            abm = GradABM(params, abm_device)

    elif "ODE" in params["model_name"]:
        if params["disease"] == "COVID":
            abm = SEIRM(params, devices[0])
        elif params["disease"] == "Flu":
            abm = SIRS(params, devices[0])

    elif "june" in params["model_name"]:
        abm = June(params, devices)

    return abm


def forward_simulator(params, param_values, abm, training_num_steps, counties, devices):
    """assumes abm contains only one simulator for covid (one county), and multiple for flu (multiple counties)"""

    if params["model_name"] == "june":
        predictions = []
        predictions = abm.step(param_values)
        return predictions
    else:
        if params["joint"]:
            num_counties = len(counties)
            predictions = torch.empty((num_counties, training_num_steps)).to(devices[0])
            for time_step in range(training_num_steps):
                if "time-varying" in params["model_name"]:
                    param_t = param_values[:, time_step // 7, :]
                else:
                    param_t = param_values
                # go over each abm
                for c in range(num_counties):
                    model_device = abm[counties[c]].device
                    population = abm[counties[c]].num_agents
                    _, pred_t = abm[counties[c]].step(
                        time_step, param_t[c].to(model_device)
                    )
                    predictions[c, time_step] = pred_t.to(devices[0])
        else:
            num_counties = 1
            param_values = param_values.squeeze(0)
            predictions = []
            for time_step in range(training_num_steps):
                if "time-varying" in params["model_name"]:
                    param_t = param_values[time_step // 7, :]
                else:
                    param_t = param_values
                model_device = abm.device
                _, pred_t = abm.step(time_step, param_t.to(model_device))
                predictions.append(pred_t.to(devices[0]))
            predictions = torch.stack(predictions, 0).reshape(
                1, -1
            )  # num counties, seq len

    # post-process predictions for flu
    # targets are weekly, so we have to convert from daily to weekly
    if not params["model_name"].startswith("june"):
        if params["disease"] == "Flu":
            predictions = predictions.reshape(num_counties, -1, 7).sum(2)
        else:
            predictions = predictions.reshape(num_counties, -1)
    return predictions.unsqueeze(2)


def runner(params, devices, verbose):
    if params["model_name"].startswith("june"):
        abm = build_simulator(copy(params), devices, None)
        parameters_to_calibrate = abm.parameters_to_calibrate
    else:
        parameters_to_calibrate = None
    for run_id in range(params["num_runs"]):
        print("Run: ", run_id)

        # set batch size depending on the number of devices
        batch_size = max(len(devices) - 1, 1)

        # get data loaders and ground truth targets
        if params["model_name"].startswith("june"):
            district_data = DistrictData.from_file(initial_day=params["initial_day"])
            abm._assign_district_to_agents(district_data)
            (
                train_loader,
                metas_train_dim,
                X_train_dim,
                seqlen,
            ) = district_data.prepare_data_for_training(
                june=abm,
                number_of_weeks=params["number_of_weeks"],
                districts_map=abm.districts_map,
            )
            params["num_steps"] = seqlen * 7
            n_districts = abm.number_of_districts
        else:
            n_districts = None
            if params["disease"] == "COVID":
                if params["joint"]:
                    (
                        train_loader,
                        metas_train_dim,
                        X_train_dim,
                        seqlen,
                    ) = fetch_county_data_covid(
                        params["state"],
                        "all",
                        pred_week=params["pred_week"],
                        batch_size=batch_size,
                        noise_level=params["noise_level"],
                    )
                else:
                    (
                        train_loader,
                        metas_train_dim,
                        X_train_dim,
                        seqlen,
                    ) = fetch_county_data_covid(
                        params["state"],
                        params["county_id"],
                        pred_week=params["pred_week"],
                        batch_size=batch_size,
                        noise_level=params["noise_level"],
                    )
                params["num_steps"] = seqlen
            elif params["disease"] == "Flu":
                if params["joint"]:
                    (
                        train_loader,
                        metas_train_dim,
                        X_train_dim,
                        seqlen,
                    ) = fetch_county_data_flu(
                        params["state"],
                        "all",
                        pred_week=params["pred_week"],
                        batch_size=batch_size,
                        noise_level=params["noise_level"],
                    )
                else:
                    (
                        train_loader,
                        metas_train_dim,
                        X_train_dim,
                        seqlen,
                    ) = fetch_county_data_flu(
                        params["state"],
                        params["county_id"],
                        pred_week=params["pred_week"],
                        batch_size=batch_size,
                        noise_level=params["noise_level"],
                    )
                params["num_steps"] = seqlen * 7

        # add days ahead to num steps because num steps is used for forward pass of param model
        training_num_steps = params["num_steps"]
        params["num_steps"] += DAYS_HEAD
        param_model = build_param_model(
            params,
            metas_train_dim,
            X_train_dim,
            devices[0],
            CUSTOM_INIT=True,
            parameters_to_calibrate=parameters_to_calibrate,
            n_districts=n_districts,
        )
        # filename to save/load model
        file_name = "param_model" + "_" + params["model_name"]
        # do not train ABM because it uses a different calibration procedure
        train_flag = (
            False
            if params["model_name"].startswith("ABM") or params["inference_only"]
            else True
        )

        num_epochs = NUM_EPOCHS_DIFF
        CLIP = 10
        # if "learnable-params" in params["model_name"]:
        #    lr = 1e-2  # obtained after tuning
        #    num_epochs *= 2
        # else:
        #    lr = 1e-4 if params["model_name"].startswith("GradABM") else 1e-4
        lr = 1e-4

        """ step 1: training  """
        if train_flag:
            assert param_model != None
            opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, param_model.parameters()),
                lr=lr,
                weight_decay=0.01,
            )
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #    opt, T_max=NUM_EPOCHS_DIFF
            # )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                opt, milestones=[50, 500], gamma=0.5
            )
            loss_fcn = torch.nn.MSELoss(reduction="mean")
            best_loss = np.inf
            losses = []
            save_path = default_data_path / "June/london_fits/losses"
            save_path.mkdir(exist_ok=True, parents=True)
            i = 0
            while True:
                losses_save_file = f"losses_{i:03d}.csv"
                losses_save_file_path = save_path / losses_save_file
                if losses_save_file_path.exists():
                    i += 1
                    continue
                break
            df = pd.DataFrame(columns=["loss"])
            df.to_csv(losses_save_file_path)
            for epi in range(num_epochs):
                print(epi)
                start = time.time()
                batch_predictions = []
                if verbose:
                    print("\n", "=" * 60)
                    print("Epoch: ", epi)
                epoch_loss = 0
                for batch, (counties, meta, x, y_deaths, y_seroprev) in enumerate(
                    train_loader
                ):
                    print("--------------")
                    print(batch, counties)
                    # construct abm for each forward pass
                    if not params["model_name"].startswith("june"):
                        abm = build_simulator(copy(params), devices, counties)
                    # forward pass param model
                    meta = meta.to(devices[0])
                    x = x.to(devices[0])
                    param_values = param_model_forward(param_model, params, x, meta)
                    if verbose:
                        if param_values.dim() > 2:
                            print(param_values[:, [0, -1], :])
                        else:
                            print(param_values)
                    # forward simulator for several time steps
                    if BENCHMARK_TRAIN:
                        start_bench = time.time()
                    predictions_deaths, predictions_seroprev = forward_simulator(
                        params, param_values, abm, training_num_steps, counties, devices
                    )
                    y_deaths = y_deaths.to(predictions_deaths.device)
                    y_seroprev = y_seroprev.to(predictions_deaths.device)
                    # print("*"*10)
                    # print(predictions)
                    # print(y)
                    # print("*"*10)
                    if BENCHMARK_TRAIN:
                        # quit after 1 epoch
                        print("No steps:", training_num_steps)
                        print("time (s): ", time.time() - start_bench)
                        quit()
                    # loss
                    if verbose:
                        print(torch.cat((y, predictions), 2))
                    # normalize
                    predictions_deaths = predictions_deaths  # / (y_deaths.sum(0)[-1])
                    y_deaths = y_deaths  # / (y_deaths.sum(0)[-1])
                    # print(predictions_deaths)
                    # print(y_deaths)
                    predictions_seroprev = predictions_seroprev
                    y_seroprev = y_seroprev
                    # print("deaths")
                    # print(y_deaths.sum(0))
                    # print(predictions_deaths.sum(0))
                    # print("seroprev")
                    # print(y_seroprev[0,:])
                    # print(predictions_seroprev)
                    loss = loss_fcn(y_deaths, predictions_deaths)
                    print(loss)
                    print(y_seroprev[0, :])
                    print(predictions_seroprev)
                    loss_seroprev = 1000 * loss_fcn(
                            y_seroprev[0, 1:], predictions_seroprev[1:]
                    )
                    print(loss_seroprev)
                    #loss += loss_seroprev
                    # print(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(param_model.parameters(), CLIP)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    epoch_loss += torch.sqrt(loss.detach()).item()
                scheduler.step()
                losses.append(epoch_loss / (batch + 1))  # divide by number of batches
                df.loc[epi, "loss"] = epoch_loss / (batch + 1)

                if verbose:
                    print("epoch_loss", epoch_loss)

                if torch.isnan(loss):
                    break
                """ save best model """
                df.to_csv(losses_save_file_path, index=False)
                print((epoch_loss / (batch + 1)))
                if (epoch_loss / (batch + 1)) < 20:
                    save_parameters(abm)
                    break
                #if epoch_loss < best_loss:
                #    if params["joint"]:
                #        save_model(
                #            param_model,
                #            file_name,
                #            params["disease"],
                #            "joint",
                #            params["pred_week"],
                #        )
                #    else:
                #        save_model(
                #            param_model,
                #            file_name,
                #            params["disease"],
                #            params["county_id"],
                #            params["pred_week"],
                #        )
                #    best_loss = epoch_loss

                print("epoch {} time (s): {:.2f}".format(epi, time.time() - start))

        """ step 2: inference step  """
        """ upload best model in inference """
        print("TRAINING ENDED")
        param_model = None
        param_model = build_param_model(
            copy(params),
            metas_train_dim,
            X_train_dim,
            devices[0],
            CUSTOM_INIT=True,
            parameters_to_calibrate=parameters_to_calibrate,
            n_districts=n_districts,
        )
        if not params["model_name"].startswith("ABM"):
            # load param model if it is not ABM-expert
            if params["joint"]:
                param_model = load_model(
                    param_model,
                    file_name,
                    params["disease"],
                    "joint",
                    params["pred_week"],
                    devices[0],
                )
            else:
                param_model = load_model(
                    param_model,
                    file_name,
                    params["disease"],
                    params["county_id"],
                    params["pred_week"],
                    devices[0],
                )
        elif "ABM-pred-correction" in params["model_name"]:
            # pred-correction, uses param model from ODE
            file_name = "param_model" + "_" + "DiffODE-learnable-params"
            if params["noise_level"] > 0:
                file_name = (
                    "param_model"
                    + "_"
                    + "DiffODE-learnable-params"
                    + "-noise"
                    + str(params["noise_level"])
                )
            param_model = load_model(
                param_model,
                file_name,
                params["disease"],
                params["county_id"],
                params["pred_week"],
                devices[0],
            )

        num_step = training_num_steps + DAYS_HEAD
        batch_predictions = []
        counties_predicted = []
        learned_params = []
        with torch.no_grad():
            for batch, (counties, meta, x, y) in enumerate(train_loader):
                # construct abm for each forward pass
                if not params["model_name"].startswith("june"):
                    abm = build_simulator(params, devices, counties)
                # forward pass param model
                meta = meta.to(devices[0])
                x = x.to(devices[0])
                param_values = param_model_forward(param_model, params, x, meta)
                # forward simulator for several time steps
                preds = forward_simulator(
                    params, param_values, abm, num_step, counties, devices
                )
                batch_predictions.append(preds)
                counties_predicted.extend(counties)
                learned_params.extend(np.array(param_values.cpu().detach()))
        predictions = torch.cat(batch_predictions, axis=0)
        # we only care about the last predictions
        # predictions are weekly, so we only care about the last 4
        if params["disease"] == "Flu":
            predictions = predictions.squeeze(2)[:, -DAYS_HEAD // 7 :]
        else:
            predictions = predictions.squeeze(2)[:, -DAYS_HEAD:]
        """ remove grad """
        predictions = predictions.cpu().detach()

        """ release memory """
        param_model = None
        abm = None
        torch.cuda.empty_cache()

        """ plot losses """
        # only if trained
        if train_flag:
            disease = params["disease"]
            if params["joint"]:
                FIGPATH = f"./Figures/{disease}/joint/"
            else:
                county_id = params["county_id"]
                FIGPATH = f"./Figures/{disease}/{county_id}/"
            if not os.path.exists(FIGPATH):
                os.makedirs(FIGPATH)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(losses)
            pred_week = params["pred_week"]
            fig.savefig(FIGPATH + f"/losses_{pred_week}.png")
        print("-" * 60)
        return counties_predicted, np.array(predictions), learned_params


def train_predict(args):

    # Setting seed
    print("=" * 60)
    if args.joint:
        print(f"state {args.state} week {args.pred_week}")
    else:
        print(f"county {args.county_id} week {args.pred_week}")
    # print("Seed used for python random, numpy and torch is {}".format(args.seed))
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    params = {}
    params["seed"] = args.seed
    params["num_runs"] = args.num_runs
    params["disease"] = args.disease
    params["pred_week"] = args.pred_week
    params["joint"] = args.joint
    params["inference_only"] = args.inference_only
    params["noise_level"] = args.noise  # for robustness experiments
    params["model_name"] = args.model_name
    params["initial_day"] = args.initial_day
    params["number_of_weeks"] = int(args.number_of_weeks)
    # state
    params["state"] = args.state
    if params["model_name"] != "june":
        if params["joint"]:
            # verify it is a state
            assert params["state"] in states
        else:
            params["county_id"] = args.county_id
            # verify county belong to state
            assert params["county_id"] in counties[params["state"]]

    if args.dev == ["cpu"]:
        devices = [torch.device("cpu")]
    else:
        devices = [torch.device(f"cuda:{i}") for i in args.dev]

    print("devices used:", devices)
    verbose = False
    counties_predicted, predictions, learned_params = runner(params, devices, verbose)

    return counties_predicted, predictions, learned_params

def save_parameters(abm):
    save_path = Path(os.path.abspath(__file__)).parent / "Data/June/london_fits"
    # find place to save.
    i = 0
    while True:
        save_file = f"london_fit_{i:03d}.yaml"
        save_file_path = save_path / save_file
        if save_file_path.exists():
            i += 1
            continue
        break

    # save best config
    base_config = yaml.safe_load(open(save_path.parent / "june_default.yaml"))
    best_params = abm.param_values_df.loc[len(abm.param_values_df)-1]
    for name, param in best_params.iteritems():
        if "networks" in name:
            network = name.split(".")[-2]
            if network == "leisure":
                for leisure_name in ["pub", "grocery", "gym", "cinema", "visit"]:
                    base_config["networks"][leisure_name]["log_beta"] = param
            else:
                base_config["networks"][network]["log_beta"] = param
        if name == "log_fraction_initial_cases":
            base_config["infection_seed"]["log_fraction_initial_cases"] = param
        if "interaction_policies" in name:
            number = int(name.split(".")[-3])
            base_config["policies"]["interaction"]["social_distancing"][number+1]["beta_factors"]["all"] = param
    base_config["system"]["device"] = "cuda:6"
    yaml.dump(base_config, open(save_file_path, "w"))
    return



