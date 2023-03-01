from dis import dis
from tkinter import W
import os
import torch
import pickle
from pathlib import Path
import numpy as np
import yaml
import pandas as pd
from datetime import datetime, timedelta
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

from grad_june import Runner
from grad_june.utils import read_path

from one_shot_sensitivity.paths import default_data_path
from one_shot_sensitivity.model_utils import SeqDataJune

default_june_data_path = default_data_path / "June"
default_june_config_path = default_june_data_path / "june_default.yaml"
default_daily_deaths_filename = default_june_data_path / "deaths_by_lad.csv"
default_mobility_data_filename = default_june_data_path / "london_mobility_data.csv"
default_area_to_district_filename = default_june_data_path / "area_district.csv"
default_seroprevalence_filename = default_june_data_path / "seroprevalence_ward.csv"


def get_attribute(base, path):
    paths = path.split(".")
    for p in paths:
        if not any(i.isdigit() for i in p):
            base = getattr(base, p)
        else:
            pn = int(p)
            base = base[pn]
    return base


def set_attribute(base, path, target):
    paths = path.split(".")
    _base = base
    for p in paths[:-1]:
        if not any(i.isdigit() for i in p):
            _base = getattr(_base, p)
        else:
            pn = int(p)
            _base = _base[pn]
    if type(_base) == dict:
        _base[paths[-1]] = target
    else:
        setattr(_base, paths[-1], target)


class June:
    r"""
    Wrapper around torch_june
    """

    def __init__(self, params, device: str):
        self.runner = Runner.from_file(default_june_config_path)
        self.number_of_districts = None
        self.districts_map = None
        with open(default_june_config_path, "r") as f:
            june_params = yaml.safe_load(f)
            self.parameters_to_calibrate = june_params["parameters_to_calibrate"]
        self.param_values_df = pd.DataFrame(columns=self.parameters_to_calibrate.keys())
        save_path = default_june_data_path / "london_fits/losses"
        save_path.mkdir(parents=True, exist_ok=True)
        i = 0
        while True:
            params_save_file = f"params_{i:03d}.csv"
            params_save_file_path = save_path / params_save_file
            if params_save_file_path.exists():
                i += 1
                continue
            break
        self.params_save_file_path = params_save_file_path
        self.param_values_df.to_csv(params_save_file_path)

    @property
    def device(self):
        return self.runner.device

    def _assign_district_to_agents(self, district_data):
        district_ids = district_data.area_to_district.loc[
            self.runner.data["agent"].area, "id"
        ].values
        district_nums = torch.arange(0, np.unique(district_ids).shape[0])
        ret_idcs = np.searchsorted(np.sort(np.unique(district_ids)), district_ids)
        ret = district_nums[ret_idcs]
        self.runner.data["agent"].district = ret.to(self.device)
        data_path = read_path(self.runner.input_parameters["data_path"])
        pickle.dump(self.runner.data, open(data_path, "wb"))  # save districts
        self.number_of_districts = self.runner.data["agent"].district.unique().shape[0]
        self.districts_map = np.sort(np.unique(district_ids))
        print("NUMBER OF DISTRICTS")
        print(self.runner.data["agent"].district.unique().shape[0])

    def _set_param_values(self, param_values):
        param_values = param_values.flatten()
        for param_name, param_value in zip(self.parameters_to_calibrate, param_values):
            if param_name == "model.infection_networks.networks.leisure.log_beta":
                for leisure_name in ["pub", "grocery", "gym", "cinema", "visit"]:
                    pname = param_name.split(".")
                    pname[-2] = leisure_name
                    pname = ".".join(pname)
                    set_attribute(self.runner, pname, param_value)
            else:
                set_attribute(self.runner, param_name, param_value)

    def _get_deaths_per_week(self):
        deaths_by_district_timestep = self.runner.data["results"][
            "daily_deaths_by_district"
        ].transpose(0, 1)
        mask = torch.zeros(
            deaths_by_district_timestep.shape[1], dtype=torch.long, device=self.device
        )
        mask[::7] = 1  # take every 7 days.
        mask = mask.to(torch.bool)
        ret = deaths_by_district_timestep[:, mask]
        return ret

    def _save_param_values(self, param_values):
        self.param_values_df.loc[len(self.param_values_df)] = (
            param_values.flatten().detach().cpu().numpy()
        )
        #self.param_values_df.to_csv("./param_values.csv", index=False)
        self.param_values_df.to_csv(self.params_save_file_path, index=False)

    def _get_seroprevalence(self, results):
        ret = torch.zeros(8, device=results["cases_per_timestep"].device)
        for i, age in enumerate(("18", "25", "35", "45", "55", "65", "75", "100")):
            cases_by_age = results[f"cases_by_age_{age}"][-1]
            population_by_age = self.runner.population_by_age[i]
            # this ignores deaths correction...
            seroprev = cases_by_age / population_by_age.to(cases_by_age.device) * 100
            ret[i] = seroprev
        ret = ret
        return ret

    def step(self, param_values):
        self._set_param_values(param_values)
        self._save_param_values(param_values)
        results, _ = self.runner()
        predictions_deaths = self._get_deaths_per_week()
        predictions_deaths = predictions_deaths.unsqueeze(2)
        predictions_seroprev = self._get_seroprevalence(results)
        return predictions_deaths, predictions_seroprev


class DistrictData:
    def __init__(
        self,
        initial_day: str,
        daily_deaths: pd.DataFrame,
        mobility_data: pd.DataFrame,
        area_to_district: pd.DataFrame,
        seroprevalence: pd.DataFrame,
    ):
        """
        Handles data at the district level, like deaths per day or mobility data.
        """
        self.initial_day = initial_day
        self.daily_deaths = (
            daily_deaths.set_index("date")
            .fillna(method="ffill")
            .fillna(method="backfill")
            .fillna(0)
        )
        self.weekly_deaths = self._get_weekly_deaths(self.daily_deaths, initial_day)
        self.daily_mobility_data = (
            mobility_data.set_index("date")
            .fillna(method="ffill")
            .fillna(method="backfill")
            .fillna(0)
        )

        self.weekly_mobility_data = self._get_weekly_mobility(
            self.daily_mobility_data, initial_day
        )
        self.area_to_district = area_to_district
        self.seroprevalence = seroprevalence.values.flatten()

    @classmethod
    def from_file(
        cls,
        initial_day: str,
        daily_deaths_filename: str = default_daily_deaths_filename,
        mobility_data_filename: str = default_mobility_data_filename,
        area_to_district_filename: str = default_area_to_district_filename,
        seroprevalence_filename: str = default_seroprevalence_filename,
    ):
        daily_deaths = pd.read_csv(daily_deaths_filename)
        mobility_data = pd.read_csv(mobility_data_filename)
        area_to_district = pd.read_csv(area_to_district_filename, index_col=0)
        seroprevalence = pd.read_csv(seroprevalence_filename)["middle"]
        return cls(
            initial_day=initial_day,
            daily_deaths=daily_deaths,
            mobility_data=mobility_data,
            area_to_district=area_to_district,
            seroprevalence=seroprevalence,
        )

    def _get_weekly_deaths(self, daily_deaths: pd.DataFrame, initial_day: pd.DataFrame):
        """
        Groups daily deaths by week, summing them.

        Args:
            daily_deaths: dataframe with number of deaths per day and district.
            initial_day: When to start counting weeks.
        """
        df = daily_deaths.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.reset_index()
        df = df.loc[df.date >= initial_day]
        df["weekly_deaths"] = df.groupby(["district_id"])["daily_deaths"].cumsum()
        df = df[["date", "district_id", "weekly_deaths"]]
        df = df.sort_values(["date", "district_id"]).set_index("date")
        return df

    def _get_weekly_mobility(
        self, daily_mobility: pd.DataFrame, initial_day: pd.DataFrame
    ):
        """
        Groups daily mobility by week, taking the average.
        Args:
            daily_mobility: dataframe with mobility reductions per day and district.
            initial_day: When to start counting weeks.
        """
        data_weekly = daily_mobility.copy()
        data_weekly = data_weekly.reset_index()
        data_weekly = data_weekly.loc[data_weekly.date >= initial_day]
        data_weekly["date"] = pd.to_datetime(data_weekly["date"]) - pd.to_timedelta(
            7, unit="d"
        )
        data_weekly = (
            data_weekly.groupby(["district_id", pd.Grouper(key="date", freq="W-MON")])
            .mean()
            .reset_index()
            .sort_values(["date", "district_id"])
            .set_index("date")
        )
        return data_weekly

    def get_data(self, june, district: int, week_1: int, week_2: int):
        """
        Gets data between week_1 and week_2

        Args:
            district: district id
            week_1: first week
            week_2: last week (included)
        """
        timer = june.runner.timer
        n_weeks = week_2 - week_1
        initial_day = timer.initial_date + timedelta(days=week_1 * 7)

        # mobility data
        district_data = self.weekly_mobility_data.loc[
            self.weekly_mobility_data.district_id == district
        ].drop(columns="district_id")
        features_mobility = []
        for n in range(n_weeks):
            day = initial_day + timedelta(days=n * 7)
            if day not in district_data.index:
                features_mobility.append(np.zeros(len(district_data.columns)))
            else:
                features_mobility.append(district_data.loc[day].values)
        features_mobility = np.array(features_mobility)

        # deaths data
        district_data = self.weekly_deaths.loc[
            self.weekly_deaths.district_id == district
        ].drop(columns="district_id")
        features_deaths = []
        for n in range(n_weeks):
            day = initial_day + timedelta(days=n * 7)
            if day not in district_data.index:
                features_deaths.append(0.0)
            else:
                features_deaths.append(district_data.loc[day].values[0])
        features_deaths = np.array(features_deaths)
        features = features_mobility
        targets = features_deaths
        return features, targets

    def get_train_data_district(self, june, district: int, number_of_weeks: int):
        """
        Gets training data for the specified district from the `initial_day`
        to `initial_day + number_of_weeks`.

        Args:
            district: district id
            number_of_weeks: Number of weeks from initial_day.
        """
        return self.get_data(june, district, 0, number_of_weeks)

    def get_test_data_district(
        self,
        june,
        district: int,
        number_of_training_weeks: int,
        number_of_testing_weeks: int,
    ):
        """
        Gets testing data

        Args:
            district: district id
            number_of_weeks: Number of weeks from initial_day.
        """
        return self.get_data(
            june,
            district,
            number_of_training_weeks + 1,
            number_of_training_weeks + number_of_testing_weeks + 1,
        )

    def get_train_data(self, june, number_of_weeks: int, districts):
        # districts = np.sort(self.weekly_mobility_data.district_id.unique())
        features = []
        targets = []
        for district in districts:
            district_features, district_targets = self.get_train_data_district(
                june, district, number_of_weeks
            )
            district_features = StandardScaler().fit_transform(district_features)
            features.append(district_features)
            targets.append(district_targets)
        targets = {
            "deaths": torch.tensor(np.array(targets), dtype=torch.float),
            "seroprevalence": torch.tensor(self.seroprevalence, dtype=torch.float),
        }
        return np.array(features), targets

    def get_static_metadata(self):
        """
        Creates static metadata for each county.
        """
        all_counties = np.sort(self.daily_deaths.reset_index().district_id.unique())
        county_idx = {r: i for i, r in enumerate(all_counties)}
        metadata = np.diag(np.ones(len(all_counties)))
        return metadata

    def create_window_seqs(self, X: np.array, y: np.array, min_sequence_length: int):
        """
        Creates windows of fixed size with appended zeros

        Args:
            X: features
            y: targets, in synchrony with features
                (i.e. x[t] and y[t] correspond to the same time)
        """
        # convert to small sequences for training, starting with length 10
        seqs = []
        targets = []
        mask_ys = []

        # starts at sequence_length and goes until the end
        # for idx in range(min_sequence_length, X.shape[0]+1, 7): # last in range is step
        for idx in range(min_sequence_length, X.shape[0] + 1, 1):
            # Sequences
            seqs.append(torch.from_numpy(X[:idx, :]))
            # Targets
            y_ = y[:idx]
            mask_y = torch.ones(len(y_))
            targets.append(torch.from_numpy(y_))
            mask_ys.append(mask_y)
        seqs = pad_sequence(seqs, batch_first=True, padding_value=0).type(torch.float)
        ys = pad_sequence(targets, batch_first=True, padding_value=-999).type(
            torch.float
        )
        mask_ys = pad_sequence(mask_ys, batch_first=True, padding_value=0).type(
            torch.float
        )

        return seqs, ys, mask_ys

    def prepare_data_for_training(self, june, number_of_weeks: int, districts_map):
        """
        Prepare train and validation dataset
        """
        metadata = self.get_static_metadata()
        X_train, y_train = self.get_train_data(june, number_of_weeks, districts_map)
        X_train = torch.tensor(X_train, dtype=torch.float)
        metadata = torch.tensor(metadata, dtype=torch.float)
        all_counties = np.sort(self.daily_deaths.district_id.unique())
        y_train["deaths"] = y_train["deaths"].unsqueeze(2)

        train_dataset = SeqDataJune(
            all_counties,
            metadata,
            X_train,
            y_deaths=y_train["deaths"],
            y_seroprev=y_train["seroprevalence"],
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=X_train.shape[0], shuffle=False
        )
        seqlen = X_train.shape[1]
        return train_loader, metadata.shape[1], X_train.shape[2], seqlen
