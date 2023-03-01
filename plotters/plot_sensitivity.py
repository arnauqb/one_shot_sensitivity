import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import numpy as np
import random
import yaml
import pandas as pd
from collections import defaultdict
from datetime import datetime
import torch

from .paths import data_path
from .paths import figures_path

from .analyser import Analyser
from grad_june import Runner

def set_runner_parameters(runner):
    for name in runner.model.infection_networks.networks:
        runner.model.infection_networks.networks[
            name
        ].log_beta = torch.nn.Parameter(
            runner.model.infection_networks.networks[name].log_beta
        )

def get_sensitivity_ethnicity_multiple(files, days):
    dfs = []
    for file in files:
        ret = get_sensitivity_ethnicity(file, days)
        dfs.append(pd.DataFrame(ret))
    return dfs

def get_sensitivity_ethnicity(file, days):
    parameters = yaml.safe_load(open(file))
    parameters["system"]["device"] = "cuda:5"
    parameters["timer"]["total_days"] = days
    runner = Runner.from_parameters(parameters)
    set_runner_parameters(runner)
    results, _ = runner()
    ethnicities = ["A", "B", "C", "D", "E"]
    ret = {}
    total_infected = runner.data["agent"].is_infected.sum()
    for ethnicity in ethnicities:
        runner.zero_grad()
        ret[ethnicity] = {}
        mask = np.char.startswith(runner.data["agent"].ethnicity, ethnicity)
        fraction_cases = (
            runner.data["agent"].is_infected[mask].sum() / total_infected
        )
        fraction_cases.backward(retain_graph=True)
        for name in runner.model.infection_networks.networks:
            ret[ethnicity][name] = runner.model.infection_networks.networks[
                name
            ].log_beta.grad.item()
    return ret

def get_sensitivity_age_multiple(files, days):
    dfs = []
    for file in files:
        ret = get_sensitivity_age(file, days)
        dfs.append(pd.DataFrame(ret))
    return dfs

def get_sensitivity_age(file, days):
    parameters = yaml.safe_load(open(file))
    parameters["system"]["device"] = "cuda:5"
    parameters["timer"]["total_days"] = days
    runner = Runner.from_parameters(parameters)
    set_runner_parameters(runner)
    results, _ = runner()
    age_bins = runner.age_bins
    ret = {}
    for age_bin in age_bins[1:]:
        runner.zero_grad()
        age_bin = age_bin.item()
        ret[age_bin] = {}
        cases = (
            results[f"cases_by_age_{age_bin:02d}"][-1]
            / results["cases_per_timestep"][-1]
        )
        cases.backward(retain_graph=True)
        for name in runner.model.infection_networks.networks:
            ret[age_bin][name] = runner.model.infection_networks.networks[
                name
            ].log_beta.grad.item()
    return ret

def get_sensitivity_social_multiple(files, days):
    dfs = []
    for file in files:
        ret = get_sensitivity_social(file, days)
        dfs.append(pd.DataFrame(ret))
    return dfs

def get_sensitivity_social(file, days):
    parameters = yaml.safe_load(open(file))
    parameters["system"]["device"] = "cuda:5"
    parameters["timer"]["total_days"] = days
    runner = Runner.from_parameters(parameters)
    set_runner_parameters(runner)
    results, _ = runner()
    quintiles = range(1, 6)
    ret = {}
    for quintile in quintiles:
        runner.zero_grad()
        ret[quintile] = {}
        mask = runner.data["agent"].socioeconomic_index == quintile
        cases = (
            runner.data["agent"].is_infected[mask].sum()
            / runner.data["agent"].is_infected.sum()
        )
        cases.backward(retain_graph=True)
        for name in runner.model.infection_networks.networks:
            ret[quintile][name] = runner.model.infection_networks.networks[
                name
            ].log_beta.grad.item()
    return ret

def get_sensitivity_district_multiple(files, days):
    dfs = []
    for file in files:
        ret = get_sensitivity_district(file, days)
        dfs.append(pd.DataFrame(ret))
    return dfs

def get_sensitivity_district(file, days):
    parameters = yaml.safe_load(open(file))
    parameters["system"]["device"] = "cuda:5"
    parameters["timer"]["total_days"] = days
    runner = Runner.from_parameters(parameters)
    set_runner_parameters(runner)
    results, _ = runner()
    districts = np.sort(np.unique(runner.data["agent"].district.cpu()))
    ret = {}
    for district in districts:
        runner.zero_grad()
        ret[district] = {}
        mask = runner.data["agent"].district == district
        cases = (
            runner.data["agent"].is_infected[mask].sum()
            / runner.data["agent"].is_infected.sum()
        )
        cases.backward(retain_graph=True)
        for name in runner.model.infection_networks.networks:
            ret[district][name] = runner.model.infection_networks.networks[
                name
            ].log_beta.grad.item()
    return ret



def generate_age_df(runner, date, n=2):
    analyser = Analyser(runner, save_path=figures_path)
    analyser.run()
    # Ethnicity beta household
    df = None
    for i in range(n):
        dd_norm_age = analyser.get_gradient_normalised_cases_by_age_location(date)
        if df is None:
            df = pd.DataFrame(dd_norm_age)
        else:
            df = df + pd.DataFrame(dd_norm_age)
    df = df / n
    return df


def generate_ethnicity_df(runner, n=2):
    analyser = Analyser(runner, save_path=figures_path)
    analyser.run()
    # Ethnicity beta household
    df = None
    for i in range(n):
        dd_norm_ethn = analyser.get_gradient_normalised_cases_by_ethnicity_location()
        if df is None:
            df = pd.DataFrame(dd_norm_ethn)
        else:
            df = df + pd.DataFrame(dd_norm_ethn)
    df = df / n
    return df


def generate_social_df(runner, n=2):
    analyser = Analyser(runner, save_path=figures_path)
    analyser.run()
    # Ethnicity beta household
    df = None
    for i in range(n):
        dd_norm_imd = analyser.get_gradient_normalised_cases_by_socio_location()
        if df is None:
            df = pd.DataFrame(dd_norm_imd)
        else:
            df = df + pd.DataFrame(dd_norm_imd)
    df = df / n
    return df


def generate_district_df(runner, n=2):
    analyser = Analyser(runner, save_path=figures_path)
    analyser.run()
    # Ethnicity beta household
    df = None
    for i in range(n):
        dd_norm = analyser.get_gradient_normalised_cases_by_district_location()
        if df is None:
            df = pd.DataFrame(dd_norm)
        else:
            df = df + pd.DataFrame(dd_norm)
    df = df / n
    return df


def plot_sensitivity_analysis_multiple(files, days=5):
    def run_file(file):
        parameters = yaml.safe_load(open(file))
        parameters["system"]["device"] = "cuda:5"
        parameters["timer"]["total_days"] = days
        runner = Runner.from_parameters(parameters)
        set_runner_parameters(runner)
        results, _ = runner()
        res = results["cases_per_timestep"][-1]
        res.backward()
        ret = {}
        for name in runner.model.infection_networks.networks:
            ret[name] = runner.model.infection_networks.networks[
                name
            ].log_beta.grad.item()
        return ret

    dd = defaultdict(list)
    for file in files:
        dd_ = run_file(file)
        for key in dd_:
            dd[key].append(dd_[key])
    df = pd.DataFrame(dd)
    means = df.mean(0)
    std = df.std(0)
    tor = 1e-9
    lower_std = [a - tor if a<b else b for a,b in zip(means,std)]
    f, ax = plt.subplots()
    ax.bar(df.columns, means, yerr=(lower_std, std), capsize=5)
    plt.xticks(rotation=90)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta_i$")
    ax.set_ylim(1e-2, 1e6)
    ax.set_ylabel(r"Sensitivity ( cases / $\beta_i$ )")
    f.savefig(figures_path / "total_sensitivity.png", dpi=150, bbox_inches="tight")


def plot_sensitivity_analysis(runner, date="2020-04-10"):
    analyser = Analyser(runner, save_path=figures_path)
    analyser.run()

    # General sensitivity
    dd = analyser.get_gradient_cases_by_location(date)
    f, ax = plt.subplots()
    ax.bar(dd.keys(), dd.values())
    plt.xticks(rotation=90)
    ax.set_yscale("log")
    ax.set_ylabel(r"Sensitivity (cases / location)")
    f.savefig(figures_path / "total_sensitivity.png", dpi=150, bbox_inches="tight")

    # Ethnicity beta household
    # df = generate_ethnicity_df(runner)
    # names = ["White", "Mixed", "Asian", "Black", "Other"]
    # toplot = df.loc[["household", "school", "company"]]
    # toplot = toplot.rename(
    #    columns={"A": "White", "B": "Mixed", "C": "Asian", "D": "Black", "E": "Other"}
    # )
    # toplot = toplot.rename(
    #    {"company": "Company", "school": "School", "household": "Household"}
    # )
    # fig, ax = plt.subplots()
    # toplot.plot.barh(ax=ax, width=0.75)
    # ax.set_xlim(-0.07, 0.07)
    # ax.set_xlabel("Sensitivity")
    # ax.legend(title="Ethnicity")
    # fig.savefig(
    #    figures_path / "sensitivity_ethnicity.png", dpi=150, bbox_inches="tight"
    # )

    ## imd quintile
    # df = generate_social_df(runner)
    # f, ax = plt.subplots()
    # names = [i + 1 for i in range(5)]
    # toplot = toplot.rename(
    #    {"company": "Company", "school": "School", "household": "Household"}
    # )
    # toplot.plot.barh(ax=ax, width=0.75)
    # ax.set_xlabel(r"Sensitivity")
    # ax.set_ylabel("IMD quintile")
    # f.savefig(figures_path / "sensitivity_socio.png", dpi=150)


def _get_network_names(runner):
    return [n for n in runner.model.infection_networks.networks]


def get_gradient_cases(runner, parameters, date, network_names):
    runner.zero_grad()
    date = datetime.strptime(date, "%Y-%m-%d")
    parameters = torch.nn.Parameter(parameters)

    def run_for_params(params):
        for i, name in enumerate(network_names):
            runner.model.infection_networks.networks[name].log_beta = params[i]
        results, _ = runner()
        date_idx = np.array(results["dates"]) == date
        return results["cases_per_timestep"][date_idx]

    res = run_for_params(parameters)
    res.backward()
    return parameters.grad.detach().cpu().numpy()


def get_hessian_cases(runner, parameters, date, network_names):
    runner.zero_grad()
    date = datetime.strptime(date, "%Y-%m-%d")

    def run_for_params(params):
        for i, name in enumerate(network_names):
            runner.model.infection_networks.networks[name].log_beta = params[i]
        results, _ = runner()
        date_idx = np.array(results["dates"]) == date
        return results["cases_per_timestep"][date_idx]

    hessian = torch.autograd.functional.hessian(run_for_params, parameters)
    return hessian.detach().cpu().numpy()


def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_newton_step(file, date):
    runner = Runner.from_file(file)
    network_names = [
        "school",
        "company",
        "university",
    ]  # _get_network_names(runner) #["school", "company"]
    parameters = torch.tensor(
        [
            runner.model.infection_networks.networks[name].log_beta.item()
            for name in network_names
        ]
    )
    n_seed = random.randint(0, 100)
    fix_seed(n_seed)
    grad = get_gradient_cases(
        runner=runner, parameters=parameters, date=date, network_names=network_names
    )
    fix_seed(n_seed)
    hess = get_hessian_cases(
        runner=runner, parameters=parameters, date=date, network_names=network_names
    )
    invhess = np.linalg.inv(hess)
    step = -invhess.dot(grad)
    ret = {}
    for i, name in enumerate(network_names):
        ret[name] = step[i]
    return ret
