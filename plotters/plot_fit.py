import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import dates as mdates
import numpy as np
import pandas as pd
import yaml
import torch

from grad_june import Runner
from .paths import data_path
from .paths import figures_path


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def read_june_data():
    path = figures_path.parent / "june_london_fit.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def get_data(runner, results, window):
    data = pd.read_csv(data_path / "deaths_by_lad.csv")
    data.index = pd.to_datetime(data.index)
    area_district = pd.read_csv(data_path / "area_district.csv")
    areas = runner.data["agent"].area
    district_codes = (
        area_district.set_index("oa11cd").loc[areas].drop_duplicates()["id"].values
    )
    data = data.loc[data.district_id.isin(district_codes)]
    data = data.groupby("date").sum()
    data.index = pd.to_datetime(data.index)
    data = data.loc[results["dates"][0] : results["dates"][-1]]
    data = data.rolling(window=window).mean()
    return data


def get_seroprev_data():
    ret = pd.read_csv(data_path / "seroprevalence_ward.csv").iloc[1:]
    return ret


def get_seroprev_june(runner, results):
    ret = torch.zeros(7, device=results["cases_per_timestep"].device)
    for i, age in enumerate(("25", "35", "45", "55", "65", "75", "100")):
        cases_by_age = results[f"cases_by_age_{age}"][-1]
        population_by_age = runner.population_by_age[i + 1]  # 18
        # this ignores deaths correction...
        seroprev = cases_by_age / population_by_age.to(cases_by_age.device) * 100
        ret[i] = seroprev
    ret = ret.detach().cpu()
    return ret


def get_by_socio_index(runner):
    runner.data["agent"].socioeconomic_index[
        runner.data["agent"].socioeconomic_index == 0
    ] = 1
    runner.data["agent"].socioeconomic_index[
        runner.data["agent"].socioeconomic_index == 6
    ] = 5
    infected_agents = (
        runner.data["agent"]
        .socioeconomic_index[runner.data["agent"].is_infected.bool()]
        .detach()
        .cpu()
        .numpy()
    )
    bins, counts = np.unique(infected_agents, return_counts=True)
    counts = counts[np.argsort(bins)]
    agents = runner.data["agent"].socioeconomic_index.detach().cpu().numpy()
    bins2, counts2 = np.unique(agents, return_counts=True)
    counts2 = counts2[np.argsort(bins2)]
    total_cases = runner.data["agent"].is_infected.sum().item()
    return (counts / total_cases) / (counts2 / runner.n_agents)


def get_fraction_ethnicity(runner):
    total_agents = runner.n_agents
    ethnicities = runner.data["agent"].ethnicity.astype("<U1")  # gets 1st letter
    ethn, eth_counts = np.unique(ethnicities, return_counts=True)
    sort = np.argsort(ethn)
    ethn = ethn[sort]
    eth_counts = eth_counts[sort]

    total_cases = runner.data["agent"].is_infected.sum().item()
    infected_agents = ethnicities[runner.data["agent"].is_infected.bool().cpu()]
    infected, inf_counts = np.unique(infected_agents, return_counts=True)
    sort = np.argsort(infected)
    infected = infected[sort]
    inf_counts = inf_counts[sort]
    fraction = (inf_counts / total_cases) / (eth_counts / total_agents)
    return fraction


def get_percentage_ethnicity():
    df = pd.read_csv(data_path / "seroprevalence_ehtnicity.csv")
    df = df.drop(columns="missing")
    df = df / df.values.sum() * 100  # renormalise
    return df


def plot_runs(files, window=5, days=90):
    results_array = []
    with torch.no_grad():
        for file in tqdm(files):
            parameters = yaml.safe_load(open(file))
            parameters["system"]["device"] = "cuda:5"
            parameters["timer"]["total_days"] = days
            runner = Runner.from_parameters(parameters)
            results, _ = runner()

            # deaths
            res = results["deaths_per_timestep"].cpu().detach().numpy()
            res = np.diff(res, prepend=0)
            results_array.append(res)

    data = get_data(runner, results, window)
    fig, ax = plt.subplots()
    ax.plot(data.index, data.daily_deaths, label="Observed data", color="black")
    for i, res in enumerate(results_array):
        ax.plot(results["dates"], res, label=i)
    ax.legend()
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_major_locator(fmt_month)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(pd.to_datetime("2020-03-01"), pd.to_datetime("2020-05-01"))
    ax.set_ylabel("Daily deaths")

def plot_fit_multiple_alex(files, window=5, days=90, errors=True):
    results_array = None
    seroprev_array = None
    ethn_array = None
    socio_array = None
    with torch.no_grad():
        for file in tqdm(files):
            parameters = yaml.safe_load(open(file))
            parameters["system"]["device"] = "cuda:5"
            parameters["timer"]["total_days"] = days
            runner = Runner.from_parameters(parameters)
            results, _ = runner()

            # deaths
            res = results["deaths_per_timestep"].cpu().detach().numpy()
            res = np.diff(res, prepend=0)
            if results_array is None:
                results_array = res
            else:
                results_array = np.vstack((results_array, res))
    # deaths
    df = pd.DataFrame(results_array.T, columns=range(len(files)))
    df = df.rolling(window=window, axis=0).mean()
    june_mean = df.mean(1)

    data = get_data(runner, results, window)
    old_fit = read_june_data()
    old_fit.loc[str(data.index[0].date()) : str(data.index[-1].date())]
    return data, results["dates"], june_mean, old_fit

def plot_fit_multiple(files, window=5, days=90, errors=True):
    results_array = None
    seroprev_array = None
    ethn_array = None
    socio_array = None
    with torch.no_grad():
        for file in tqdm(files):
            parameters = yaml.safe_load(open(file))
            parameters["system"]["device"] = "cuda:5"
            parameters["timer"]["total_days"] = days
            runner = Runner.from_parameters(parameters)
            results, _ = runner()

            # deaths
            res = results["deaths_per_timestep"].cpu().detach().numpy()
            res = np.diff(res, prepend=0)
            if results_array is None:
                results_array = res
            else:
                results_array = np.vstack((results_array, res))
            # seroprev
            res_seroprev = get_seroprev_june(runner, results)
            if seroprev_array is None:
                seroprev_array = res_seroprev
            else:
                seroprev_array = np.vstack((seroprev_array, res_seroprev))

            # ethn
            res_ethn = get_fraction_ethnicity(runner)
            if ethn_array is None:
                ethn_array = res_ethn
            else:
                ethn_array = np.vstack((ethn_array, res_ethn))

            # socio_index
            res_socio = get_by_socio_index(runner)
            if socio_array is None:
                socio_array = res_socio
            else:
                socio_array = np.vstack((socio_array, res_socio))

    # deaths
    df = pd.DataFrame(results_array.T, columns=range(len(files)))
    df = df.rolling(window=window, axis=0).mean()

    june_mean = df.mean(1)
    june_std = df.std(1)

    data = get_data(runner, results, window)
    data_std = np.sqrt(data.daily_deaths.values)
    old_fit = read_june_data()
    old_fit.loc[str(data.index[0].date()) : str(data.index[-1].date())]
    fig, ax = plt.subplots()
    ax.plot(data.index, data.daily_deaths, label="Observed Data", color="black")
    if errors:
        ax.fill_between(
            data.index,
            data.daily_deaths - data_std,
            data.daily_deaths + data_std,
            color="black",
            alpha=0.25,
            linewidth=0,
        )
    ax.plot(results["dates"], june_mean, color="C0", label="GradABM-JUNE")
    if errors:
        ax.fill_between(
            results["dates"],
            june_mean - june_std,
            june_mean + june_std,
            alpha=0.25,
            color="C0",
            linewidth=0,
        )
    ax.plot(old_fit.index, old_fit["mean"], label="JUNE", color="C3")
    if errors:
        ax.fill_between(
            old_fit.index,
            old_fit["mean"] - old_fit["std"],
            old_fit["mean"] + old_fit["std"],
            color="C3",
            alpha=0.25,
            linewidth=0,
        )
    ax.legend()
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_major_locator(fmt_month)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(pd.to_datetime("2020-03-01"), pd.to_datetime("2020-05-01"))
    ax.set_ylabel("Daily deaths")
    fig.savefig(figures_path / "fit.png", dpi=150, bbox_inches="tight")

    # seroprev fit
    seroprev_mean = np.mean(seroprev_array, 0)
    seroprev_std = np.std(seroprev_array, 0)
    data_seroprev = get_seroprev_data()
    ages = ("18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+")
    fig, ax = plt.subplots()
    ax.errorbar(ages, seroprev_mean, yerr=seroprev_std, capsize=5, label="GradABM-JUNE")
    yerr_data = np.vstack(
        (
            data_seroprev.middle - data_seroprev.lower,
            data_seroprev.upper - data_seroprev.middle,
        )
    )
    ax.errorbar(
        ages,
        data_seroprev.middle.values,
        yerr=yerr_data,
        capsize=5,
        label="Ward +21",
        color="black",
    )
    ax.legend()
    ax.set_ylabel("Prevalence [%]")
    ax.set_xlabel("Age bin")
    fig.savefig(figures_path / "seroprev_fit.png", dpi=150, bbox_inches="tight")

    # ethnicity fit
    # data_ethn = get_percentage_ethnicity()

    june_mean = np.mean(ethn_array, 0)
    june_std = np.std(ethn_array, 0)

    fig, ax = plt.subplots()
    names = ["White", "Mixed", "Asian", "Black", "Other"]
    # data_ethn = data_ethn.values.flatten() #/ population_by_ethnicity
    ax.bar(
        names,
        june_mean,
        yerr=june_std,
        capsize=5,
        label="GradABM-JUNE",
        color="C0",
        ecolor="black",
    )
    ax.axhline(1, linestyle="--", linewidth=1, color="black")
    ax.set_ylim(0.7, 1.3)
    # ax.bar(names, data_ethn, color = "black", label = "Williamson +20", alpha=0.5)
    ax.set_ylabel("Fraction infected / fraction population")
    ax.set_xlabel("Ethnicity")
    fig.savefig(figures_path / "ethn_fit.png", dpi=150, bbox_inches="tight")

    # socioecon_index

    june_mean = np.mean(socio_array, 0)
    june_std = np.std(socio_array, 0)
    # names = [i + 1 for i in range(5)]
    names = ["1 (most deprived)", "2", "3", "4", "5 (least deprived)"]
    fig, ax = plt.subplots()
    ax.bar(
        names,
        june_mean,
        yerr=june_std,
        capsize=5,
        label="JUNE",
        color="C0",
        ecolor="black",
    )
    ax.axhline(1, linestyle="--", linewidth=1, color="black")
    ax.set_ylabel("Fraction infected / fraction population")
    ax.set_ylim(0.7, 1.3)
    ax.set_xlabel("IMD quintile")
    fig.savefig(figures_path / "socio_fit.png", dpi=150, bbox_inches="tight")

    return


def plot_fit(runner, n_reps=10, window=5):

    results_array = None
    seroprev_array = None
    ethn_array = None
    socio_array = None
    # population_by_ethnicity = get_population_by_ethnicity(runner)
    for i in range(n_reps):
        results, _ = runner()

        # deaths
        res = results["deaths_per_timestep"].cpu().detach().numpy()
        res = np.diff(res, prepend=0)
        if results_array is None:
            results_array = res
        else:
            results_array = np.vstack((results_array, res))

        # seroprev
        res_seroprev = get_seroprev_june(runner, results)
        if seroprev_array is None:
            seroprev_array = res_seroprev
        else:
            seroprev_array = np.vstack((seroprev_array, res_seroprev))

        # ethn
        res_ethn = get_fraction_ethnicity(runner)
        if ethn_array is None:
            ethn_array = res_ethn
        else:
            ethn_array = np.vstack((ethn_array, res_ethn))

        # socio_index
        res_socio = get_by_socio_index(runner)
        if socio_array is None:
            socio_array = res_socio
        else:
            socio_array = np.vstack((socio_array, res_socio))

    # Daily deaths fit.
    df = pd.DataFrame(results_array.T, columns=range(n_reps))
    df = df.rolling(window=window, axis=0).mean()

    june_mean = df.mean(1)
    june_std = df.std(1)

    data = get_data(runner, results, window)
    fig, ax = plt.subplots()
    ax.plot(data.index, data.daily_deaths, label="Observed data", color="black")
    ax.plot(results["dates"], june_mean, color="C0")
    ax.fill_between(
        results["dates"],
        june_mean - june_std,
        june_mean + june_std,
        label="GradJUNE",
        alpha=0.5,
        color="C0",
        linewidth=0,
    )
    ax.legend()
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_major_locator(fmt_month)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(pd.to_datetime("2020-03-01"), pd.to_datetime("2020-05-15"))
    ax.set_ylabel("Daily deaths")
    fig.savefig(figures_path / "fit.png", dpi=150, bbox_inches="tight")

    # seroprev fit
    seroprev_mean = np.mean(seroprev_array, 0)
    seroprev_std = np.std(seroprev_array, 0)
    data_seroprev = get_seroprev_data()
    ages = ("18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+")
    fig, ax = plt.subplots()
    ax.errorbar(ages, seroprev_mean, yerr=seroprev_std, capsize=5, label="JUNE")
    yerr_data = np.vstack(
        (
            data_seroprev.middle - data_seroprev.lower,
            data_seroprev.upper - data_seroprev.middle,
        )
    )
    ax.errorbar(
        ages,
        data_seroprev.middle.values,
        yerr=2 * yerr_data,
        capsize=5,
        label="Ward +21",
        color="black",
    )
    ax.legend()
    ax.set_ylabel("Prevalence [%]")
    ax.set_xlabel("Age bin")
    fig.savefig(figures_path / "seroprev_fit.png", dpi=150, bbox_inches="tight")

    # ethnicity fit
    # data_ethn = get_percentage_ethnicity()

    june_mean = np.mean(ethn_array, 0)
    june_std = np.std(ethn_array, 0)

    fig, ax = plt.subplots()
    names = ["White", "Mixed", "Asian", "Black", "Other"]
    # data_ethn = data_ethn.values.flatten() #/ population_by_ethnicity
    ax.bar(
        names,
        june_mean,
        yerr=june_std,
        capsize=5,
        label="JUNE",
        color="C0",
        ecolor="black",
    )
    ax.axhline(1, linestyle="--", linewidth=1, color="black")
    ax.set_ylim(0.7, 1.3)
    # ax.bar(names, data_ethn, color = "black", label = "Williamson +20", alpha=0.5)
    ax.set_ylabel("Fraction infected / fraction population")
    ax.legend()
    ax.set_xlabel("Ethnicity")
    fig.savefig(figures_path / "ethn_fit.png", dpi=150, bbox_inches="tight")

    # socioecon_index

    june_mean = np.mean(socio_array, 0)
    june_std = np.std(socio_array, 0)
    # names = [i + 1 for i in range(5)]
    names = ["1 (most deprived)", "2", "3", "4", "5 (least deprived)"]
    fig, ax = plt.subplots()
    ax.bar(
        names,
        june_mean,
        yerr=2 * june_std,
        capsize=5,
        label="JUNE",
        color="C0",
        ecolor="black",
    )
    ax.axhline(1, linestyle="--", linewidth=1, color="black")
    ax.set_ylabel("Fraction infected / fraction population")
    ax.set_ylim(0.7, 1.3)
    ax.set_xlabel("IMD quintile")
    fig.savefig(figures_path / "socio_fit.png", dpi=150, bbox_inches="tight")

    return
