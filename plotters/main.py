from . import plot_fit

from grad_june import Runner

def plot():
    runner = Runner.from_file("./best_fit.yaml")

    plot_fit(runner)
