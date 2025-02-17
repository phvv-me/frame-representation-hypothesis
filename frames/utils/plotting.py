"""Utility functions for creating and saving statistical plots using Seaborn and Matplotlib.

This module provides wrapper functions around Seaborn's plotting capabilities with
consistent styling and automatic saving of both the plot and the underlying data.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def save_plot_and_show(name: str, df: pd.DataFrame) -> None:
    """Save both the plot and the underlying data, then display the plot.

    Args:
        name: Base filename (without extension) for saving plot and data
        df: DataFrame containing the plotted data

    The function saves:
        - Plot as PNG file in resources/{name}.png
        - Data as pickle file in resources/{name}.pkl
    """
    basepath = Path("resources")
    basepath.mkdir(exist_ok=True)
    df.to_pickle(basepath / f"{name}.pkl")
    plt.savefig(basepath / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def histplot_and_save(
    name: str, df: pd.DataFrame, x: str, hue: str, *args: Any, **kwargs: Any
) -> None:
    """Create a stacked histogram using Seaborn and save the results.

    Args:
        name: Base filename for saving the plot and data
        df: DataFrame containing the data to plot
        x: Column name for x-axis values
        hue: Column name for color grouping
        *args: Additional positional arguments passed to sns.histplot
        **kwargs: Additional keyword arguments passed to sns.histplot
    """
    sns.histplot(
        df, x=x, hue=hue, palette="colorblind", multiple="stack", *args, **kwargs
    )
    save_plot_and_show(name, df)


def lineplot_and_save(
    name: str, df: pd.DataFrame, x: str, y: str, hue: str, *args: Any, **kwargs: Any
) -> None:
    """Create a line plot using Seaborn with consistent styling and save the results.

    Args:
        name: Base filename for saving the plot and data
        df: DataFrame containing the data to plot
        x: Column name for x-axis values
        y: Column name for y-axis values
        hue: Column name for color and style grouping
        *args: Additional positional arguments passed to sns.lineplot
        **kwargs: Additional keyword arguments passed to sns.lineplot
    """
    sns.lineplot(
        df,
        x=x,
        y=y,
        hue=hue,
        style=hue,
        markers=True,
        dashes=False,
        errorbar=("ci", 99),
        palette="colorblind",
        linewidth=2,
        *args,
        **kwargs,
    )
    plt.xlim(0, df.groupby(hue)[x].max().min())
    save_plot_and_show(name, df)
