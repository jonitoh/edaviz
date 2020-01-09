"""
    Functions to visualize matrices of data.
    It is a custom version of a Heatmap allowing
    cells size's customization.
    It is based on matrix.py in https://github.com/mwaskom/seaborn
    by Michael L. Waskom
    ( commit id: https://github.com/mwaskom/seaborn/pull/1830 )
"""

from __future__ import division
import itertools
import datetime

import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

import seaborn as sns
from seaborn import cm
from seaborn.axisgrid import Grid
from seaborn.utils import (
    despine, axis_ticklabels_overlap, relative_luminance, to_utf8)
from seaborn.external.six import string_types


__all__ = ['afficher_heatmap', 'afficher_hetmap_avec_cellules_variables']


def _index_to_label(index):
    """
        (Unchanged funtions)
        Convert a pandas index or multiindex to an axis label.
    """
    if isinstance(index, pd.MultiIndex):
        return "-".join(map(to_utf8, index.names))
    else:
        return index.name


def _index_to_ticklabels(index):
    """
        (Unchanged funtions)
        Convert a pandas index or multiindex into ticklabels.
    """
    if isinstance(index, pd.MultiIndex):
        return ["-".join(map(to_utf8, i)) for i in index.values]
    else:
        return index.values


def _convert_colors(colors):
    """
        (Unchanged funtions)
        Convert either a list of colors or nested lists of colors to RGB.
    """
    to_rgb = mpl.colors.colorConverter.to_rgb

    if isinstance(colors, pd.DataFrame):
        # Convert dataframe
        return pd.DataFrame({col: colors[col].map(to_rgb)
                             for col in colors})
    elif isinstance(colors, pd.Series):
        return colors.map(to_rgb)
    else:
        try:
            to_rgb(colors[0])
            # If this works, there is only one level of colors
            return list(map(to_rgb, colors))
        except ValueError:
            # If we get here, we have nested lists
            return [list(map(to_rgb, l)) for l in colors]


def _matrix_mask(data, mask):
    """
        (Unchanged funtions)
        Ensure that data and mask are compatible and add missing values.
        Values will be plotted for cells where ``mask`` is ``False``.
        ``data`` is expected to be a DataFrame; ``mask`` can be an array or
        a DataFrame.
    """
    if mask is None:
        mask = np.zeros(data.shape, np.bool)

    if isinstance(mask, np.ndarray):
        # For array masks, ensure that shape matches data then convert
        if mask.shape != data.shape:
            raise ValueError("Mask must have the same shape as data.")

        mask = pd.DataFrame(mask,
                            index=data.index,
                            columns=data.columns,
                            dtype=np.bool)

    elif isinstance(mask, pd.DataFrame):
        # For DataFrame masks, ensure that semantic labels match data
        if not mask.index.equals(data.index) \
           and mask.columns.equals(data.columns):
            err = "Mask must have the same index and columns as data."
            raise ValueError(err)

    # Add any cells with missing data to the mask
    # This works around an issue where `plt.pcolormesh` doesn't represent
    # missing data properly
    mask = mask | pd.isnull(data)

    return mask


def _normalize_cell_size(size, size_min, size_max, size_true, size_false, size_nan):
    """ """
    if isinstance(size, bool):
        return size_true if size else size_false
    elif np.isnan(size):
        return size_nan
    elif size <= size_min:
        return size_min
    elif size >= size_max:
        return size_max
    else:
        return size


class _CustomisedCellHeatMapper(object):
    """Custom version of _HeatMapper adding the control of the cell size."""

    DEFAULT_VMIN_CELLS = .1
    DEFAULT_VMAX_CELLS = 1

    def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
                 annot_kws, cbar, cbar_kws, shape_kws,
                 data_cells, vmin_cells, vmax_cells, robust_cells,
                 xticklabels=True, yticklabels=True, mask=None, normalize_cells=True,
                 square_shaped_cells=True):
        """
            Initialize the plotting object.
        """
        # We always want to have a DataFrame with semantic information
        # and an ndarray to pass to matplotlib
        if isinstance(data, pd.DataFrame):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)
            data = pd.DataFrame(plot_data)

        # We always want to have a DataFrame with semantic information
        # and an ndarray to pass to matplotlib
        if data_cells is None:
            data_cells = pd.DataFrame(data=np.ones(data.shape, dtype=float),
                                      columns=data.columns,
                                      index=data.index)

        if isinstance(data_cells, pd.DataFrame):
            plot_cells = data_cells.values
        else:
            plot_cells = np.asarray(data_cells)
            data_cells = pd.DataFrame(plot_cells)

        # Validate the mask and convert to DataFrame
        mask = _matrix_mask(data, mask)

        plot_data = np.ma.masked_where(np.asarray(mask), plot_data)
        plot_cells = np.ma.masked_where(np.asarray(mask), plot_cells)

        # Get good names for the rows and columns
        xtickevery = 1
        if isinstance(xticklabels, int):
            xtickevery = xticklabels
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is True:
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is False:
            xticklabels = []

        ytickevery = 1
        if isinstance(yticklabels, int):
            ytickevery = yticklabels
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is True:
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is False:
            yticklabels = []

        # Get the positions and used label for the ticks
        nx, ny = data.T.shape

        if not len(xticklabels):
            self.xticks = []
            self.xticklabels = []
        elif isinstance(xticklabels, string_types) and xticklabels == "auto":
            self.xticks = "auto"
            self.xticklabels = _index_to_ticklabels(data.columns)
        else:
            self.xticks, self.xticklabels = self._skip_ticks(xticklabels,
                                                             xtickevery)

        if not len(yticklabels):
            self.yticks = []
            self.yticklabels = []
        elif isinstance(yticklabels, string_types) and yticklabels == "auto":
            self.yticks = "auto"
            self.yticklabels = _index_to_ticklabels(data.index)
        else:
            self.yticks, self.yticklabels = self._skip_ticks(yticklabels,
                                                             ytickevery)

        # Get good names for the axis labels
        xlabel = _index_to_label(data.columns)
        ylabel = _index_to_label(data.index)
        self.xlabel = xlabel if xlabel is not None else ""
        self.ylabel = ylabel if ylabel is not None else ""

        # Determine good default values for the colormapping
        self._determine_cmap_params(plot_data, vmin, vmax,
                                    cmap, center, robust)

        # Determine good default values for the sizemapping
        self._determine_cells_params(plot_cells, vmin_cells,
                                     vmax_cells, robust_cells,
                                     normalize_cells, square_shaped_cells)

        # Sort out the annotations
        if annot is None:
            annot = False
            annot_data = None
        elif isinstance(annot, bool):
            if annot:
                annot_data = plot_data
            else:
                annot_data = None
        else:
            try:
                annot_data = annot.values
            except AttributeError:
                annot_data = annot
            if annot.shape != plot_data.shape:
                raise ValueError('Data supplied to "annot" must be the same '
                                 'shape as the data to plot.')
            annot = True

        # Save other attributes to the object
        self.data = data
        self.plot_data = plot_data

        self.data_cells = data_cells
        self.plot_cells = plot_cells

        self.annot = annot
        self.annot_data = annot_data

        self.fmt = fmt
        self.annot_kws = {} if annot_kws is None else annot_kws
        self.cbar = cbar
        self.cbar_kws = {} if cbar_kws is None else cbar_kws
        self.cbar_kws.setdefault('ticks', mpl.ticker.MaxNLocator(6))
        self.shape_kws = {} if shape_kws is None else shape_kws

    def _determine_cmap_params(self, plot_data, vmin, vmax,
                               cmap, center, robust):
        """Use some heuristics to set good defaults for colorbar and range."""
        calc_data = plot_data.data[~np.isnan(plot_data.data)]
        if vmin is None:
            vmin = np.percentile(calc_data, 2) if robust else calc_data.min()
        if vmax is None:
            vmax = np.percentile(calc_data, 98) if robust else calc_data.max()
        self.vmin, self.vmax = vmin, vmax

        # Choose default colormaps if not provided
        if cmap is None:
            if center is None:
                self.cmap = cm.rocket
            else:
                self.cmap = cm.icefire
        elif isinstance(cmap, string_types):
            self.cmap = mpl.cm.get_cmap(cmap)
        elif isinstance(cmap, list):
            self.cmap = mpl.colors.ListedColormap(cmap)
        else:
            self.cmap = cmap

        # Recenter a divergent colormap
        if center is not None:
            vrange = max(vmax - center, center - vmin)
            normlize = mpl.colors.Normalize(center - vrange, center + vrange)
            cmin, cmax = normlize([vmin, vmax])
            cc = np.linspace(cmin, cmax, 256)
            self.cmap = mpl.colors.ListedColormap(self.cmap(cc))

    def _determine_cells_params(self, plot_cells, vmin_cells, vmax_cells, robust_cells, normalize_cells):
        """Use some heuristics to set good defaults for colorbar and range."""
        # ( NEW )
        if plot_cells is None:
            self.plot_cells = np.ones(plot_cells.shape)
            self.vmax_cells, self.vmin_cells = self.DEFAULT_VMAX_CELLS, self.DEFAULT_VMIN_CELLS
        else:
            # Handle incorrect types (only accepted or np.bool and np.numeric)
            type_cells = plot_cells.applymap(type)
            available_types = set(type_cells.values.flatten())

            invalid_types = [
                ctype for ctype in available_types if not isinstance(ctype, (bool, float))]

            if invalid_types:
                raise TypeError(f"Incorrect types: {invalid_types} ")

            # Format into a unique type with the right imputation
            plot_cells = plot_cells.replace({True: 1.0, False: 0})

            # Normalize the the range of values
            calc_cells = plot_cells.data[~np.isnan(plot_cells.data)]

            if vmin_cells is None:
                vmin_cells = 0
            if vmax_cells is None:
                vmax_cells = 1.0

            robust_vmin_cells = np.percentile(
                calc_cells, 5) if robust else calc_cells.min()
            robust_vmax_cells = np.percentile(
                calc_cells, 95) if robust else calc_cells.max()

            if robust_vmin_cells == 0:
                robust_vmin_cells = self.DEFAULT_VMIN_CELLS

            # Normalize the values
            plot_cells = plot_cells.applymap(_normalize_cell_size,
                                             vmin=robust_vmin_cells,
                                             vmax=robust_vmax_cells,
                                             true_value=robust_vmax_cells,
                                             false_value=robust_vmin_cells,
                                             nan_value=0.0
                                             )

            # Store the values
            self.plot_cells = plot_cells
            self.vmax_cells = robust_vmax_cells
            self. vmin_cells = robust_vmin_cells

    def _annotate_and_size_cells(self, ax, mesh, square_shaped_cells):
        """Add textual labels with the value in each cell."""
        # ( MODIFY: former _annotate_heatmap )
        mesh.update_scalarmappable()
        height, width = self.annot_data.shape
        xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
        for x, y, m, color, val, cell_size in zip(xpos.flat, ypos.flat,
                                                  mesh.get_array(), mesh.get_facecolors(),
                                                  self.annot_data.flat, self.plot_cells.flat):
            if m is not np.ma.masked:
                # vv = (val - self.vmin) / (self.vmax - self.vmin)# done
                # size = np.clip(s / self.cellsize_vmax, 0.1, 1.0)
                shape = None
                if square_shaped_cells:
                    shape = plt.Rectangle((x - cell_size / 2, y - cell_size / 2),
                                          cell_size,
                                          cell_size,
                                          facecolor=color,
                                          **self.shape_kws)
                else:
                    shape = plt.Circle((x - cell_size / 2, y - cell_size / 2),
                                       cell_size,
                                       facecolor=color,
                                       fill=True,
                                       **self.shape_kws)

                ax.add_patch(shape)

                if self.annot:
                    lum = relative_luminance(color)
                    text_color = ".15" if lum > .408 else "w"
                    annotation = ("{:" + self.fmt + "}").format(val)
                    text_kwargs = dict(
                        color=text_color, ha="center", va="center")
                    text_kwargs.update(self.annot_kws)
                    ax.text(x, y, annotation, **text_kwargs)

    def _skip_ticks(self, labels, tickevery):
        """Return ticks and labels at evenly spaced intervals."""
        n = len(labels)
        if tickevery == 0:
            ticks, labels = [], []
        elif tickevery == 1:
            ticks, labels = np.arange(n) + .5, labels
        else:
            start, end, step = 0, n, tickevery
            ticks = np.arange(start, end, step) + .5
            labels = labels[start:end:step]
        return ticks, labels

    def _auto_ticks(self, ax, labels, axis):
        """Determine ticks and ticklabels that minimize overlap."""
        transform = ax.figure.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(transform)
        size = [bbox.width, bbox.height][axis]
        axis = [ax.xaxis, ax.yaxis][axis]
        tick, = axis.set_ticks([0])
        fontsize = tick.label1.get_size()
        max_ticks = int(size // (fontsize / 72))
        if max_ticks < 1:
            return [], []
        tick_every = len(labels) // max_ticks + 1
        tick_every = 1 if tick_every == 0 else tick_every
        ticks, labels = self._skip_ticks(labels, tick_every)
        return ticks, labels

    def _plot_custom_pcolormesh(self, ax, **kwargs):
        """ """
        mesh = ax.pcolormesh(self.plot_data, vmin=self.vmin, vmax=self.vmax,
                             cmap=self.cmap, **kws)
        pass

    def plot(self, ax, cax, kws):
        """Draw the heatmap on the provided Axes."""
        # Remove all the Axes spines
        despine(ax=ax, left=True, bottom=True)

        # Draw the heatmap
        # mesh = self._plot_custom_pcolormesh(ax, **kws)
        mesh = ax.pcolormesh(self.plot_data, vmin=self.vmin, vmax=self.vmax,
                             cmap=self.cmap, **kws)

        # Set the axis limits
        ax.set(xlim=(0, self.data.shape[1]), ylim=(0, self.data.shape[0]))

        # Invert the y axis to show the plot in matrix form
        ax.invert_yaxis()

        # Possibly add a colorbar
        if self.cbar:
            cb = ax.figure.colorbar(mesh, cax, ax, **self.cbar_kws)
            cb.outline.set_linewidth(0)
            # If rasterized is passed to pcolormesh, also rasterize the
            # colorbar to avoid white lines on the PDF rendering
            if kws.get('rasterized', False):
                cb.solids.set_rasterized(True)

        # Add row and column labels
        if isinstance(self.xticks, string_types) and self.xticks == "auto":
            xticks, xticklabels = self._auto_ticks(ax, self.xticklabels, 0)
        else:
            xticks, xticklabels = self.xticks, self.xticklabels

        if isinstance(self.yticks, string_types) and self.yticks == "auto":
            yticks, yticklabels = self._auto_ticks(ax, self.yticklabels, 1)
        else:
            yticks, yticklabels = self.yticks, self.yticklabels

        ax.set(xticks=xticks, yticks=yticks)
        xtl = ax.set_xticklabels(xticklabels)
        ytl = ax.set_yticklabels(yticklabels, rotation="vertical")

        # Possibly rotate them if they overlap
        if hasattr(ax.figure.canvas, "get_renderer"):
            ax.figure.draw(ax.figure.canvas.get_renderer())
        if axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")
        if axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")

        # Add the axis labels
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)

        # Annotate the cells with the formatted values
        self._annotate_and_size_cells(ax, mesh)


def customised_cells_heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
                             annot=None, fmt=".2g", annot_kws=None,
                             cbar=True, cbar_kws=None, cbar_ax=None,
                             data_cells=None, robust_cells=True,
                             vmin_cells=None, vmax_cells=None,
                             square=False, xticklabels="auto", yticklabels="auto",
                             mask=None, ax=None, ax_kws=None, shape_kws=None,
                             normalize_cells=True, square_shaped_cells=True):

    # Initialize the plotter object
    plotter = _CustomisedCellHeatMapper(data, vmin, vmax,
                                        cmap, center, robust,
                                        annot, fmt, annot_kws,
                                        cbar, cbar_kws, shape_kws,
                                        data_cells, vmin_cells, vmax_cells,
                                        robust_cells, xticklabels, yticklabels, mask, normalize_cells,
                                        square_shaped_cells)

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")

    # delete grid
    ax.grid(False)

    plotter.plot(ax, cbar_ax)
    return ax
