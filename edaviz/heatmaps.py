"""
    Functions to visualize matrices of data.
    It is a custom version of a Heatmap allowing
    cells size's customization.
    It is based on matrix.py in https://github.com/mwaskom/seaborn
    ( commit 065d3c1 ) by Michael L. Waskom .
"""

from __future__ import division
#import itertools
import functools
#import datetime
import numbers
import operator

import matplotlib as mpl
#from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
#from matplotlib import gridspec
#import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
#from scipy.cluster import hierarchy

import seaborn as sns
from seaborn import cm
#from seaborn.axisgrid import Grid
from seaborn.utils import (
    despine, axis_ticklabels_overlap, relative_luminance, to_utf8)
from seaborn.external.six import string_types

__all__  = [ 'custom_cells_heatmap' ]


def _replace_bool(value):
    """NEW-- Custom function to replace boolean values"""
    if not isinstance(value, bool):
        return value
    return 1.0 if value else 0.0


def _index_to_label(index):
    """Convert a pandas index or multiindex to an axis label."""
    if isinstance(index, pd.MultiIndex):
        return "-".join(map(to_utf8, index.names))
    else:
        return index.name


def _index_to_ticklabels(index):
    """Convert a pandas index or multiindex into ticklabels."""
    if isinstance(index, pd.MultiIndex):
        return ["-".join(map(to_utf8, i)) for i in index.values]
    else:
        return index.values


def _convert_colors(colors):
    """Convert either a list of colors or nested lists of colors to RGB."""
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
    """NEW--"""
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


class _CustomCellHeatMapper(object):
    """NEW--Custom version of _HeatMapper adding the control of the cell size."""

    DEFAULT_VMIN_CELLS = .1
    DEFAULT_VMAX_CELLS = 1

    def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
                 annot_kws, cbar, cbar_kws, shape_kws,
                 data_cells, vmin_cells, vmax_cells, robust_cells, robust_type='percentile',
                 xticklabels=True, yticklabels=True, mask=None, normalize_cells=True):
        """
            Initialize the plotting object.
        """
        # NEW-- PLOT_DATA
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

        
        # NEW-- PLOT_CELLS
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

        # Determine good default values for the sizemapping according to a feature
        self._determine_cells_params(plot_cells, vmin_cells,
                                     vmax_cells, robust_cells,
                                     robust_type, normalize_cells)

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

        # NEW-- Save other attributes to the object
        self.data_cells = data_cells
        self.plot_cells = plot_cells

        self.annot = annot
        self.annot_data = annot_data

        self.fmt = fmt
        self.annot_kws = {} if annot_kws is None else annot_kws
        self.cbar = cbar
        self.cbar_kws = {} if cbar_kws is None else cbar_kws
        self.cbar_kws.setdefault('ticks', mpl.ticker.MaxNLocator(6))
        # NEW-- 
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

    def _determine_cells_params(self, plot_cells, vmin_cells, vmax_cells, robust_cells, robust_type, normalize_cells):
        """ NEW-- Use some heuristics to set good defaults for cells' size."""
        # Handle unknown robust methods
        if robust_type not in [ 'percentile', 'boundary' ] and robust_cells:
            raise ValueError(f"Incorrect robust_type: {robust_type} instead of 'percentile' or 'boundary.")

        # Handle incorrect types (only accepted or np.bool and np.numeric)
        type_cells = [ list(map(type,l)) for l in plot_cells ]
        available_types = functools.reduce(operator.xor, map(set, type_cells))
        invalid_types = [ _type for _type in available_types if not issubclass(_type, (bool, numbers.Number)) ]

        if invalid_types:
            raise TypeError(f"Incorrect types: {invalid_types}.")

        calc_cells = plot_cells.data[~(np.isnan(plot_cells.data))]

        # Compute vmin_cells and vmax_cells according the method
        robust_vmax_cells, robust_vmin_cells = None, None 
        if robust_cells:
            if robust_type == 'percentile':
                robust_vmax_cells = np.percentile(calc_cells, 5)
                robust_vmin_cells = np.percentile(calc_cells, 95)
            
            if robust_type == 'boundary':
                robust_vmax_cells = calc_cells.min()
                robust_vmin_cells = calc_cells.max()
        
        self.vmax_cells = robust_vmax_cells or self.DEFAULT_VMAX_CELLS
        self.vmin_cells = robust_vmin_cells or self.DEFAULT_VMIN_CELLS

        # Normalize the values and format into a unique type with the right imputation
        normalize = lambda x: _normalize_cell_size(
                                         size=x,
                                         size_min=self.vmin_cells,
                                         size_max=self.vmax_cells,
                                         size_true=self.vmax_cells,
                                         size_false=self.vmin_cells,
                                         size_nan=0.0
                                        )                             

        plot_cells = np.ma.masked_array(data=[ list(map(normalize, l)) for l in plot_cells.data ],
                                        mask=plot_cells.mask)

        # Store the values
        self.plot_cells = plot_cells

    def _annotate_and_size_cells(self, ax, mesh, square_shaped_cells):
        """Add textual labels with the value in each cell."""
        # ( MODIFY: former _annotate_heatmap )
        mesh.update_scalarmappable()
        annot_data = self.annot_data or np.full(self.plot_data.shape, np.nan)
        height, width = self.plot_data.shape
        xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
        for x, y, m, color, annotation, cell_size in zip(xpos.flat, ypos.flat,
                                                  mesh.get_array(), mesh.get_facecolors(),
                                                  annot_data.flat, self.plot_cells.flat):
            if m is not np.ma.masked:
                size = np.clip ( cell_size / self.vmax_cells, 0.1, 1.0)
                shape = None
                if square_shaped_cells:
                    shape = plt.Rectangle((x - size / 2, y - size / 2),
                                          size,
                                          size,
                                          facecolor=color,
                                          **self.shape_kws)
                else:
                    shape = plt.Circle((x - size / 2, y - size / 2),
                                       size,
                                       facecolor=color,
                                       fill=True,
                                       **self.shape_kws)

                ax.add_patch(shape)

                if self.annot and not np.isnan(annotation):
                    lum = relative_luminance(color)
                    text_color = ".15" if lum > .408 else "w"
                    annotation = ("{:" + self.fmt + "}").format(annotation)
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
                             cmap=self.cmap, **kwargs)
        pass

    def plot(self, ax, cax, square_shaped_cells, kws=None):
        """Draw the heatmap on the provided Axes."""
        # Remove all the Axes spines
        despine(ax=ax, left=True, bottom=True)

        # Draw the heatmap
        kws = {} if kws is None else kws
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
        self._annotate_and_size_cells(ax, mesh, square_shaped_cells)


def custom_cells_heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
                        annot=None, fmt=".2g", annot_kws=None,
                        linewidths=0, linecolor="white",
                        cbar=True, cbar_kws=None, cbar_ax=None,
                        square=False, xticklabels="auto", yticklabels="auto",
                        mask=None, ax=None, data_cells=None, robust_cells=True,
                        vmin_cells=None, vmax_cells=None, ax_kws=None,
                        shape_kws=None, normalize_cells=True,
                        square_shaped_cells=True**kwargs):
                        callable(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
                 annot_kws, cbar, cbar_kws, shape_kws,
                 data_cells, vmin_cells, vmax_cells, robust_cells, robust_type='percentile',
                 xticklabels=True, yticklabels=True, mask=None, normalize_cells=True)
    """Plot rectangular data as a color-encoded matrix.

    This is an Axes-level function and will draw the heatmap into the
    currently-active Axes if none is provided to the ``ax`` argument.  Part of
    this Axes space will be taken and used to plot a colormap, unless ``cbar``
    is False or a separate Axes is provided to ``cbar_ax``.

    Parameters
    ----------
    data : rectangular dataset
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame
        is provided, the index/column information will be used to label the
        columns and rows.
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments.
    cmap : matplotlib colormap name or object, or list of colors, optional
        The mapping from data values to color space. If not provided, the
        default will depend on whether ``center`` is set.
    center : float, optional
        The value at which to center the colormap when plotting divergant data.
        Using this parameter will change the default ``cmap`` if none is
        specified.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with robust quantiles instead of the extreme values.
    annot : bool or rectangular dataset, optional
        If True, write the data value in each cell. If an array-like with the
        same shape as ``data``, then use this to annotate the heatmap instead
        of the raw data.
    fmt : string, optional
        String formatting code to use when adding annotations.
    annot_kws : dict of key, value mappings, optional
        Keyword arguments for ``ax.text`` when ``annot`` is True.
    linewidths : float, optional
        Width of the lines that will divide each cell.
    linecolor : color, optional
        Color of the lines that will divide each cell.
    cbar : boolean, optional
        Whether to draw a colorbar.
    cbar_kws : dict of key, value mappings, optional
        Keyword arguments for `fig.colorbar`.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar, otherwise take space from the
        main Axes.
    square : boolean, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped.
    xticklabels, yticklabels : "auto", bool, list-like, or int, optional
        If True, plot the column names of the dataframe. If False, don't plot
        the column names. If list-like, plot these alternate labels as the
        xticklabels. If an integer, use the column names but plot only every
        n label. If "auto", try to densely plot non-overlapping labels.
    mask : boolean array or DataFrame, optional
        If passed, data will not be shown in cells where ``mask`` is True.
        Cells with missing values are automatically masked.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.
    kwargs : other keyword arguments
        All other keyword arguments are passed to ``ax.pcolormesh``.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap.

    See also
    --------
    clustermap : Plot a matrix using hierachical clustering to arrange the
                 rows and columns.

    Examples
    --------

    Plot a heatmap for a numpy array:

    .. plot::
        :context: close-figs

        >>> import numpy as np; np.random.seed(0)
        >>> import seaborn as sns; sns.set()
        >>> uniform_data = np.random.rand(10, 12)
        >>> ax = sns.heatmap(uniform_data)

    Change the limits of the colormap:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(uniform_data, vmin=0, vmax=1)

    Plot a heatmap for data centered on 0 with a diverging colormap:

    .. plot::
        :context: close-figs

        >>> normal_data = np.random.randn(10, 12)
        >>> ax = sns.heatmap(normal_data, center=0)

    Plot a dataframe with meaningful row and column labels:

    .. plot::
        :context: close-figs

        >>> flights = sns.load_dataset("flights")
        >>> flights = flights.pivot("month", "year", "passengers")
        >>> ax = sns.heatmap(flights)

    Annotate each cell with the numeric value using integer formatting:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, annot=True, fmt="d")

    Add lines between each cell:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, linewidths=.5)

    Use a different colormap:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, cmap="YlGnBu")

    Center the colormap at a specific value:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, center=flights.loc["January", 1955])

    Plot every other column label and don't plot row labels:

    .. plot::
        :context: close-figs

        >>> data = np.random.randn(50, 20)
        >>> ax = sns.heatmap(data, xticklabels=2, yticklabels=False)

    Don't draw a colorbar:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, cbar=False)

    Use different axes for the colorbar:

    .. plot::
        :context: close-figs

        >>> grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
        >>> f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
        >>> ax = sns.heatmap(flights, ax=ax,
        ...                  cbar_ax=cbar_ax,
        ...                  cbar_kws={"orientation": "horizontal"})

    Use a mask to plot only part of a matrix

    .. plot::
        :context: close-figs

        >>> corr = np.corrcoef(np.random.randn(10, 200))
        >>> mask = np.zeros_like(corr)
        >>> mask[np.triu_indices_from(mask)] = True
        >>> with sns.axes_style("white"):
        ...     ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)


    """
    # Initialize the plotter object
    plotter = _HeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
                          annot_kws, cbar, cbar_kws, xticklabels,
                          yticklabels, mask)

    # Add the pcolormesh kwargs here
    kwargs["linewidths"] = linewidths
    kwargs["edgecolor"] = linecolor

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")
    plotter.plot(ax, cbar_ax, kwargs)
    return ax


def custom_cells_heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
                             annot=None, fmt=".2g", annot_kws=None,
                             cbar=True, cbar_kws=None, cbar_ax=None,
                             data_cells=None, robust_cells=True,
                             vmin_cells=None, vmax_cells=None,
                             square=False, xticklabels="auto", yticklabels="auto",
                             mask=None, ax=None, ax_kws=None, shape_kws=None,
                             normalize_cells=True, square_shaped_cells=True):

    # Initialize the plotter object
    plotter = _CustomCellHeatMapper(data, vmin, vmax,
                                    cmap, center, robust,
                                    annot, fmt, annot_kws,
                                    cbar, cbar_kws, shape_kws,
                                    data_cells, vmin_cells, vmax_cells,
                                    robust_cells, xticklabels, yticklabels, mask, normalize_cells
                                    )

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")

    # delete grid
    ax.grid(False)

    plotter.plot(ax, cbar_ax, square_shaped_cells, ax_kws)

    return ax
