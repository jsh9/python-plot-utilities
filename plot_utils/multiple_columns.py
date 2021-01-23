# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.version import LooseVersion

from . import helper as hlp
from . import colors_and_lines as cl

#%%============================================================================
def missing_value_counts(X, fig=None, ax=None, figsize=None, dpi=100, rot=45):
    '''
    Visualize the number of missing values in each column of ``X``.

    Parameters
    ----------
    X : pandas.DataFrame or pandas.Series
        Input data set whose every row is an observation and every column is
        a variable.
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    figsize: (float, float)
        Figure size in inches, as a tuple of two numbers. The figure
        size of ``fig`` (if not ``None``) will override this parameter.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    rot : float
        Rotation (in degrees) of the x axis labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    null_counts : pandas.Series
        A pandas Series whose every element is the number of missing values
        corresponding to each column of ``X``.
    '''
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        raise TypeError('`X` should be pandas DataFrame or Series.')

    if isinstance(X, pd.Series): X = pd.DataFrame(X)

    ncol = X.shape[1]
    null_counts = X.isnull().sum()  # a pd Series containing number of non-null numbers

    if not figsize:
        figsize = (ncol * 0.5, 2.5)

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    ax.bar(range(ncol), null_counts)
    ax.set_xticks(range(ncol))

    ha = 'center' if (0 <= rot < 30 or rot == 90) else 'right'
    ax.set_xticklabels(null_counts.index, rotation=rot, ha=ha)
    plt.ylabel('Number of missing values')
    plt.grid(ls=':')
    ax.set_axisbelow(True)

    alpha = null_counts.max()*0.02  # vertical offset for the texts

    for j, col in enumerate(null_counts.index):
        if null_counts[col] != 0:  # show count of missing values on top of bars
            plt.text(
                j, null_counts[col] + alpha, str(null_counts[col]),
                ha='center', va='bottom', rotation=90,
            )

    return fig, ax, null_counts

#%%============================================================================
def histogram3d(
        X, bins=10, fig=None, ax=None, figsize=(8,4), dpi=100,
        elev=30, azim=5, alpha=0.6, data_labels=None,
        plot_legend=True, plot_xlabel=False, color=None,
        dx_factor=0.4, dy_factor=0.8,
        ylabel='Data', zlabel='Counts',
        **legend_kwargs,
):
    '''
    Plot 3D histograms. 3D histograms are best used to compare the distribution
    of more than one set of data.

    Parameters
    ----------
    X : numpy.ndarray, list<list<float>>, pandas.Series, pandas.DataFrame
        Input data. ``X`` can be:
           (1) a 2D numpy array, where each row is one data set;
           (2) a 1D numpy array, containing only one set of data;
           (3) a list of lists, e.g., [[1,2,3],[2,3,4,5],[2,4]], where each
               element corresponds to a data set (can have different lengths);
           (4) a list of 1D numpy arrays.
               [Note: Robustness is not guaranteed for X being a list of
                      2D numpy arrays.]
           (5) a pandas Series, which is treated as a 1D numpy array;
           (5) a pandas DataFrame, where each column is one data set.
    bins : int, list, numpy.ndarray, or pandas.Series
        Bin specifications. Can be:
           (1) An integer, which indicates number of bins;
           (2) An array or list, which specifies bin edges.
               [Note: If an integer is used, the widths of bars across data
                      sets may be different. Thus array/list is recommended.]
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    figsize: (float, float)
        Figure size in inches, as a tuple of two numbers. The figure
        size of ``fig`` (if not ``None``) will override this parameter.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    elev : float
        Elevation of the 3D view point.
    azim : float
        Azimuth angle of the 3D view point (unit: degree).
    alpha : float
        Opacity of bars
    data_labels : list of str
        Names of different datasets, e.g., ['Simulation', 'Measurement'].
        If not provided, generic names ['Dataset #1', 'Dataset #2', ...]
        are used. The data_labels are only shown when either plot_legend or
        plot_xlabel is ``True``.
        If not provided, and X is a pandas DataFrame/Series, data_labels will
        be overridden by the column names (or name) of ``X``.
    plot_legend : bool
        Whether to show legends or not.
    plot_xlabel : str
        Whether to show data_labels of each data set on their respective x
        axis position or not.
    color : list<list>, or tuple<tuples>
        Colors of each distributions. Needs to be at least the same length as
        the number of data series in ``X``. Can be RGB colors, HEX colors,
        or valid color names in Python. If ``None``,
        get_colors(N=N, color_scheme='tab10') will be queried.
    dx_factor : float
        Width factor of 3D bars in x direction.
    dy_factor : float
        Width factor of 3D bars in y direction. For example, if ``dy_factor``
        is 0.9, there will be a small gap between bars in y direction.
    ylabel : str
        Label of Y axes.
    zlabel : str
        Labels of Z axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.

    Notes
    -----
    x direction :
        Across data sets (i.e., if we have three datasets, the bars will
        occupy three different x values).
    y direction :
        Within dataset.

    Illustration::

                    ^ z
                    |
                    |
                    |
                    |
                    |
                    |--------------------> y
                   /
                  /
                 /
                /
               V  x

    '''
    from mpl_toolkits.mplot3d import Axes3D

    #---------  Data type checking for X  -------------------------------------
    if isinstance(X, np.ndarray):
        if X.ndim <= 1:
            N = 1
            X = [list(X)]  # np.array([1,2,3])-->[[1,2,3]], so that X[0]=[1,2,3]
        elif X.ndim == 2:
            N = X.shape[0]  # number of separate distribution to be compared
            X = list(X)  # turn X into a list of numpy arrays
        else:  # 3D numpy array or above
            raise TypeError('If `X` is a numpy array, it should be a 1D or 2D array.')
    elif isinstance(X, pd.Series):
        data_labels = [X.name]
        X = [list(X)]
        N = 1
    elif isinstance(X, pd.DataFrame):
        N = X.shape[1]
        if data_labels is None:
            data_labels = X.columns  # override data_labels with column names
        X = list(X.values.T)
    elif len(list(X)) > 1:  # adding list() to X to make sure len() does not throw an error
        N = len(X)  # number of separate distribution to be compared
    else:  # X is a scalar
        raise TypeError(
            '`X` must be a list, 2D numpy array, or pandas Series/DataFrame.'
        )

    #------------  NaN checking for X  ----------------------------------------
    for j in range(N):
        if not all(np.isfinite(X[j])):
            raise ValueError(
                f'X[{j}] contains non-finite values (not accepted by `histogram3d()`).'
            )

    if data_labels is None:
        data_labels = [[None]] * N
        for j in range(N):
            data_labels[j] = 'Dataset #%d' % (j+1)  # use generic data set names

    #------------ Prepare figure, axes and colors -----------------------------
    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi, '3d')
    ax.view_init(elev, azim)  # set view elevation and angle

    proxy = [[None]] * N  # create a 'proxy' to help generate legends
    if not color:
        c_ = cl.get_colors(color_scheme='tab10', N=N)  # get a list of colors
    else:
        valid_color_flag, msg = cl._check_color_types(color, N)
        if not valid_color_flag:
            raise TypeError(msg)
        c_ = color

    #------------ Plot one data set at a time ---------------------------------
    xpos_list = [[None]] * N
    for j in range(N):  # loop through each dataset
        if isinstance(bins, (list, np.ndarray)):
            if len(bins) == 0:
                raise ValueError('`bins` must not be empty.')
            else:
                all_bin_widths = np.array(bins[1:]) - np.array(bins[:-1])
                bar_width = np.min(all_bin_widths)
        elif isinstance(bins, (int, np.integer)):  # i.e., number of bins
            if bins <= 0:
                raise ValueError('`bins` must be a positive integer.')
            bar_width = np.ptp(X[j])/float(bins)  # most narrow bin width --> bar_width
        else:
            raise ValueError('`bins` must be an integer, list, or np.ndarray.')

        dz, ypos_ = np.histogram(X[j], bins)  # calculate counts and bin edges
        ypos = np.mean(np.array([ypos_[:-1],ypos_[1:]]), axis=0)  # mid-point of all bins
        xpos = np.ones_like(ypos) * (j-0.5)  # location of each data set
        zpos = np.zeros_like(xpos)  # zpos is where the bars stand
        dx = dx_factor  # width of bars in x direction (across data sets)
        dy = bar_width * dy_factor  # width of bars in y direction (within data set)
        if LooseVersion(mpl.__version__) >= LooseVersion('2.0'):
            bar3d_kwargs = {'alpha':alpha}  # lw clashes with alpha in 2.0+ versions
        else:
            bar3d_kwargs = {'alpha':alpha, 'lw':0.5}
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=c_[j], **bar3d_kwargs)
        proxy[j] = plt.Rectangle((0, 0), 1, 1, fc=c_[j])  # generate proxy for plotting legends
        xpos_list[j] = xpos[0] + dx/2.0  # '+dx/2.0' makes x ticks pass through center of bars

    #-------------- Legends, labels, etc. -------------------------------------
    if plot_legend is True:
        default_kwargs = {
            'loc':9, 'fancybox':True, 'framealpha':0.5, 'ncol':N, 'fontsize':10,
        }
        if legend_kwargs == {}:
            legend_kwargs.update(default_kwargs)
        else:  # if user provides some keyword arguments
            default_kwargs.update(legend_kwargs)
            legend_kwargs = default_kwargs
        ax.legend(proxy, data_labels, **legend_kwargs)

    if plot_xlabel is True:
        ax.set_xticks(xpos_list)
        ax.set_xticklabels(data_labels)
    else:
        ax.set_xticks([])

    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.invert_xaxis()  # make X[0] appear in front, and X[-1] appear at back

    plt.tight_layout(pad=0.3)

    return fig, ax

#%%============================================================================
def correlation_matrix(
        X, color_map='RdBu_r', fig=None, ax=None, figsize=None,
        dpi=100, variable_names=None, rot=45, scatter_plots=False,
):
    '''
    Plot correlation matrix of a dataset ``X``, whose columns are different
    variables (or a sample of a certain random variable).

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        The data set.
    color_map : str or matplotlib.colors.Colormap
        The color scheme to show high, low, negative high correlations. Valid
        names are listed in https://matplotlib.org/users/colormaps.html. Using
        diverging color maps is recommended: PiYG, PRGn, BrBG, PuOr, RdGy,
        RdBu, RdYlBu, RdYlGn, Spectral, coolwarm, bwr, seismic.
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    figsize: (float, float)
        Figure size in inches, as a tuple of two numbers. The figure
        size of ``fig`` (if not ``None``) will override this parameter.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    variable_names : list<str>
        Names of the variables in ``X``. If ``X`` is a pandas DataFrame, this
        argument is not needed: column names of ``X`` is automatically used as
        variable names. If ``X`` is a numpy array, and this argument is not
        provided, then ``X``'s column indices are used. The length of
        ``variable_names`` should match the number of columns in ``X``; if
        not, a warning will be thrown (not error).
    rot : float
        The rotation of the x axis labels, in degrees.
    scatter_plots : bool
        Whether or not to show the scatter plots of pairs of variables.

    Returns
    -------
    correlations : pandas.DataFrame
        The correlation matrix.
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise TypeError('`X` must be a numpy array or a pandas DataFrame.')

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, copy=True)

    correlations = X.corr()
    variable_list = list(correlations.columns)
    nr = len(variable_list)

    if not figsize:
        figsize = (0.7 * nr, 0.7 * nr)  # every column of X takes 0.7 inches

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    im = ax.matshow(correlations, vmin=-1, vmax=1, cmap=color_map)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)  # 'cb' is a Colorbar instance
    cb.set_label("Pearson's correlation")

    ticks = np.arange(0,correlations.shape[1],1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if variable_names is None:
        variable_names = variable_list

    if len(variable_names) != len(variable_list):
        print('*****  Warning: feature_names may not be valid!  *****')

    ha = 'center' if (0 <= rot < 30 or rot == 90) else 'left'
    ax.set_xticklabels(variable_names, rotation=rot, ha=ha)
    ax.set_yticklabels(variable_names)

    if scatter_plots:
        pd.plotting.scatter_matrix(X, figsize=(1.8 * nr, 1.8 * nr))

    return fig, ax, correlations

#%%============================================================================
def violin_plot(
        X, fig=None, ax=None, figsize=None, dpi=100, nan_warning=False,
        showmeans=True, showextrema=False, showmedians=False, vert=True,
        data_names=[], rot=45, name_ax_label=None, data_ax_label=None,
        sort_by=None, title=None, **violinplot_kwargs,
):
    '''
    Generate violin plots for each data set within ``X``.

    Parameters
    ----------
    X : pandas.DataFrame, pandas.Series, numpy.ndarray, or dict
        The data to be visualized. It can be of the following types:

        - pandas.DataFrame:
            + Each column contains a set of data
        - pandas.Series:
            + Contains only one set of data
        - numpy.ndarray:
            + 1D numpy array: only one set of data
            + 2D numpy array: each column contains a set of data
            + Higher dimensional numpy array: not allowed
        - dict:
            + Each key-value pair is one set of data
        - list of lists:
            + Each sub-list is a data set

        Note that the NaN values in the data are implicitly excluded.

    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    figsize: (float, float)
        Figure size in inches, as a tuple of two numbers. The figure
        size of ``fig`` (if not ``None``) will override this parameter.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    nan_warning : bool
        Whether to show a warning if there are NaN values in the data.
    showmeans : bool
        Whether to show the mean values of each data group.
    showextrema : bool
        Whether to show the extrema of each data group.
    showmedians : bool
        Whether to show the median values of each data group.
    vert : bool
        Whether to show the violins as vertical.
    data_names : list<str>, ``[]``, or ``None``
        The names of each data set, to be shown as the axis tick label of each
        data set. If ``[]`` or ``None``, it will be determined automatically.
        If ``X`` is a:
            - numpy.ndarray:
                + data_names = ['data_0', 'data_1', 'data_2', ...]
            - pandas.Series:
                + data_names = X.name
            - pd.DataFrame:
                + data_names = list(X.columns)
            - dict:
                + data_names = list(X.keys())
    rot : float
        The rotation (in degrees) of the data_names when shown as the tick
        labels. If vert is False, rot has no effect.
    name_ax_label : str
        The label of the "name axis". ("Name axis" is the axis along which
        different violins are presented.)
    data_ax_label : str
        The labels of the "data axis". ("Data axis" is the axis along which
        the data values are presented.)
    sort_by : {'name', 'mean', 'median', ``None``}
        Option to sort the different data groups in ``X`` in the violin plot.
        ``None`` means no sorting, keeping the violin plot order as provided;
        'mean' and 'median' mean sorting the violins according to the
        mean/median values of each data group; 'name' means sorting the violins
        according to the names of the groups.
    title : str
        The title of the plot.
    **violinplot_kwargs : dict
        Other keyword arguments to be passed to ``matplotlib.pyplot.violinplot()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    _check_violin_plot_or_hist_multi_input(X, data_names, nan_warning)

    data, data_names, n_datasets = _preprocess_violin_plot_data(
        X, data_names=data_names, nan_warning=nan_warning,
    )

    data_with_names = _prepare_violin_plot_data(
        data, data_names, sort_by=sort_by, vert=vert,
    )

    fig, ax = _violin_plot_helper(
        data_with_names, fig=fig, ax=ax,
        figsize=figsize, dpi=dpi, showmeans=showmeans,
        showmedians=showmedians, vert=vert, rot=rot,
        data_ax_label=data_ax_label,
        name_ax_label=name_ax_label,
        title=title, **violinplot_kwargs,
    )

    return fig, ax

#%%============================================================================
def _check_violin_plot_or_hist_multi_input(X, data_names, nan_warning):
    '''
    Check that the input, `X`, for violin_plot() or hist_multi() is valid.
    '''
    if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray, dict, list)):
        raise TypeError(
            '`X` must be pandas.DataFrame, pandas.Series, np.ndarray, dict, or list.'
        )
    if not isinstance(data_names, (list, type(None))):
        raise TypeError('`data_names` must be a list of names, empty list, or None.')
    if nan_warning and isinstance(X, (pd.DataFrame, pd.Series)) and X.isnull().any().any():
        print('WARNING in violin_plot(): X contains NaN values.')
    if nan_warning and isinstance(X, np.ndarray) and np.isnan(X).any():
        print('WARNING in violin_plot(): X contains NaN values.')
    if isinstance(X, list) and not all([isinstance(_, list) for _ in X]):
        raise TypeError('If `X` is a list, it must be a list of lists.')

#%%============================================================================
def _preprocess_violin_plot_data(X, data_names=None, nan_warning=False):
    '''
    Helper function. Preprocess raw data (``X``) for violin plot or
    multi-histogram plot.
    '''
    if isinstance(X, pd.Series):
        n_datasets = 1
        data = X.dropna().values
    elif isinstance(X, pd.DataFrame):
        n_datasets = X.shape[1]
        data = []
        for j in range(n_datasets):
            data.append(X.iloc[:,j].dropna().values)
    elif isinstance(X, np.ndarray):  # use columns
        if X.ndim == 1:  # 1D numpy array
            n_datasets = 1
            data = X[np.isfinite(X)].copy()
        elif X.ndim == 2:  # 2D numpy array
            n_datasets = X.shape[1]
            data = []
            for j in range(n_datasets):  # go through every column
                x = X[:,j]
                data.append(x[np.isfinite(x)])  # remove NaN values
        else:
            raise hlp.DimensionError('`X` should be a 1D or 2D numpy array.')
    elif isinstance(X, list):  # list of lists
        data = X.copy()
        n_datasets = len(data)
    else:  # dict --> extract its values
        n_datasets = len(X)
        data = []
        key_list = []
        for key in X:
            x = X[key]
            key_list.append(key)
            if isinstance(x, pd.Series):
                x_ = x.values
            elif isinstance(x, np.ndarray) and x.ndim == 1:
                x_ = x.copy()
            elif isinstance(x, list):
                x_ = np.array(x)
            else:
                raise TypeError(
                    'Unknown data type in X["%s"]. Should be either '
                    'pandas.Series, 1D numpy array, or a list.' % key
                )
            if nan_warning and np.isnan(x_).any():
                print(
                    'WARNING in violin_plot() or hist_multi(): '
                    'X[%s] contains NaN values.' % key
                )
            data.append(x_[np.isfinite(x_)])

    if not data_names and isinstance(X, dict):
        data_names = key_list

    assert(len(data) == n_datasets)
    if len(data_names) != 0 and len(data_names) != n_datasets:
        raise hlp.LengthError('Length of `data_names` must equal the number of datasets.')

    if not data_names:  # [] or None
        if isinstance(X, pd.Series):
            data_names = [X.name]
        elif isinstance(X, pd.DataFrame):
            data_names = list(X.columns)
        elif isinstance(X, dict):
            data_names = list(X.keys())
        else:  # numpy array or list of lists
            data_names = ['data_' + str(_) for _ in range(n_datasets)]

    return data, data_names, n_datasets

#%%============================================================================
def _prepare_violin_plot_data(data, data_names, sort_by=None, vert=False):
    '''
    Package ``data`` and ``data_names`` into a dictionary with the specified
    sorting option.

    Parameters
    ----------
    data : list<list>
        All the data. Each element of ``data`` is an array of data points.
    data_names : list<str>
        The names of the data. It should have the same length as ``data``.
    sort_by : [None, 'name', 'mean', 'median']
        The method by which to sort the data sets.  If ``None``, then use the
        original order of ``data`` (i.e., left to right if ``vert`` is ``True``,
        top to bottom if ``vert`` is ``False``).
    vert : bool
        Whether to show the histograms as vertical.

    Returns
    -------
    data_with_names_dict : OrderedDict<str, list>
        A mapping from data names to data, ordered by the specification in
        ``sort_by``.
    '''
    from collections import OrderedDict

    assert(len(data) == len(data_names))
    n = len(data)

    data_with_names = []
    for j in range(n):
        data_with_names.append((data_names[j], data[j]))

    reverse = not vert

    if not sort_by:
        if not reverse:
            sorted_list = data_with_names.copy()
        else:  # for "not vert" histograms, we want the first data set on top
            sorted_list = data_with_names[::-1]
    elif sort_by == 'name':
        sorted_list = sorted(
            data_with_names, key=lambda x: x[0], reverse=reverse,
        )
    elif sort_by == 'mean':
        sorted_list = sorted(
            data_with_names, key=lambda x: np.mean(x[1]), reverse=reverse,
        )
    elif sort_by == 'median':
        sorted_list = sorted(
            data_with_names, key=lambda x: np.median(x[1]), reverse=reverse,
        )
    else:
        raise NameError(
            "`sort_by` must be one of {`None`, 'name', 'mean', "
            "'median'}, not '%s'." % sort_by
        )

    data_with_names_dict = OrderedDict()
    for j in range(n):
        data_with_names_dict[sorted_list[j][0]] = sorted_list[j][1]

    return data_with_names_dict

#%%============================================================================
def _violin_plot_helper(
        data_with_names, fig=None, ax=None, figsize=None,
        dpi=100, showmeans=True, showextrema=False,
        showmedians=False, vert=False, rot=45,
        data_ax_label=None, name_ax_label=None, title=None,
        **violinplot_kwargs,
):
    '''
    Helper function for violin plot.

    Parameters
    ----------
    data_with_names : OrderedDict<str, list>
        A dictionary whose keys are the names of the categories and values are
        the actual data.
    '''
    data = []
    data_names = []
    for key, val in data_with_names.items():
        data.append(val)
        data_names.append(key)

    n_datasets = len(data)

    if not figsize:
        l1 = max(3, 0.5 * n_datasets)
        l2 = 3.5
        figsize = (l1, l2) if vert else (l2, l1)

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)
    ax.violinplot(
        data, vert=vert, showmeans=showmeans, showextrema=showextrema,
        showmedians=showmedians, **violinplot_kwargs,
    )
    ax = hlp.__axes_styling_helper(
        ax, vert, rot, data_names, n_datasets,
        data_ax_label, name_ax_label, title,
    )
    return fig, ax

#%%============================================================================
def hist_multi(
        X, bins=10, fig=None, ax=None, figsize=None, dpi=100,
        nan_warning=False, showmeans=True, showmedians=False, vert=True,
        data_names=[], rot=45, name_ax_label=None, data_ax_label=None,
        sort_by=None, title=None, show_vals=True, show_pct_diff=False,
        baseline_data_index=0, legend_loc='best',
        show_counts_on_data_ax=True, **extra_kwargs,
):
    '''
    Generate multiple histograms, one for each data set within ``X``.

    Parameters
    ----------
    X : pandas.DataFrame, pandas.Series, numpy.ndarray, or dict
        The data to be visualized. It can be of the following types:

        - pandas.DataFrame:
            + Each column contains a set of data
        - pandas.Series:
            + Contains only one set of data
        - numpy.ndarray:
            + 1D numpy array: only one set of data
            + 2D numpy array: each column contains a set of data
            + Higher dimensional numpy array: not allowed
        - dict:
            + Each key-value pair is one set of data
        - list of lists:
            + Each sub-list is a data set

        Note that the NaN values in the data are implicitly excluded.

    bins : int or sequence or str
        If an integer is given, the whole range of data (i.e., all the numbers
        within ``X``) is divided into ``bins`` segments. If sequence or str,
        they will be passed to the ``bins`` argument of ``matplotlib.pyplot.hist()``.
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    figsize: (float, float)
        Figure size in inches, as a tuple of two numbers. The figure
        size of ``fig`` (if not ``None``) will override this parameter.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    nan_warning : bool
        Whether to show a warning if there are NaN values in the data.
    showmeans : bool
        Whether to show the mean values of each data group.
    showmedians : bool
        Whether to show the median values of each data group.
    vert : bool
        Whether to show the "base" of the histograms as vertical.
    data_names : list<str>, ``[]``, or ``None``
        The names of each data set, to be shown as the axis tick label of each
        data set. If ``[]`` or ``None``, it will be determined automatically.
        If ``X`` is a:
            - numpy.ndarray:
                + data_names = ['data_0', 'data_1', 'data_2', ...]
            - pandas.Series:
                + data_names = X.name
            - pd.DataFrame:
                + data_names = list(X.columns)
            - dict:
                + data_names = list(X.keys())
    rot : float
        The rotation (in degrees) of the data_names when shown as the tick
        labels. If vert is False, rot has no effect.
    name_ax_label : str
        The label of the "name axis". ("Name axis" is the axis along which
        different violins are presented.)
    data_ax_label : str
        The labels of the "data axis". ("Data axis" is the axis along which
        the data values are presented.)
    sort_by : {'name', 'mean', 'median', ``None``}
        Option to sort the different data groups in ``X`` in the violin plot.
        ``None`` means no sorting, keeping the violin plot order as provided;
        'mean' and 'median' mean sorting the violins according to the
        mean/median values of each data group; 'name' means sorting the violins
        according to the names of the groups.
    title : str
        The title of the plot.
    show_vals : bool
        Whether to show mean and/or median values along the mean/median bars.
        Only effective if ``showmeans`` and/or ``showmedians`` are turned on.
    show_pct_diff : bool
        Whether to show percent difference of mean and/or median values
        between different data sets. Only effective when ``show_vals`` is
        set to ``True``.
    baseline_data_index : int
        Which data set is considered the "baseline" when showing percent
        differences.
    legend_loc : str
        The location specification for the legend.
    show_counts_on_data_ax : bool
        Whether to show counts besides the histograms.
    **extra_kwargs : dict
        Other keyword arguments to be passed to ``matplotlib.pyplot.bar()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    _check_violin_plot_or_hist_multi_input(X, data_names, nan_warning)

    data, data_names, n_datasets = _preprocess_violin_plot_data(
        X, data_names=data_names, nan_warning=nan_warning,
    )

    data_with_names = _prepare_violin_plot_data(
        data, data_names, sort_by=sort_by, vert=vert,
    )

    if isinstance(bins, int):
        flattened_data = []
        for data_i in data:
            flattened_data.extend(data_i)
        all_X_max = np.max(flattened_data)
        all_X_min = np.min(flattened_data)
        bins = np.linspace(all_X_min, all_X_max, num=bins, endpoint=True)

    fig, ax = _hist_multi_helper(
        data_with_names, bins=bins, fig=fig, ax=ax,
        figsize=figsize, dpi=dpi, showmeans=showmeans,
        showmedians=showmedians, vert=vert, rot=rot,
        data_ax_label=data_ax_label,
        name_ax_label=name_ax_label,
        title=title, show_vals=show_vals,
        show_pct_diff=show_pct_diff,
        baseline_data_index=baseline_data_index,
        legend_loc=legend_loc,
        show_counts_on_data_ax=show_counts_on_data_ax,
        **extra_kwargs,
    )

    return fig, ax

#%%============================================================================
def _hist_multi_helper(
        data_with_names, bins=10, fig=None, ax=None,
        figsize=None, dpi=100, showmeans=True, showmedians=False,
        vert=False, rot=45, data_ax_label=None,
        name_ax_label=None, show_legend=True, title=None,
        show_vals=True, show_pct_diff=False,
        baseline_data_index=0, legend_loc='best',
        show_counts_on_data_ax=True,
        **extra_kwargs,
):
    '''
    Helper function to multi_hist().

    Parameters
    ----------
    data_with_names : OrderedDict<str, list>
        A dictionary whose keys are the names of the categories and values are
        the actual data.
    (Other parameters are the same as multi_hist().)

    Returns
    -------
    Same sa multi_hist()
    '''
    data = []
    data_names = []
    for key, val in data_with_names.items():
        data.append(val)
        data_names.append(key)

    n_datasets = len(data)

    if not figsize:
        l1 = max(3, 1.0 * n_datasets)
        l2 = 3.5
        figsize = (l1, l2) if vert else (l2, l1)

    MAX_RELATIVE_BAR_HEIGHT = 0.8  # limit tallest bar height to 90%

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    mean_vals = []
    median_vals = []
    max_count_each_dataset = []
    for i, data_i in enumerate(data):
        freq_bar_heights, bin_edges = np.histogram(data_i, bins=bins)
        max_bar_height = max(freq_bar_heights)
        bar_full_width = bin_edges[1] - bin_edges[0]
        bar_half_width = bar_full_width / 2.0

        max_count_each_dataset.append(freq_bar_heights.max())

        bin_centers = bin_edges[:-1] + bar_half_width
        bar_heights = freq_bar_heights / max_bar_height * MAX_RELATIVE_BAR_HEIGHT
        extra_kwarg = {'bottom': i + 1} if not vert else {'left': i + 1}

        plot_bar_func = ax.bar if not vert else ax.barh  # flipped compared to violin plot!
        plot_bar_func(
            bin_centers, bar_heights,
            bar_full_width * 0.9,  # leave some space between bars
            align='center', alpha=0.75, lw=0.5, ec='w', **extra_kwarg,
        )

        mbl = 0.8  # mean/median bar length
        if showmeans:
            mean_val = np.mean(data_i)
            mean_vals.append(mean_val)
            label_1 = 'mean' if i == 0 else None
            if vert:
                ax.plot(
                    [i+1, i+1+mbl], [mean_val] * 2, c='k', label=label_1,
                    alpha=0.6,
                )
            else:
                ax.plot(
                    [mean_val] * 2, [i+1, i+1+mbl], c='k', label=label_1,
                    alpha=0.6,
                )
        if showmedians:
            median_val = np.median(data_i)
            median_vals.append(median_val)
            label_2 = 'median' if i == 0 else None
            if vert:
                ax.plot(
                    [i+1, i+1+mbl], [median_val] * 2, c='k', ls='--',
                    alpha=0.6, label=label_2,
                )
            else:
                ax.plot(
                    [median_val] * 2, [i+1, i+1+mbl], c='k', ls='--',
                    alpha=0.6, label=label_2,
                )

    #~~~~~~~~~~ Print values of mean and/or median ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if show_vals and (len(mean_vals) > 0 or len(median_vals) > 0):
        if len(mean_vals) == 0:
            mean_vals = [None] * n_datasets
        # END
        if len(median_vals) == 0:
            median_vals = [None] * n_datasets
        # END

        bdi = baseline_data_index
        if not isinstance(bdi, int):
            raise TypeError('`baseline_data_index` should be an int.')
        # END
        if bdi > n_datasets - 1:
            raise ValueError(f'`baseline_data_index` not in [0, {n_datasets}).')
        # END

        def _annotate(value, ax, i, base_val, vert=True, below=True):
            if i != bdi:
                if base_val != 0:
                    pct_diff = (value - base_val) / abs(base_val) * 100
                else:
                    pct_diff = None
                # END
            else:  # this value is the base value; no need to calculate pct_diff
                pct_diff = None
            # END

            if not show_pct_diff or pct_diff is None:
                fmt = '%.3g' if abs(value) < 10 else '%.2f'
                txt = fmt % value
            else:
                fmt1 = '%.3g' if abs(value) < 10 else '%.2f'
                fmt2 = '%.1f'
                sign = '+' if pct_diff > 0 else ''
                txt = f'{fmt1} ({sign}{fmt2}%%)' % (value, pct_diff)
            # END

            if vert:
                y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
                gap = y_span / 50
                x_position = i + 1.5
                y_position = value - gap if below else value + gap
                ha = 'center'
                va = 'top' if below else 'bottom'
            else:
                x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
                gap = x_span / 50
                x_position = value - gap if below else value + gap
                y_position = i + 1.5
                ha = 'right' if below else 'left'
                va = 'center'
            # END

            ax.annotate(
                txt, xy=(x_position, y_position), xycoords='data',
                ha=ha, va=va,
            )
            return

        for i in range(n_datasets):
            mean_val = mean_vals[i]
            median_val = median_vals[i]
            if median_val is None:
                _annotate(mean_val, ax, i, mean_vals[bdi], vert=vert, below=False)
            elif mean_val is None:
                _annotate(median_val, ax, i, median_vals[bdi], vert=vert, below=False)
            elif mean_val > median_val:
                _annotate(mean_val, ax, i, mean_vals[bdi], vert=vert, below=False)
                _annotate(median_val, ax, i, median_vals[bdi], vert=vert, below=True)
            elif mean_val < median_val:
                _annotate(mean_val, ax, i, mean_vals[bdi], vert=vert, below=True)
                _annotate(median_val, ax, i, median_vals[bdi], vert=vert, below=False)
            else:  # mean val = median val
                _annotate(mean_val, ax, i, mean_vals[bdi], vert=vert, below=True)
            # END IF
        # END FOR
    # END IF
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if show_legend:
        ax.legend(loc=legend_loc)
    ax = hlp.__axes_styling_helper(
        ax, vert, rot, data_names, n_datasets,
        data_ax_label, name_ax_label, title,
    )

    if show_counts_on_data_ax:

        def get_ticks_and_labels(
            n_datasets_, max_count_each_dataset, max_relative_bar_height,
        ):
            ticks = []
            tick_labels = []
            for i in range(n_datasets_):
                ticks.extend([1 + i, 1 + i + max_relative_bar_height])
                tick_labels.extend([0, max_count_each_dataset[i]])
            # END
            ticks.append(1 + n_datasets_)
            return ticks, tick_labels

        if not vert:
            ax2 = ax.twinx()
            ax2.set_ylabel('Counts')
            ticks, tick_labels = get_ticks_and_labels(
                n_datasets, max_count_each_dataset, MAX_RELATIVE_BAR_HEIGHT,
            )
            ax2.set_yticks(ticks)
            ax2.set_yticklabels(tick_labels)
            ax.set_ylim(1, n_datasets + 1)
            ax2.set_ylim(1, n_datasets + 1)
        else:
            ax2 = ax.twiny()
            ax2.set_xlabel('Counts')
            ticks, tick_labels = get_ticks_and_labels(
                n_datasets, max_count_each_dataset, MAX_RELATIVE_BAR_HEIGHT,
            )
            ax2.set_xticks(ticks)
            ax2.set_xticklabels(tick_labels, rotation=45)
            ax.set_xlim(1, n_datasets + 1)
            ax2.set_xlim(1, n_datasets + 1)

    return fig, ax
