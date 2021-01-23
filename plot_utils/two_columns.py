# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pandas as pd
from scipy import stats

from . import misc
from . import helper as hlp
from . import multiple_columns as mc
from . import colors_and_lines as cl

#%%============================================================================
def category_means(
        categorical_array, continuous_array, fig=None, ax=None,
        figsize=None, dpi=100, title=None, xlabel=None, ylabel=None,
        rot=0, dropna=False, show_stats=True, sort_by='name',
        vert=True, plot_violins=True, **extra_kwargs,
):
    '''
    Summarize the mean values of entries of ``continuous_array`` corresponding
    to each distinct category in ``categorical_array``, and show a violin plot
    to visualize it. The violin plot will show the distribution of values in
    ``continuous_array`` corresponding to each category in
    ``categorical_array``.

    Also, a one-way ANOVA test (H0: different categories in ``categorical_array``
    yield the same average values in ``continuous_array``) is performed, and
    F statistics and p-value are returned.

    Parameters
    ----------
    categorical_array : list, numpy.ndarray, or pandas.Series
        An vector of categorical values.
    continuous_array : list, numpy.ndarray, or pandas.Series
        The target variable whose values correspond to the values in x. Must
        have the same length as x. It is natural that y contains continuous
        values, but if y contains categorical values (expressed as integers,
        not strings), this function should also work.
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
    title : str
        The title of the violin plot, usually the name of ``categorical_array`
    xlabel : str
        The label for the x axis (i.e., categories) of the violin plot. If
        ``None`` and ``categorical_array`` is a pandas Series, use the 'name'
        attribute of ``categorical_array`` as xlabel.
    ylabel : str
        The label for the y axis (i.e., average ``continuous_array`` values)
        of the violin plot. If ``None`` and ``continuous_array`` is a pandas
        Series, use the 'name' attribute of ``continuous_array`` as ylabel.
    rot : float
        The rotation (in degrees) of the x axis labels.
    dropna : bool
        Whether or not to exclude N/A records in the data.
    show_stats : bool
        Whether or not to show the statistical test results (F statistics
        and p-value) on the figure.
    sort_by : {'name', 'mean', 'median', None}
        Option to arrange the different categories in `categorical_array` in
        the violin plot. ``None`` means no sorting, i.e., using the hashed
        order of the category names; 'mean' and 'median' mean sorting the
        violins according to the mean/median values of each category; 'name'
        means sorting the violins according to the category names.
    vert : bool
        Whether to show the violins as vertical.
    plot_violins : bool
        If ``True``, use violin plots to illustrate the distribution of groups.
        Otherwise, use multi-histogram (hist_multi()).
    **extra_kwargs :
        Keyword arguments to be passed to plt.violinplot() or hist_multi().
        (https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.violinplot.html)
        Note that this subroutine overrides the default behavior of violinplot:
        showmeans is overriden to True and showextrema to False.

    Return
    ------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    mean_values : dict
        A dictionary whose keys are the categories in x, and their corresponding
        values are the mean values in y.
    F_test_result : tuple<float>
        A tuple in the order of (F_stat, p_value), where F_stat is the computed
        F-value of the one-way ANOVA test, and p_value is the associated
        p-value from the F-distribution.
    '''
    x = categorical_array
    y = continuous_array

    if not isinstance(x, hlp._array_like):
        raise TypeError(
            '`categorical_array` must be pandas.Series, numpy.ndarray, or list.'
        )
    if not isinstance(y, hlp._array_like):
        raise TypeError(
            '`continuous_array` must be pandas.Series, numpy.ndarray, or list.'
        )
    if len(x) != len(y):
        raise hlp.LengthError(
            'Lengths of `categorical_array` and `continuous_array` must be the same.'
        )
    if isinstance(x, np.ndarray) and x.ndim > 1:
        raise hlp.DimensionError('`categorical_array` must be a 1D numpy array.')
    if isinstance(y, np.ndarray) and y.ndim > 1:
        raise hlp.DimensionError('`continuous_array` must be a 1D numpy array..')

    if not xlabel and isinstance(x, pd.Series): #xlabel = x.name
        if vert:
            xlabel = x.name
        else:
            xlabel = y.name
        # END IF-ELSE
    # END IF
    if not ylabel and isinstance(y, pd.Series): #ylabel = y.name
        if vert:
            ylabel = y.name
        else:
            ylabel = x.name
        # END IF-ELSE
    # END IF

    if isinstance(x, (list, np.ndarray)): x = pd.Series(x)
    if isinstance(y, (list, np.ndarray)): y = pd.Series(y)

    x = hlp._upcast_dtype(x)

    if not dropna: x = x.fillna('N/A')  # input arrays are unchanged

    x_classes = x.unique()
    x_classes_copy = list(x_classes.copy())
    y_values = []  # each element in y_values represent the values of a category
    mean_values = {}  # each entry in the dict corresponds to a category
    for cat in x_classes:
        cat_index = (x == cat)
        y_cat = y[cat_index]
        mean_values[cat] = y_cat.mean(skipna=True)
        if not y_cat.isnull().all():
            y_values.append(list(y_cat[np.isfinite(y_cat)]))  # convert to list to avoid 'reshape' deprecation warning
        else:  # all the y values in the current category is NaN
            print('*****WARNING: category %s contains only NaN values.*****' % str(cat))
            x_classes_copy.remove(cat)

    F_stat, p_value = stats.f_oneway(*y_values)  # pass every group into f_oneway()

    if plot_violins:
        if 'showextrema' not in extra_kwargs:
            extra_kwargs['showextrema'] = False  # override default behavior of violinplot
        if 'showmeans' not in extra_kwargs:
            extra_kwargs['showmeans'] = True

    data_names = [str(_) for _ in x_classes_copy]

    if plot_violins:
        fig, ax = mc.violin_plot(
            y_values, fig=fig, ax=ax, figsize=figsize,
            dpi=dpi, data_names=data_names,
            sort_by=sort_by, vert=vert, **extra_kwargs,
        )
    else:
        fig, ax = mc.hist_multi(
            y_values, bins='auto',
            fig=fig, ax=ax, figsize=figsize, dpi=dpi,
            data_names=data_names, sort_by=sort_by, vert=vert,
            show_legend=False, **extra_kwargs,
        )

    if show_stats:
        ha = 'left' if vert else 'right'
        xy = (0.05, 0.92) if vert else (0.95, 0.92)
        ax.annotate(
            'F=%.2f, p_val=%.2g' % (F_stat, p_value), ha=ha,
            xy=xy, xycoords='axes fraction',
        )

    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    return fig, ax, mean_values, (F_stat, p_value)

#%%============================================================================
def positive_rate(
        categorical_array, two_classes_array, fig=None, ax=None,
        figsize=None, dpi=100, barh=True, top_n=None, dropna=False,
        xlabel=None, ylabel=None, show_stats=True,
):
    '''
    Calculate the proportions of the different categories in
    ``categorical_array`` that fall into class "1" (or ``True``) in
    ``two_classes_array``, and optionally show a figure.

    Also, a Pearson's chi-squared test is performed to test the independence
    between ``categorical_array`` and ``two_classes_array``. The chi-squared
    statistics, p-value, and degree-of-freedom are returned.

    Parameters
    ----------
    categorical_array : list, numpy.ndarray, or pandas.Series
        An array of categorical values.
    two_class_array : list, numpy.ndarray, or pandas.Series
        The target variable containing two classes. Each value in this
        parameter correspond to a value in ``categorical_array`` (at the same
        index). It must have the same length as ``categorical_array``. The
        second unique value in this parameter will be considered as the
        positive class (for example, "True" in [True, False, True], or "3" in
        [1, 1, 3, 3, 1]).
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
    barh : bool
        Whether or not to show the bars as horizontal (otherwise, vertical).
    top_n : int
        Only shows ``top_n`` categories (ranked by their positive rate) in the
        figure. Useful when there are too many categories. If ``None``, show
        all categories.
    dropna : bool
        If ``True``, ignore entries (in both arrays) where there are missing
        values in at least one array. If ``False``, the missing values are
        treated as a new category: "N/A".
    xlabel : str
        X axes label.
    ylabel : str
        Y axes label.
    show_stats : bool
        Whether or not to show the statistical test results (chi2 statistics
        and p-value) on the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    pos_rate : pandas.Series
        The positive rate of each categories in x
    chi2_results : tuple<float>
        A tuple in the order of (chi2, p_value, degree_of_freedom)
    '''
    import collections

    x = categorical_array
    y = two_classes_array

    if not isinstance(categorical_array, hlp._array_like):
        raise TypeError(
            '`categorical_array` must be pandas.Series, numpy.ndarray, or list.'
        )
    if not isinstance(two_classes_array, hlp._array_like):
        raise TypeError(
            '`two_classes_array` must be pandas.Series, numpy.array, or list.'
        )
    if len(x) != len(y):
        raise hlp.LengthError(
            'Lengths of `categorical_array` and `two_classes_array` must be the same.'
        )
    if isinstance(x, np.ndarray) and x.ndim > 1:
        raise hlp.DimensionError('`categorical_array` must be a 1D numpy array.')
    if isinstance(y, np.ndarray) and y.ndim > 1:
        raise hlp.DimensionError('`two_classes_array` must be a 1D numpy array.')

    if isinstance(x, (list, np.ndarray)): x = pd.Series(x)
    if isinstance(y, (list, np.ndarray)): y = pd.Series(y)

    x = hlp._upcast_dtype(x)
    y = hlp._upcast_dtype(y)

    if dropna:
        x = x[pd.notnull(x) & pd.notnull(y)]  # input arrays are not changed
        y = y[pd.notnull(x) & pd.notnull(y)]
    else:
        x = x.fillna('N/A')  # input arrays are not changed
        y = y.fillna('N/A')

    if len(np.unique(y)) != 2:
        raise ValueError('`two_classes_array` should have only two unique values.')

    nr_classes = len(x.unique())  # this is not sorted
    y_classes = list(np.unique(y))  # use numpy's unique() to get sorted classes
    y_pos_index = (y == y_classes[1])  # treat the last class as the positive class

    count_all_classes = collections.Counter(x)
    count_pos_class = collections.Counter(x[y_pos_index])

    pos_rate = pd.Series(count_pos_class)/pd.Series(count_all_classes)
    pos_rate = pos_rate.fillna(0.0)  # keys not in count_pos_class show up as NaN

    observed = pd.crosstab(y, x)
    chi2, p_val, dof, expected = stats.chi2_contingency(observed)

    if not figsize:
        if barh:
            figsize = (5, nr_classes * 0.26)  # 0.26 inch = height for each category
        else:
            figsize = (nr_classes * 0.26, 5)

    if xlabel is None and isinstance(x, pd.Series): xlabel = x.name
    if ylabel is None and isinstance(y, pd.Series):
        char = '\n' if (not barh and figsize[1] <= 1.5) else ' '
        ylabel = 'Positive rate%sof "%s"' % (char, y.name)

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)
    fig, ax = misc.plot_ranking(
        pos_rate, fig=fig, ax=ax, top_n=top_n, barh=barh,
        score_ax_label=ylabel, name_ax_label=xlabel,
    )

    if show_stats:
        ax.annotate(
            'chi^2=%.2f, p_val=%.2g' % (chi2, p_val), ha='right',
            xy=(0.99, 1.05), xycoords='axes fraction', va='bottom',
        )

    return fig, ax, pos_rate, (chi2, p_val, dof)

#%%============================================================================
def _crosstab_to_arrays(cross_tab):
    '''
    Helper function. Convert a contingency table to two arrays, which is the
    reversed operation of pandas.crosstab().

    Parameter
    ---------
    cross_tab : numpy.ndarray or pandas.DataFrame
        A contingency table to be converted to two arrays

    Returns
    -------
    x : list<float>
        The first output array
    y : list<float>
        The second output array
    '''
    if isinstance(cross_tab, (list, pd.Series)):
        raise hlp.DimensionError('Please pass a 2D data structure.')
    if isinstance(cross_tab,(np.ndarray,pd.DataFrame)) and min(cross_tab.shape)==1:
        raise hlp.DimensionError('Please pass a 2D data structure.')

    if isinstance(cross_tab, np.ndarray): cross_tab = pd.DataFrame(cross_tab)

    factor_1 = list(cross_tab.columns)
    factor_2 = list(cross_tab.index)

    combinations = itertools.product(factor_1, factor_2)
    result = []
    for fact_1 ,fact_2 in combinations:
        lst = [[fact_2, fact_1]] * cross_tab.loc[fact_2,fact_1]
        result.extend(lst)

    x, y = list(zip(*result))

    return list(x), list(y)  # convert tuple into list

#%%============================================================================
def contingency_table(
        array_horizontal, array_vertical, fig=None, ax=None,
        figsize='auto', dpi=100, color_map='auto', xlabel=None,
        ylabel=None, dropna=False, rot=45, normalize=True,
        symm_cbar=True, show_stats=True,
):
    '''
    Calculate and visualize the contingency table from two categorical arrays.
    Also perform a Pearson's chi-squared test to evaluate whether the two
    arrays are independent.

    Parameters
    ----------
    array_horizontal : list, numpy.ndarray, or pandas.Series
        Array to show as the horizontal margin in the contigency table (i.e.,
        its categories are the column headers).
    array_vertical : list, numpy.ndarray, or pandas.Series
        Array to show as the vertical margin in the contigency table (i.e.,
        its categories are the row names).
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
    color_map : str or matplotlib.colors.Colormap
        The color scheme specifications. Valid names are listed in
        https://matplotlib.org/users/colormaps.html.
        If relative_color is True, use diverging color maps (e.g., PiYG, PRGn,
        BrBG, PuOr, RdGy, RdBu, RdYlBu, RdYlGn, Spectral, coolwarm, bwr,
        seismic). Otherwise, use sequential color maps (e.g., viridis, jet).
    xlabel : str
        The label for the horizontal axis. If ``None`` and ``array_horizontal``
        is a pandas Series, use the 'name' attribute of ``array_horizontal``
        as xlabel.
    ylabel : str
        The label for the vertical axis. If ``None`` and ``array_vertical``
        is a pandas Series, use the 'name' attribute of ``array_vertical`` as
        ylabel.
    dropna : bool
        If ``True``, ignore entries (in both arrays) where there are missing
        values in at least one array. If ``False``, the missing values are
        treated as a new category: "N/A".
    rot : float or 'vertical' or 'horizontal'
        The rotation of the x axis labels (in degrees).
    normalize : bool
        If ``True``, plot the contingency table as the relative difference
        between the observed and the expected (i.e., (obs. - exp.)/exp. ).
        If ``False``, plot the original "observed frequency".
    symm_cbar : bool
        If ``True``, the limits of the color bar are symmetric. Otherwise, the
        limits are the natural minimum/maximum of the table to be plotted.
        It has no effect if "normalize" is set to ``False``.
    show_stats : bool
        Whether or not to show the statistical test results (chi2 statistics
        and p-value) on the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    chi2_results : tuple<float>
        A tuple in the order of (chi2, p_value, degree_of_freedom).
    correlation_metrics : tuple<float>
        A tuple in the order of (phi coef., coeff. of contingency, Cramer's V).
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    x = array_horizontal
    y = array_vertical

    if not isinstance(x, hlp._array_like):
        raise TypeError(
            'The input `array_horizontal` must be pandas.Series, '
            'numpy.ndarray, or list.'
        )
    if not isinstance(y, hlp._array_like):
        raise TypeError(
            'The input `array_vertical` must be pandas.Series, '
            'numpy.array, or list.'
        )
    if len(x) != len(y):
        raise hlp.LengthError(
            'Lengths of `array_horizontal` and `array_vertical` '
            'must be the same.'
        )
    if isinstance(x, np.ndarray) and len(x.shape) > 1:
        raise hlp.DimensionError('`array_horizontal` must be a 1D numpy array.')
    if isinstance(y, np.ndarray) and len(y.shape) > 1:
        raise hlp.DimensionError('`array_vertical` must be a 1D numpy array.')

    if xlabel is None and isinstance(x, pd.Series): xlabel = x.name
    if ylabel is None and isinstance(y, pd.Series): ylabel = y.name

    if isinstance(x, (list, np.ndarray)): x = pd.Series(x)
    if isinstance(y, (list, np.ndarray)): y = pd.Series(y)

    x = hlp._upcast_dtype(x)
    y = hlp._upcast_dtype(y)

    if not dropna:  # keep missing values: replace them with actual string "N/A"
        x = x.fillna('N/A')  # this is to avoid changing the input arrays
        y = y.fillna('N/A')

    observed = pd.crosstab(np.array(y), x)  # use at least one numpy array to avoid possible index matching errors
    chi2, p_val, dof, expected = stats.chi2_contingency(observed)
    expected = pd.DataFrame(
        expected, index=observed.index, columns=observed.columns,
    )
    relative_diff = (observed - expected) / expected

    if figsize == 'auto':
        figsize = observed.shape

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    table = relative_diff if normalize else observed
    peak = max(abs(table.min().min()), abs(table.max().max()))
    max_val = table.max().max()
    min_val = table.min().min()

    if color_map == 'auto':
        color_map = 'RdBu_r' if normalize else 'viridis'

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.08)

    if normalize:
        if symm_cbar:
            if peak <= 1:
                peak = 1.0  # still set color bar limits to [-1.0, 1.0]
            norm = hlp._MidpointNormalize(midpoint=0.0, vmin=-peak, vmax=peak)
        else:  # limits of color bar are the natural minimum/maximum of "table"
            norm = hlp._MidpointNormalize(midpoint=0.0, vmin=min_val, vmax=max_val)
    else:
        norm = None  # no need to set midpoint of color bar

    im = ax.matshow(table, cmap=color_map, norm=norm)
    cb = fig.colorbar(im, cax=cax)  # 'cb' is a Colorbar instance
    if normalize:
        cb.set_label('(Obs$-$Exp)/Exp')
    else:
        cb.set_label('Observed freq.')

    ax.set_xticks(range(table.shape[1]))
    ax.set_yticks(range(table.shape[0]))

    ha = 'center' if (0 <= rot < 30 or rot == 90) else 'left'
    ax.set_xticklabels(table.columns, rotation=rot, ha=ha)
    ax.set_yticklabels(table.index)

    fmt = '.2f' if normalize else 'd'

    if normalize:
        text_color = lambda x: 'white' if abs(x) > peak/2.0 else 'black'
    else:
        lo_3 = min_val + (max_val - min_val)/3.0  # lower-third boundary
        up_3 = max_val - (max_val - min_val)/3.0  # upper-third boundary
        text_color = lambda x: 'k' if x > up_3 else ('y' if x > lo_3 else 'w')

    for i, j in itertools.product(range(table.shape[0]), range(table.shape[1])):
        ax.text(
            j, i, format(table.iloc[i, j], fmt), ha="center", va='center',
            fontsize=9, color=text_color(table.iloc[i, j]),
        )

    if xlabel:
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.yaxis.set_label_position('left')
        ax.set_ylabel(ylabel)

    tables = (observed, expected, relative_diff)
    chi2_results = (chi2, p_val, dof)

    phi = np.sqrt(chi2 / len(x))  # https://en.wikipedia.org/wiki/Phi_coefficient
    cc = np.sqrt(chi2 / (chi2 + len(x)))  # http://www.statisticshowto.com/contingency-coefficient/
    R, C = table.shape
    V = np.sqrt(phi**2. / min(C-1, R-1))  # https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    correlation_metrics = (phi, cc, V)

    if show_stats:
        ax.annotate(
            '$\chi^2$=%.2f, p_val=%.2g\n'
            '$\phi$=%.2g, CoC=%.2g, V=%.2g' % (chi2, p_val, phi, cc, V),
            ha='center', xy=(0.5, -0.09), xycoords='axes fraction',
            va='top',
        )

    return fig, ax, tables, chi2_results, correlation_metrics

#%%============================================================================
def scatter_plot_two_cols(
        X, two_columns, fig=None, ax=None,
        figsize=(3,3), dpi=100, alpha=0.5, color=None,
        grid_on=True, logx=False, logy=False,
):
    '''
    Produce scatter plots of two of the columns in ``X`` (the data matrix).
    The correlation between the two columns are shown on top of the plot.

    Parameters
    ----------
    X : pandas.DataFrame
        The dataset. Currently only supports pandas dataframe.
    two_columns : [str, str] or [int, int]
        The names or indices of the two columns within ``X``. Must be a list of
        length 2. The elements must either be both integers, or both strings.
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
    alpha : float
        Opacity of the scatter points.
    color : str, list<float>, tuple<float>, or ``None``
        Color of the scatter points. If ``None``, default matplotlib color
        palette will be used.
    grid_on : bool
        Whether or not to show grids on the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    if not isinstance(X, pd.DataFrame):
        raise TypeError('`X` must be a pandas DataFrame.')

    if not isinstance(two_columns, list):
        raise TypeError('`two_columns` must be a list of length 2.')
    if len(two_columns) != 2:
        raise hlp.LengthError('Length of `two_columns` must be 2.')

    if isinstance(two_columns[0], str):
        x = X[two_columns[0]]
        xlabel = two_columns[0]
    elif isinstance(two_columns[0], (int, np.integer)):
        x = X.iloc[:, two_columns[0]]
        xlabel = X.columns[two_columns[0]]
    else:
        raise TypeError('`two_columns` must be a list of str or int.')

    if isinstance(two_columns[1], str):
        y = X[two_columns[1]]
        ylabel = two_columns[1]
    elif isinstance(two_columns[1], (int, np.integer)):
        y = X.iloc[:,two_columns[1]]
        ylabel = X.columns[two_columns[1]]
    else:
        raise TypeError('`two_columns` must be a list of str or int.')

    x = np.array(x)  # convert to numpy array so that x[ind] runs correctly
    y = np.array(y)

    try:
        nan_index_in_x = np.where(np.isnan(x))[0]
    except TypeError:
        raise TypeError('Cannot cast the first column safely into numerical types.')
    try:
        nan_index_in_y = np.where(np.isnan(y))[0]
    except TypeError:
        raise TypeError('Cannot cast the second column safely into numerical types.')
    nan_index = set(nan_index_in_x) | set(nan_index_in_y)
    not_nan_index = list(set(range(len(x))) - nan_index)
    _, _, r_value, _, _ = stats.linregress(x[not_nan_index], y[not_nan_index])

    ax.scatter(x, y, alpha=alpha, color=color)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.set_title('$r$ = %.2f' % r_value)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if grid_on == True:
        ax.grid(ls=':', lw=0.5)
        ax.set_axisbelow(True)

    return fig, ax

#%%============================================================================
def bin_and_mean(
        xdata, ydata, bins=10, distribution='normal', show_fig=True,
        fig=None, ax=None, figsize=None, dpi=100, show_bins=True,
        raw_data_label='raw data', mean_data_label='average',
        xlabel=None, ylabel=None, logx=False, logy=False, grid_on=True,
        error_bounds=True, err_bound_type='shade', legend_on=True,
        subsamp_thres=None, show_stats=True, show_SE=False,
        err_bound_shade_opacity=0.5,
):
    '''
    Calculate the "bin-and-mean" results and optionally show the "bin-and-mean"
    plot.

    A "bin-and-mean" plot is a more salient way to show the dependency of
    ``ydata`` on ``xdata``. The data points (``xdata``, ``ydata``) are divided
    into different bins according to the values in ``xdata`` (via ``bins``),
    and within each bin, the mean values of x and y are calculated, and treated
    as the representative x and y values.

    "Bin-and-mean" is preferred when data points are highly skewed (e.g.,
    a lot of data points for when x is small, but very few for large x). The
    data points when x is large are usually not noises, and could be even more
    valuable (think of the case where x is earthquake magnitude and y is the
    related economic loss). If we want to study the relationship between
    economic loss and earthquake magnitude, we need to bin-and-mean raw data
    and draw conclusions from the mean data points.

    The theory that enables this method is the assumption that the data points
    with similar x values follow the same distribution. Naively, we assume the
    data points are normally distributed, then y_mean is the arithmetic mean of
    the data points within a bin. We also often assume the data points follow
    log-normal distribution (if we want to assert that y values are all
    positive), then y_mean is the expected value of the log-normal distribution,
    while x_mean for any bins are still just the arithmetic mean.

    Notes:
      (1) For log-normal distribution, the expective value of y is:
                    E(Y) = exp(mu + (1/2)*sigma^2)
          and the variance is:
                 Var(Y) = [exp(sigma^2) - 1] * exp(2*mu + sigma^2)
          where mu and sigma are the two parameters of the distribution.
      (2) Knowing E(Y) and Var(Y), mu and sigma can be back-calculated::

                              ___________________
             mu = ln[ E(Y) / V 1 + Var(Y)/E^2(Y)  ]

                      _________________________
             sigma = V ln[ 1 + Var(Y)/E^2(Y) ]

          (Reference: https://en.wikipedia.org/wiki/Log-normal_distribution)

    Parameters
    ----------
    xdata : list, numpy.ndarray, or pandas.Series
        X data.
    ydata : list, numpy.ndarray, or pandas.Series
        Y data.
    bins : int, list, numpy.ndarray, or pandas.Series
        Number of bins (an integer), or an array representing the actual bin
        edges. If ``bins`` means bin edges, the edges are inclusive on the
        lower bound, e.g., a value 2 shall fall into the bin [2, 3), but not
        the bin [1, 2). Note that the binning is done according to the X values.
    distribution : {'normal', 'lognormal'}
        Specifies which distribution the Y values within a bin follow. Use
        'lognormal' if you want to assert all positive Y values. Only supports
        normal and log-normal distributions at this time.
    show_fig : bool
        Whether or not to show a bin-and-mean plot.
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
    show_bins : bool
        Whether or not to show the bin edges as vertical lines on the plots.
    raw_data_label : str
        The label name of the raw data to be shown in the legend (such as
        "raw data"). It has no effects if ``show_legend`` is ``False``.
    mean_data_label : str
        The label name of the mean data to be shown in the legend (such as
        "averaged data"). It has no effects if ``show_legend`` is ``False``.
    xlabel : str or ``None``
        X axis label. If ``None`` and ``xdata`` is a pandas Series, use
        ``xdata``'s "name" attribute as ``xlabel``.
    ylabel : str of ``None``
        Y axis label. If ``None`` and ``ydata`` is a pandas Series, use
        ``ydata``'s "name" attribute as ``ylabel``.
    logx : bool
        Whether or not to show the X axis in log scale.
    logy : bool
        Whether or not to show the Y axis in log scale.
    grid_on : bool
        Whether or not to show grids on the plot.
    error_bounds : bool
        Whether or not to show error bounds of each bin.
    err_bound_type : {'shade', 'bar'}
        Type of error bound: shaded area or error bars. It has no effects if
        error_bounds is set to ``False``.
    legend_on : bool
        Whether or not to show a legend.
    subsamp_thres : int
        A positive integer that defines the number of data points in each bin
        to show in the scatter plot. The smaller this number, the faster the
        plotting process. If larger than the number of data points in a bin,
        then all data points from that bin are plotted. If ``None``, then all
        data points from all bins are plotted.
    show_stats : bool
        Whether or not to show R^2 scores, correlation coefficients of the raw
        data and the binned averages on the plot.
    show_SE : bool
        If ``True``, show the standard error of y_mean (orange dots) of each
        bin as the shaded area beneath the mean value lines. If ``False``, show
        the standard deviation of raw Y values (gray dots) within each bin.
    err_bound_shade_opacity : float
        The opacity of the shaded area representing the error bound. 0 means
        completely transparent, and 1 means completely opaque. It has no effect
        if ``error_bound_type`` is ``'bar'``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
        ``None``, if ``show_fig`` is set to ``False``.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
        ``None``, if ``show_fig`` is set to ``False``.
    x_mean : numpy.ndarray
        Mean X values of each data bin (in terms of X values).
    y_mean : numpy.ndarray
        Mean Y values of each data bin (in terms of X values).
    y_std : numpy.ndarray
        Standard deviation of Y values or each data bin (in terms of X values).
    y_SE : numpy.ndarray
        Standard error of ``y_mean``. It describes how far ``y_mean`` is from
        the population mean (or the "true mean value") within each bin, which
        is a different concept from ``y_std``.
        See https://en.wikipedia.org/wiki/Standard_error#Standard_error_of_mean_versus_standard_deviation
        for further information.
    stats_ : tuple<float>
        A tuple in the order of (r2_score_raw, corr_coeff_raw, r2_score_binned,
        corr_coeff_binned), which are the R^2 score and correlation coefficient
        of the raw data (``xdata`` and ``ydata``) and the binned averages
        (``x_mean`` and ``y_mean``).
    '''
    if not isinstance(xdata, hlp._array_like) or not isinstance(ydata, hlp._array_like):
        raise TypeError(
            '`xdata` and `ydata` must be lists, numpy arrays, or pandas Series.'
        )

    if len(xdata) != len(ydata):
        raise hlp.LengthError('`xdata` and `ydata` must have the same length.')

    if isinstance(xdata, list): xdata = np.array(xdata)  # otherwise boolean
    if isinstance(ydata, list): ydata = np.array(ydata)  # indexing won't work

    #------------Pre-process "bins"--------------------------------------------
    if isinstance(bins,(int,np.integer)):  # if user specifies number of bins
        if bins <= 0:
            raise ValueError('`bins` must be a positive integer.')
        else:
            nr = bins + 1  # create bins with percentiles in xdata
            x_uni = np.unique(xdata)
            bins = [np.nanpercentile(x_uni,(j+0.)/bins*100) for j in range(nr)]
            if not all(x <= y for x,y in zip(bins,bins[1:])):  # https://stackoverflow.com/a/4983359/8892243
                print(
                    '\nWARNING: Resulting "bins" array is not monotonically '
                    'increasing. Please use a smaller "bins" to avoid potential '
                    'issues.\n'
                )
    elif isinstance(bins,(list,np.ndarray)):  # if user specifies array
        nr = len(bins)
    else:
        raise TypeError('`bins` must be either an integer or an array.')

    #-----------Pre-process xlabel and ylabel----------------------------------
    if not xlabel and isinstance(xdata, pd.Series):  # xdata has 'name' attr
        xlabel = xdata.name
    if not ylabel and isinstance(ydata, pd.Series):  # ydata has 'name' attr
        ylabel = ydata.name

    #-----------Group data into bins-------------------------------------------
    inds = np.digitize(xdata, bins)
    x_mean = np.zeros(nr-1)
    y_mean = np.zeros(nr-1)
    y_std  = np.zeros(nr-1)
    y_SE   = np.zeros(nr-1)
    x_subs = []  # subsampled x data (for faster scatter plots)
    y_subs = []
    for j in range(nr-1):  # loop over every bin
        x_in_bin = xdata[inds == j+1]
        y_in_bin = ydata[inds == j+1]

        #------------Calculate mean and std------------------------------------
        if len(x_in_bin) == 0:  # no point falls into current bin
            x_mean[j] = np.nan  # this is to prevent numpy from throwing...
            y_mean[j] = np.nan  #...confusing warning messages
            y_std[j]  = np.nan
            y_SE[j] = np.nan
        else:
            x_mean[j] = np.nanmean(x_in_bin)
            if distribution == 'normal':
                y_mean[j] = np.nanmean(y_in_bin)
                y_std[j] = np.nanstd(y_in_bin)
                y_SE[j] = stats.sem(y_in_bin)
            elif distribution == 'lognormal':
                s, loc, scale = stats.lognorm.fit(y_in_bin, floc=0)
                estimated_mu = np.log(scale)
                estimated_sigma = s
                y_mean[j] = np.exp(estimated_mu + estimated_sigma**2.0/2.0)
                y_std[j]  = np.sqrt(
                    np.exp(2.*estimated_mu + estimated_sigma**2.) \
                    * (np.exp(estimated_sigma**2.) - 1)
                )
                y_SE[j] = y_std[j] / np.sqrt(len(y_in_bin))
            else:
                raise ValueError(
                    "Valid values of `distribution` are "
                    "{'normal', 'lognormal'}. Not '%s'." % distribution
                )

        #------------Pick subsets of data, for faster plotting-----------------
        #------------Note that this does not affect mean and std---------------
        if subsamp_thres is not None and show_fig:
            if not isinstance(subsamp_thres, (int, np.integer)) or subsamp_thres <= 0:
                raise TypeError('`subsamp_thres` must be a positive integer or None.')
            if len(x_in_bin) > subsamp_thres:
                x_subs.extend(np.random.choice(x_in_bin,subsamp_thres,replace=False))
                y_subs.extend(np.random.choice(y_in_bin,subsamp_thres,replace=False))
            else:
                x_subs.extend(x_in_bin)
                y_subs.extend(y_in_bin)

    #-------------Calculate R^2 and corr. coeff.-------------------------------
    non_nan_indices = ~np.isnan(xdata) & ~np.isnan(ydata)
    xdata_without_nan = xdata[non_nan_indices]
    ydata_without_nan = ydata[non_nan_indices]

    r2_score_raw = hlp._calc_r2_score(ydata_without_nan, xdata_without_nan)  # treat "xdata" as "y_pred"
    corr_coeff_raw = np.corrcoef(xdata_without_nan, ydata_without_nan)[0, 1]
    r2_score_binned = hlp._calc_r2_score(y_mean, x_mean)
    corr_coeff_binned = np.corrcoef(x_mean, y_mean)[0, 1]
    stats_ = (r2_score_raw, corr_coeff_raw, r2_score_binned, corr_coeff_binned)

    #-------------Plot data on figure------------------------------------------
    if show_fig:
        fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

        if subsamp_thres: xdata, ydata = x_subs, y_subs
        ax.scatter(xdata,ydata,c='gray',alpha=0.3,label=raw_data_label,zorder=1)
        if error_bounds:
            if err_bound_type == 'shade':
                ax.plot(
                    x_mean, y_mean, '-o', c='orange', lw=2,
                    label=mean_data_label, zorder=3,
                )
                if show_SE:
                    ax.fill_between(
                        x_mean, y_mean + y_SE, y_mean - y_SE,
                        label='$\pm$ S.E.', facecolor='orange',
                        alpha=err_bound_shade_opacity, zorder=2.5,
                    )
                else:
                    ax.fill_between(
                        x_mean, y_mean + y_std, y_mean - y_std,
                        label='$\pm$ std', facecolor='orange',
                        alpha=err_bound_shade_opacity, zorder=2.5,
                    )
                # END IF-ELSE
            elif err_bound_type == 'bar':
                if show_SE:
                    mean_data_label += '$\pm$ S.E.'
                    ax.errorbar(
                        x_mean, y_mean, yerr=y_SE, ls='-', marker='o',
                        c='orange', lw=2, elinewidth=1, capsize=2,
                        label=mean_data_label, zorder=3,
                    )
                else:
                    mean_data_label += '$\pm$ std'
                    ax.errorbar(
                        x_mean, y_mean, yerr=y_std, ls='-', marker='o',
                        c='orange', lw=2, elinewidth=1, capsize=2,
                        label=mean_data_label, zorder=3,
                    )
                # END IF-ELSE
            else:
                raise ValueError(
                    'Valid "err_bound_type" name are {"bound", '
                    '"bar"}, not "%s".' % err_bound_type
                )
        else:
            ax.plot(
                x_mean, y_mean, '-o', c='orange', lw=2, label=mean_data_label,
                zorder=3,
            )

        ax.set_axisbelow(True)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        if grid_on:
            ax.grid(ls=':')
            ax.set_axisbelow(True)
        if show_bins:
            ylims = ax.get_ylim()
            for k, edge in enumerate(bins):
                lab_ = 'bin edges' if k==0 else None  # only label 1st edge
                ec = cl.get_colors(N=1)[0]
                ax.plot([edge]*2,ylims,'--',c=ec,lw=1.0,zorder=2,label=lab_)
        if legend_on:
            ax.legend(loc='best')
        if show_stats:
            stats_text = "$R^2_{\mathrm{raw}}$=%.2f, $r_{\mathrm{raw}}$=%.2f, " \
                         "$R^2_{\mathrm{avg}}$=%.2f, " \
                         "$r_{\mathrm{avg}}$=%.2f" % stats_
            ax.set_title(stats_text)

        return fig, ax, x_mean, y_mean, y_std, y_SE, stats_
    else:
        return None, None, x_mean, y_mean, y_std, y_SE, stats_
