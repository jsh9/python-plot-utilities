# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl

from . import helper as hlp

#%%============================================================================
def piechart(
        target_array, class_names=None, dropna=False, top_n=None,
        sort_by='counts', fig=None, ax=None, figsize=(3,3),
        dpi=100, colors=None, display='percent', title=None,
        fontsize=None, verbose=True, **piechart_kwargs,
):
    '''
    Plot a pie chart demonstrating proportions of different categories within
    an array.

    Parameters
    ----------
    target_array : array_like
        An array containing categorical values (could have more than two
        categories). Target value can be numeric or texts.
    class_names : sequence of str
        Names of different classes. The order should correspond to that in the
        target_array. For example, if target_array has 0 and 1 then class_names
        should be ['0', '1']; and if target_array has "pos" and "neg", then
        class_names should be ['neg','pos'] (i.e., alphabetical).
        If None, values of the categories will be used as names. If [], then
        no class names are displayed.
    dropna : bool
        Whether to drop NaN values or not. If ``False``, they show up as 'N/A'.
    top_n : int
        An integer between 1 and the number of unique categories in
        ``target_array``. Useful for preventing plotting too many unique
        categories (very slow). If None, plot all categories.
    sort_by : {'counts', 'name'}
        An option to control whether the pie slices are arranged by the counts
        of each unique categories, or by the names of those categories.
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
    colors : list or ``None``
        A list of colors (can be RGB values, hex strings, or color names) to be
        used for each class. The length can be longer or shorter than the number
        of classes. If longer, only the first few colors are used; if shorter,
        colors are wrapped around. If ``None``, automatically use the Pastel2
        color map (8 colors total).
    display : {'percent', 'count', 'both', ``None``}
        An option of what to show on top of each pie slices: percentage of each
        class, or count of each class, or both percentage and count, or nothing.
    title : str or ``None``
        The text to be shown on the top of the pie chart.
    fontsize : scalar or tuple/list of two scalars
        Font size. If scalar, both the class names and the percentages are set
        to the specified size. If tuple of two scalars, the first value sets
        the font size of class names, and the last value sets the font size
        of the percentages.
    verbose : bool
        Whether or to show a "Plotting more than 100 slices; please be patient"
        message when the number of categories exceeds 100.
    **piechart_kwargs :
        Keyword arguments to be passed to matplotlib.pyplot.pie function,
        except for "colors", "labels"and  "autopct" because this subroutine
        re-defines these three arguments.
        (See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pie.html)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    if not isinstance(target_array, hlp._array_like):
        raise TypeError(
            '`target_array` must be a numpy array, pandas.Series, or list.'
        )

    if sort_by not in ['count', 'counts', 'name', 'names']:
        raise ValueError(
            f"`sort_by` must be one of {'count', 'counts', 'name', 'names'}, not '{sort_by}'."
        )

    y = target_array
    if ~isinstance(y, pd.Series):
        y = pd.Series(y)

    y = hlp._upcast_dtype(y)

    if dropna:
        print('****** WARNING: NaNs in target_array dropped. ******')
        y = y[y.notnull()]  # only keep non-null entries
    else:  # need to fill with some str, otherwise the count will be 0
        y.fillna('N/A', inplace=True)

    #----------- Count occurrences --------------------------------------------
    val_count = y.value_counts()  # index: unique values; values: their counts
    if sort_by in ['names', 'name']:
        val_count.sort_index(inplace=True)
    vals = list(val_count.index)
    counts = list(val_count)

    #----------- (Optional) truncation of less common categories --------------
    if top_n is not None:
        if not isinstance(top_n, (int, np.integer)) or top_n <= 0:
            raise ValueError('`top_n` must be a positive integer.')
        if top_n > len(vals):
            raise ValueError(
                '`top_n` larger than the total number of '
                'categories (i.e., %d).' % len(vals)
            )

        occurrences = pd.Series(
            index=vals, data=counts).sort_values(ascending=False)
        truncated = occurrences.iloc[:top_n]  # first top_n entries

        combined_category_name = 'others'
        while combined_category_name in vals:
            combined_category_name += '_'  # must not clash with current category names

        other = pd.Series(
            index=[combined_category_name],  # just one row of data
            data=[occurrences.iloc[top_n:].sum()],
        )
        new_array = truncated.append(other, verify_integrity=True)
        counts = new_array.values
        vals = new_array.index

    thres = 100
    if len(counts) > thres and verbose:
        print('Plotting more than %d slices. Please be very patient.' % thres)

    #---------- Set colors ----------------------------------------------------
    if not colors:  # set default color cycle to 'Pastel2'
        colors_4 = mpl.cm.Pastel2(range(8))  # RGBA values ("8" means Pastel2 has maximum 8 colors)
        colors = [list(_)[:3] for _ in colors_4]  # remove the fourth value

    #---------- Set class names -----------------------------------------------
    if class_names is None:
        class_names = [str(val) for val in vals]
    if class_names == []:
        class_names = None

    #---------- Whether to display percentage or counts (or both) on pie ------
    if display == 'percent':
        autopct = '%1.1f%%'
    elif display == 'count':
        total = np.sum(counts)  # https://stackoverflow.com/a/14171272/8892243
        autopct = lambda p: '{:,d}'.format(int(round(p * total / 100.0)))
    elif display == 'both':
        def make_autopct(values):  # https://stackoverflow.com/a/6170354/8892243
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return '{p:.1f}%  ({v:,d})'.format(p=pct, v=val)
            return my_autopct
        autopct = make_autopct(counts)
    elif display == None:
        autopct = ''
    else:
        raise ValueError(
            "`display` can only be one of {'percent', 'count', "
            "'both', None}, not '%s'." % display
        )

    #------------ Plot pie chart ----------------------------------------------
    _, texts, autotexts = ax.pie(
        counts, labels=class_names, colors=colors,
        autopct=autopct, **piechart_kwargs,
    )
    if isinstance(fontsize, (list, tuple)):
        for t_ in texts: t_.set_fontsize(fontsize[0])
        for t_ in autotexts: t_.set_fontsize(fontsize[1])
    elif fontsize:
        for t_ in texts: t_.set_fontsize(fontsize)
        for t_ in autotexts: t_.set_fontsize(fontsize)

    ax.axis('equal')

    if title: ax.set_title(title)

    return fig, ax

#%%============================================================================
def discrete_histogram(
        x, fig=None, ax=None, figsize=(5,3), dpi=100, color=None,
        alpha=None, rot=0, logy=False, title=None, xlabel=None,
        ylabel='Number of occurrences', show_xticklabel=True,
):
    '''
    Plot a discrete histogram based on the given data ``x``, such as below::


      N ^
        |
        |           ____
        |           |  |   ____
        |           |  |   |  |
        |    ____   |  |   |  |
        |    |  |   |  |   |  |
        |    |  |   |  |   |  |   ____
        |    |  |   |  |   |  |   |  |
        |    |  |   |  |   |  |   |  |
       -|--------------------------------------->  x
              x1     x2     x3     x4    ...

    In the figure, N is the number of occurences for x1, x2, x3, x4, etc.
    And x1, x2, x3, x4, etc. are the discrete values within ``x``.

    Parameters
    ----------
    x : list, numpy.ndarray, pandas.Series, or dict
        Data to be visualized. If ``x`` is a list, numpy arrary, or pandas
        Series, the content of ``x`` is analyzed and counts of ``x``'s values
        are plotted. If ``x`` is a dict, then ``x``'s keys are treated as
        discrete values and ``x``'s values are treated as counts.
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
    color : str, list<float>, or ``None``
        Color of bar. If ``None``, the default color (muted blue) is used.
    alpha : float or ``None``
        Opacity of bar. If ``None``, the default value (1.0) is used.
    rot : float or int
        Rotation angle (degrees) of X axis label. Default = 0 (upright label).
    logy : bool
        Whether or not to use log scale for the Y axis.
    title : str
        The title of the plot.
    xlabel : str
        The X axis label.
    ylabel : str
        The Y axis label.
    show_xticklabel : bool
        Whether or not to show the X tick labels (the names of the classes).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    value_count : pandas.Series
        The counts of each discrete values within ``x`` (if ``x`` is an array)
        with each values sorted in ascending order, or the pandas Series
        generated from ``x`` (if ``x`` is a dict).

    Notes
    -----
    References:

    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.plot.html
    http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html#bar-plots

    See Also
    --------
    plot_ranking :
        Plot bars showing the ranking of the data
    '''
    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    if not isinstance(x, (list, pd.Series, np.ndarray, dict)):
        raise TypeError(
            '`x` should be a list, pandas.Series, numpy.ndarray, or dict.'
        )

    if isinstance(x, dict):
        value_count = pd.Series(x, name='counts').sort_index()
    else:
        X = pd.Series(x)
        value_count = X.value_counts().sort_index()  # count distinct values and sort
        name = 'counts' if value_count.name is None else value_count.name + '_counts'
        value_count.rename(name, inplace=True)

    if color is None:
        ax = value_count.plot.bar(alpha=alpha,ax=ax,rot=rot)
    else:
        ax = value_count.plot.bar(color=color,alpha=alpha,ax=ax,rot=rot)

    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    if show_xticklabel:
        ha = 'center' if (0 <= rot < 30 or rot == 90) else 'right'
        ax.set_xticklabels(value_count.index,rotation=rot,ha=ha)
    else:
        ax.set_xticklabels([])
    if logy:   # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.yscale
        ax.set_yscale('log', nonposy='clip')  # https://stackoverflow.com/a/17952890
    if title: ax.set_title(title)

    return fig, ax, value_count

