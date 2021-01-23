# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from . import helper as hlp

#%%============================================================================
def plot_ranking(
        ranking, fig=None, ax=None, figsize='auto', dpi=100,
        barh=True, top_n=None, score_ax_label=None, name_ax_label=None,
        invert_name_ax=False, grid_on=True,
):
    '''
    Plot rankings as a bar plot (in descending order), such as::

                ^
                |
        dolphin |||||||||||||||||||||||||||||||
                |
        cat     |||||||||||||||||||||||||
                |
        rabbit  ||||||||||||||||
                |
        dog     |||||||||||||
                |
               -|------------------------------------>  Age of pet
                0  1  2  3  4  5  6  7  8  9  10  11

    Parameters
    ----------
    ranking : dict or pandas.Series
        The ranking information, for example:
            {'rabbit': 5, 'cat': 8, 'dog': 4, 'dolphin': 10}
        It does not need to be sorted externally.
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
        Whether or not to show the bars as horizontal (otherwise, vertical)
    top_n : int
        If ``None``, show all categories. ``top_n`` > 0 means showing the
        highest ``top_n`` categories. ``top_n`` < 0 means showing the lowest
        |``top_n``| categories.
    score_ax_label : str
        Label of the score axis (e.g., "Age of pet").
    name_ax_label : str
        Label of the "category name" axis (e.g., "Pet name").
    invert_name_ax : bool
        Whether to invert the "category name" axis. For example, if
        ``invert_name_ax`` is ``False``, then higher values are shown on the
        top if ``barh`` is ``True``.
    grid_on : bool
        Whether or not to show grids on the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    if not isinstance(ranking, (dict, pd.Series)):
        raise TypeError('`ranking` must be a Python dict or pandas Series.')

    if top_n is not None and not isinstance(top_n, (int, np.integer)):
        raise ValueError('`top_n` must be an integer of None.')

    if top_n == None:
        nr_classes = len(ranking)
        top_n = len(ranking)
    else:
        nr_classes = np.abs(top_n)

    if figsize == 'auto':
        if barh:
            figsize = (5, nr_classes * 0.26)  # 0.26 inch = height for each category
        else:
            figsize = (nr_classes * 0.26, 5)

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    if isinstance(ranking,dict):
        ranking = pd.Series(ranking)

    if barh:
        kind = 'barh'
        xlabel, ylabel = score_ax_label, name_ax_label
        ax = ranking.sort_values(
            ascending=(top_n >= 0)
        ).iloc[-np.abs(top_n):].plot(kind=kind, ax=ax)
    else:
        kind = 'bar'
        xlabel, ylabel = name_ax_label, score_ax_label
        ax = ranking.sort_values(
            ascending=(top_n < 0)
        ).iloc[:np.abs(top_n) if top_n != 0 else None].plot(
            kind=kind, ax=ax,
        )

    if invert_name_ax:
        if barh is True:
            ax.invert_yaxis()
        else:
            ax.invert_xaxis()
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if grid_on:
        ax.grid(ls=':')
        ax.set_axisbelow(True)

    return fig, ax

#%%============================================================================
def plot_with_error_bounds(
        x, y, upper_bound, lower_bound,
        fig=None, ax=None, figsize=None, dpi=100,
        line_color=[0.4]*3, shade_color=[0.7]*3,
        shade_alpha=0.5, linewidth=2.0, legend_loc='best',
        line_label='Data', shade_label='$\mathregular{\pm}$STD',
        logx=False, logy=False, grid_on=True,
):
    '''
    Plot a graph with one line and its upper and lower bounds, with areas between
    bounds shaded. The effect is similar to this illustration below::

      y ^            ...                         _____________________
        |         ...   ..........              |                     |
        |         .   ______     .              |  ---  Mean value    |
        |      ...   /      \    ..             |  ...  Error bounds  |
        |   ...  ___/        \    ...           |_____________________|
        |  .    /    ...      \    ........
        | .  __/   ...  ....   \________  .
        |  /    ....       ...          \  .
        | /  ....            .....       \_
        | ...                    ..........
       -|--------------------------------------->  x


    Parameters
    ----------
    x : list, numpy.ndarray, or pandas.Series
        X data points to be plotted as a line.
    y : list, numpy.ndarray, or pandas.Series
        Y data points to be plotted as a line.
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
    upper_bound : list, numpy.ndarray, or pandas.Series
        Upper bound of the Y values.
    lower_bound : list, numpy.ndarray, or pandas.Series
        Lower bound of the Y values.
    line_color : str, list, or tuple
        Color of the line.
    shade_color : str, list, or tuple
        Color of the underlying shades.
    shade_alpha : float
        Opacity of the shades.
    linewidth : float
        Width of the line.
    legend_loc : int, str
        Location of the legend, to be passed directly to ``plt.legend()``.
    line_label : str
        Label of the line, to be used in the legend.
    shade_label : str
        Label of the shades, to be used in the legend.
    logx : bool
        Whether or not to show the X axis in log scale.
    logy : bool
        Whether or not to show the Y axis in log scale.
    grid_on : bool
        Whether or not to show grids on the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    if not isinstance(x, hlp._array_like) or not isinstance(y, hlp._array_like):
        raise TypeError('`x` and `y` must be arrays.')

    if len(x) != len(y):
        raise hlp.LengthError('`x` and `y` must have the same length.')

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    hl1 = ax.fill_between(
        x, lower_bound, upper_bound,
        color=shade_color, facecolor=shade_color,
        linewidth=0.01, alpha=shade_alpha, interpolate=True,
        label=shade_label,
    )
    hl2, = ax.plot(x, y, color=line_color, linewidth=linewidth, label=line_label)
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')

    if grid_on:
        ax.grid(ls=':',lw=0.5)
        ax.set_axisbelow(True)

    plt.legend(handles=[hl2,hl1],loc=legend_loc)

    return fig, ax

#%%============================================================================
def visualize_cv_scores(
        fig=None, ax=None, dpi=100, n_folds=5,
        cv_scores=None, box_height=0.6, box_width=0.9,
        gap_frac=0.05, metric_name='AUC', avg_cv_score=None,
        no_holdout_set=False, holdout_score=None, fontsize=9,
        flip_yaxis=True,
):
    '''
    Visualize K-fold cross-validation scores as well as hold-out set performance
    in an intuitive way.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    n_folds : int
        Number of CV folds.
    cv_scores : list<float> or ``None``
        The validation score of each fold. If ``None``, no scores will be shown
        on the small boxes.
    box_height : float
        The height of the the small box, in inches.
    box_width : float
        The width of the small box, in inches.
    gap_frac : float
        How much gap should there be between each small box.
    metric_name : str
        The name of the metric to be shown in the figure.
    avg_cv_score : float or ``None``
        The average cross-validation score. If ``None`` (recommended), it will
        be calculated by numpy.mean(cv_scores).
    no_holdout_set : bool
        If ``False``, the hold-out data set will be visualized alongside the
        training data set. This parameter supersedes ``holdout_score``.
    holdout_score : float or ``None``
        The performance on the hold-out data set. If ``no_holdout_set`` is
        ``True``, this parameter has no effect.
    fontsize : float
        The font size of all the texts.
    flip_yaxis : bool
        If ``True``, everything will be flipped upside down. This parameter is
        for diagnosis and and debugging purpose only. It is recommended to leave
        it as ``True``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    hlp.assert_type(n_folds, int, name='n_folds')
    hlp.assert_type(cv_scores, (type(None), list), name='cv_scores')
    if cv_scores is not None:
        hlp.assert_element_type(cv_scores, hlp._scalar_like, name='cv_scores')
    # END IF
    hlp.assert_type(avg_cv_score, (type(None), hlp._scalar_like), name='avg_cv_score')
    hlp.assert_type(gap_frac, hlp._scalar_like, name='gap_frac')
    if gap_frac < 0 or gap_frac > 1:
        raise ValueError('`gap_frac` must be within (0, 1).')
    # END IF
    hlp.assert_type(metric_name, str, name='metric_name')
    hlp.assert_type(holdout_score, (type(None), hlp._scalar_like), name='holdout_score')

    GRAY_COLOR_ALPHA = 0.25
    OTHER_COLOR_ALPHA = 0.5

    total_width = n_folds * box_width
    total_height = n_folds * box_height
    fig_width = total_width * 1.5
    fig_height = total_height * 1.5

    if metric_name is None:
        metric_name = 'score'
    # END IF

    if cv_scores is not None:
        assert(len(cv_scores) == n_folds)
    # END IF
    figsize = (fig_width, fig_height)
    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)
    for j in range(n_folds):
        text_ = [''] * n_folds
        if cv_scores is not None:
            if j == 0:
                text_[j] = '%s\n= %.4g' % (metric_name, cv_scores[j])
            else:
                text_[j] = '%.4g' % cv_scores[j]
        else:
            for k in range(n_folds):
                if k == j:
                    text_[k] = 'eval.'
                else:
                    text_[k] = 'train'
                # END IF-ELS
            # END FOR
        # END IF-ELSE
        ax = _plot_one_row_of_rectangles(
            ax, n_boxes=n_folds,
            southwest_corner=(0, j * box_height),
            box_height=box_height, fontsize=fontsize,
            box_width=box_width, gap_frac=gap_frac,
            show_which_box_as_test=j, text=text_,
        )
    # END FOR

    text_list = ['Fold %d' % (_ + 1) for _ in range(n_folds)]
    ax = _plot_one_row_of_rectangles(
        ax, n_boxes=n_folds,
        southwest_corner=(0, n_folds * box_height),
        box_height=box_height, box_width=box_width,
        gap_frac=gap_frac, fontsize=fontsize,
        show_which_box_as_test=-1,
        train_set_color='gray', alpha=GRAY_COLOR_ALPHA,
        text=text_list,
    )

    ax = _plot_one_row_of_rectangles(
        ax, n_boxes=1,
        southwest_corner=(0, -1.5 * box_height),
        box_height=box_height,
        box_width=total_width,
        gap_frac=0.0, text=['Training data'],
        fontsize=fontsize,
        train_set_color='#6baed6',
        alpha=OTHER_COLOR_ALPHA,
    )

    if not no_holdout_set:
        if holdout_score is not None:
            holdout_txt = 'Hold-out data\n%s = %.4g' % (metric_name, holdout_score)
        else:
            holdout_txt = 'Hold-out data'
        # END IF-ELSE
        holdout_box_gap_frac = 0.01
        r1 = 1 + holdout_box_gap_frac
        holdout_box_width = total_width * 0.5
        ax = _plot_one_row_of_rectangles(
            ax, n_boxes=1,
            southwest_corner=(total_width * r1, -1.5 * box_height),
            box_width=holdout_box_width,
            box_height=box_height,
            gap_frac=0.0, text=[holdout_txt],
            fontsize=fontsize,
            train_set_color='yellow',
            alpha=OTHER_COLOR_ALPHA,
        )
        ax = _plot_one_row_of_rectangles(
            ax, n_boxes=1,
            southwest_corner=(0, -2.7 * box_height),
            box_height=box_height,
            box_width=total_width * r1 + holdout_box_width,
            gap_frac=0.0, text=['All data'],
            fontsize=fontsize,
            train_set_color='gray',
            alpha=GRAY_COLOR_ALPHA,
        )
    # END IF

    if avg_cv_score is not None or cv_scores is not None:
        avg_cv_score = np.mean(cv_scores) if avg_cv_score is None else avg_cv_score
        char = '\n' if n_folds <= 4 else ' '  # too few folds: display text in two lines
        avg_score_txt = 'Mean %s%s= %.4g' % (metric_name, char, avg_cv_score)
    else:
        avg_score_txt = 'Take average'
        # END IF-ELSE
    # END IF-ELSE
    _plot_bracket(
        ax, n_folds, total_width, total_height, avg_score_txt, fontsize=fontsize,
    )

    if flip_yaxis:
        ax.invert_yaxis()
    # END IF
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')

    return fig, ax

#------------------------------------------------------------------------------
def _plot_one_row_of_rectangles(
        ax, n_boxes=5, southwest_corner=(0, 0),
        box_height=0.6, box_width=0.9, gap_frac=0.05,
        show_which_box_as_test=-1,
        train_set_color='green', test_set_color='orange',
        alpha=0.3, text=None, fontsize=None,
):
    '''
    Plot one row of rectangles (small boxes).

    Parameters
    ----------
    ax :
        Figure axes object.
    n_boxes : int
        Number of boxes to plot on this row.
    southwest_corner : (float, float)
        A tuple of two floats. The south-west corner coordinate of the box.
    box_height : float
        Height of a small box.
    box_width : float
        Width of a small box.
    gap_frac : float
        How much gap should there be between each small box.
    show_which_box_as_test : int
        The 0-based index of one of the ``n_boxes`` boxes to show as the
        "test" box. If -1, treat all boxes as the "train" boxes.
    train_set_color : str or tuple<float>
        The color of the "train" boxes. Can be a color name or rgb.
    test_set_color : str or tuple<float>
        The color of the "test" boxes. Can be a color name or rgb.
    alpha : float
        Opacity of the box color.
    text : list<str> or ``None``
        The text to show on each box. If ``None``, do not show text.
    fontsize : float
        The font size of the texts.

    Returns
    -------
    ax :
        Figure axes object.
    '''
    patches_train = []
    patches_test = []
    for i in range(n_boxes):
        x0, y0 = southwest_corner
        x1, y1, width, height = __add_gap_to_coord(
            x0 + i * box_width, y0, box_width, box_height, gap_frac=gap_frac,
        )
        rect = Rectangle((x1, y1), width, height)
        if i == show_which_box_as_test:
            patches_test.append(rect)
        else:
            patches_train.append(rect)
    # END IF
    box_edge_width = 0.7
    pc_train = PatchCollection(
        patches_train, edgecolor='k', lw=box_edge_width,
        facecolor=train_set_color, alpha=alpha,
    )
    pc_test = PatchCollection(
        patches_test, edgecolor='k', lw=box_edge_width,
        facecolor=test_set_color, alpha=alpha,
    )
    ax.add_collection(pc_train)
    ax.add_collection(pc_test)
    if text is not None:
        __add_text(
            ax, text, n_boxes=n_boxes, southwest_corner=southwest_corner,
            box_height=box_height, box_width=box_width, fontsize=fontsize,
        )
    # END IF
    return ax

#------------------------------------------------------------------------------
def __add_gap_to_coord(x0, y0, width, height, gap_frac=0.05):
    x1 = x0 + width * gap_frac / 2.0
    y1 = y0 + height * gap_frac / 2.0
    new_width = width * (1 - gap_frac)
    new_height = height * (1 - gap_frac)
    return x1, y1, new_width, new_height

#------------------------------------------------------------------------------
def __add_text(
        ax, text, n_boxes=5, southwest_corner=(0, 0),
        box_height=0.6, box_width=0.9, fontsize=10,
    ):
    assert(len(text) == n_boxes)
    x_mid, y_mid = ___get_mid_points(
        n_boxes=n_boxes, southwest_corner=southwest_corner,
        box_height=box_height, box_width=box_width,
    )
    for i in range(n_boxes):
        ax.text(x_mid[i], y_mid[i], text[i], ha='center', va='center', fontsize=fontsize)
    # END FOR
    return ax

#------------------------------------------------------------------------------
def ___get_mid_points(n_boxes=5, southwest_corner=(0, 0), box_height=0.6, box_width=0.9):
    x_mid = []
    y_mid = []
    x0, y0 = southwest_corner
    for i in range(n_boxes):
        x_mid.append(x0 + box_width / 2.0 + i * box_width)
        y_mid.append(y0 + box_height / 2.0)
    # END FOR
    return x_mid, y_mid

#------------------------------------------------------------------------------
def _plot_bracket(
        ax, n_boxes, total_width, total_height, text, gap_frac=0.02,
        c='gray', lw=1.0, fontsize=10,
):
    bar_len = total_width * gap_frac * 2

    x1 = total_width * (1 + gap_frac)
    x1_ = x1 + bar_len
    y1 = 0
    y1_ = y1 + bar_len

    x2 = x1
    x2_ = x2 + bar_len
    y2 = total_height
    y2_ = y2 - bar_len

    x0 = x1_
    x0_ = x0 + bar_len
    x0__ = x0_ + bar_len / 2.0  # where to put text
    y0 = (y1 + y2) / 2.0

    ax.plot([x1, x1_], [y1, y1_], c=c, lw=lw)
    ax.plot([x2, x2_], [y2, y2_], c=c, lw=lw)
    ax.plot([x1_, x1_], [y1_, y2_], c=c, lw=lw)
    ax.plot([x0, x0_], [y0, y0], c=c, lw=lw)
    ax.text(
        x0__, y0, text, ha='left', va='center', fontsize=fontsize, rotation=270,
    )
