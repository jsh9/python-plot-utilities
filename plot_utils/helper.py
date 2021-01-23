# -*- coding: utf-8 -*-

import collections
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

#%%----------------------------------------------------------------------------
_array_like = (list, np.ndarray, pd.Series)  # define a "compound" data type
_scalar_like = (int, float, np.number)  # "compound" data type

#%%----------------------------------------------------------------------------
class LengthError(Exception):
    pass

class DimensionError(Exception):
    pass

#%%============================================================================
def assert_type(something, desired_type, name='something'):
    '''
    Assert ``something`` is a ``desired_type``.

    Parameters
    ----------
    something :
        Any Python object.
    desired_type : type or typle<type>
        A valid Python type, such as float, or a tuple of Python types, such
        as (float, int).
    name : str
        The name of ``something`` to show in the error message (if applicable).
    '''
    if not isinstance(something, desired_type):
        msg = '"%s" must be %s, rather than %s.' % (name, desired_type, type(something))
        raise TypeError(msg)
    # END IF

#%%============================================================================
def assert_element_type(some_iterable, desired_element_type, name='some_iterable'):
    '''
    Assert all elements of ``some_iterable`` is of ``desired_type``.

    Parameters
    ----------
    some_iterable : Python iterable
        An iterable object, such as a list, numpy array.
    desired_element_type : type or tuple<type>
        Desired element type.
    name : str
        The name of ``something`` to show in the error message (if applicable).
    '''
    msg = 'All elements of "%s" must be %s.' % (name, desired_element_type)
    assert_type(some_iterable, collections.abc.Iterable, name=name)
    if isinstance(desired_element_type, type):
        if not all([isinstance(_, desired_element_type) for _ in some_iterable]):
            raise TypeError(msg)
        # END IF
    elif isinstance(desired_element_type, tuple):
        success = False
        for this_type in desired_element_type:
            if all([isinstance(_, this_type) for _ in some_iterable]):
                success = True
                continue
            # END IF
        # END FOR
        if not success:
            raise TypeError(msg)
        # END IF
    else:
        raise TypeError('`desired_element_type` must be a type or a tuple of types.')
    # END IF-ELSE

#%%============================================================================
def _process_fig_ax_objects(fig, ax, figsize=None, dpi=None, ax_proj=None):
    '''
    Processes figure and axes objects. If ``fig`` and ``ax`` are None, creates
    new figure and new axes according to ``figsize``, ``dpi``, and ``ax_proj``.
    Otherwise, uses the passed-in ``fig`` and/or ``ax``.

    Parameters
    ----------
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
    ax_proj : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}
        The projection type of the axes. The default None results in a
        'rectilinear' projection.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    if fig is None:  # if a figure handle is not provided, create new figure
        fig = pl.figure(figsize=figsize,dpi=dpi)
    else:   # if provided, plot to the specified figure
        pl.figure(fig.number)

    if ax is None:  # if ax is not provided
        ax = plt.axes(projection=ax_proj)  # create new axes and plot lines on it
    else:
        ax = ax  # plot lines on the provided axes handle

    return fig, ax

#%%============================================================================
def _upcast_dtype(x):
    '''
    Cast dtype of x (a pandas Series) as string or float in-place.

    Parameter
    ---------
    x : pandas.Series
        An array whose elements are to be upcast.

    Returns
    -------
    x : pandas.Series
        The array whose elements are now upcast.
    '''
    assert(type(x) == pd.Series)

    if x.dtype.name in ['category', 'bool', 'datetime64[ns]', 'datetime64[ns, tz]']:
        x = x.astype(str)

    if x.dtype.name == 'timedelta[ns]':
        x = x.astype(float)

    return x

#%%============================================================================
def _find_axes_lim(data_limit, tick_base_unit, direction='upper'):
    '''
    Return a "whole" number to be used as the upper or lower limit of axes.

    For example, if the maximum x value of the data is 921.5, and you would
    like the upper x_limit to be a multiple of 50, then this function returns
    950.

    Parameters
    ----------
    data_limit : float, int, list<float>, list<int>, tuple<float>, tuple<int>
        The upper and/or lower limit(s) of data.
            (1) If a tuple (or list) of two elements is provided, then the
                upper and lower axis limits are automatically determined.
                (The order of the two elements does not matter.)
            (2) If a float or an int is provided, then the axis limit is
                determined based on the ``direction`` provided.
    tick_base_unit : float
        For example, if you want your axis limit(s) to be a multiple of 20
        (such as 80, 120, 2020, etc.), then use 20.
    direction : {'upper', 'lower'}
        The direction of the limit to be found. For example, if the maximum
        of the data is 127, and ``tick_base_unit`` is 50, then a ``direction``
        of lower yields a result of 100. This parameter is effective only when
        ``data_limit`` is a scalar.

    Returns
    -------
    axes_lim : list<float> or float
        If ``data_limit`` is a list/tuple of length 2, return a list:
        [min_limit, max_limit] (always ordered no matter what the order of
        ``data_limit`` is). If ``data_limit`` is a scalar, return the axis
        limit according to ``direction``.
    '''
    if isinstance(data_limit, _scalar_like):
        if direction == 'upper':
            return tick_base_unit * (int(data_limit/tick_base_unit)+1)
        elif direction == 'lower':
            return tick_base_unit * (int(data_limit/tick_base_unit))
        else:
            raise LengthError('Length of `data_limit` should be at least 1.')
    elif isinstance(data_limit, (tuple, list)):
        if len(data_limit) > 2:
            raise LengthError('Length of `data_limit` should be at most 2.')
        elif len(list(data_limit)) == 2:
            min_data = min(data_limit)
            max_data = max(data_limit)
            max_limit = tick_base_unit * (int(max_data/tick_base_unit)+1)
            min_limit = tick_base_unit * (int(min_data/tick_base_unit))
            return [min_limit, max_limit]
        elif len(data_limit) == 1:  # such as [2.14]
            return _find_axes_lim(data_limit[0],tick_base_unit,direction)
    elif isinstance(data_limit, np.ndarray):
        data_limit = data_limit.flatten()  # convert np.array(2.5) into np.array([2.5])
        if data_limit.size == 1:
            return _find_axes_lim(data_limit[0],tick_base_unit,direction)
        elif data_limit.size == 2:
            return _find_axes_lim(list(data_limit),tick_base_unit,direction)
        elif data_limit.size >= 3:
            raise LengthError('Length of `data_limit` should be at most 2.')
        else:
            raise TypeError(
                '`data_limit` should be a scalar or a tuple/list of length 2.'
            )
    else:
        raise TypeError(
            '`data_limit` should be a scalar or a tuple/list of length 2.'
        )

#%%============================================================================
class _FixedOrderFormatter(mpl.ticker.ScalarFormatter):
    '''
    Formats axis ticks using scientific notation with a constant order of
    magnitude.

    (Reference: https://stackoverflow.com/a/3679918)

    Note: this class is not currently being used.
    '''
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        self._order_of_mag = order_of_mag
        mpl.ticker.ScalarFormatter.__init__(
            self, useOffset=useOffset, useMathText=useMathText,
        )

    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag

#%%============================================================================
def _calc_bar_width(width):
    '''
    Calculate width (in points) of bar plot from figure width (in inches).
    '''
    if width <= 7:
        bar_width = width * 3.35  # these numbers are manually fine-tuned
    elif width <= 9:
        bar_width = width * 2.60
    elif width <= 10:
        bar_width = width * 2.10
    else:
        bar_width = width * 1.2

    return bar_width

#%%============================================================================
def _get_ax_size(fig, ax, unit='inches'):
    '''
    Get size of axes within a figure, given fig and ax objects.

    https://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
    '''
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    if unit == 'pixels':
        width *= fig.dpi  # convert from inches to pixels
        height *= fig.dpi

    return width, height

#%%============================================================================
def _calc_r2_score(y_true, y_pred):
    '''
    Calculate the coefficient of determination between two arrays. The best
    possible value is 1.0. The result can be negative, because the model
    predicted value (``y_pred``) can be arbitrarily bad. A naive prediction,
    i.e., ``y_pred`` equals the mean value of ``y_true`` produces a negative
    infinity R2 score.

    Parameters
    ----------
    y_true : list, numpy.ndarray, or pandas.Series
        The "true values", or the dependent variable, or "y axis".
    y_pred : list, numpy.ndarray, or pandas.Series
        The "predicted values", or the independent variable, or "x axis".

    Returns
    -------
    r2_score : float
        The coefficient of determination.

    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    '''
    if not isinstance(y_true, _array_like):
        raise TypeError('`y_true` needs to be a list, numpy array, or pandas Series.')
    if not isinstance(y_pred, _array_like):
        raise TypeError('`y_pred` needs to be a list, numpy array, or pandas Series.')
    if len(y_true) != len(y_pred):
        raise LengthError('`y_true` and `y_pred` should have the same length.')

    f = np.array(y_pred)  # follow the notation in the wikipedia page
    y = np.array(y_true)

    y_bar = np.mean(y)
    SS_tot = np.sum((y - y_bar)**2.0)
    SS_res = np.sum((y - f)**2.0)
    r2_score = 1 - SS_res / SS_tot
    return r2_score

#%%============================================================================
def __axes_styling_helper(
        ax, vert, rot, data_names, n_datasets, data_ax_label,
        name_ax_label, title,
):
    '''
    Helper function. Used by _violin_plot_helper() and _multi_hist_helper().

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Matplotlib axes object.
    vert : bool
        Whether to show the violins or the "base" of the histograms as vertical.
    rot : float
        The rotation (in degrees) of the data_names when shown as the tick
        labels. If ``vert`` is ``False``, ``rot`` has no effect.
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
    n_datasets : int
        Number of sets of data.
    data_ax_label : str
        The labels of the "data axis". ("Data axis" is the axis along which
        the data values are presented.)
    name_ax_label : str
        The label of the "name axis". ("Name axis" is the axis along which
        different violins are presented.)
    title : str
        The title of the plot.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Matplotlib axes object.
    '''
    ax.grid(ls=':')
    ax.set_axisbelow(True)

    if vert:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
        ax.set_xticks(np.arange(n_datasets) + 1)
        ha = 'center' if (0 <= rot < 30 or rot == 90) else 'right'
        ax.set_xticklabels(data_names, rotation=rot, ha=ha)
    else:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
        ax.set_yticks(np.arange(n_datasets) + 1)
        ax.set_yticklabels(data_names)

    if data_ax_label:
        if not vert:
            ax.set_xlabel(data_ax_label)
        else:
            ax.set_ylabel(data_ax_label)
    if name_ax_label:
        if not vert:
            ax.set_ylabel(name_ax_label)
        else:
            ax.set_xlabel(name_ax_label)

    if title:
        ax.set_title(title)

    return ax

#%%============================================================================
class _MidpointNormalize(Normalize):
    '''
    Auxiliary class definition. Copied from:
    https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib/20146989#20146989
    '''
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
