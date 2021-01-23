# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.version import LooseVersion

# Explicitly register matplotlib converters:
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from . import helper as hlp
from . import colors_and_lines as cl

#%%============================================================================
def plot_timeseries(
        time_series, date_fmt=None, fig=None, ax=None, figsize=(10,3),
        dpi=100, xlabel='Time', ylabel=None, label=None, color=None,
        lw=2, ls=None, marker=None, fontsize=12, xgrid_on=True,
        ygrid_on=True, title=None, zorder=None, alpha=1.0,
        month_grid_width=None,
):
    '''
    Plot time series (i.e., values a function of dates).

    You can plot multiple time series by supplying a multi-column pandas
    Dataframe, but you cannot use custom line specifications (colors, width,
    and styles) for each time series. It is recommended to use
    :func:`~plot_multiple_timeseries` in stead.

    Parameters
    ----------
    time_series : pandas.Series or pandas.DataFrame
        A pandas Series, with index being date; or a pandas DataFrame, with
        index being date, and each column being a different time series.
    date_fmt : str
        Date format specifier, e.g., '%Y-%m' or '%d/%m/%y'.
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
    xlabel : str
        Label of X axis. Usually "Time" or "Date".
    ylabel : str
        Label of Y axis. Usually the meaning of the data, e.g., "Gas price [$]".
    label : str
        Label of data, for plotting legends.
    color : list<float> or str
        Color of line. If None, let Python decide for itself.
    xgrid_on : bool
        Whether or not to show vertical grid lines (default: ``True``).
    ygrid_on : bool
        Whether or not to show horizontal grid lines (default: ``True``).
    title : str
        Figure title (optional).
    zorder : float
        Set the zorder for lines. Higher zorder are drawn on top.
    alpha : float
        Opacity of the line.
    month_grid_width : float
        the on-figure "horizontal width" that each time interval occupies.
        This value determines how X axis labels are displayed (e.g., smaller
        width leads to date labels being displayed with 90 deg rotation).
        Do not change this unless you really know what you are doing.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.

    See also
    --------
    :func:`~plot_multiple_timeseries` :
        Plot multiple time series, with the ability to specify different
        line specifications for each line.
    '''
    if not isinstance(time_series, (pd.Series, pd.DataFrame)):
        raise TypeError('`time_series` must be a pandas Series or DataFrame.')

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)
    ax_size = hlp._get_ax_size(fig, ax)

    ts = time_series.copy()  # shorten the name + avoid changing input
    ts.index = _as_date(ts.index, date_fmt)  # batch-convert index to Timestamp format of pandas

    if zorder:
        ax.plot(
            ts.index, ts, color=color, lw=lw, ls=ls, marker=marker,
            label=label, zorder=zorder, alpha=alpha,
        )
    else:
        ax.plot(
            ts.index, ts, color=color, lw=lw, ls=ls, marker=marker,
            label=label, alpha=alpha,
        )
    ax.set_label(label)  # set label for legends using argument 'label'
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if month_grid_width == None:  # width of each month in inches
        month_grid_width = float(ax_size[0])/_calc_month_interval(ts.index)
    ax = _format_xlabel(ax,month_grid_width)

    if ygrid_on == True:
        ax.yaxis.grid(ls=':', color=[0.75]*3)
    if xgrid_on == True:
        ax.xaxis.grid(False, 'major')
        ax.xaxis.grid(xgrid_on, 'minor', ls=':', color=[0.75]*3)
    ax.set_axisbelow(True)

    if title is not None:
        ax.set_title(title)

    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig, ax

#%%============================================================================
def plot_multiple_timeseries(
        multiple_time_series, show_legend=True,
        fig=None, ax=None, figsize=(10,3), dpi=100,
        ncol_legend=5, **kwargs,
):
    '''
    Plot multiple time series.

    Note that setting keyword arguments such as ``color`` or ``linestyle`` will
    force all time series to have the same color or line style. So we recommend
    letting this function generate distinguishable line specifications (color/
    linestyle/linewidth combinations) by itself. (Although the more time series,
    the less the distinguishability. 240 time series or less is recommended.)

    Parameters
    ----------
    multiple_time_series : pandas.DataFrame or pandas.Series
        If it is a pandas DataFrame, its index is the date, and each column
        is a different time series.
        If it is a pandas Series, it will be internally converted into a
        1-column pandas DataFrame.
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
    ncol_legend : int
        Number of columns of the legend.
    **kwargs :
        Other keyword arguments to be passed to :func:`~plot_timeseries()`, such
        as color, marker, fontsize, alpha, etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.

    See also
    --------
    :func:`~plot_timeseries` :
        Plot a single set of time series.
    '''
    if not isinstance(multiple_time_series, (pd.Series, pd.DataFrame)):
        raise TypeError(
            '`multiple_time_series` must be a pandas Series or DataFrame.'
        )

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    if not show_legend:  # if no need to show legends, just pass everything
        fig, ax = plot_timeseries(multiple_time_series, fig, ax, dpi, **kwargs)
    else:
        if isinstance(multiple_time_series,pd.Series):
            nr_timeseries = 1
            multiple_time_series = pd.DataFrame(multiple_time_series,copy=True)
        else:
            nr_timeseries = multiple_time_series.shape[1]

        if nr_timeseries <= 40:  # 10 colors x 4 linestyles = 40, so use lw=2
            linespecs = cl.get_linespecs(range_linewidth=[2])
        elif nr_timeseries <= 120:  # need multiple line widths
            linespecs = cl.get_linespecs(range_linewidth=[1,3,5])
        elif nr_timeseries <= 240:
            linespecs = cl.get_linespecs(
                color_scheme='tab20', range_linewidth=[1,3,5],
            )
        else:
            linespecs = cl.get_linespecs(
                color_scheme='tab20',  # use more line widths
                range_linewidth=range(1, (nr_timeseries - 1) // 240 + 5, 2),
            )

        for j in range(nr_timeseries):
            tmp_dict = linespecs[j % nr_timeseries].copy()
            tmp_dict.update(kwargs)  # kwargs overwrites tmp_dict if key already exists in tmp_dict
            if 'lw' in tmp_dict:  # thinner lines above thicker lines
                zorder = 1 + 1.0/tmp_dict['lw']  # and "+1" to put all lines above grid line

            plot_timeseries(
                multiple_time_series.iloc[:,j],
                fig=fig,
                ax=ax,
                zorder=zorder,
                label=multiple_time_series.columns[j],
                **tmp_dict,
            )

        if 'title' not in kwargs:
            bbox_anchor_loc = (0., 1.02, 1., .102)
        else:
            bbox_anchor_loc = (0., 1.08, 1., .102)
        ax.legend(
            bbox_to_anchor=bbox_anchor_loc, loc='lower center', ncol=ncol_legend,
        )

    ax.set_axisbelow(True)
    return fig, ax

#%%============================================================================
def fill_timeseries(
        time_series, upper_bound, lower_bound, date_fmt=None,
        fig=None, ax=None, figsize=(10,3), dpi=100,
        xlabel='Time', ylabel=None, label=None,
        color=None, lw=3, ls='-', fontsize=12, title=None,
        xgrid_on=True, ygrid_on=True,
):
    '''
    Plot time series as a line and then plot the upper and lower bounds as
    shaded areas.

    Parameters
    ----------
    time_series : pandas.Series
        A pandas Series, with index being date.
    upper_bound : pandas.Series
        Upper bounds of the time series, must have the same length as
        ``time_series``.
    lower_bound : pandas.Series
        Lower bounds of the time series, must have the same length as
        ``time_series``.
    date_fmt : str
        Date format specifier, e.g., '%Y-%m' or '%d/%m/%y'.
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
    xlabel : str
        Label of X axis. Usually "Time" or "Date".
    ylabel : str
        Label of Y axis. Usually the meaning of the data (e.g., "Gas price [$]").
    label : str
        Label of data, for plotting legends.
    color : str or list or tuple
        Color of line. If None, let Python decide for itself.
    lw : scalar
        Line width of the line that represents time_series.
    ls : str
        Line style of the line that represents time_series.
    fontsize : scalar
        Font size of the texts in the figure.
    title : str
        Figure title.
    xgrid_on : bool
        Whether or not to show vertical grid lines (default: ``True``).
    ygrid_on : bool
        Whether or not to show horizontal grid lines (default: ``True``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    if not isinstance(time_series, pd.Series):
        raise TypeError(
            '`time_series` must be a pandas Series with index being dates.'
        )

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    ts = time_series.copy()  # shorten the name + avoid changing some_time_series
    ts.index = _as_date(ts.index, date_fmt)  # batch-convert index to Timestamp format of pandas
    lb = lower_bound.copy()
    ub = upper_bound.copy()

    ax.fill_between(
        ts.index, lb, ub, color=color, facecolor=color,
        linewidth=0.01, alpha=0.5, interpolate=True,
    )
    ax.plot(ts.index, ts, color=color, lw=lw, ls=ls, label=label)
    ax.set_label(label)  # set label for legends using argument 'label'
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    month_grid_width = float(figsize[0])/_calc_month_interval(ts.index) # width of each month in inches
    ax = _format_xlabel(ax, month_grid_width)

    if ygrid_on == True:
        ax.yaxis.grid(ygrid_on, ls=':', color=[0.75]*3)
    if xgrid_on == True:
        ax.xaxis.grid(False, 'major')
        ax.xaxis.grid(xgrid_on, 'minor', ls=':', color=[0.75]*3)
    ax.set_axisbelow(True)

    if title is not None:
        ax.set_title(title)

    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig, ax

#%%============================================================================
def _calc_month_interval(date_array):
    '''
    Calculate how many months are there between the first month and the last
    month of the given date_array.
    '''
    date9 = list(date_array)[-1]
    date0 = list(date_array)[0]
    delta_days = (date9 - date0).days
    if delta_days < 30:  # within one month
        delta_months = delta_days/30.0  # return a float between 0 and 1
    else:
        delta_months = delta_days//30
    return delta_months

#%%============================================================================
def _format_xlabel(ax, month_width):
    '''
    Format the x axis label (which represents dates) in accordance to the width
    of each time interval (month or day).

    For narrower cases, year will be put below month.

    For even narrower cases, not every month will be displayed as a label.

    For very narrow cases (e.g., many years), months will not be displayed, and
    sometimes not every year will be displayed.
    '''
    rot = None  # degree of rotation
    y_int = None  # year interval
    d_int = None  # day interval
    if month_width < 0.038:
        m_int = None  # month interval
        y_int = 3 * int(0.038/month_width)  # interval increases with narrower size
    elif month_width < 0.043:
        m_int = None
        y_int = 2
    elif month_width < 0.05:
        m_int = None
        rot = 30
    elif month_width < 0.055:
        m_int = 6  # display month label for every 6 months
        rot = 30
    elif month_width < 0.06:
        m_int = 6
    elif month_width < 0.09:
        m_int = 5
    elif month_width < 0.11:
        m_int = 4
    elif month_width < 0.16:
        m_int = 3
    elif month_width < 0.29:
        m_int = 2
    else:
        m_int = 1
        if month_width < 1.9:
            pass  # d_int is still None
        elif month_width < 2.9:
            d_int = 15
        elif month_width < 3.5:
            d_int = 10
        elif month_width < 4:
            d_int = 9
        elif month_width < 5:
            d_int = 8
        elif month_width < 6:
            d_int = 7
        elif month_width < 7:
            d_int = 6
        elif month_width < 8.5:
            d_int = 5
        elif month_width < 11:
            d_int = 4
        elif month_width < 15:
            d_int = 3
            rot = 30
        elif month_width < 25.5:
            d_int = 2
            rot = 30
        else:
            d_int = 1
            rot = 30

    if y_int:  # show only every 'y_int' years
        years = mpl.dates.YearLocator(base=y_int)
    else:  # show year on January of every year
        years = mpl.dates.YearLocator()

    xlim = ax.get_xlim()  # number of days since 0001/Jan/1-00:00:00 UTC plus one
    xlim_ = [mpl.dates.num2date(i) for i in xlim]  # convert to datetime object
    if xlim_[0].year == xlim_[1].year:  # if date range is within same year
        if xlim_[0].day > 1:  # not first day of month: show year on next month
            years = mpl.dates.YearLocator(base=1,month=xlim_[0].month+1,day=1)
        else:   # first day of month: show year on this month
            years = mpl.dates.YearLocator(base=1,month=xlim_[0].month  ,day=1)

    if not d_int:  # no day labels will be shown
        months_fmt = mpl.dates.DateFormatter('%m')
    else:
        months_fmt = mpl.dates.DateFormatter('%m/%d')

    if m_int:  # show every 'm_int' months
        if d_int:  # day labels will be shown
            months = mpl.dates.DayLocator(interval=d_int)
        else:
            months = mpl.dates.MonthLocator(interval=m_int)
        ax.xaxis.set_minor_locator(months)
        ax.xaxis.set_minor_formatter(months_fmt)
        if d_int and rot:  # days are shown as rotated
            years_fmt = mpl.dates.DateFormatter('\n\n%Y')  # show year on next next line
        else:
            years_fmt = mpl.dates.DateFormatter('\n%Y')  # show year on next line
    else:  # do not show months in x axis label
        years_fmt = mpl.dates.DateFormatter('%Y')  # show year on current line

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.tick_params(labelright=True)  # also show y axis on right edge of figure

    if rot and d_int:  # days/months are shown as rotated
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=rot)
    elif rot and not d_int:  # only show years as rotated
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=rot)

    return ax

#%%============================================================================
def _as_date(raw_date, date_fmt=None):
    '''
    Convert raw_date to datetime array.

    It can handle:
    (A) A list of str, int, or float, such as:
        [1] ['20150101', '20150201', '20160101']
        [2] ['2015-01-01', '2015-02-01', '2016-01-01']
        [3] [201405, 201406, 201407]
        [4] [201405.0, 201406.0, 201407.0]
    (B) A list of just a single element, such as:
        [1] [201405]
        [2] ['2014-05-25']
        [3] [201412.0]
    (C) A single element of: str, int, float, such as:
        [1] 201310
        [2] 201210.0
    (D) A pandas Series, of length 1 or length larger than 1

    Parameters
    ----------
    raw_date : (see above for acceptable formats)
        The raw date information to be processed
    date_fmt : str
        The format of each individual date entry, e.g., '%Y-%m-%d' or '%m/%d/%y'.
        To be passed directly to pd.to_datetime()
        (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html)

    Returns
    -------
    date_list :
        A variable with the same structure (list or scaler-like) as raw_date,
        whose contents have the data type "pandas._libs.tslib.Timestamp".

    Reference
    ---------
    https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
    '''
    if LooseVersion(pd.__version__) <= LooseVersion('0.17.1'):
        timestamp_type = pd.tslib.Timestamp
    else:
        timestamp_type = pd._libs.tslib.Timestamp

    if isinstance(raw_date,timestamp_type):  # if already a pandas Timestamp obj
        date_list = raw_date  # return raw_date as is
    else:
        # -----------  Convert to list for pd.Series or np.ndarray objects  -------
        if isinstance(raw_date,(pd.Series,np.ndarray,pd.Index)):
            raw_date = list(raw_date)

        # ----------  Element-wise checks and conversion  -------------------------
        if isinstance(raw_date,list):   # if input is a list
            if len(raw_date) == 0:  # empty list
                date_list = None   # return an empty object
            elif len(raw_date) == 1:  # length of string is 1
                date_ = str(int(raw_date[0]))  # simply unpack it and convert to str int
                date_list = pd.to_datetime(date_, format=date_fmt)
            else:  # length is larger than 1
                nr = len(raw_date)
                date_list = [[None]] * nr
                for j in range(nr):  # loop every element in raw_date
                    j_th = raw_date[j]
                    if isinstance(j_th, str) and j_th.isdigit():
                        date_ = str(int(j_th))
                    elif isinstance(j_th, str) and not j_th.isdigit():
                        date_ = j_th
                    elif isinstance(j_th,(int,np.integer,np.float)):
                        date_ = str(int(j_th))  # robustness not guarenteed!
                    else:
                        raise TypeError('Date type of the element(s) in '
                                        '`raw_date` not recognized.')
                    date_list[j] = pd.to_datetime(date_, format=date_fmt)
        elif type(raw_date) == dt.date:  # if a datetime.date object
            date_list = raw_date  # no need for conversion
        elif isinstance(raw_date, hlp._scalar_like):
            date_ = str(int(raw_date))
            date_list = pd.to_datetime(date_, format=date_fmt)
        elif isinstance(raw_date, str):  # a single string, such as '2015-04'
            date_ = raw_date  # no conversion needed
            date_list = pd.to_datetime(date_, format=date_fmt)
        else:
            raise TypeError('Input data type of `raw_date` not recognized.')
            print('\ntype(raw_date) is: %s' % type(raw_date))
            try:
                print('Length of raw_date is: %s' % len(raw_date))
            except TypeError:
                print('raw_date has no length.')

    return date_list

#%%============================================================================
def _str2date(date_):
    '''
    Convert date_ into a datetime object. date_ must be a string (not a list
    of strings).

    Currently accepted date formats:
    (1) Aug-2014
    (2) August 2014
    (3) 201407
    (4) 2016-07
    (5) 2015-02-21

    Note: This subroutine is no longer being used.
    '''
    day = None
    if ('-' in date_) and (len(date_) == 8):  # for date style 'Aug-2014'
        month, year = date_.split('-')  # split string by character
        month = dt.datetime.strptime(month,'%b').month  # from 'Mar' to '3'
    elif ' ' in date_:  # for date style 'August 2014'
        month, year = date_.split(' ')  # split string by character
        month = dt.datetime.strptime(month,'%B').month  # from 'March' to '3'
        year = int(year)
    elif (len(date_) == 6) and date_.isdigit():  # for cases like '201205'
        year  = int(date_[:4])  # first four characters
        month = int(date_[4:])  # remaining characters
    elif (len(date_) == 7) and (date_[4]=='-') and not date_.isdigit():  # such as '2015-03' [NOT 100% ROBUST!]
        year, month = date_.split('-')
        year = int(year)
        month = int(month)
    elif (len(date_) == 10) and not date_.isdigit():  # such as '2012-02-01' [NOT 100% ROBUST!!]
        year, month, day = date_.split('-')  # split string by character
        year = int(year)
        month = int(month)
        day = int(day)
    elif (len(date_)==6) and (date_[3]=='-') and (date_[:3].isalpha()) \
         and (date_[4:].isdigit()):  # such as 'May-12'
        month, year = date_.split('-')
        month = dt.datetime.strptime(month,'%b').month  # from 'Mar' to '3'
        year = int(year) + 2000  # from '13' to '2013'
    else:
        print('*****  Edge case encountered! (Date format not recognized.)  *****')
        print('\nUser supplied %s, which is not recognized.\n' % date_)

    if day is None:  # if day is not defined in the if statements
        return dt.date(year,month,1)
    else:
        return dt.date(year,month,day)
