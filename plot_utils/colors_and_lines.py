# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import helper as hlp

#%%============================================================================
class Color():
    '''
    A class that defines a color.

    Parameters
    ----------
    color : str or <tuple> or <list>
        The color information to initialize the Color object. Can be a list
        or tuple of 3 elements (i.e., the RGB information), or a HEX string
        such as "#00FF00", or XKCD color names (https://xkcd.com/color/rgb/)
        or X11 color names  (http://cng.seas.rochester.edu/CNG/docs/x11color.html).
    is_rgb_normalized : bool
        Whether or not the input information (if RGB) contains the normalized
        values (such as [0, 0.5, 0.5]). This parameter has no effect if
        the input is not RGB.
    '''
    def __init__(self, color, is_rgb_normalized=True):

        import matplotlib._color_data as mcd

        if not isinstance(color, (list, tuple, str)):
            raise TypeError('`color` must be a list/tuple of length 3 or a str.')

        if isinstance(color, (list, tuple)):
            if len(color) != 3:
                raise TypeError('If `color` is a list/tuple, its length must be 3.')
            else:
                self.__color = self.__rgb_to_hex(color, is_rgb_normalized)

        if isinstance(color, str):
            color = color.lower()  # convert all to lower case
            if len(color) == 1:  # base color specification, such as 'w' or 'b'
                _rgb_color = mcd.BASE_COLORS[color]
                self.__color = self.__rgb_to_hex(_rgb_color, is_normalized=True)
            elif color[0] == '#':  # HEX color specification, such as '#00FFFF'
                self.__color = color
            elif color.startswith('xkcd:'):
                self.__color = mcd.XKCD_COLORS[color]
            elif color.startswith('tab:'):
                self.__color = mcd.TABLEAU_COLORS[color]
            else:
                try:
                    self.__color = mcd.CSS4_COLORS[color]
                except KeyError:
                    raise ValueError("Unrecognized color: '%s'" % color)

    def __repr__(self):
        return 'RGB color: %s' % str(self.as_rgb());

    def __rgb_to_hex(self, rgb, is_normalized=True):
        '''
        Private method. Converts RGB values into HEX.

        Parameters
        ----------
        rgb : list<float> or tuple<float>
            RGB values
        is_normalized : bool
            Whether the RGB values are normalized (i.e., from 0 to 1)

        Returns
        -------
        hex_val : str
            HEX value
        '''
        if np.any(np.array(rgb) > 255):
            raise ValueError('`rgb` values should not exceed 255.')

        if np.any(np.array(rgb) < 0):
            raise ValueError('`rgb` values should not be negative.')

        if max(rgb) > 1.0 and is_normalized == True:
            rgb = [_ / 255.0 for _ in rgb]

        if is_normalized:
            rgb_255 = [int(_ * 255) for _ in rgb]
        else:
            rgb_255 = [int(_) for _ in rgb]

        return u'#{:02x}{:02x}{:02x}'.format(*rgb_255)

    def __hex_to_rgb(self, hex_, normalize=True):
        '''
        Private method. Converts HEX values into RGB.

        Reference: https://stackoverflow.com/a/29643643/8892243

        Parameters
        ----------
        hex_ : str
            HEX representation of a color
        normalize : bool
            Whether or not to return the normalized (between 0 and 1) RGB

        Returns
        -------
        rgb : tuple<float>
            RGB values in three numbers.
        '''
        h = hex_[1:]  # strip the '#' in the front
        if normalize:
            rgb = tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2 ,4))
        else:
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))

        return rgb

    def as_rgb(self, normalize=True):
        '''
        Export thes color as RGB values.

        Parameter
        ---------
        normalize : bool
            Whether or not to return the normalized (between 0 and 1) RGB.

        Returns
        -------
        rgb_val : tuple<float>
            RGB values in three numbers.
        '''
        return self.__hex_to_rgb(self.__color, normalize=normalize)

    def as_rgba(self, alpha=1.0):
        '''
        Exports the color object as RGBA values. The R, G, and B values are
        always normalized (between 0 and 1).

        Parameter
        ---------
        alpha : float
            The transparency (0 being completely transparent and 1 opaque).

        Returns
        -------
        rgba_val : tuple<float>
            RGBA values in four numbers.
        '''
        if alpha < 0 or alpha > 1:
            raise ValueError('`alpha` must be between 0 and 1 (inclusive).')

        rgb = self.__hex_to_rgb(self.__color, normalize=True)
        rgba = (rgb[0], rgb[1], rgb[2], alpha)

        return rgba

    def as_hex(self):
        '''
        Exports the color object as HEX values.

        Returns
        -------
        hex_val : str
            HEX value.
        '''
        return self.__color

    def show(self):
        '''
        Shows color as a square patch.
        '''
        import matplotlib.patches as mpatch

        fig = plt.figure(figsize=(0.5, 0.5))
        ax = fig.add_axes([0, 0, 1, 1])
        p = mpatch.Rectangle((0, 0), 1, 1, color=self.__color)
        ax.add_patch(p)
        ax.axis('off')

#%%============================================================================
class Multiple_Colors():
    '''
    A class that defines multiple colors.

    Parameters
    ----------
    colors : list
        A list of color information to initialize the Multiple_Colors object.
        The list elements can be:
            - a list or tuple of 3 elements (i.e., the RGB information)
            - a HEX string such as "#00FF00"
            - an XKCD color name (https://xkcd.com/color/rgb/)
            - an X11 color name (http://cng.seas.rochester.edu/CNG/docs/x11color.html)
        Different elements of colors do not need to be of the same type.
    is_rgb_normalized : bool
        Whether or not the input information (if RGB) contains the normalized
        values (such as [0, 0.5, 0.5]). This parameter has no effect if
        the input is not RGB.
    '''
    def __init__(self, colors, is_rgb_normalized=True):
        if not isinstance(colors, list):
            raise TypeError('`colors` must be a list.')
        if len(colors) == 0:
            raise hlp.LengthError('Length of `colors` must non-zero.')

        self.__length = len(colors)
        self.__Colors = [None] * self.__length
        for j, color in enumerate(colors):
            self.__Colors[j] = Color(color, is_rgb_normalized)

    def __repr__(self):
        return self.as_rgb()

    def as_rgb(self, normalize=True):
        '''
        Exports the colors as a list of RGB values

        Parameter
        ---------
        normalize : bool
            Whether or not to return the normalized (between 0 and 1) RGB.

        Returns
        -------
        rgb_list : list<list<float>>
            A list of list: each sub-list represents a RGB color in three
            numbers.
        '''
        result = [None] * self.__length
        for j in range(self.__length):
            result[j] = self.__Colors[j].as_rgb(normalize=normalize)

        return result

    def as_rgba(self, alpha=1.0):
        '''
        Exports the colors as a list of RGBA values

        Parameter
        ---------
        alpha : float
            The transparency (0 being completely transparent and 1 opaque).

        Returns
        -------
        rgba_list : list<list<float>>
            A list of list: each sub-list represents a RGBA color in four
            numbers.
        '''
        result = [None] * self.__length
        for j in range(self.__length):
            result[j] = self.__Colors[j].as_rgba(alpha=alpha)

        return result

    def as_hex(self):
        '''
        Exports the colors as a list of HEX values

        Returns
        -------
        hex_list : list<str>
            A list of HEX colors
        '''
        result = [None] * self.__length
        for j in range(self.__length):
            result[j] = self.__Colors[j].as_hex()

        return result

    def show(self, vertical=False, text=None):
        '''
        Shows the colors as square patches

        Parameters
        ----------
        vertical : bool
            Whether or not to show the patches vertically
        text : str
            The text to show next to the colors
        '''
        import matplotlib.patches as mpatch

        figsize = (.5, self.__length/2) if vertical else (self.__length/2, .5)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        for j in range(self.__length):
            loc = (j, 0) if not vertical else (0, self.__length - j - 1)
            p = mpatch.Rectangle(loc, 1, 1, color=self.__Colors[j].as_hex())
            ax.add_patch(p)

        ax.axis('off')
        if not vertical:
            ax.set_xlim(0, self.__length)
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, self.__length)
            ax.set_xlim(0, 1)

        if text:
            if not vertical:
                ax.text(j + 1.5, 0.5, text, va='center')
            else:
                ax.text(0.5, j + 1.5, text, ha='center')

#%%============================================================================
def _check_color_types(color, n=None):
    '''
    Helper function that checks whether a Python object ``color`` is indeed a
    valid list (or tuple) of length n that defines ``n`` colors.

    Returns ``True`` (valid) or ``False`` (otherwise), and an error message
    (empty message if ``True``).
    '''
    if not isinstance(color,(list,tuple)):
        is_valid = False
        err_msg = '"color" must be a list of lists (or tuple of tuples).'
    elif not all([isinstance(c_, (list, tuple)) for c_ in color]) and \
         not all([isinstance(c_, str) for c_ in color]):
            is_valid = False
            err_msg = '"color" must be a list of lists (or tuple of tuples).'
    else:
        if n and len(color) < n:
            is_valid = False
            err_msg = 'Length of "color" must be at least the same length as "n".'
        else:
            is_valid = True
            err_msg = ''

    return is_valid, err_msg

#%%============================================================================
def get_colors(N=None, color_scheme='tab10'):
    '''
    Return a list of N distinguisable colors. When N is larger than the color
    scheme capacity, the color cycle is wrapped around.

    What does each color_scheme look like?
        https://matplotlib.org/mpl_examples/color/colormaps_reference_04.png
        https://matplotlib.org/users/dflt_style_changes.html#colors-color-cycles-and-color-maps
        https://github.com/vega/vega/wiki/Scales#scale-range-literals
        https://www.mathworks.com/help/matlab/graphics_transition/why-are-plot-lines-different-colors.html

    Parameters
    ----------
    N : int or ``None``
        Number of qualitative colors desired. If None, returns all the colors
        in the specified color scheme.
    color_scheme : str or {8.3, 8.4}
        Color scheme specifier. Valid specifiers are:
        (1) Matplotlib qualitative color map names:
            'Pastel1'
            'Pastel2'
            'Paired'
            'Accent'
            'Dark2'
            'Set1'
            'Set2'
            'Set3'
            'tab10'
            'tab20'
            'tab20b'
            'tab20c'
            (https://matplotlib.org/mpl_examples/color/colormaps_reference_04.png)
        (2) 'tab10_muted':
            A set of 10 colors that are the muted version of "tab10"
        (3) '8.3' and '8.4': old and new MATLAB color scheme
            Old: https://www.mathworks.com/help/matlab/graphics_transition/transition_colororder_old.png
            New: https://www.mathworks.com/help/matlab/graphics_transition/transition_colororder.png
        (4) 'rgbcmyk': old default Matplotlib color palette (v1.5 and earlier)
        (5) 'bw' (or 'bw3'), 'bw4', and 'bw5'
            Black-and-white (grayscale colors in 3, 4, and 5 levels)

    Returns
    -------
    colors : list<list<float>>, list<str>
        A list of colors (as RGB, or color name, or hex)
    '''
    nr_c = {
        'Pastel1': 9,  # number of qualitative colors in each color map
        'Pastel2': 8,
        'Paired': 12,
        'Accent': 8,
        'Dark2': 8,
        'Set1': 9,
        'Set2': 8,
        'Set3': 12,
        'tab10': 10,
        'tab20': 20,
        'tab20b': 20,
        'tab20c': 20,
    }

    qcm_names = list(nr_c.keys())  # valid names of qualititative color maps
    qcm_names_lower = [
        'pastel1','pastel2','paired','accent','dark2','set1',
        'set2','set3',  # lower case version (without 'tab' ones)
    ]

    if not isinstance(color_scheme,(str, int, float, np.number)):
        raise TypeError('`color_scheme` must be str, int, or float.')

    d = {
        'rgbcmyk': ['b','g','r','c','m','y','k'], # matplotlib v1.5 palette
        'bw':  [[0]*3,[0.4]*3,[0.75]*3], # black and white: 3 levels
        'bw3': [[0]*3,[0.4]*3,[0.75]*3], # black and white: 3 levels
        'bw4': [[0]*3,[0.25]*3,[0.5]*3,[0.75]*3],  # b and w, 4 levels
        'bw5': [[0]*3,[0.15]*3,[0.3]*3,[0.5]*3,[0.7]*3],  # b and w, 5 levels
        'tab10': [
            '#1f77b4','#ff7f0e','#2ca02c','#d62728',  # old Tableau palette
            '#9467bd', '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
        ],
        '8.3': [
            [0, 0, 1.0000],  # blue (MATLAB ver 8.3 (R2014a) or earlier)
            [0, 0.5000, 0],  # green
            [1.0000, 0, 0],  # red
            [0, 0.7500, 0.7500],  # cyan
            [0.7500, 0, 0.7500],  # magenta
            [0.7500, 0.7500, 0],  # dark yellow
            [0.2500, 0.2500, 0.2500],  # dark gray
        ],
        '8.4': [
            [0.0000, 0.4470, 0.7410],  # MATLAB ver 8.4 (R2014b) or later
            [0.8500, 0.3250, 0.0980],
            [0.9290, 0.6940, 0.1250],
            [0.4940, 0.1840, 0.5560],
            [0.4660, 0.6740, 0.1880],
            [0.3010, 0.7450, 0.9330],
            [0.6350, 0.0780, 0.1840],
        ],
    }

    if color_scheme in d:
        palette = d[color_scheme]
    else:
        if color_scheme in qcm_names:
            c_s = color_scheme  # short hand [Note: no wrap-around behavior in mpl.cm functions]
            rgba = eval('mpl.cm.%s(range(%d))' % (c_s, nr_c[c_s]))  # e.g., mpl.cm.Set1(range(10))
            palette = [list(_)[:3] for _ in rgba]  # remove alpha value from each sub-list
        elif color_scheme in qcm_names_lower:
            c_s = color_scheme.title()  # first letter upper case
            rgba = eval('mpl.cm.%s(range(%d))' % (c_s, nr_c[c_s]))
            palette = [list(_)[:3] for _ in rgba]
        elif color_scheme == 'tab10_muted':
            rgba_tmp = mpl.cm.tab20(range(nr_c['tab20']))
            palette_tmp = [list(_)[:3] for _ in rgba_tmp]
            palette = palette_tmp[1::2]
        else:
            raise ValueError(
                f"You provided an invalid `color_scheme` ('{color_scheme}'). "
                "A valid `color_scheme` must be one of "
                "{'pastel1', 'pastel2', 'paired', 'accent', "
                "'dark2', 'set1', 'set2', 'set3', 'tab10', "
                "'tab10_muted', 'tab20', 'tab20b', 'tab20c', "
                "'rgbcmyk', 'bw', 'bw3', 'bw4', 'bw5', "
                "'8.3', '8.4'}."
            )

    L = len(palette)
    if N is None:
        N = L
    elif not isinstance(N, (int, np.integer)):
        raise TypeError('`N` should be either None or integers.')

    return [palette[i % L] for i in range(N)]  # wrap around 'palette' if N > L

#%%============================================================================
def get_linespecs(
        color_scheme='tab10', n_linestyle=4, range_linewidth=[1,2,3],
        priority='color',
):
    '''
    Return a list of distinguishable line specifications (color, line style,
    and line width combinations).

    Parameters
    ----------
    color_scheme : str or {8.3, 8.4}
        Color scheme specifier. See documentation of ``get_colors()`` for
        valid specifiers.
    n_linestyle : {1, 2, 3, 4}
        Number of different line styles to use. There are only four available
        line stylies in Matplotlib: (1) - (2) -- (3) -. and (4) ..
        For example, if you use 2, you choose only - and --
    range_linewidth : list, numpy.ndarray, or pandas.Series
        The range of different line width values to use.
    priority : {'color', 'linestyle', 'linewidth'}
        Which one of the three line specification aspects (i.e., color, line
        style, or line width) should change first in the resulting list of
        line specifications.

    Returns
    -------
    style_cycle_list : list<dict>
        A list whose every element is a dictionary that looks like this:
            {'color': '#1f77b4', 'ls': '-', 'lw': 1}.
        Each element can then be passed as keyword arguments to
        ``matplotlib.pyplot.plot()`` or other similar functions.

    Example
    -------
    >>> import plot_utils as pu
    >>> import matplotlib.pyplot as plt
    >>> plt.plot([0,1], [0,1], **pu.get_linespecs()[53])
    '''
    import cycler

    colors = get_colors(N=None, color_scheme=color_scheme)
    if n_linestyle in [1,2,3,4]:
        linestyles = ['-', '--', '-.', ':'][:n_linestyle]
    else:
        raise ValueError('`n_linestyle` should be 1, 2, 3, or 4.')

    color_cycle = cycler.cycler(color=colors)
    ls_cycle = cycler.cycler(ls=linestyles)
    lw_cycle = cycler.cycler(lw=range_linewidth)

    if priority == 'color':
        style_cycle = lw_cycle * ls_cycle * color_cycle
    elif priority == 'linestyle':
        style_cycle = lw_cycle * color_cycle * ls_cycle
    elif priority == 'linewidth':
        style_cycle = color_cycle * lw_cycle * ls_cycle

    return list(style_cycle)

#%%============================================================================
def linespecs_demo(line_specs, horizontal_plot=False):
    '''
    Demonstrate line specifications generated by :func:~`get_linespecs()`.

    Parameter
    ---------
    line_spec : list<dict>
        A list of line specifications. It can be the returned value of
        :func:`~get_linespecs()`.
    horizontal_plot : bool
        Whether or not to demonstrate the line specifications in a horizontal
        plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    x = np.arange(0,10,0.05)  # define x and y points to plot
    y = np.sin(x)
    fig_width = 8
    fig_height = len(line_specs) * 0.2
    if horizontal_plot:
        x, y = y, x
        fig_width, fig_height = fig_height, fig_width

    figsize = (fig_width, fig_height)
    fig = plt.figure(figsize=figsize)
    ax =  plt.axes()

    for j, linespec in enumerate(line_specs):
        if horizontal_plot:
            plt.plot(x+j, y, **linespec)
        else:
            plt.plot(x, y-j, **linespec)

    ax.axis('off')  # no coordinate axes box

    return fig, ax
