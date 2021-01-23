# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from distutils.version import LooseVersion

from . import helper as hlp

basemap_error_msg = '\nPlease install Basemap in order to use ' \
                    '`choropleth_map_state()`.\n' \
                    'To install with conda (recommended):\n' \
                    '    >>> conda install basemap\n' \
                    'To install without conda, refer to:\n' \
                    '    https://matplotlib.org/basemap/users/installing.html'

proj_lib_error_msg = 'Due to a bug in conda, the environmental variable ' \
                     '"PROJ_LIB" is not correctly configured. Please ' \
                     'manually add the path to "proj4" as an environmental' \
                     ' variable "PROJ_LIB" on your system.\n\n' \
                     'The path to "proj4" is usually in ' \
                     '<anaconda_path>/pkgs/proj4-<version>/Library/share' \
                     '\n\n' \
                     'To add environmental variables in Windows:\n' \
                     '  Control panel --> System properties --> ' \
                     'Environment variables... \n    --> "User variables for USERNAME"' \
                     '--> New...\n    ' \
                     '--> Use "PROJ_LIB" as variable name, and paste in the path above.\n' \
                     'Then, restart your computer.\n' \
                     '(You only need to do this once.)'

#%%============================================================================
def choropleth_map_state(
        data_per_state, fig=None, ax=None, figsize=(10,7),
        dpi=100, vmin=None, vmax=None, map_title='USA map',
        unit='', cmap='OrRd', fontsize=14, cmap_midpoint=None,
        shapefile_dir=None,
):
    '''
    Generate a choropleth map of the US (including Alaska and Hawaii), on a
    state level.

    According to Wikipedia, a choropleth map is a thematic map in which areas
    are shaded or patterned in proportion to the measurement of the statistical
    variable being displayed on the map, such as population density or
    per-capita income.

    Parameters
    ----------
    data_per_state : dict or pandas.Series or pandas.DataFrame
        Numerical data of each state, to be plotted onto the map.
        Acceptable data types include:
            - pandas Series: Index should be valid state identifiers (i.e.,
                             state full name, abbreviation, or FIPS code)
            - pandas DataFrame: The dataframe can have only one column (with
                                the index being valid state identifiers), two
                                columns (with one of the column named 'state',
                                'State', or 'FIPS_code', and containing state
                                identifiers).
            - dictionary: with keys being valid state identifiers, and values
                          being the numerical values to be visualized
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
    vmin : float
        Minimum value to be shown on the map. If ``vmin`` is larger than the
        actual minimum value in the data, some of the data values will be
        "clipped". This is useful if there are extreme values in the data
        and you do not want those values to complete skew the color
        distribution.
    vmax : float
        Maximum value to be shown on the map. Similar to ``vmin``.
    map_title : str
        Title of the map, to be shown on the top of the map.
    unit : str
        Unit of the numerical (for example, "population per km^2"), to be
        shown on the right side of the color bar.
    cmap : str or matplotlib.colors.Colormap
        Color map name. Suggested names: 'hot_r', 'summer_r', and 'RdYlBu'
        for plotting deviation maps.
    fontsize : float
        Font size of all the texts on the map.
    cmap_midpoint : float
        A numerical value that specifies the "deviation point". For example,
        if your data ranges from -200 to 1000, and you want negative values
        to appear blue-ish, and positive values to appear red-ish, then you
        can set ``cmap_midpoint`` to 0.0. If ``None``, then the "deviation
        point" will be the median value of the data values.
    shapefile_dir : str
        Directory where shape files are stored. Shape files (state level and
        county level) should be organized as follows:
            ``shapefile_dir``/usa_states/st99_d00.(...)
            ``shapefile_dir``/usa_counties/cb_2016_us_county_500k.(...)
        If ``None``, the shapefile directory within this library will be used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.

    References
    ----------
    This function is based partly on an example in the Basemap repository
    (https://github.com/matplotlib/basemap/blob/master/examples/fillstates.py)
    as well as a modification on Stack Overflow
    (https://stackoverflow.com/questions/39742305).
    '''
    try:
        from mpl_toolkits.basemap import Basemap as Basemap
    except ImportError:
        raise ImportError(basemap_error_msg)
    except KeyError as e:
        if e.args[0] == 'PROJ_LIB':
            raise Exception(proj_lib_error_msg)
        else:
            raise e

    import pkg_resources
    from matplotlib.colors import rgb2hex
    from matplotlib.patches import Polygon
    from matplotlib.colorbar import ColorbarBase
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if isinstance(data_per_state, pd.Series):
        data_per_state = data_per_state.to_dict()  # convert to dict
    elif isinstance(data_per_state, pd.DataFrame):
        if data_per_state.shape[1] == 1:  # only one column
            data_per_state = data_per_state.iloc[:,0].to_dict()
        elif data_per_state.shape[1] == 2:  # two columns
            if 'FIPS_code' in data_per_state.columns:
                data_per_state = data_per_state.set_index('FIPS_code')
            elif 'state' in data_per_state.columns:
                data_per_state = data_per_state.set_index('state')
            elif 'State' in data_per_state.columns:
                data_per_state = data_per_state.set_index('State')
            else:
                raise ValueError('`data_per_state` has unrecognized column name.')
            data_per_state = data_per_state.iloc[:,0].to_dict()
        else:  # more than two columns
            raise hlp.DimensionError('`data_per_state` should have only two columns.')
    elif isinstance(data_per_state,dict):
        pass
    else:
        raise TypeError('`data_per_state` should be pandas.Series, '
                        'pandas.DataFrame, or dict.')

    #  if dict keys are state abbreviations such as "AK", "CA", etc.
    if len(list(data_per_state.keys())[0])==2 and list(data_per_state.keys())[0].isalpha():
        data_per_state = _translate_state_abbrev(data_per_state) # convert from 'AK' to 'Alaska'

    #  if dict keys are state FIPS codes such as "01", "45", etc.
    if len(list(data_per_state.keys())[0])==2 and list(data_per_state.keys())[0].isdigit():
        data_per_state = _convert_FIPS_to_state_name(data_per_state) # convert from '01' to 'Alabama'

    data_per_state = _check_all_states(data_per_state)  # see function definition of _check_all_states()

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    # Lambert Conformal map of lower 48 states.
    m = Basemap(
        llcrnrlon=-119, llcrnrlat=20, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
    )

    # Mercator projection, for Alaska and Hawaii
    m_ = Basemap(
        llcrnrlon=-190, llcrnrlat=20, urcrnrlon=-143, urcrnrlat=46,
        projection='merc', lat_ts=20,  # do not change these numbers
    )

    #---------   draw state boundaries  ----------------------------------------
    if shapefile_dir is None:
        shapefile_dir = pkg_resources.resource_filename('plot_utils', 'shapefiles/')
    shp_path_state = os.path.join(shapefile_dir, 'usa_states', 'st99_d00')
    try:
        shp_info = m.readshapefile(
            shp_path_state, 'states', drawbounds=True,
            linewidth=0.45, color='gray',
        )
        shp_info_ = m_.readshapefile(shp_path_state, 'states', drawbounds=False)
    except IOError:
        raise IOError('Shape files not found. Specify the location of the "shapefiles" folder.')

    #-------- choose a color for each state based on population density. -------
    colors={}
    statenames=[]
    cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.nanmin(list(data_per_state.values()))
    if vmax is None:
        vmax = np.nanmax(list(data_per_state.values()))
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        # skip DC and Puerto Rico.
        if statename not in ['District of Columbia','Puerto Rico']:
            data_ = data_per_state[statename]
            if not np.isnan(data_):
                # calling colormap with value between 0 and 1 returns rgba value.
                colors[statename] = cmap(float(data_-vmin)/(vmax-vmin))[:3]
            else:  # if data_ is NaN, set color to light grey, and with hatching pattern
                colors[statename] = None #np.nan#[0.93]*3
        statenames.append(statename)

    #---------  cycle through state names, color each one.  --------------------
    ax = plt.gca() # get current axes instance

    for nshape, seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['Puerto Rico', 'District of Columbia']:
            if colors[statenames[nshape]] == None:
                color = rgb2hex([0.93] * 3)
                poly = Polygon(seg, facecolor=color, edgecolor=[0.4]*3, hatch='\\')
            else:
                color = rgb2hex(colors[statenames[nshape]])
                poly = Polygon(seg, facecolor=color, edgecolor=color)

            ax.add_patch(poly)

    AREA_1 = 0.005  # exclude small Hawaiian islands that are smaller than AREA_1
    AREA_2 = AREA_1 * 30.0  # exclude Alaskan islands that are smaller than AREA_2
    AK_SCALE = 0.19  # scale down Alaska to show as a map inset
    HI_OFFSET_X = -1900000  # X coordinate offset amount to move Hawaii "beneath" Texas
    HI_OFFSET_Y = 250000    # similar to above: Y offset for Hawaii
    AK_OFFSET_X = -250000   # X offset for Alaska (These four values are obtained
    AK_OFFSET_Y = -750000   # via manual trial and error, thus changing them is not recommended.)

    for nshape, shapedict in enumerate(m_.states_info):  # plot Alaska and Hawaii as map insets
        if shapedict['NAME'] in ['Alaska', 'Hawaii']:
            seg = m_.states[int(shapedict['SHAPENUM'] - 1)]
            if shapedict['NAME']=='Hawaii' and float(shapedict['AREA'])>AREA_1:
                seg = [
                    (x + HI_OFFSET_X, y + HI_OFFSET_Y)
                    for x, y in seg
                ]
            elif shapedict['NAME']=='Alaska' and float(shapedict['AREA'])>AREA_2:
                seg = [
                    (x * AK_SCALE + AK_OFFSET_X, y * AK_SCALE + AK_OFFSET_Y)
                    for x, y in seg
                ]

            if colors[statenames[nshape]] == None:
                color = rgb2hex([0.93] * 3)
                poly = Polygon(
                    seg, facecolor=color, edgecolor='gray',
                    linewidth=.45, hatch='\\',
                )
            else:
                color = rgb2hex(colors[statenames[nshape]])
                poly = Polygon(
                    seg, facecolor=color, edgecolor='gray', linewidth=.45,
                )

            ax.add_patch(poly)

    ax.set_title(map_title)

    #---------  Plot bounding boxes for Alaska and Hawaii insets  --------------
    light_gray = [0.8] * 3
    m_.plot(
        np.linspace(170, 177), np.linspace(29, 29), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(177, 180), np.linspace(29, 26), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(180, 180), np.linspace(26, 23), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(-180, -177), np.linspace(23, 20), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(-180, -175), np.linspace(26, 26), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(-175, -171), np.linspace(26, 22), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(-171, -171), np.linspace(22, 20), linewidth=1.,
        color=light_gray, latlon=True,
    )

    #---------   Show color bar  ---------------------------------------
    if cmap_midpoint is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = hlp._MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=cmap_midpoint)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical', label=unit)

    if LooseVersion(mpl.__version__) >= LooseVersion('2.1.0'):
        cb = _adjust_colorbar_tick_labels(
            cb,
            np.nanmax(list(data_per_state.values())) > vmax,
            np.nanmin(list(data_per_state.values())) < vmin,
        )

    #---------   Set overall font size  --------------------------------
    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig, ax  # return figure and axes handles

#%%============================================================================
def choropleth_map_county(
        data_per_county, fig=None, ax=None, figsize=(10,7),
        dpi=100, vmin=None, vmax=None, unit='', cmap='OrRd',
        map_title='USA county map', fontsize=14,
        cmap_midpoint=None, shapefile_dir=None,
    ):
    '''
    Generate a choropleth map of the US (including Alaska and Hawaii), on a
    county level.

    According to Wikipedia, a choropleth map is a thematic map in which areas
    are shaded or patterned in proportion to the measurement of the statistical
    variable being displayed on the map, such as population density or
    per-capita income.

    Parameters
    ----------
    data_per_county : dict or pandas.Series or pandas.DataFrame
        Numerical data of each county, to be plotted onto the map.
        Acceptable data types include:
            - pandas Series: Index should be valid county identifiers (i.e.,
                             5-digit county FIPS codes)
            - pandas DataFrame: The dataframe can have only one column (with
                                the index being valid county identifiers), two
                                columns (with one of the column named 'state',
                                'State', or 'FIPS_code', and containing county
                                identifiers).
            - dictionary: with keys being valid county identifiers, and values
                          being the numerical values to be visualized
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
    vmin : float
        Minimum value to be shown on the map. If ``vmin`` is larger than the
        actual minimum value in the data, some of the data values will be
        "clipped". This is useful if there are extreme values in the data
        and you do not want those values to complete skew the color
        distribution.
    vmax : float
        Maximum value to be shown on the map. Similar to ``vmin``.
    map_title : str
        Title of the map, to be shown on the top of the map.
    unit : str
        Unit of the numerical (for example, "population per km^2"), to be
        shown on the right side of the color bar.
    cmap : str or <matplotlib.colors.Colormap>
        Color map name. Suggested names: 'hot_r', 'summer_r', and 'RdYlBu'
        for plotting deviation maps.
    fontsize : scalar
        Font size of all the texts on the map.
    cmap_midpoint : float
        A numerical value that specifies the "deviation point". For example,
        if your data ranges from -200 to 1000, and you want negative values
        to appear blue-ish, and positive values to appear red-ish, then you
        can set ``cmap_midpoint`` to 0.0. If ``None``, then the "deviation
        point" will be the median value of the data values.
    shapefile_dir : str
        Directory where shape files are stored. Shape files (state level and
        county level) should be organized as follows:
            ``shapefile_dir``/usa_states/st99_d00.(...)
            ``shapefile_dir``/usa_counties/cb_2016_us_county_500k.(...)
        If ``None``, the shapefile directory within this library will be used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.

    References
    ----------
    This function is based partly on an example in the Basemap repository
    (https://github.com/matplotlib/basemap/blob/master/examples/fillstates.py)
    as well as a modification on Stack Overflow
    (https://stackoverflow.com/questions/39742305).
    '''
    try:
        from mpl_toolkits.basemap import Basemap as Basemap
    except ImportError:
        raise ImportError(basemap_error_msg)
    except KeyError as e:
        if e.args[0] == 'PROJ_LIB':
            raise Exception(proj_lib_error_msg)
        else:
            raise e

    import pkg_resources
    from matplotlib.colors import rgb2hex
    from matplotlib.patches import Polygon
    from matplotlib.colorbar import ColorbarBase
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if isinstance(data_per_county, pd.Series):
        data_per_county = data_per_county.to_dict()  # convert to dict
    elif isinstance(data_per_county, pd.DataFrame):
        if data_per_county.shape[1] == 1:  # only one column
            data_per_county = data_per_county.iloc[:,0].to_dict()
        elif data_per_county.shape[1] == 2:  # two columns
            if 'FIPS_code' in data_per_county.columns:
                data_per_county = data_per_county.set_index('FIPS_code')
            else:
                raise ValueError(
                    '`data_per_county` should have a column named "FIPS_code".'
                )
            data_per_county = data_per_county.iloc[:,0].to_dict()
        else:  # more than two columns
            raise hlp.DimensionError('`data_per_county` should have only two columns.')
    elif isinstance(data_per_county,dict):
        pass
    else:
        raise TypeError(
            '`data_per_county` should be pandas.Series, pandas.DataFrame, or dict.'
        )

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    # Lambert Conformal map of lower 48 states.
    m = Basemap(
        llcrnrlon=-119, llcrnrlat=20, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
    )

    # Mercator projection, for Alaska and Hawaii
    m_ = Basemap(
        llcrnrlon=-190, llcrnrlat=20, urcrnrlon=-143, urcrnrlat=46,
        projection='merc', lat_ts=20,  # do not change these numbers
    )

    #---------   draw state and county boundaries  ----------------------------
    if shapefile_dir is None:
        shapefile_dir = pkg_resources.resource_filename('plot_utils', 'shapefiles/')
    shp_path_state = os.path.join(shapefile_dir, 'usa_states', 'st99_d00')
    try:
        shp_info = m.readshapefile(shp_path_state, 'states', drawbounds=True,
                                   linewidth=0.45, color='gray')
        shp_info_ = m_.readshapefile(shp_path_state, 'states', drawbounds=False)
    except IOError:
        raise IOError(
            'Shape files not found. Specify the location of the "shapefiles" folder.'
        )

    cbc = [0.75] * 3  # county boundary color
    cbw = 0.15  # county boundary line width
    shp_path_county = os.path.join(shapefile_dir, 'usa_counties', 'cb_2016_us_county_500k')
    try:
        shp_info_cnty = m.readshapefile(
            shp_path_county, 'counties', drawbounds=True, linewidth=cbw, color=cbc,
        )
        shp_info_cnty_ = m_.readshapefile(
            shp_path_county, 'counties', drawbounds=False,
        )
    except IOError:
        raise IOError(
            'Shape files not found. Specify the location of the "shapefiles" folder.'
        )

    #-------- choose a color for each county based on unemployment rate -------
    colors={}
    county_FIPS_code_list=[]
    cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.nanmin(list(data_per_county.values()))
    if vmax is None:
        vmax = np.nanmax(list(data_per_county.values()))
    for shapedict in m.counties_info:
        county_FIPS_code = shapedict['GEOID']
        if county_FIPS_code in data_per_county.keys():
            data_ = data_per_county[county_FIPS_code]
        else:
            data_ = np.nan

        # calling colormap with value between 0 and 1 returns rgba value.
        if not np.isnan(data_):
            colors[county_FIPS_code] = cmap(float(data_-vmin)/(vmax-vmin))[:3]
        else:
            colors[county_FIPS_code] = None

        county_FIPS_code_list.append(county_FIPS_code)

    #---------  cycle through county names, color each one.  --------------------
    AK_SCALE = 0.19  # scale down Alaska to show as a map inset
    HI_OFFSET_X = -1900000  # X coordinate offset amount to move Hawaii "beneath" Texas
    HI_OFFSET_Y = 250000    # similar to above: Y offset for Hawaii
    AK_OFFSET_X = -250000   # X offset for Alaska (These four values are obtained
    AK_OFFSET_Y = -750000   # via manual trial and error, thus changing them is not recommended.)

    for j, seg in enumerate(m.counties):  # for 48 lower states
        shapedict = m.counties_info[j]  # query shape dict at j-th position
        if shapedict['STATEFP'] not in ['02','15']:  # not Alaska or Hawaii
            if colors[county_FIPS_code_list[j]] == None:
                color = rgb2hex([0.93] * 3)
                poly = Polygon(seg, facecolor=color, edgecolor=color)
            else:
                color = rgb2hex(colors[county_FIPS_code_list[j]])
                poly = Polygon(seg, facecolor=color, edgecolor=color)
            ax.add_patch(poly)

    for j, seg in enumerate(m_.counties):  # for Alaska and Hawaii
        shapedict = m.counties_info[j]  # query shape dict at j-th position
        if shapedict['STATEFP'] == '02':  # Alaska
            seg = [
                (x * AK_SCALE + AK_OFFSET_X, y * AK_SCALE + AK_OFFSET_Y)
                for x, y in seg
            ]
            if colors[county_FIPS_code_list[j]] == None:
                color = rgb2hex([0.93]*3)
                poly = Polygon(seg, facecolor=color, edgecolor=cbc, lw=cbw)
            else:
                color = rgb2hex(colors[county_FIPS_code_list[j]])
                poly = Polygon(seg, facecolor=color, edgecolor=cbc, lw=cbw)
            ax.add_patch(poly)
        if shapedict['STATEFP'] == '15':  # Hawaii
            seg = [(x + HI_OFFSET_X, y + HI_OFFSET_Y) for x, y in seg]
            if colors[county_FIPS_code_list[j]] == None:
                color = rgb2hex([0.93] * 3)
                poly = Polygon(seg, facecolor=color, edgecolor=cbc, lw=cbw)
            else:
                color = rgb2hex(colors[county_FIPS_code_list[j]])
                poly = Polygon(seg, facecolor=color, edgecolor=cbc, lw=cbw)
            ax.add_patch(poly)

    ax.set_title(map_title)

    #------------  Plot bounding boxes for Alaska and Hawaii insets  --------------
    light_gray = [0.8] * 3
    m_.plot(
        np.linspace(170, 177), np.linspace(29, 29), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(177, 180), np.linspace(29, 26), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(180, 180), np.linspace(26, 23), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(-180, -177), np.linspace(23, 20), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(-180, -175), np.linspace(26, 26), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(-175, -171), np.linspace(26, 22), linewidth=1.,
        color=light_gray, latlon=True,
    )
    m_.plot(
        np.linspace(-171, -171), np.linspace(22, 20), linewidth=1.,
        color=light_gray, latlon=True,
    )

    #------------   Show color bar   ---------------------------------------
    if cmap_midpoint is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = hlp._MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=cmap_midpoint)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cb = ColorbarBase(cax,cmap=cmap,norm=norm,orientation='vertical',label=unit)

    if LooseVersion(mpl.__version__) >= LooseVersion('2.1.0'):
        cb = _adjust_colorbar_tick_labels(
            cb,
            np.nanmax(list(data_per_county.values())) > vmax,
            np.nanmin(list(data_per_county.values())) < vmin,
        )

    #------------   Set overall font size  --------------------------------
    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig, ax  # return figure and axes handles

#%%============================================================================
def _adjust_colorbar_tick_labels(colorbar_obj, adjust_top=True, adjust_bottom=True):
    '''
    Given a colorbar object (colorbar_obj), change the text of the top (and/or
    bottom) tick label text.

    For example, the top tick label of the color bar is originally "1000", then
    this function change it to ">1000", to represent the cases where the colors
    limits are manually clipped at a certain level (useful for cases with
    extreme values in only some limited locations in the color map).

    Similarly, this function adjusts the lower limit.
    For example, the bottom tick label is originally "0", then this function
    changes it to "<0".

    The second and third parameters control whether or not this function adjusts
    top/bottom labels, and which one(s) to adjust.

    Note: get_ticks() only exists in matplotlib version 2.1.0+, and this function
          does not check for matplotlib version. Use with caution.
    '''
    cbar_ticks = colorbar_obj.get_ticks()  # get_ticks() is only added in ver 2.1.0
    new_ticks = [str(int(a)) if int(a)==a else str(a) for a in cbar_ticks]  # convert to int if possible

    if (adjust_top == True) and (adjust_bottom == True):
        new_ticks[-1] = '>' + new_ticks[-1]   # adjust_top and adjust_bottom may
        new_ticks[0] = '<' + new_ticks[0]     # be numpy.bool_ type, which is
    elif adjust_top == True:                  # different from Python bool type!
        new_ticks[-1] = '>' + new_ticks[-1]   # Thus 'adjust_top == True' is used
    elif adjust_bottom == True:               # here, instead of 'adjust_top is True'.
        new_ticks[0] = '<' + new_ticks[0]
    else:
        pass

    colorbar_obj.ax.set_yticklabels(new_ticks)

    return colorbar_obj

#%%============================================================================
def _convert_FIPS_to_state_name(dict1):
    '''
    Convert state FIPS codes such as '01' and '45' into full state names.

    Parameter
    ---------
    dict1 : dict
        A dictionary whose keys are 2-digit FIPS codes of state names.

    Returns
    -------
    dict3 : dict
        A dictionary whose keys are state abbreviations. Its values of each
        state come from ``dict``.
    '''
    assert(isinstance(dict1, dict))

    fips2state = {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
        "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
        "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
        "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
        "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
        "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
        "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
        "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
        "56": "WY",  # dictionary mapping FIPS code to state abbreviation
    }

    dict2 = {}  # create empty dict
    for FIPS_code in dict1:
        new_state_name = fips2state[FIPS_code]  # convert state name
        dict2.update({new_state_name: dict1[FIPS_code]})

    dict3 = _translate_state_abbrev(dict2, abbrev_to_full=True)

    return dict3

#%%============================================================================
def _translate_state_abbrev(dict1, abbrev_to_full=True):
    '''
    Convert state full names into state abbreviations, or the other way.
    Overseas territories (except Puerto Rico) cannot be converted.

    Robustness is not guaranteed: if invalide state names (full or abbreviated)
    exist in dict1, a KeyError will be raised.

    Parameters
    ----------
    dict1 : dict
        A mapping between state name and some data, e.g., {'AK': 1, 'AL': 2, ...}
    abbrev_to_full : bool
        If ``True``, translate {'AK': 1, 'AL': 2, ...} into
        {'Alaska': 1, 'Alabama': 2, ...}. If ``False``, the opposite way.

    Returns
    -------
    dict2 : dict
        The converted dictionary
    '''
    assert(isinstance(dict1, dict))

    if abbrev_to_full is True:
        translation = {
            'AK': 'Alaska',
            'AL': 'Alabama',
            'AR': 'Arkansas',
            'AS': 'American Samoa',
            'AZ': 'Arizona',
            'CA': 'California',
            'CO': 'Colorado',
            'CT': 'Connecticut',
            'DC': 'District of Columbia',
            'DE': 'Delaware',
            'FL': 'Florida',
            'GA': 'Georgia',
            'GU': 'Guam',
            'HI': 'Hawaii',
            'IA': 'Iowa',
            'ID': 'Idaho',
            'IL': 'Illinois',
            'IN': 'Indiana',
            'KS': 'Kansas',
            'KY': 'Kentucky',
            'LA': 'Louisiana',
            'MA': 'Massachusetts',
            'MD': 'Maryland',
            'ME': 'Maine',
            'MI': 'Michigan',
            'MN': 'Minnesota',
            'MO': 'Missouri',
            'MP': 'Northern Mariana Islands',
            'MS': 'Mississippi',
            'MT': 'Montana',
            'NA': 'National',
            'NC': 'North Carolina',
            'ND': 'North Dakota',
            'NE': 'Nebraska',
            'NH': 'New Hampshire',
            'NJ': 'New Jersey',
            'NM': 'New Mexico',
            'NV': 'Nevada',
            'NY': 'New York',
            'OH': 'Ohio',
            'OK': 'Oklahoma',
            'OR': 'Oregon',
            'PA': 'Pennsylvania',
            'PR': 'Puerto Rico',
            'RI': 'Rhode Island',
            'SC': 'South Carolina',
            'SD': 'South Dakota',
            'TN': 'Tennessee',
            'TX': 'Texas',
            'UT': 'Utah',
            'VA': 'Virginia',
            'VI': 'Virgin Islands',
            'VT': 'Vermont',
            'WA': 'Washington',
            'WI': 'Wisconsin',
            'WV': 'West Virginia',
            'WY': 'Wyoming'
        }
    else:
        translation = {
            'Alabama': 'AL',
            'Alaska': 'AK',
            'Arizona': 'AZ',
            'Arkansas': 'AR',
            'California': 'CA',
            'Colorado': 'CO',
            'Connecticut': 'CT',
            'Delaware': 'DE',
            'District of Columbia': 'DC',
            'Florida': 'FL',
            'Georgia': 'GA',
            'Hawaii': 'HI',
            'Idaho': 'ID',
            'Illinois': 'IL',
            'Indiana': 'IN',
            'Iowa': 'IA',
            'Kansas': 'KS',
            'Kentucky': 'KY',
            'Louisiana': 'LA',
            'Maine': 'ME',
            'Maryland': 'MD',
            'Massachusetts': 'MA',
            'Michigan': 'MI',
            'Minnesota': 'MN',
            'Mississippi': 'MS',
            'Missouri': 'MO',
            'Montana': 'MT',
            'Nebraska': 'NE',
            'Nevada': 'NV',
            'New Hampshire': 'NH',
            'New Jersey': 'NJ',
            'New Mexico': 'NM',
            'New York': 'NY',
            'North Carolina': 'NC',
            'North Dakota': 'ND',
            'Ohio': 'OH',
            'Oklahoma': 'OK',
            'Oregon': 'OR',
            'Pennsylvania': 'PA',
            'Puerto Rico': 'PR',
            'Rhode Island': 'RI',
            'South Carolina': 'SC',
            'South Dakota': 'SD',
            'Tennessee': 'TN',
            'Texas': 'TX',
            'Utah': 'UT',
            'Vermont': 'VT',
            'Virginia': 'VA',
            'Washington': 'WA',
            'West Virginia': 'WV',
            'Wisconsin': 'WI',
            'Wyoming': 'WY',
        }

    dict2 = {}
    for state_name in dict1:
        new_state_name = translation[state_name]  # convert state name
        dict2.update({new_state_name: dict1[state_name]})

    return dict2

#%%============================================================================
def _check_all_states(dict1):
    '''
    Check whether dict1 has all 50 states of USA as well as District of
    Columbia. If not, append missing state(s) to the dictionary and assign
    np.nan value as its value.

    The state names of dict1 must be full names.
    '''
    assert(type(dict1) == dict)

    full_state_list = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
        'Connecticut', 'Delaware', 'District of Columbia', 'Florida',
        'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas',
        'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
        'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana',
        'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
        'New York', 'North Carolina', 'North Dakota', 'Ohio',
        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
        'South Carolina', 'South Dakota', 'Tennessee', 'Texas','Utah',
        'Vermont', 'Virginia', 'Washington', 'West Virginia',
        'Wisconsin', 'Wyoming',
    ]

    if dict1.keys() != set(full_state_list):
        dict2 = {}
        for state in full_state_list:
            if state in dict1:
                dict2[state] = dict1[state]
            else:
                print('%s data missing (replaced with NaN).'%state)
                dict2[state] = np.nan
    else:
        dict2 = dict1

    return dict2

