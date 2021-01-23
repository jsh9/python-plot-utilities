.. plot_utils documentation master file, created by
   sphinx-quickstart on Fri Apr 19 13:56:10 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

plot_utils documentation
========================

Welcome! This is a Python module that contains some useful data visualization functions.


Installation
------------
Recommended method (in the terminal or command window, execute the following command):

.. code-block:: bash

    pip install git+https://github.com/jsh9/python-plot-utilities@v0.6.7

For other installation alternatives, see the `installation guide <installation_guide.html>`_.


Dependencies
------------

* Python 2.7 or 3.5+
* matplotlib 1.5.0+, or 2.0.0+ (Version 2.1.0+ is strongly recommended.)
* numpy: 1.11.0+
* scipy: 0.19.0+
* pandas: 0.20.0+
* cycler: 0.10.0+
* matplotlib/basemap: 1.0.7 (only if you want to plot the two choropleth maps)
* PIL (only if you want to use the ``trim_img()`` function)


API Documentation
-----------------

1. Visualizing one column of data

    .. toctree::
       :maxdepth: 1

       api_docs/pie_chart
       api_docs/discrete_histogram

2. Visualizing two columns of data

    .. toctree::
       :maxdepth: 1

       api_docs/bin_and_mean
       api_docs/category_means
       api_docs/positive_rate
       api_docs/contingency_table
       api_docs/scatter_plot_two_cols

3. Visualizing multiple columns of data

    .. toctree::
       :maxdepth: 1

       api_docs/3d_histograms
       api_docs/hist_multi
       api_docs/violin_plot
       api_docs/correlation_matrix
       api_docs/missing_values

4. Map plotting

    .. toctree::
       :maxdepth: 1

       api_docs/choropleth_map

5. Time series plotting

    .. toctree::
       :maxdepth: 1

       api_docs/plot_time_series
       api_docs/plot_multiple_timeseries
       api_docs/fill_timeseries

6. Miscellaneous

    .. toctree::
       :maxdepth: 1

       api_docs/get_colors
       api_docs/get_linespecs
       api_docs/linespecs_demo
       api_docs/color_classes
       api_docs/plot_with_bounds
       api_docs/trim_image
       api_docs/pad_image
       api_docs/plot_ranking
       api_docs/visualize_cv_scores

7. Other helper functions

    .. toctree::
       :maxdepth: 1

       api_docs/_convert_FIPS_to_state_name
       api_docs/_translate_state_abbrev
       api_docs/_find_axes_lim


Gallery
-------

See `here <https://github.com/jsh9/python-plot-utilities#gallery>`_.


Examples
--------

Examples are presented as Jupyter notebooks `here <https://github.com/jsh9/python-plot-utilities/tree/master/examples>`_.


Copyright and license
---------------------

Copyright: |copy| 2017-2019, Jian Shi

License: `GPL v3.0 <https://github.com/jsh9/python-plot-utilities/blob/master/LICENSE>`_


GitHub repository
-----------------

https://github.com/jsh9/python-plot-utilities

Bug reports and/or suggestions are welcome!


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. |copy|   unicode:: U+000A9 .. COPYRIGHT SIGN

