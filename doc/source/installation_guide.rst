Installation guide
------------------

1. Default method (install the most up-to-date changes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install git+https://github.com/jsh9/python-plot-utilities

2. Install a specific release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install git+https://github.com/jsh9/python-plot-utilities@v0.6.7

3. The portable way
^^^^^^^^^^^^^^^^^^^

Just download this repository, and you can put ``plot_utils.py`` anywhere within your Python search path.

Note
^^^^

If you run into the following issue on Mac OS X (or macOS) when importing ``plot_utils``:

.. code-block:: bash

    RuntimeError: Python is not installed as a framework.
    The Mac OS X backend will not be able to function correctly if Python
    is not installed as a framework.

Then please follow this solution to fix the issue: https://stackoverflow.com/a/21789908/8892243.
