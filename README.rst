viscm
=====

This is a little tool for analyzing colormaps and creating new colormaps.

Downloads:
  * https://pypi.python.org/pypi/viscm/
  * https://anaconda.org/conda-forge/viscm/

Code and bug tracker:
  https://github.com/matplotlib/viscm

Contact:
  Nathaniel J. Smith <njs@pobox.com> and Stéfan van der Walt <stefanv@berkeley.edu>

Dependencies:
  * Python 3.10+
  * `colorspacious <https://pypi.python.org/pypi/colorspacious>`_ 1.1+
  * Matplotlib 3.5+
  * NumPy 1.22+
  * SciPy 1.8+
  * one of PyQt6, PySide6

License:
  MIT, see `LICENSE <LICENSE>`__ for details.


Installation
------------

This is a GUI application, and requires Qt Python bindings.
They can be provided by PyQt (GPL) or PySide (LGPL)::

  $ pip install viscm[PySide]

...or::

  $ pip install viscm[PyQt]


Usage
-----

::

  $ viscm view jet
  $ viscm edit

There is some information available about how to interpret the
resulting visualizations and use the editor tool `on this website
<https://bids.github.io/colormap/>`_.


Reproducing viridis
^^^^^^^^^^^^^^^^^^^

Load `viridis AKA option_d.py <https://github.com/BIDS/colormap/>`__ using::

  viscm --uniform-space buggy-CAM02-UCS -m Bezier edit /tmp/option_d.py

Note that there was a small bug in the assumed sRGB viewing conditions
while designing viridis. It does not affect the outcome by much. Also
see :code:`viscm --help`.
