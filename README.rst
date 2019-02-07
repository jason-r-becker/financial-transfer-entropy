financial-transfer-entropy
==========================

|python-Versions| |LICENSE|

``financial-transfer-entropy`` is a Python package for uncovering comovements
and clusters in financial time series with transfer entropy.
 Adaptive directional networks (i.e., asset graphs) are constructed using
 transfer entropy to threshold edges between each (asset) node. Advanced
 visualizations of both the formulated networks and standard unconditional
 correlations are included.


.. contents:: Table of contents
   :backlinks: top
   :local:

Installation
------------

Install Repo
~~~~~~~~~~~~


From terminal:

.. code:: sh

   git clone https://github.com/jason-r-becker/financial-transfer-entropy.git


Set up venv
~~~~~~~~~~~

Using Anaconda, from terminal:

.. code:: sh

   cd fte/
   conda create -n fte python=3.7
   source activate fte
   pip install -U pip
   pip install -r requirements.txt



Contributions
-------------

|GitHub-Commits| |GitHub-Issues| |GitHub-PRs|

All source code is hosted on `GitHub <https://github.com/jason-r-becker/financial-transfer-entropy>`__.
Contributions are welcome.


LICENSE
-------

Open Source (OSI approved): |LICENSE|


Authors
-------

The main developer(s):

- Jason R Becker (`jrbecker <https://github.com/jason-r-becker>`__)
- Jack St. Clair (`JackStC <https://github.com/JackStC>`__)
- John Hurford (`fruhj <https://github.com/fruhj>`__)
- Jerry Fan

.. |GitHub-Status| image:: https://img.shields.io/github/tag/jason-r-becker/financial-transfer-entropy.svg?maxAge=86400
   :target: https://github.com/jason-r-becker/financial-transfer-entropy/releases
.. |GitHub-Forks| image:: https://img.shields.io/github/forks/jason-r-becker/financial-transfer-entropy.svg
   :target: https://github.com/jason-r-becker/financial-transfer-entropy/network
.. |GitHub-Stars| image:: https://img.shields.io/github/stars/jason-r-becker/financial-transfer-entropy.svg
   :target: https://github.com/jason-r-becker/financial-transfer-entropy/stargazers
.. |GitHub-Commits| image:: https://img.shields.io/github/commit-activity/m/jason-r-becker/financial-transfer-entropy.svg
   :target: https://github.com/jason-r-becker/financial-transfer-entropy/graphs/commit-activity
.. |GitHub-Issues| image:: https://img.shields.io/github/issues-closed/jason-r-becker/financial-transfer-entropy.svg
   :target: https://github.com/jason-r-becker/financial-transfer-entropy/issues
.. |GitHub-PRs| image:: https://img.shields.io/github/issues-pr-closed/jason-r-becker/financial-transfer-entropy.svg
   :target: https://github.com/jason-r-becker/financial-transfer-entropy/pulls
.. |GitHub-Contributions| image:: https://img.shields.io/github/contributors/jason-r-becker/financial-transfer-entropy.svg
   :target: https://github.com/jason-r-becker/financial-transfer-entropy/graphs/contributors
.. |Python-Versions| image:: https://img.shields.io/badge/python-3.7-blue.svg
.. |LICENSE| image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://raw.githubusercontent.com/jason-r-becker/financial-transfer-entropy/master/License.txt
