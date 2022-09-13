Bart Ouwehand 13-09-2022

Contact info
Bart - b.k.ouwehand@gmail.com
Any questons can be emailed to this adress

The contents of this folder are created for my first master project of the Astronomy Research master at Leiden University. The project was supervised by Elena Rossi and Oliver Jennrich. During the project we tested if we could use the LISA verification binaries to determine the amplitude of the OMS and acceleration noise, the two dominating noise sources of LISA.


Most of the notebooks in this folder were used during the project but ended up not working. The important files are the following:
Analytical_Amps.ipynb - Contains code for semi-analytical model. Used to calculate error in verification binary amplitude & compare with fully simulated code
Phase shift testing.ipynb - The phases of the GW's are different when using the instrument class of lisainstrument vs the from_gws class. This notebook calculates the difference in phase.

The bulk of the simulation code can be found in the pyscripts folder. See README.txt in that folder for furhter instructions.

Below you find a list of packages used for simulations. Note that a modified version of pytdi was used which added a parameter interpolate to the function. If this goes wrong, please remove the call to the interpolate parameter. The code should work again after this. 

Package                       Version   Editable project location
----------------------------- --------- --------------------------------
alabaster                     0.7.12
anyio                         3.5.0
appnope                       0.1.2
argon2-cffi                   21.3.0
argon2-cffi-bindings          21.2.0
astroid                       2.4.2
astropy                       4.2
asttokens                     2.0.5
attrs                         20.3.0
Babel                         2.9.1
backcall                      0.2.0
black                         21.12b0
bleach                        4.1.0
certifi                       2021.10.8
cffi                          1.15.0
charset-normalizer            2.0.9
click                         8.0.3
corner                        2.2.1
cycler                        0.10.0
debugpy                       1.5.1
decorator                     5.1.0
defusedxml                    0.7.1
dill                          0.3.4
docutils                      0.17.1
emcee                         3.1.1
entrypoints                   0.3
executing                     0.8.2
ffmpeg                        1.4
fonttools                     4.28.4
graphviz                      0.19.1
h5py                          3.1.0
healpy                        1.14.0
idna                          3.3
imagesize                     1.3.0
importlib-resources           5.4.0
iniconfig                     1.1.1
ipykernel                     6.4.2
ipython                       7.28.0
ipython-genutils              0.2.0
ipywidgets                    7.6.5
isort                         5.7.0
jedi                          0.18.0
Jinja2                        3.0.3
joblib                        1.1.0
json5                         0.9.6
jsonschema                    4.4.0
jupyter                       1.0.0
jupyter-client                7.0.6
jupyter-console               6.4.0
jupyter-core                  4.9.0
jupyter-server                1.13.4
jupyterlab                    3.2.8
jupyterlab-pygments           0.1.2
jupyterlab-server             2.10.3
jupyterlab-widgets            1.0.2
kiwisolver                    1.3.1
lazy-object-proxy             1.4.3
lisaconstants                 1.1.3
lisaglitch                    1.0
lisagwresponse                1.0.1
lisainstrument                1.0.3
lisanode                      1.2.1     /home/bart/Desktop/lisa/LISANode
lisaorbits                    1.0.2     /home/bart/Desktop/lisa/orbits
llvmlite                      0.38.0
lxml                          4.6.3
markdown-it-py                2.0.0
MarkupSafe                    2.0.1
matplotlib                    3.3.3
matplotlib-inline             0.1.3
mccabe                        0.6.1
mdit-py-plugins               0.3.0
mdurl                         0.1.0
mistune                       0.8.4
multiprocess                  0.70.12.2
mypy                          0.910
mypy-extensions               0.4.3
myst-parser                   0.16.1
nbclassic                     0.3.5
nbclient                      0.5.10
nbconvert                     6.4.0
nbformat                      5.1.3
nest-asyncio                  1.5.1
notebook                      6.4.7
numba                         0.55.1
numpy                         1.19.5
oem                           0.3.2
packaging                     21.0
pandas                        1.4.1
pandocfilters                 1.5.0
parso                         0.8.2
pathspec                      0.9.0
pexpect                       4.8.0
pickleshare                   0.7.5
Pillow                        8.1.0
pip                           22.1.2
pkg_resources                 0.0.0
platformdirs                  2.4.0
pluggy                        0.13.1
prometheus-client             0.12.0
prompt-toolkit                3.0.21
psd                           1.5.2
psutil                        5.9.0
ptyprocess                    0.7.0
pure-eval                     0.2.2
py                            1.9.0
pycparser                     2.21
pyerfa                        1.7.1.1
Pygments                      2.10.0
pylint                        2.6.0
pyparsing                     2.4.7
pyplnoise                     1.3
pyrsistent                    0.18.1
pytdi                         1.1       /home/bart/Desktop/lisa/pytdi
pytest                        6.2.1
python-dateutil               2.8.1
pytz                          2021.3
PyYAML                        6.0
pyzmq                         22.3.0
qtconsole                     5.2.2
QtPy                          2.0.0
requests                      2.26.0
rstcheck                      3.3.1
scikit-learn                  1.0.2
scipy                         1.6.0
seaborn                       0.11.2
Send2Trash                    1.8.0
setuptools                    44.0.0
six                           1.15.0
sklearn                       0.0
sniffio                       1.2.0
snowballstemmer               2.2.0
Sphinx                        4.3.1
sphinx-rtd-theme              1.0.0
sphinxcontrib-applehelp       1.0.2
sphinxcontrib-devhelp         1.0.2
sphinxcontrib-htmlhelp        2.0.0
sphinxcontrib-jsmath          1.0.1
sphinxcontrib-qthelp          1.0.3
sphinxcontrib-serializinghtml 1.1.5
stack-data                    0.1.4
terminado                     0.12.1
testpath                      0.5.0
threadpoolctl                 3.1.0
toml                          0.10.2
tomli                         2.0.0
torch                         1.10.1
tornado                       6.1
tqdm                          4.62.3
traitlets                     5.1.1
typing-extensions             3.10.0.2
urllib3                       1.26.7
wcwidth                       0.2.5
webencodings                  0.5.1
websocket-client              1.2.3
widgetsnbextension            3.5.2
wrapt                         1.12.1
zeus-mcmc                     2.4.1
zipp                          3.7.0

