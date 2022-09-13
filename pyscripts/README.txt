Bart Ouwehand 13-09-2022

The contents of this folder can be used to generate the results of the fully simulated part of the thesis. The code must be run in the following order:

1) gen_fsdata.py - generates sample (signal + noise) and model (only signal) data for the maximum duration specified in config.py

2) duration_mod.py - truncates the long data for all shorter durations specified in config.py

3) matched_filtering.py - calculates the error in amplitude of the verification binaries using a matched filtering method

4) gen_masterplots.py - generates the final plot which can be compared to the semi-analytical model

5) The results can be found in plots/masterplots/


Note that many .py files contain parameters that can be set to True or False to turn on/off the generation of plots or to reuse previous calculations which are saved, therefore decreasing runtime.
