## Dependencies
Python3 and some python3 libraries:
 - numpy (v1.15.0 used)
 - scipy (v1.1.0 used)
 - sklearn (v1.19.1 used)
 - ot (the Optimal Transport library, v0.4.0 used) https://github.com/rflamary/POT
 - matplotlib (to plot, v2.0.2 used)


## Content
file '1_baseExperiment.py':
 - launch all the experiments on the office caltech dataset, and save the results in a .pkl file in the 'results' folder.

file '1_plot.py':
 - process the results stored in the .pkl file created by 1_baseExperiment.py.
 - this file contains four functions, one to plot the results, and 3 others to print code in Latex to generate arrays of the results

folder 'features':
 - contains all the data required for the experiments in the file '1_baseExperiment.py'

folder 'results':
 - contains the results obtained

file 'plotComparisonOT.py':
 - plot a figure to compare the different Optimal Transport algorithms and store the image in the 'results' folder

file 'plotToy.py':
 - plot a figure comparing different example selection and store the image in the 'results' folder
