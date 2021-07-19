Learning-Augmented Dynamic Power Management
===========================================

This repository contains source code accompanying paper *Learning-Augmented
Dynamic Power Management with Multiple States via New Ski Rental Bounds* by
Antonios Antoniadis, Christian Coester, Marek Eliáš, Adam Polak, Bertrand Simon.

Dependencies
------------

You need Python 3 with additional packages: matplotlib, networkx, numpy.
You also need pdflatex in order to produce the plots.

To install them using pip run:

    sudo apt install python3-pip texlive-latex-base
    pip3 install matplotlib networkx numpy


Datasets
--------

We run our experiments on synthetic randomly generated instances. To obtain
them in the data/ folder, run:

    python3 build_datasets.py


Experiments
-----------


To generate the plots present in the paper, simply run the file:

    ./make_plots.sh

That should take about 9 hours on a modern 8-core CPU. You can change -n 10 to
-n 1 in make_plots.sh file to run one instead of ten repetitions of each
experiment and get preliminary results quicker.

The plots are generated in the pdf format, and the raw data is stored in json
files. The results we obtained are stored in the results/ folder.

