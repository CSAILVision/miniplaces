# Sample training code for miniplaces challenge

## Setup

Please download the [MiniPlaces dataset](http://miniplaces.csail.mit.edu/data/data.tar.gz) and untar the data.

    tar -xvf data.tar.gz

You will need to modify the data paths accordingly in `alexnet_train.py` and `alexnet_bn_train.py`.

Then install the following `pip` dependencies:

    pip install h5py
    pip install pillow
    pip install scipy

## Getting started

To run AlexNet training script:

    python alexnet_train.py

To run AlexNet with batch normalization:

    python alexnet_bn_train.py

## Advanced

For faster data loading when training on cluster, preprocess data into .h5 and uncomment the lines in the code to use the h5 data loader instead of the disk data loader. You will need to modify the data paths accordingly in `prepro_data.py`.

    python prepro_data.py

## Acknowledgement
Thanks to [Hang Zhao](http://www.mit.edu/~hangzhao/) for developing this baseline training code.
