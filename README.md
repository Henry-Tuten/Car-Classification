# Car Classification

## Overview

This project uses a dataset of car and video gameplay images from kaggle and uses them to train and test a custom CNN to predict whether or not a car is in an image.

The data is organized into two subfolers in order to utilize PyTorch's ImageFolder. Ninety percent of the data is used for training and ten is used for validation.

Each image was resized prior to training.

The repository contains source code for splitting the data, resizing it, and labelling.

## Results

Overall accuracy was 67.4% with the current hyperparameters 
