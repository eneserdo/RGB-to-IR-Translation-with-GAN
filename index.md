# The Project

CURRENTLY IMPLEMENTING

This project aims to create novel generative adversarial network model that can translate RGB image to the IR image

## The Dataset

I used [FLIR Thermal Dataset](https://www.flir.com/oem/adas/adas-dataset-form/). I resized the dataset to have same size 640x512. 

To improve the result further, I used semantic segmantation of the input image, and concatanate it to the input. For this purpose, I used keras-segmentation module which let us to use pretrained segmentation network.

## The Generator Model

The generator was adopted from style transfer network. Additionally, feature matching loss and perceptual loss was used.

## The Multi-Scale Discriminator Model

I designed generic classification network except that it yields a matrix of values, so it works like a PatchGAN. I downsampled the image and used second discriminator so that first scale discriminator scores the local features and second scale discriminator scores the relatively global featureas. 

## Training Details

I trained the model for a 80 epochs with and without segmented input.
I used Adam optimizer with learning rate of 0.0002 and 0.00004 for generator and discriminator respectively. 

## Result

...
