# RGB-to-IR-Translation-with-GAN

Currently implementing...

I designed and implemented a GAN network to translate RGB image to IR image. 

This project was done within the scope of my graduation project.

Check out the [project webpage](https://eneserdo.github.io/RGB-to-IR-Translation-with-GAN/) for details

## Environment

1. Python 3.8
2. Pytorch 1.6
3. Opencv 

# Usage

## Train

```
python train.py -i $dataset_directory
```

Note that dataset must have two folder named "rgb" and "ir"

## Test

```
python test.py -ce $Epoch_number_of_last_saved_model -i $testset_dir
```

## Resize

If you are using FLIR thermal image dataset, keep in mind that images in the dataset have different sizes, and they are unregistered.
To deal with this problem, you can use the following command to make the images registered (approximately). 

```
python img_resize.py -rgb $RGB_image_dir -ir $IR_image_dir
```
