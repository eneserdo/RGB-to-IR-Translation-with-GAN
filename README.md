# RGB-to-IR-Translation-with-GAN

CURRENTLY IMPLEMENTING! THIS IS AN EXPERIMENTAL MODEL SO NOT YET READY TO USE!

I designed and implemented a generative adversarial network (GAN) to translate optical (RGB) image to infrared (IR) image. 

This project was done within the scope of my graduation project.

Check out the [project webpage](https://eneserdo.github.io/RGB-to-IR-Translation-with-GAN/) for details

## Requirements

* Python 3.8
* Pytorch 1.6 or higher
* [keras-segmentation](https://github.com/divamgupta/image-segmentation-keras) if you want to use image segmentations 
* Opencv 
* tqdm
* scikit-image

# Usage

Please use the ```-h ```command without further ado. This code designed for FLIR thermal image dataset. If you will train with different dataset, you must modify the utils/dataset.py

## Train

The dataset directory must have two folder named "rgb", "ir, and also optionally "segment".

dataset  
├───ir  
├───rgb  
└───segment  
```
python train.py -i $dataset_directory
```

Note that for resumption of the training, use the following command

```
python train.py -i $dataset_directory -ce $epoch_number_to_continue
```



## Test

```
python test.py -ce $epoch_number_of_last_saved_model -i $testset_dir
```

## Resize

If you are using FLIR thermal image dataset, keep in mind that images in the dataset have different sizes, and they are unregistered.
To deal with this problem, you can use the following command to make the images registered (approximately). Output will be 640x512

```
python img_resize.py -rgb $RGB_image_dir -ir $IR_image_dir
```
