# Dog Segmentation

This is a project for segmenting dog images with mobilenet unet model.

 # Dataset
The dataset is the Oxford-IIIT Pet Dataset, created by Parkhi et al.
The dataset consists of images, their corresponding labels, and pixel-wise masks. 
The masks are basically labels for each pixel. Each pixel is given one of three categories :

    Class 1 : Pixel belonging to the pet.
    Class 2 : Pixel bordering the pet.
    Class 3 : None of the above/ Surrounding pixel.


# Credits
[Tensorflow Tutorials](https://www.tensorflow.org/tutorials/images/segmentation)