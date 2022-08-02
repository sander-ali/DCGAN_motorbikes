# DCGAN_motorbikes
The repository provides the code for training a DCGAN to generate images.

The code has been tested on Caltech 101 dataset that can be accessed through the following link  
https://data.caltech.edu/records/20086

The repository structure should be as follows:

working directory  
data  
|__ 101_objectCategories


    |___ Motorbike  
        |___ image_0001  
        |___ image_0002  
        |___ ...  
        |___ image_0798  
|___ dcgan_mbikes  
|___ utils  

The code is based on Tensorflow. Before running the code make sure you install the following packages  
tensorflow  
numpy  
sklearn  
opencv  
matplotlib  
graphviz  

Generated results from the network are displayed below:  
![res1](https://user-images.githubusercontent.com/26203136/182378982-4ea37cb4-2511-4ba2-81fd-4c349715b2b0.png)

For details please refer to the original paper  
[1] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks - https://arxiv.org/abs/1511.06434
