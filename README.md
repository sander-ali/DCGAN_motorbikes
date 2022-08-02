# DCGAN_motorbikes
The repository provides the code for training a DCGAN to generate images.

The code has been tested on Caltech 101 dataset that can be accessed through the following link  
https://data.caltech.edu/records/20086

The repository structure should be as follows:

working directory  
data  
---| 101_objectCategories


    ---| Motorbike  
        ---| image_0001  
        ---| image_0002  
        ---| ...  
        ---| image_0798  
---| dcgan_mbikes  
---| utils  

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
