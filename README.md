# GradCAM-pytorch

To activate map for Gradient descent weights in deep learning models



## Available Models
- ResNet-101
- Inception-ResNet-V2
- EfficientNet
- NASNet-A-Large
- DenseNet-161
- DenseNet-169
- ...

## Code Explanation
#### Main Code List:

- gradcam.py 
- gradcam_for_video.py
- models.py
- CLAHE_Aug.py

#### Folder and Code contents

##### Codes

- gradcam.py

  - Image folderlist use making GradCAM images
  - folder name has label
  - GradCAM image name is groundtruth_prediction_probability_imagename.jpg
- gradcam_for_video.py

  - video image frame use making GradCAM images
- models.py
  - selected model definition code
- CLAHE_Aug.py
  - video image frame convert CLAHE image 

##### Folders

- CAM

  - it saves CAM output image 

- Model

  - Saving models to predict probability and output image

- test

  - test set images to predict
  
  - ```buildoutcfg
  ## folder structure
    +-- test
    |   +-- class1
    |           +-- img1.jpg
    |           +-- img2.jpg
    |           +-- img3.jpg
    |   +-- class2
    |   +-- class3
    ```



#### Contents

- This GradCAM  codes read PIL image library
- All I want various models can make GradCAM

#### Examples

![Screenshot from 2020-12-10 14-26-56](https://user-images.githubusercontent.com/26396102/101725366-d9c59280-3af3-11eb-8308-6633c08a20c8.png)



#### How to run 

> all gradcam code should control main function
>
> - python gradcam.py
>- python gradcam_for_video.py



#### Reference

- main refernce
  - https://github.com/kazuto1011
- I modified hyunseok oh codes
  - https://github.com/hyunseokOh



