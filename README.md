# Introduction 
Stitching images and estimate the bending

# Getting Started

##### Required Libraries:
- numpy 1.21.5
- matplotlib 3.5.1
- scipy 1.7.3
- OpenCV-python 4.7.0
- imutils 0.5.4
- python 3.9

## Usage

``` python
>> python StraightnessCalculator.py -i images/wood2 -o images/outputs/wood.png -d false
```

where the arguments are:

&nbsp; **-i**: input folder with images to stitch (required)  
&nbsp; **-o**: output folder where to save the result image (required)  
&nbsp; **-d**: display all the available annotations in the result image (optional)  
&nbsp; **-c**: crop the image (optional and not recommended)  

# Example

![results](https://raw.githubusercontent.com/chacoff/Straightness/master/images/outputs/Figure_1.png)
