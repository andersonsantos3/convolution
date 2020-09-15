# Convolution
Given an input image I and a kernel K, this application performs convolution of a filter on an image.

## Getting started

### How to run the application:
In the terminal, clone the project:
> git clone https://github.com/andersonsantos3/convolution.git

After downloading, inside the project folder, run the following command:
> python3 main.py -h

This command will return the positional and optional arguments to run the application. They are:
#### Positional arguments
- input_image: Path to input image.
- kernel: type Gaussian or Sobel. If Gaussian, three filters of different sizes (5x5, 7x7, 11x11) will be applied to the images, generating three results. If Sobel, two filters (3x3) will be applied from the Sobel edge detector (horizontal detector and vertical detector) and the pixel magnitude will be calculated.

#### Optional arguments
- out_path: Path to save the results.

The images folder contained in this project's directory has two images. An example of a command to run the application is:
> python3 main.py images/noisy.jpg Gaussian --out_path ./results

This command will open the image noisy.jpg and apply three Gaussian filters (5x5, 7x7 and 11x11) by means of the convolution operation. To maintain the image size during convolution, zero padding is used. After convolution, the results will be saved and shown on the screen.

If you do not want to save the resulting images, simply do not enter the --out_path parameter.
