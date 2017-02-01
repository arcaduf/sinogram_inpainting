###################################################################
###################################################################
####                                                           ####
####                      SINOGRAM INPAINTING                  ####
####                                                           ####
###################################################################
###################################################################



##  Brief description
This repository contains different algorithms to double the number of views
of a sinogram, such that its reconstruction with filtered backprojection 
siffers from less severe artifacts.



##  Installation
Basic compilers like gcc and g++ are required.
The simplest way to install all the code is to use Anaconda with python-2.7 and to 
add the installation of the python package scipy, scikit-image, Cython and opencv.

On a terminal, just type:
	1) conda create -n sino-inp python=2.7 anaconda
	2) conda install -n sino-inp scipy Cython scikit-image opencv
	3) source activate sino-inp
	4) download the repo and type: python setup.py

If setup.py runs without giving any error all subroutines in C have been installed and
your python version meets all dependencies.



##  Test the package
Go inside the folder "scripts/" and run the tests.
Every time this script creates an image, the script is halted. To run the successive tests
just close the image.

