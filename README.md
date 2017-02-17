SINOGRAM INPAINTING
===================



##  Brief description
This repository contains different algorithms to double the number of views
of a view-undersampled sinogram, such that its reconstruction with filtered backprojection 
suffers from less severe artifacts.



##  Installation
Basic compilers like gcc and g++ and the FFTW library are required.
The simplest way to use the code is with an Anaconda environment equipped with
python-2.7, scipy, scikit-image, Cython and opencv.

Procedure:
	
1. Create the Anaconda environment (if not already existing): `conda create -n sino-inp python=2.7 anaconda`.
2. Install necessary Python packages: `conda install -n sino-inp scipy Cython scikit-image opencv`.
3. Activate environment: `source activate sino-inp`.
4. Download the repository: `git clone git@github.com:arcaduf/sinogram_inpainting.git`.
5. Install the subroutines in C: `python setup.py`.

If `setup.py` runs without giving any error all subroutines in C have been installed and
your python version meets all dependencies.

If you run `python setup.py 1` (you can use any other character than 1), the 
all executables, temporary and build folders are deleted, the test data are 
placed in .zip files. In this way, the repository is restored to its original
status, right after the download.


##  Test the package
Go inside the folder "scripts/" and run the tests.

When a plot is produced during the execution of a test, the script is halted until
the plot window is manually closed.