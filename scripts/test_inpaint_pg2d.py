from __future__ import division , print_function
import os

os.chdir( '../algorithms/' )
command = 'python sinogram_inpainting.py -Di ../data/ -i shepp_logan_pix0256_ang0040_sino.tif -a pg1d -o shepp_logan_pix0256_ang0040_sino_inpaint.tif -b 70 -p'
print( command )
os.system( command )