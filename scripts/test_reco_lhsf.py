from __future__ import division , print_function
import os
import sys

cwd = os.getcwd()
os.chdir( '../algorithms/' )

command = 'python gridding_adjoint.py -Di ../data/ -i shepp_logan_pix0256_ang0040_sino.tif -f hann -o shepp_logan_pix0256_ang0040_sino_reco.tif -p'
print( command )
os.system( command )

command = 'python gridding_adjoint.py -Di ../data/ -i shepp_logan_pix0256_ang0040_sino.tif -a lhsf -f hann -o shepp_logan_pix0256_ang0040_sino_inpaint_reco.tif -p'
print( command )
os.system( command )
