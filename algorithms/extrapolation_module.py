########################################################################
########################################################################
####                                                                #### 
####                        Extrapolation algorithms                ####
####                                                                ####
####     Author: Filippo Arcadu, arcusfil@gmail.com, 30/08/2015     ####  
####                                                                ####
########################################################################
########################################################################




####  PYTHON MODULES
from __future__ import division , print_function
import numpy as np
from scipy import interpolate as intp
from scipy import fftpack as fp
import sys
import time
from operator import itemgetter
from itertools import groupby
import cv2




####  MY PYTHON MODULES
sys.path.append( '../common/myimage/' )
import my_image_display as dis




####  MY FORMAT VARIABLES
mycomplex = np.complex64
myfloat   = np.float32
myint     = np.int16 




###########################################################
###########################################################
####                                                   #### 
####      AUTOMATIC DETERMINATION OF THE BANDWIDTH     ####
####                                                   ####
###########################################################
###########################################################

def get_bandwidth( f , thr=0.1 ):
    n  = int( 0.5 * len( f ) )
    ft = np.log10( np.abs( np.fft.fft( f ) )[:n] )
    fm = np.max( ft )
    ii = np.argwhere( ft/fm < thr ).reshape( -1 )

    if len( ii ) == 0:
        thr += 0.1
        while len( ii ) == 0:
            ii = np.argwhere( ft/fm < thr ).reshape( -1 ) 
            thr += 0.1 

    bins = []
    #for k, g in groupby( enumerate( ii ), lambda(i,x):i-x ):
    #    bins.append( map( itemgetter(1) , g ) )
    print( 'bins = ' , bins )
    if len( bins ) > 1:
        ind = bins[-1][0]
    else:
        ind = bins[0][0]

    print( 'ii: ' , ind )
    bd = myfloat( ind ) / myfloat( n ) * 100
    print( 'Bandwidth: ' , bd )
    return bd




###########################################################
###########################################################
####                                                   #### 
####               PAPOULIS-GERCHBERG METHOD           ####
####                                                   ####
###########################################################
###########################################################

def papoulis_gerchberg_1d( signal , tracer , bd=None , niter=50 , eps=1e-10 ):
    ##  Get traced and non-traced elements
    ib       = np.argwhere( signal == tracer )
    ig      = np.argwhere( signal != tracer )
    reco       = signal.copy()
    reco[ib] = 0.0


    ##  Bandlimitation setting
    if bd is None:
        bd = get_bandwidth( signal )
    n     = len( signal ) 
    w     = bd * 0.5 / 100.0
    freq  = np.fft.fftfreq( n ) 
    ifreq = np.argwhere( np.abs( freq ) > w ) 


    ##  Loop
    it   = 0
    err  = 1000

    while it < niter and err > eps:
        if it == 0:
            freco = np.fft.fft( reco )
        else:
            freco[:] = np.fft.fft( reco )
        freco[ifreq] = 0.0
        reco_old     = reco.copy()
        reco[:]      = np.fft.ifft( freco )
        reco[ig]     = signal[ig]

        err = np.linalg.norm( reco - reco_old )
        it += 1
        print( '\n    Iteration n. ', it,' ---> || x_{k+1} - x{k}|| = ', err, end='' )

    return np.real( reco )



def papoulis_gerchberg_2d( signal , tracer , bd=None , niter=50 , eps=1e-10 ):
    ##  Get traced and non-traced elements
    ib       = np.argwhere( signal == tracer )
    ig       = np.argwhere( signal != tracer )
    reco     = signal.copy()
    reco[ib[:,0],ib[:,1]] = 0.0


    ##  Bandlimitation setting
    if bd is None:
        bd = get_bandwidth( signal )
    m , n = signal.shape
    w     = bd * 0.5 / 100.0
    freq1 = np.fft.fftfreq( m )
    freq2 = np.fft.fftfreq( n )
    freq1 , freq2 = np.meshgrid( freq2 , freq1 )
    ifreq = np.argwhere( np.sqrt( freq1**2 + freq2**2 ) > w ) 


    ##  Loop
    it   = 0
    err  = 1000

    while it < niter and err > eps:
        if it == 0:
            freco = np.fft.fft2( reco )
        else:
            freco[:] = np.fft.fft2( reco )
        freco[ifreq[:,0],ifreq[:,1]] = 0.0
        reco_old = reco.copy()
        reco[:]  = np.real( np.fft.ifft2( freco ) )
        reco[ig[:,0],ig[:,1]] = signal[ig[:,0],ig[:,1]]

        err = np.linalg.norm( reco - reco_old )
        it += 1
        print( '\n    Iteration n. ', it,' ---> || x_{k+1} - x{k}|| = ', err, end='' )

    return np.real( reco ) 




###########################################################
###########################################################
####                                                   #### 
####                  ZERO-PADDING METHOD              ####
####                                                   ####
###########################################################
###########################################################

##  The variable 'mode' is used when the input is a 2D signal.
##  mode='a' ---> zero-padding applied to the FFT-2 of the signal
##  mode='r' ---> zero-padding applied to the FFT-1 along the rows of the signal
##  mode='c' ---> zero-padding applied to the FFT-1 along the columns of the signal 

def zero_padding( signal , mode='c'):
    ##  Dealing with 1D signals
    if signal.ndim == 1:
        n = len( signal );  nh = myint( 0.5 * n )
        fsignal = np.fft.fft( signal )
        fsignal_pad = np.concatenate( [ fsignal[:nh] , 
                                        np.zeros( n ) ,
                                        fsignal[nh:] ] ,
                                        axis=0 )
        reco = 2 * np.real( np.fft.ifft( fsignal_pad ) )



    ##  Dealing with 2D signals
    elif signal.ndim == 2:
        ##  Check the necessary dimensions are even 
        nr , nc = signal.shape
       
        ##  Zero-padding along rows and columns
        if mode == 'a':
            Op_r = np.zeros( ( nc , 2 * nc ) )
            ii = np.arange( int( nc * 0.5 ) )
            Op_r[ii,ii] = 1.0;  Op_r[ nc  -1 - ii , 2 * nc - 1 - ii ] = 1.0  

            Op_c = np.zeros( ( 2 * nr , nr ) )
            ii = np.arange( int( nr * 0.5 ) )
            Op_c[ii,ii] = 1.0;  Op_c[ 2 * nr - 1 - ii , nr - 1 - ii ] = 1.0   

            fsignal = np.fft.fft2( signal )
            fsignal = np.dot( fsignal , Op_r ) 
            fsignal_pad = np.dot( Op_c , fsignal )
            reco = 4 * np.real( np.fft.ifft2( fsignal_pad ) )   

        ##  Zero-padding along the rows
        elif mode == 'r':
            Op_r = np.zeros( ( nc , 2 * nc ) )
            ii = np.arange( int( nc * 0.5 ) )
            Op_r[ii,ii] = 1.0;  Op_r[ nc - 1 - ii , 2 * nc - 1 - ii ] = 1.0
            
            fsignal = np.fft.fft( signal , axis=1 ) 
            fsignal_pad = np.dot( fsignal , Op_r )
            reco = 2 * np.real( np.fft.ifft( fsignal_pad , axis=1 ) )   

        ##  Zero-padding along the columns
        elif mode == 'c':
            Op_c = np.zeros( ( 2 * nr , nr ) )
            ii = np.arange( int( nr * 0.5 ) )
            Op_c[ii,ii] = 1.0;  Op_c[ 2 * nr - 1 - ii , nr - 1 - ii ] = 1.0

            fsignal = np.fft.fft( signal , axis=0 ) 
            fsignal_pad = np.dot( Op_c , fsignal )
            reco = 2 * np.real( np.fft.ifft( fsignal_pad , axis=0 ) )  
    
    return reco




###########################################################
###########################################################
####                                                   #### 
####             INTERPOLATION IN REAL-SPACE           ####
####                                                   ####
###########################################################
###########################################################

def get_interp_neigh( signal , ib , side=3 ):
    ##  Convert signal to float
    signal = signal.astype( myfloat )


    ##  Get signal size
    nr , nc = signal.shape
    aux = np.zeros( ( nr , nc ) )


    ##  Determine mask for the data to use for interpolation
    i1 = np.max( [ np.min( ib[:,0] - side ) , 0 ] )
    i2 = np.min( [ np.max( ib[:,0] + 1 + side ) , nr ] )
    j1 = np.max( [ np.min( ib[:,1] - side ) , 0 ] ) 
    j2 = np.min( [ np.max( ib[:,1] + 1 + side ) , nc ] )
 
    aux[i1:i2,j1:j2] = 1.0
    aux[ib[:,0],ib[:,1]] = 0.0


    ##  Get rectangular grid with data and unknown fro griddata
    grid_x, grid_y = np.mgrid[i1:i2,j1:j2]


    ##  Get indeces of data to be used for interpolation
    ig = np.argwhere( aux == 1.0 )

    return ig , grid_x , grid_y , [i1,i2,j1,j2]




def interp( signal , tracer , side=0 , itype='cubic' , ref=None ):
    ##  Get coordinates of data and points to interpolate
    ib  = np.argwhere( signal == tracer )
    ig = np.argwhere( signal != tracer ) 

    
    ##  Interpolate 1D signal
    if signal.ndim == 1:
        n = len( signal )
        x = np.arange( n )
        
        ##  Get cubic polynomial fitting the data
        func = intp.InterpolatedUnivariateSpline( x[ig].reshape( -1 ) ,
                                                  signal[ig].reshape( -1 ) ,
                                                  k=3 )

        ##  Get interpolated data
        signal[x[ib]] = func( x[ib].reshape( -1 ) )

    
    ##  Interpolate 2D signal
    if signal.ndim == 2:
        ##  Get signal shape
        nr , nc = signal.shape

        ##  Get neighbourhood of data for interpolation
        if side > 0:
            ig , grid_x , grid_y , ii = get_interp_neigh( signal , ib , side=side )
            aux = intp.griddata( ig , signal[ig[:,0],ig[:,1]] , 
                                 ( grid_x , grid_y ) ,  
                                 method=itype )
            signal[ii[0]:ii[1],ii[2]:ii[3]] = aux
        else:
            grid_x , grid_y = np.mgrid[:nr,:nc]
            signal = intp.griddata( ig , signal[ig[:,0],ig[:,1]] , 
                                    ( grid_x , grid_y ) ,  
                                    method=itype )

    return signal


    
    
###########################################################
###########################################################
####                                                   #### 
####               CONVERT IMAGE TO UINT 8             ####
####                                                   ####
###########################################################
###########################################################

def convert_image_to_uint8( image ):
    maxv = np.max( image )
    minv = np.min( image )

    image_uint8 = ( image - minv ) * 255.0 / ( maxv - minv )
    image_uint8 = image_uint8.astype( np.uint8 )

    return image_uint8 , maxv , minv



def repristine_image( image_uint8 , maxv , minv ):
    image_uint8 = image_uint8.astype( myfloat ) 
    image = image_uint8 * ( maxv - minv ) / 255.0 + minv

    return image




 ###########################################################
###########################################################
####                                                   #### 
####            INPAINTING WITH NAVIER-STOKES          ####
####                                                   ####
###########################################################
###########################################################

def inpainting_navier_stokes( image , marker , radius=2 ):
    ##  Image size
    nrows , ncols = image.shape


    ##  Create binary mask
    mask = np.zeros( ( nrows , ncols ) , dtype=np.uint8 )
    mask[ image == marker ]  = 1
    image[ image == marker ] = 0
    

    ##  Convert image to uint8
    image , maxv , minv = convert_image_to_uint8( image )


    ##  Inpainting
    reco = cv2.inpaint( image , mask , radius , cv2.INPAINT_NS )


    ##  Repristine image
    reco = repristine_image( reco , maxv , minv )

    return reco




###########################################################
###########################################################
####                                                   #### 
####                INPAINTING WITH TELEA              ####
####                                                   ####
###########################################################
###########################################################

def inpainting_telea( image , marker , radius=2.0 ):  
    ##  Image size
    nrows , ncols = image.shape


    ##  Create binary mask
    mask = np.zeros( ( nrows , ncols ) , dtype=np.uint8 )
    mask[ image == marker ]  = 1
    image[ image == marker ] = 0
    

    ##  Convert image to uint8
    image , maxv , minv = convert_image_to_uint8( image )


    ##  Inpainting
    reco = cv2.inpaint( image , mask , radius , cv2.INPAINT_TELEA )


    ##  Repristine image
    reco = repristine_image( reco , maxv , minv )

    return reco 
    
    
    
    
###########################################################
###########################################################
####                                                   #### 
####                  Anti-aliasing filter             ####
####                                                   ####
###########################################################
########################################################### 

def anti_alias_filt( sino , op='cubic' , factor=2 , bd=0.7 , radius=2 , reassign=True , plot=False ):
    nang , npix = sino.shape

    tracer = np.min( sino ) - 100
    factor = np.int( factor )
    if factor <= 1:
        factor = 2
    nang_new = nang * factor

    ig = np.arange( 0 , nang_new , factor )
    ib = np.setdiff1d( np.arange( nang_new ) , ig )

    sino_new       = np.zeros( ( nang_new , npix ) , dtype=myfloat )
    sino_new[ig,:] = sino
    sino_new[ib,:] = tracer

    if plot is True:
        dis.plot( sino_new , 'Sinogram to inpaint' )

    if op == 'pg1d':
        for i in range( npix ):
            proj = sino_new[:,i]
            sino_new[:,i] = papoulis_gerchberg_1d( proj , tracer , bd=bd , niter=50 , eps=1e-10 )
            
    elif op == 'pg2d':
        sino_new[:] = papoulis_gerchberg_2d( sino_new , tracer , bd=bd , niter=50 , eps=1e-10 )
        
    elif op == 'zp1d':
        for i in range( npix ):
            proj = sino[:,i]        
            sino_new[:,i] = zero_padding( proj , mode='c')
            
    elif op == 'zp2d':
        sino_new[:] = zero_padding( sino , mode='a')[:,::2]    
        
    elif op == 'cubic':
        for i in range( npix ):
            proj = sino_new[:,i]        
            sino_new[:,i] = interp( proj , tracer , side=0 , itype='cubic' )

    elif op == 'navier':
        sino_new[:] = inpainting_navier_stokes( sino_new , tracer , radius=radius )
        
    elif op == 'telea':
        sino_new[:] = inpainting_telea( sino_new , tracer , radius=radius )        
    
    if reassign is True:
        sino_new[ig,:]  = sino
    
    return sino_new 
