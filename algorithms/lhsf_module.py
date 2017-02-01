########################################################################
########################################################################
####                                                                #### 
####             Ludwig-Helgason sinogram filters (LWSF)            ####
####                                                                ####
####     Author: Filippo Arcadu, arcusfil@gmail.com, 30/08/2015     ####  
####                                                                ####
########################################################################
########################################################################




####  PYTHON MODULES
from __future__ import division , print_function
import numpy as np
import utils as ut
from scipy import fftpack as fp
from scipy import interpolate as intp
from scipy import sparse as ss
import scipy.sparse.linalg as ssl




####  MY PYTHON MODULES
import my_image_display as dis
import my_image_process as proc





####  MY FORMAT VARIABLES
mycomplex = np.complex64
myfloat   = np.float32
myint     = np.int16




###########################################################
###########################################################
####                                                   #### 
####   EXPLICIT FOURIER MATRIX FOR FOURIER TRANSFORM   ####
####            OF A 2D ARRAY ALONG THE ROWS           ####
####                                                   ####
###########################################################
###########################################################

def ftmx( n1 , n2 ):
    ##  Create combination of all possible couples k1 , k2
    k = np.arange( n2 )
    k1 , k2 = np.meshgrid( k , k )


    ##  Base exponentials
    be = np.exp( -2j * np.pi * k1 * k2 / myfloat( n2 ) )


    ##  Initialize matrix for FT along the rows of a 2D array    
    fx        = np.zeros( ( n2 , n1 * n2 ) , dtype=mycomplex )
    fx[:,:n2] = be
    fxa       = fx.copy()


    ##  Create matrix
    for i in range( n2 , n1 * n2 , n2 ):
        fxa = np.roll( fxa , n2 , axis=1 )
        fx  = np.concatenate( ( fx , fxa ) , axis=0 )

    fx = ss.coo_matrix( fx )

    return fx




###########################################################
###########################################################
####                                                   #### 
####   EXPLICIT FOURIER MATRIX FOR FOURIER TRANSFORM   ####
####           OF A 2D ARRAY ALONG THE COLUMNS         ####
####                                                   ####
###########################################################
###########################################################

def ftmy( n1 , n2 ):
    ##  Create combination of all possible couples k1 , k2
    k = np.arange( n1 )
    k1 , k2 = np.meshgrid( k , k )


    ##  Base exponentials
    be = np.exp( -2j * np.pi * k1 * k2 / myfloat( n1 ) )


    ##  Initialize matrix for FT along the rows of a 2D array    
    fy            = np.zeros( ( n1 * n2 , n1 * n2 ) , dtype=mycomplex )
    fy[::n2,::n2] = be
    fya           = fy.copy()


    ##  Create matrix
    for i in range( n2 - 1 ):
        fya = np.roll( fya , 1 , axis=0 )
        fya = np.roll( fya , 1 , axis=1 ) 
        fy += fya

    fy = 1.0 / np.sqrt( n1 ) * np.fft.fftshift( fy , axes=0 )
    fy = ss.coo_matrix( fy )            

    return fy




###########################################################
###########################################################
####                                                   #### 
####         EXPLICIT MATRIX FOR SINE TRANSFORM        ####
####            OF A 2D ARRAY ALONG THE ROWS           ####
####                                                   ####
###########################################################
###########################################################

def fsmx( n1 , n2 ):
    ##  Create combination of all possible couples k1 , k2
    k = np.arange( n2 )
    k1 , k2 = np.meshgrid( k , k )


    ##  Base elements
    be = 2 * np.sqrt( 1.0 / ( 2 * ( n2 + 1 ) ) ) * np.sin( np.pi * ( k1 + 1 ) * ( k2 + 1 ) / myfloat( n2 + 1 ) )


    ##  Initialize matrix for FT along the rows of a 2D array    
    fx        = np.zeros( ( n2 , n1 * n2 ) , dtype=myfloat )
    fx[:,:n2] = be
    fxa       = fx.copy()


    ##  Create matrix
    for i in range( n2 , n1 * n2 , n2 ):
        fxa = np.roll( fxa , n2 , axis=1 )
        fx  = np.concatenate( ( fx , fxa ) , axis=0 )

    fx = ss.coo_matrix( fx )

    return fx   




###########################################################
###########################################################
####                                                   #### 
####   EXPLICIT MATRIX OF THE FOURIER-CHEBYSHEV MASK   ####
####                                                   ####
###########################################################
###########################################################

def fbmask( n1 , n2 ):
    nh1  = myint( 0.5 * n1 )
    ii   = get_ind_zerout( n2 , nh1 )   
    mask = np.ones( ( n1 , n2 ) , dtype=myint )
    mask[ii[:,0],ii[:,1]] = 0.0
    mask = mask.reshape( -1 )
    M    = np.diag( mask )
    M    = ss.coo_matrix( M ) 

    return M




###########################################################
###########################################################
####                                                   #### 
####             EXTEND SINOGRAM TO [0,2pi)            ####
####                                                   ####
###########################################################
###########################################################

def extend_full_period( s ):
    s_pi    = s.copy()
    s_pi[:] = s_pi[:,::-1]
    s_pi[:] = np.roll( s_pi , 1 , axis=1 ) 
    s_2pi   = np.concatenate( ( s , s_pi ) , axis=0 )    
    return s_2pi




###########################################################
###########################################################
####                                                   #### 
####       EXTEND CHANNELS TO CONNECT DST AND DFT      ####
####                                                   ####
###########################################################
###########################################################

def extend_channels( s ):
    n1 = s.shape[1]
    n2 = 2 * n1 + 2
    s2 = np.zeros( ( s.shape[0] , n2 ) , dtype=myfloat )
    s2[:,1:1+n1] = s
    s2[:,n1+2:] = -s[:,::-1]  
    return s2




###########################################################
###########################################################
####                                                   #### 
####         RE-SAMPLING AT COSINUSOIDAL NODES         ####
####                                                   ####
###########################################################
###########################################################

def get_all_nodes( n ):
    ##  Current equispaced nodes in [-1,+1] 
    x = np.linspace( -1.0 , 1.0 , n )


    ##  Cosinusoidal nodes
    nc = int( np.pi * n / 2.0 )
    xc = np.cos( np.pi * np.arange( 0 , nc ) / myfloat( nc ) )
    xc[:] = xc[::-1]

    return x , xc

                       

def resampling_cosine_nodes( s ):
    ##  Get dimensions
    m , n = s.shape


    ##  Get equispaced and cosinusoidal nodes
    x , xc = get_all_nodes( n )
    nc = len( xc )

    
    ##  Allocate memory for re-sampled projections
    sc = np.zeros( ( m , nc ) , dtype=myfloat )

    for i in range( m ):
        f = intp.interp1d( x , s[i,:] )
        sc[i,:] = f( xc )

    return sc



def resampling_equisp_nodes( q , n ):
    ##  Get dimensions
    m , nc = q.shape


    ##  Get equispaced and cosinusoidal nodes
    x , xc = get_all_nodes( n )
    x = x[1:n-1]

    
    ##  Allocate memory for re-sampled projections
    h = np.zeros( ( m , n ) , dtype=myfloat )

    for i in range( m ):
        f = intp.interp1d( xc , q[i,:] )
        h[i,1:n-1] = f( x )

    return h




###########################################################
###########################################################
####                                                   #### 
####    GET INDICES OF INCONSISTENT FB-COEFFICIENTS    ####
####                                                   ####
###########################################################
###########################################################

def get_ind_zerout( nc , na ):
    k     = np.arange( nc )
    m     = np.arange( -na , na ) 
    k , m = np.meshgrid( k , m )
    ii    = np.argwhere( ( np.abs( m ) > k ) | ( ( np.abs( m ) + k ) % 2 != 0 ) )
    mask  = np.ones( ( 2 * na , nc ) )
    mask[ii[:,0],ii[:,1]] = 0.0
    #dis.plot( mask , 'Mask' )
    return ii




###########################################################
###########################################################
####                                                   #### 
####      Ludwig-Helgason iterative filter (LWIF)      ####
####                     DST version                   ####
####                                                   ####
###########################################################
###########################################################

def lhif_dst( x0 , tracer , niter=100 , eps=1e-20 ):
    ##  Extend sinogram to [0,2pi)
    m0 , n0 = x0.shape
    s_2pi   = extend_full_period( x0 )
    ig      = np.argwhere( s_2pi != tracer )[:,0]
    ib      = np.argwhere( s_2pi == tracer )[:,0]



    ##  Resample each projection at cosinusoidal nodes
    sc = resampling_cosine_nodes( s_2pi )
    sc[ib,:] = tracer
    m1 , n1 = sc.shape
    mh1 = int( m1 * 0.5 )
    nh1 = int( n1 * 0.5 )


    
    ##  Indexes for Fourier-Chebyshev filtering
    ic = get_ind_zerout( n1 , m0 )



    ##  Get traced and non-traced elements
    sx  = sc.copy()
    sx[ib,:] = 0.0



    ##  Loop
    it   = 0
    err  = 1000
    nc   = n1 
    norm = np.sqrt( 1.0 / ( 2.0 * ( nc + 1 ) ) )


    while it < niter and err > eps:
        ##  Fourier-Chebyshev decomposition
        if it == 0:
            c = norm * fp.dst( sx , type=1 , axis=1 )
            b = np.fft.fftshift( np.fft.fft( c , axis=0 ) , axes=0 )             
            sp = sx.copy()
        else:
            c[:] = norm * fp.dst( sx , type=1 , axis=1 )
            b[:] = np.fft.fftshift( np.fft.fft( c , axis=0 ) , axes=0 )
            sp[:] = sx


        ##  Zero-out inconsistent coefficients
        b[ic[:,0],ic[:,1]] = 0.0
        
        
        ##  Reproject to real space
        c[:]  = np.real( np.fft.ifft( np.fft.ifftshift( b , axes=0 ) , axis=0 ) )
        #dis.plot( np.real( c ).reshape( m1 , n1 ) , 'C' )
        if it == 0:
            ci   = np.imag( np.fft.ifft( np.fft.ifftshift( b , axes=0 ) , axis=0 ) ) 
            sx_r = norm * fp.dst( c , type=1 , axis=1 )
            sx_i = norm * fp.dst( ci , type=1 , axis=1 )
        else:
            ci[:]   = np.imag( np.fft.ifft( np.fft.ifftshift( b , axes=0 ) , axis=0 ) )  
            sx_r[:] = norm * fp.dst( c , type=1 , axis=1 )
            sx_i[:] = norm * fp.dst( ci , type=1 , axis=1 )  
        sx[:] = np.sqrt( sx_r**2 + sx_i**2 )


        #dis.plot( np.real( sx ).reshape( m1 , n1 ) , 'Sx before reassign' )

        ##  Re-assign original values
        sx[ig,:] = sc[ig,:]


        ##  Compute error & PSNR
        err = np.linalg.norm( sx - sp )
        it += 1
        
        print( '\n    Iteration n. ', it,' ---> || x_{k+1} - x{k}|| = ', err, end='' )



    ##  Crop original sinogram interval [0,pi) and first channel half (transl. of 1)   
    sx = sx[:mh1,:]



    ##  Interpolate back on equispaced nodes
    x = resampling_equisp_nodes( sx , n0 )
    
    print('\n')

    return x




###########################################################
###########################################################
####                                                   #### 
####       Ludwig-Helgason one-step filter (LWIF)      ####
####                                                   ####
###########################################################
###########################################################

def lhif_onestep( x0 , tracer , factor=2 ):
    ##  Extend sinogram to [0,2pi)
    m0 , n0 = x0.shape
    s_2pi   = extend_full_period( x0 )
    ig      = np.unique( np.argwhere( s_2pi != tracer )[:,0] )
    ib      = np.unique( np.argwhere( s_2pi == tracer )[:,0] )



    ##  Resample each projection at cosinusoidal nodes
    sc = resampling_cosine_nodes( s_2pi )
    sc[ib,:] = tracer
    
    m1 , n1 = sc.shape
    mh1 = int( m1 * 0.5 )
    nh1 = int( n1 * 0.5 )


    
    ##  Indexes for Fourier-Chebyshev filtering
    ic = get_ind_zerout( n1 , m0 )



    ##  Get traced and non-traced elements
    sx  = sc.copy()
    sx[ib,:] = 0.0
    nc   = n1 
    norm = np.sqrt( 1.0 / ( 2.0 * ( nc + 1 ) ) )



    ##  Decomposition and filter
    c = norm * fp.dst( sx , type=1 , axis=1 )
    b = np.fft.fftshift( np.fft.fft( c , axis=0 ) , axes=0 )             
    b[ic[:,0],ic[:,1]] = 0.0
    


    ##  Reconstruction
    c[:] = np.real( np.fft.ifft( np.fft.ifftshift( b , axes=0 ) , axis=0 ) )
    ci   = np.imag( np.fft.ifft( np.fft.ifftshift( b , axes=0 ) , axis=0 ) ) 
    sx_r = norm * fp.dst( c , type=1 , axis=1 )
    sx_i = norm * fp.dst( ci , type=1 , axis=1 )
    sx[:] = np.sqrt( sx_r**2 + sx_i**2 )



    ##  Renormalize
    sx[:] = factor * sx



    ##  Crop original sinogram interval [0,pi) and first channel half (transl. of 1)   
    sx = sx[:mh1,:]



    ##  Interpolate back on equispaced nodes
    x = resampling_equisp_nodes( sx , n0 )

    return x  




###########################################################
###########################################################
####                                                   #### 
####      Ludwig-Helgason iterative filter (LWIF)      ####
####                     DFT version                   ####
####                                                   ####
###########################################################
###########################################################

def lhif_dft( x0 , tracer , niter=100 , eps=1e-20 ):
    ##  Extend sinogram to [0,2pi)
    m0 , n0 = x0.shape
    s_2pi   = extend_full_period( x0 )
    ig      = np.argwhere( s_2pi != tracer )[:,0]
    ib      = np.argwhere( s_2pi == tracer )[:,0] 



    ##  Resample each projection at cosinusoidal nodes
    sc = resampling_cosine_nodes( s_2pi )
    sc[ib,:] = tracer
    m1 , n1 = sc.shape
    mh1 = int( m1 * 0.5 )
    nh1 = int( n1 * 0.5 )


    
    ##  Indexes for Fourier-Chebyshev filtering
    ic = get_ind_zerout( n1 , m0 )



    ##  Extend detector channels to create connection DST <---> DFT
    sc = extend_channels( sc )
    sc[ib,:] = tracer



    ##  Get traced and non-traced elements
    sx  = sc.copy()
    sx[ib,:] = 0.0



    ##  Loop
    it   = 0
    err  = 1000

    while it < niter and err > eps:
        ##  Compute FF2 and save previous iteration
        if it == 0:
            fx = np.fft.fft2( sx )
            sp = sx.copy()
        else:
            fx[:] = np.fft.fft2( sx )
            sp[:] = sx


        ##   DFT-based Fourier Chebyshev filtering
        c = 1.0j * fx[:,1:n1+1]
        b = np.fft.fftshift( c , axes=0 )
        b[ic[:,0],ic[:,1]] = 0.0
        c = np.fft.ifftshift( b , axes=0 )
        fx[:,1:n1+1] = -1.0j * c
        fx[:,2*n1+1:n1+1:-1] = 1.0j * c
        

        ##  Compute IFFT2
        sx[:]  = np.real( np.fft.ifft2( fx ) )


        ##  Re-assign original values
        sx[ig,:] = sc[ig,:]


        ##  Compute error & PSNR
        err = np.linalg.norm( sx - sp )
        it += 1
        
        print( '\n    Iteration n. ', it,' ---> || x_{k+1} - x{k}|| = ', err, end='' )



    ##  Crop original sinogram interval [0,pi) and first channel half (transl. of 1)   
    sx = sx[:mh1,1:n1+1]



    ##  Interpolate back on equispaced nodes
    x = resampling_equisp_nodes( sx , n0 )
    
    print('\n')

    return x 




###########################################################
###########################################################
####                                                   #### 
####                Debugger for lhem                  ####
####                                                   ####
###########################################################
###########################################################

def debug_lhem( x0 , tracer ):
    ##  Extend sinogram to [0,2pi)
    m0 , n0 = x0.shape
    s_2pi   = extend_full_period( x0 )
    ig      = np.argwhere( s_2pi != tracer )[:,0]
    ib      = np.argwhere( s_2pi == tracer )[:,0] 
    #dis.plot( s_2pi , 'S-2pi' )



    ##  Resample each projection at cosinusoidal nodes
    sc = resampling_cosine_nodes( s_2pi )
    sc[ib,:] = tracer
    m1 , n1 = sc.shape
    mh1 = int( m1 * 0.5 )
    nh1 = int( n1 * 0.5 )
    dis.plot( sc , 'SC' )



    ##  Get matrix representation of each operator
    print( 'Creating Fy and Fya ....' )
    Fy = ftmy( m1 , n1 );  Fya = Fy.getH()
    print( 'Creating Sx ....' )  
    Sx = fsmx( m1 , n1 )
    print( 'Creating M ....' )   
    M  = fbmask( m1 , n1 )
    I  = ss.eye( m1 * n1 , m1 * n1 )


    ##  Matrix representation of D, that keeps only the
    ##  extrapolated values
    print( 'Creating D ....' )
    sx  = np.mat( sc.copy().reshape( m1 * n1 , 1 ) )
    ib2 = np.argwhere( sx == tracer )
    D   = np.zeros( ( m1 * n1 , m1 * n1 ) , dtype=myint )
    D[ib2,ib2] = 1.0
    D   = ss.coo_matrix( D )



    ##  Get traced and non-traced elements
    sx  = sc.copy()
    sx[ib,:] = 0.0



    ##  Loop
    it  = 0
    err = 1000
    nc  = n1 
    niter = 50
    eps = 1e-10

    while it < niter and err > eps:
        sx = sx.reshape( m1 * n1 , 1 )

        ##  Fourier-Chebyshev decomposition
        if it == 0:
            c = Sx.dot( sx )  #fp.dst( sx , type=1 , axis=1 )
            b = Fy.dot( c )  #np.fft.fftshift( np.fft.fft( c , axis=0 ) , axes=0 )             
            sp = sx.copy().reshape( m1 , n1 ) 
        else:
            c[:] = Sx.dot( sx )  #fp.dst( sx , type=1 , axis=1 )
            b[:] = Fy.dot( c )   #np.fft.fftshift( np.fft.fft( c , axis=0 ) , axes=0 )
            sp[:] = sx.reshape( m1 , n1 ) 


        ##  Zero-out inconsistent coefficients
        #dis.plot( np.real( b ).reshape( m1 , n1 ) , 'B before' )      
        b[:] = M.dot( b )  #b[ic[:,0],ic[:,1]] = 0.0
        #dis.plot( np.real( b ).reshape( m1 , n1 ) , 'B after' )
        
        
        ##  Reproject to real space
        c = Fya.dot( b )  #c[:]  = np.real( np.fft.ifft( np.fft.ifftshift( b , axes=0 ) , axis=0 ) )
        #dis.plot( np.real( c ).reshape( m1 , n1 ) , 'C' ) 
                             # sx_r[:] = 1.0 / ( 2.0 * ( nc + 1 ) ) * fp.dst( c , type=1 , axis=1 )
                             # sx_i[:] = 1.0 / ( 2.0 * ( nc + 1 ) ) * fp.dst( ci , type=1 , axis=1 ) 
        sx[:] = Sx.dot( c )  # sx[:] = np.sqrt( sx_r**2 + sx_i**2 )

        #dis.plot( np.real( sx ).reshape( m1 , n1 ) , 'Sx before reassign' ) 
        ##  Re-assign original values
        sx = sx.reshape( m1 , n1 )
        sx[ig,:] = sc[ig,:]
        #dis.plot( sx , 'Sx after reassign' )


        ##  Compute error & PSNR
        err = np.linalg.norm( sx - sp )
        it += 1
        
        print( '\n    Iteration n. ', it,' ---> || x_{k+1} - x{k}|| = ', err, end='' )



    ##  Crop original sinogram interval [0,pi) and first channel half (transl. of 1)   
    sx = sx[:mh1,:]



    ##  Interpolate back on equispaced nodes
    x = resampling_equisp_nodes( sx , n0 )
    
    print('\n')

    return x 




###########################################################
###########################################################
####                                                   #### 
####        Ludwig-Helgason extrapolation matrix       ####
####                                                   ####
###########################################################
###########################################################

def lhem( x0 , tracer ):
    ##  Extend sinogram to [0,2pi)
    m0 , n0 = x0.shape
    s_2pi   = extend_full_period( x0 )
    ig      = np.argwhere( s_2pi != tracer )[:,0]
    ib      = np.argwhere( s_2pi == tracer )[:,0] 



    ##  Resample each projection at cosinusoidal nodes
    sc = resampling_cosine_nodes( s_2pi )
    sc[ib,:] = tracer
    m1 , n1 = sc.shape
    mh1 = int( m1 * 0.5 )
    nh1 = int( n1 * 0.5 )  



    ##  Get matrix representation of each operator
    print( 'Creating Fy and Fya ....' )
    Fy = ftmy( m1 , n1 );  Fya = Fy.getH()
    print( 'Creating Sx ....' )  
    Sx = fsmx( m1 , n1 )
    print( 'Creating M ....' )   
    M  = fbmask( m1 , n1 )
    #I  = ss.eye( m1 * n1 , m1 * n1 )
    I = np.eye( m1 * n1 , m1 * n1 )


    ##  Matrix representation of D, that keeps only the
    ##  extrapolated values
    print( 'Creating D ....' )
    sx  = np.mat( sc.copy().reshape( m1 * n1 , 1 ) )
    ib2 = np.argwhere( sx == tracer )
    sx[ib2] = 0.0
    D   = np.zeros( ( m1 * n1 , m1 * n1 ) , dtype=myint )
    D[ib2,ib2] = 1.0
    D   = ss.coo_matrix( D )

    aux = D.dot( sx )
    dis.plot( aux.reshape( m1 , n1 ) , 'Aux' )



    ##  Complete operator
    print( 'Creating H ....' )
    H = D.dot( Fya.dot( Sx.dot( M.dot( Sx.dot( Fy ) ) ) ) )



    ##  Compute limit operator
    print( 'Creating Hl ....' )
    eps = 1e-5
    H  = H.todense()
    Hl = np.linalg.pinv( I - H )
    
    #Hl = np.eye( m1 * n1 , dtype=mycomplex )
    #for i in range( 1 , 50 ):
    #    print( 'i = ' , i )
    #    Hl += np.linalg.matrix_power( H , i )
    
    #for i in range( 1 , 30 ):
    #    print( 'i = ' , i )
    #    sx[:] = np.mat( np.dot( H , sx ) )
    #    sx[ig] = sc[ig]
    #sx = sx.reshape( m1 , n1 )

    eva , eve = np.linalg.eig( H )
    print( 'Max-eigen: ' , np.max( np.abs( eva ) ) )

    H2 = np.dot( np.transpose( np.conjugate( H ) ) , H )
    eva , eve = np.linalg.eig( H2 )
    print( 'Max-eigen: ' , np.max( np.abs( eva ) ) ) 


    
    ##  Filtered sinogram
    print( 'Creating filt. sinogram ....' )
    #sx = np.real( Hl.dot( sx ) ).reshape( m1 , n1 )
    sx = np.real( np.dot( Hl , sx ) ).reshape( m1 , n1 )


    ##  Resample on equispaced nodes
    print( 'Resample on equispaced nodes ....' ) 
    x = resampling_equisp_nodes( sx , n0 )



    ##  Halve the nuber of projections
    x = x[:m0,:]

    return x




###########################################################
###########################################################
####                                                   #### 
####                  Anti-aliasing filter             ####
####                                                   ####
###########################################################
########################################################### 

def anti_alias_filt( sino , op='one-step' , factor=2 , reassign=False ):
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

    if op == 'dft':
        sino_new[:] = lhif_dft( sino_new , tracer , niter=100 )
    elif op == 'dst':
        sino_new[:] = lhif_dst( sino_new , tracer , niter=100 )
    elif op == 'em':
        sino_new[:] = lhem( sino_new , tracer )
    elif op == 'one-step':
        sino_new[:] = lhif_onestep( sino_new , tracer , factor=factor )  
    elif op == 'debug':
        sino_new[:] = debug_lhem( sino_new , tracer )        
    
    if reassign is True:
        sino_new[ig,:]  = sino
    
    return sino_new 




###########################################################
###########################################################
####                                                   #### 
####                  Anti-aliasing filter             ####
####                                                   ####
###########################################################
########################################################### 

def anti_missing_wedge( sino , nang_or ):
    nang , npix = sino.shape
    nang_h   = myint( 0.5 * nang ) 

    tracer = np.min( sino ) - 100    

    all_ind = np.arange( nang_or ) 
    ig_l    = all_ind[:nang_h];  ig_r = all_ind[nang_or-nang_h:]
    ig = np.concatenate( ( ig_l , ig_r ) )
    ib = np.setdiff1d( np.arange( nang_or ) , ig )

    sino_new       = np.zeros( ( nang_or , npix ) , dtype=myfloat )
    sino_new[ig,:] = sino
    sino_new[ib,:] = tracer

    factor = myfloat( nang_or ) / myfloat( nang )
    sino_new[:] = lhif_onestep( sino_new , tracer , factor=factor )  
    sino_new[ig,:]  = sino
    
    return sino_new




###########################################################
###########################################################
####                                                   #### 
####       Ludwig-Helgason one-step filter (LWIF)      ####
####                                                   ####
###########################################################
###########################################################

def super_lhsf( x0 , tracer , factor=2 ):
    ##  Extend sinogram to [0,2pi)
    m0 , n0 = x0.shape
    n0_h    = np.int( 0.5 * n0 )  
    s_2pi   = extend_full_period( x0 )
    ig      = np.unique( np.argwhere( s_2pi != tracer )[:,1] )
    ib      = np.unique( np.argwhere( s_2pi == tracer )[:,1] )
    print( ig.shape )
    print( s_2pi.shape )
    print( s_2pi[:,ig].shape )
    print( n0_h )



    ##  Resample each projection at cosinusoidal nodes
    s_2pi = s_2pi[:,ig].reshape( 2 * m0 , n0_h )
    sc_a  = resampling_cosine_nodes( s_2pi )
    dis.plot( sc_a , 'SC' )
    sc    = np.zeros( ( 2 * m0 , 2 * sc_a.shape[1] ) , dtype=myfloat )
    print( sc.shape )
    print( sc_a.shape )
    sc[:,::2] = sc_a
    sc[:,1::2] = 0.0
    dis.plot( sc , 'SC' )   
    
    m1 , n1 = sc.shape
    mh1 = int( m1 * 0.5 )
    nh1 = int( n1 * 0.5 )


    
    ##  Indexes for Fourier-Chebyshev filtering
    ic = get_ind_zerout( n1 , m0 )



    ##  Get traced and non-traced elements
    sx  = sc.copy()
    #sx[:,ib] = 0.0
    nc   = n1 
    norm = np.sqrt( 1.0 / ( 2.0 * ( nc + 1 ) ) )



    ##  Decomposition and filter
    c = norm * fp.dst( sx , type=1 , axis=1 )
    b = np.fft.fftshift( np.fft.fft( c , axis=0 ) , axes=0 )             
    b[ic[:,0],ic[:,1]] = 0.0
    


    ##  Reconstruction
    c[:] = np.real( np.fft.ifft( np.fft.ifftshift( b , axes=0 ) , axis=0 ) )
    ci   = np.imag( np.fft.ifft( np.fft.ifftshift( b , axes=0 ) , axis=0 ) ) 
    sx_r = norm * fp.dst( c , type=1 , axis=1 )
    sx_i = norm * fp.dst( ci , type=1 , axis=1 )
    sx[:] = np.sqrt( sx_r**2 + sx_i**2 )



    ##  Renormalize
    sx[:] = factor * sx



    ##  Crop original sinogram interval [0,pi) and first channel half (transl. of 1)   
    sx = sx[:mh1,:]



    ##  Interpolate back on equispaced nodes
    x = resampling_equisp_nodes( sx , n0 )
    
    print('\n')

    return x




###########################################################
###########################################################
####                                                   #### 
####               Super-resolution filter             ####
####                                                   ####
###########################################################
########################################################### 

def super_filt( sino , factor=2 ):
    nang , npix = sino.shape

    tracer = np.min( sino ) - 100
    factor = np.int( factor )
    if factor <= 1:
        factor = 2
    npix_new = npix * factor

    ig = np.arange( 0 , npix_new , factor )
    ib = np.setdiff1d( np.arange( npix_new ) , ig )

    sino_new       = np.zeros( ( nang , npix_new ) , dtype=myfloat )
    sino_new[:,ig] = sino
    sino_new[:,ib] = tracer
    dis.plot( sino_new , 'Starting' )
    sino_new[:] = super_lhsf( sino_new , tracer )
    dis.plot( sino_new , 'Ending' )
    
    sino_new[:,ig]  = sino
    
    return sino_new




###########################################################
###########################################################
####                                                   #### 
####           DECOMPOSITION  & RECONSTRUCTION         ####
####                                                   ####
###########################################################
###########################################################

def decomposition( s ):
    ##  Extend sinogram to [0,2pi)
    s_2pi = extend_full_period( s )
    m , n = s_2pi.shape;  mh = myint( 0.5 * m )


    ##  Resample each projection at cosinusoidal nodes
    sc = resampling_cosine_nodes( s_2pi )


    ##  Sine transform along the pixels  --->  Chebyshev coefficients
    c = fp.dst( sc , type=1 , axis=1 )
    
    
    ##  FFT along the view direction  --->  Fourier-Chebyshev coefficients
    b = np.fft.fftshift( np.fft.fft( c , axis=0 ) , axes=0 )

    return b




def reconstruction( b , n ):
    ##  Inverse Fourier transform along the angles  --->  Chebyshev coefficients      
    c = np.fft.ifft( np.fft.ifftshift( b , axes=0 ) , axis=0 )
    nc = c.shape[1]; 


    ##  Inverse sine transform  --->  projections sampled at cosinusoidal nodes
    sc_r = 1.0 / ( 2.0 * ( nc + 1 ) ) * fp.dst( np.real( c ) , type=1 , axis=1 )
    sc_i = 1.0 / ( 2.0 * ( nc + 1 ) ) * fp.dst( np.imag( c ) , type=1 , axis=1 )
    sc   = np.sqrt( sc_r**2 + sc_i**2 )


    ##  Interpolate back on equispaced nodes
    s_2pi = resampling_equisp_nodes( sc , n )
    m = s_2pi.shape[0];  mh = myint( 0.5 * m )


    ##  Crop original sinogram interval [0,pi)
    s = s_2pi[:mh,:]
    
    return s 




###########################################################
###########################################################
####                                                   #### 
####              FOURIER-CHEBYSHEV ANALYSIS           ####
####                                                   ####
###########################################################
###########################################################

def lhsf_analysis( s ):
    ##  Get dimensions of the sinogram
    na , n = s.shape


    ##  Decompose sinogram into Fourier-Chebyshev basis
    b  = decomposition( s )
    nc = b.shape[1]
    dis.plot( np.real( b ) , 'Fourier-Chebyshev decomposition' )


    ##  Zero-out coefficients at ( m , k ) s.t ( |m| > k ) or ( |m| + k even )
    ii = get_ind_zerout( nc , na )


    ##  Compute number and power percentage of wrong coefficients
    n_wro = len( ii )
    bb = b[ii[:,0],ii[:,1]]
    iii = np.argwhere( np.abs( bb ) > 0.5 )
    value = len( iii ) / myfloat( n_wro ) * 100.0
    print( '\nFourier-Chebyshev analysis of the forward projection:')
    print( 'Wrong Fourier-Chebyshev coefficients: ', value,' %' )
    
    pb_tot = np.linalg.norm( b )
    pb_wro = np.linalg.norm( bb )
    value  = pb_wro / myfloat( pb_tot ) * 100.0
    print( 'Wrong Fourier-Chebyshev power: ', value,' %\n' )

    b[ii[:,0],ii[:,1]] = 0.0


    ##  Reverse FFT and sine transform
    sf = reconstruction( b , n )   
    return sf
