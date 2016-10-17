#!/usr/bin/env python
#
# This validates CoLoRe, by: 
#    * feeding bias propto 1/g
#    * density propto r^2dr/dz
# which should then generate a truly 
# stationary field.
#
import numpy as np
import os
import h5py as hdf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.fft import rfftn, fftfreq, rfftfreq
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--Nseed", dest="Nseed", default=1,type="int",
                  help="Number of times we run CoLoRe")
parser.add_option("--Nfft", dest="Nfft", default=512, type="int",
                  help="FFT size in validation.py")
parser.add_option("--NfftC", dest="NfftC", default=512, type="int",
                  help="FFT size in CoLoRe")
parser.add_option("--bias", dest="bias", default=1.0, type="float",
                  help="bias at z=0")
parser.add_option("--rsmooth", dest="rsmooth", default=1.0, type="float",
                  help="smoothing")
parser.add_option("--nfac", dest="nfac", default=1.0, type="float",
                  help="relative number of sources")

(o, args) = parser.parse_args()


def writeNzBz(o):
    """ Writes Nz and Bz files and gets gfactor/dist (z)"""
    gr=np.loadtxt("gr.dat")
    # redhift, rad gfact
    (zar,rr,gf)=gr.T
    # dr/dz
    drdz=rr*0.0
    # let's take derivative, but two-sided derivatives breaks at endpoints.
    drdz[1:-1]=(rr[2:]-rr[:-2])/(zar[2:]-zar[:-2])
    drdz[0]=drdz[1]
    drdz[-1]=drdz[-2]
    rofz=interp1d(zar,rr)
    ## first write the bz file
    f=open("Bz.txt","w")
    for z,g in zip(zar,gf):
        f.write("%g %g\n"%(z,o.bias/g))
    f.close()
    ## next write the dn/dz
    f=open("Nz.txt","w")
    for z,r,d in zip(zar,rr,drdz):
        f.write("%g %g \n"%(z,1*r*r*d*1e-6*o.nfac))
    f.close()
    return rofz

def getReal(o, seed, rofz):
    """ Runs CoLoRe with one seed and returns power spectrum """
    Nfft=o.Nfft
    rsmooth=o.rsmooth
    ## next write the input file
    s=open('param.proto').read()
    s=(s.replace("%seed%",str(seed))
       .replace("%rsmooth%",str(rsmooth))
       .replace("%fft%",str(o.NfftC))
       )
    open('param.cfg','w').write(s)
    fname="out__gals_0.h5"
    os.remove(fname)
    os.system("../CoLoRe param.cfg")
    #now read in the ra, dec convert to x,y,z
    print "Reading HDF"
    if not os.path.isfile(fname):
        print "CoLoRe went gray..."
        sys.exit(1)
        
    da=hdf.File(fname)
    ra=da['sources0']['RA']/180.*np.pi
    dec=da['sources0']['DEC']/180.*np.pi
    zz=da['sources0']['Z_COSMO']
    ra=rofz(zz)
    cx=ra*np.cos(dec)*np.sin(ra)
    cy=ra*np.cos(dec)*np.cos(ra)
    cz=ra*np.sin(dec)
    rr=np.sqrt(cx*cx+cy*cy+cz*cz)
    ## this is the square that fits inside
    ## the sphere -rmax..+rmax
    rmax=rr.max()/4*np.sqrt(3)
    L=2*rmax
    print "rmax=",rmax, rr.min(), rr.max()
    w=np.where((abs(cx)<rmax)&(abs(cy)<rmax)&(abs(cz)<rmax))
    cx=cx[w]
    cy=cy[w]
    cz=cz[w]
    rr=rr[w]
    Ng=len(rr)
    dx=L/Nfft
    print "Indexing..."
    ci=((cx+rmax)/dx).astype(int)
    cj=((cy+rmax)/dx).astype(int)
    cz=((cz+rmax)/dx).astype(int)
    ndx=(cz*Nfft*Nfft+cj*Nfft+ci)
    del ci,cj,cz
    print "Gridding..."
    grd=np.bincount(ndx,minlength=Nfft**3)
    print "Normalizing..."
    grd=grd.astype(float)
    mean=grd.mean()
    print "mean=",mean, Ng/L**3
    grd-=mean
    grd/=mean
    grd=grd.reshape((Nfft,Nfft,Nfft))
    # normalize
    print "FFT..."
    fgrd=rfftn(grd,norm='ortho')
    del grd
    fxy=fftfreq(Nfft)*Nfft*2*np.pi/L
    fz=rfftfreq(Nfft)*Nfft*2*np.pi/L
    print "Sorting freqs."
    kz=(np.outer(np.ones(Nfft*Nfft),fz)).reshape(fgrd.shape)
    kx=(np.outer(fxy,np.ones(Nfft*(Nfft/2+1)))).reshape(fgrd.shape)
    ky=(np.outer(np.ones(Nfft),np.outer(fxy,np.ones(Nfft/2+1)))).reshape(fgrd.shape)
    kk=np.sqrt((kx*kx+ky*ky+kz*kz))
    #Let's do linear bins in k with some sensible fft-based size
    print "Binning"
    dk=kk[3,3,3]
    kk=kk.flatten()
    fgrd=fgrd.flatten()
    fgrd=np.abs(fgrd*fgrd)
    kbins=(kk/dk).astype(int)
    ## throw first one into the last one
    kbins[0]=kbins.max()
    Nmodes=np.bincount(kbins)
    kvals=np.bincount(kbins,weights=kk)
    Pk=np.bincount(kbins,weights=fgrd)
    kvals/=Nmodes
    Pk*=(L**3)/(Nmodes*Nfft**3)
    kvals=kvals[:-5]
    Pk=Pk[:-5]
    Nmodes=Nmodes[:-5]
    print "plotting"
    Pshot=L*L*L/Ng
    Pk-=Pshot
    return kvals, Pk,


## MAIN

rofz=writeNzBz(o);
for i in range(o.Nseed):
    kvals,Pk=getReal(o,i,rofz)
    plt.plot(kvals,Pk,'o-')

kt,Pt,Px,Pl=np.loadtxt("out__pk_pop0_z0.1.dat").T
plt.plot(kt,Pt,'k--')
plt.plot(kt,Pl,'k--')
plt.loglog()
plt.show()



