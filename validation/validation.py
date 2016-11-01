#!/usr/bin/env python
#
# This validates CoLoRe, by: 
#    * feeding bias propto 1/g
#    * density propto r^2dr/dz
# which should then generate a truly 
# stationary field.
#
import numpy as np
import os,sys
import h5py as hdf
import healpy as hp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.fft import rfftn, fftfreq, rfftfreq
from optparse import OptionParser
import cPickle as cp

parser = OptionParser()
parser.add_option("--Nseed", dest="Nseed", default=2,type="int",
                  help="Number of times we run CoLoRe")
parser.add_option("--NfftC", dest="NfftC", default=512, type="int",
                  help="FFT size in CoLoRe")
parser.add_option("--Nside", dest="Nside", default=128, type="int",
                  help="healpix grid size")
parser.add_option("--bias", dest="bias", default=1.0, type="float",
                  help="bias at z=0")
parser.add_option("--zmean", dest="zmean", default=0.5, type="float",
                  help="zmean of N(z)")
parser.add_option("--zsig", dest="zsig", default=0.1, type="float",
                  help="sigma of N(z)")
parser.add_option("--nmax", dest="nmax", default=200, type="float",
                  help="sources at zmax")
parser.add_option("--rsmooth", dest="rsmooth", default=1.0, type="float",
                  help="smoothing")
parser.add_option("--plotslice", dest="plotslice", default=False, action="store_true",
                  help="Plot one slice")
parser.add_option("--scate", dest="scate", default=False, action="store_true",
                  help="Error from scatter")
parser.add_option("--lmax", dest="lmax", default=-1, type="int",
                  help="lmax for calculation of chi2, by default as high as possible")
parser.add_option("--save_to", dest="saveto", default="", type="string",
                  help="save spectra to")
parser.add_option("--load_from", dest="loadfrom", default="", type="string",
                  help="load spectra from")

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
        f.write("%g %g\n"%(z,o.bias))
    f.close()
    ## next write the dn/dz
    f=open("Nz.txt","w")
    for z,r,d in zip(zar,rr,drdz):
        f.write("%g %g \n"%(z,o.nmax*np.exp(-(z-o.zmean)**2/(2*o.zsig**2))))
    f.close()
    return rofz

def getReal(o, seed):
    """ Runs CoLoRe with one seed and returns power spectrum """
    rsmooth=o.rsmooth
    ## next write the input file
    s=open('param.proto').read()
    s=(s.replace("%seed%",str(seed))
       .replace("%rsmooth%",str(rsmooth))
       .replace("%fft%",str(o.NfftC))
       )
    open('param.cfg','w').write(s)
    fname="output/test_srcs_0.h5"
    try:
        os.remove(fname)
    except:
        pass
    os.system("../CoLoRe param.cfg")
    #now read in the ra, dec convert to x,y,z
    print "Reading HDF"
    if not os.path.isfile(fname):
        print "CoLoRe went gray..."
        sys.exit(1)
        
    da=hdf.File(fname)
    phi=da['sources0']['RA']/180.*np.pi
    theta=(90.-da['sources0']['DEC'])/180.*np.pi
    Nside=o.Nside
    Npix=Nside**2*12
    Nmean=float(len(phi))/Npix

    mp=np.bincount(hp.ang2pix(Nside, theta, phi),minlength=Npix)
    mp=mp*1.0/Nmean-1.0
    Cl=hp.anafast(mp)
    els=np.arange(len(Cl))
    Pshot=4*np.pi/float(len(phi))
    return els[2:],Cl[2:],Pshot


def getTheory(o, lognorm=False):
    """ Runs CoLoRe with one seed and returns power spectrum """
    rsmooth=o.rsmooth
    ## next write the input file
    s=open('param_limberjack.proto').read()
    s=(s.replace("%rsmooth%",str(rsmooth))
        .replace("%lnorm%",str(int(lognorm)))
       )
    open('param_limberjack.ini','w').write(s)
    os.system("../../LimberJack/LimberJack param_limberjack.ini")
    fname="output/lj_cl_dd.txt"
    #now read in the ra, dec convert to x,y,z
    print "Reading HDF"
    if not os.path.isfile(fname):
        print "Limberjack went gray..."
        sys.exit(1)
    ell,Cl=np.loadtxt(fname).T
    return ell[2:], Cl[2:]
        
    
def avgCl(ell,Cl,Cle,fact):
    nell,nCl, nCle=[],[],[]
    sw=0
    sC=0
    sL=0
    for l,C,Ce in zip (ell,Cl,Cle):
        if (sw==0):
            sw=1./Ce**2
            sC=C/Ce**2
            sL=l/Ce**2
            lmax=int(l*fact)+1
        else:
            if (l<lmax):
                sw+=1./Ce**2
                sC+=C/Ce**2
                sL+=l/Ce**2
            else:
                nell.append(sL/sw)
                nCl.append(sC/sw)
                nCle.append(np.sqrt(1/sw))
                sw=0
    print nell
    return np.array(nell), np.array(nCl), np.array(nCle)
                


## MAIN

rofz=writeNzBz(o);

if (len(o.loadfrom)==0):
    for i in range(o.Nseed):
        if i==0:
            ell,Cl,Pshot=getReal(o,i)
            Clx=Cl*Cl
        else:
            nCl=getReal(o,i)[1]
            Cl+=nCl
            Clx+=nCl*nCl


    Cl=Cl/o.Nseed
    if (o.scate):
        Cle=np.sqrt(Clx/o.Nseed-Cl*Cl)/np.sqrt(o.Nseed)
    else:
        Cle=Cl*np.sqrt(2./(2*ell+1)/o.Nseed)
    if (len(o.saveto)>0):
        cp.dump([ell,Cl,Cle,Pshot],open(o.saveto,'w'),-1)
else:
    ell,Cl,Cle,Pshot=cp.load(open(o.loadfrom))
    

pell,pCl,pCle=avgCl(ell,Cl,Cle,1.08)

plt.errorbar(pell,pCl,yerr=pCle,fmt='b.')
plt.plot(ell, np.ones(len(ell))*Pshot,'k:',label="shot noise")
ellt,Clt=getTheory(o,False)
ellt,Cltl=getTheory(o,True)
Clt+=Pshot
Cltl+=Pshot
plt.plot(ellt,Clt,'g-', label="linear theory")
plt.plot(ellt,Cltl,'r-', label="log trans theory")

lmax=o.lmax
if (lmax<0):
    lmax=min(len(ellt),len(ell))
assert(ellt[0]==ell[0])
print "chi2=", (((Cl[:lmax]-Cltl[:lmax])/Cle[:lmax])**2).sum(),"dof=",lmax
# kt,Pt,Px,Pl=np.loadtxt("out__pk_pop0_z0.1.dat").T
# plt.plot(kt,Pt,'k--')
# plt.plot(kt,Pl,'k--')
plt.loglog()
plt.legend()
plt.xlabel("$\ell")
plt.ylabel("$C_\ell$")
plt.tight_layout()
plt.show()



