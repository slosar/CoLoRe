///////////////////////////////////////////////////////////////////////
//                                                                   //
//   Copyright 2012 David Alonso                                     //
//                                                                   //
//                                                                   //
// This file is part of CoLoRe.                                      //
//                                                                   //
// CoLoRe is free software: you can redistribute it and/or modify it //
// under the terms of the GNU General Public License as published by //
// the Free Software Foundation, either version 3 of the License, or //
// (at your option) any later version.                               //
//                                                                   //
// CoLoRe is distributed in the hope that it will be useful, but     //
// WITHOUT ANY WARRANTY; without even the implied warranty of        //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU //
// General Public License for more details.                          //
//                                                                   //
// You should have received a copy of the GNU General Public License //
// along with CoLoRe.  If not, see <http://www.gnu.org/licenses/>.   //
//                                                                   //
///////////////////////////////////////////////////////////////////////
#include "common.h"

static inline void cart2sph(double x,double y,double z,double *r,double *cth,double *phi)
{
  *r=sqrt(x*x+y*y+z*z);

  if((*r)==0) {
    *cth=1;
    *phi=0;
  }
  else {
    double xn=x/(*r);
    double yn=y/(*r);

    *cth=z/(*r);
    *phi=atan2(yn,xn);
    if((*phi)<0)
      (*phi)+=2*M_PI;
  }
}

static double get_rvel(ParamCoLoRe *par,int ix,int iy,int iz,
		       double x0,double y0,double z0,double rr)
{
  double v[3],u[3];
  double idx=par->n_grid/par->l_box;
  long ngx=2*(par->n_grid/2+1);
  long iz_hi=iz+1,iz_lo=iz-1,iz_0=iz;
  long iy_hi=iy+1,iy_lo=iy-1,iy_0=iy;
  long ix_hi=ix+1,ix_lo=ix-1,ix_0=ix;
  if(iy==0) iy_lo=par->n_grid-1;
  if(iy==par->n_grid-1) iy_hi=0;
  if(ix==0) ix_lo=par->n_grid-1;
  if(ix==par->n_grid-1) ix_hi=0;
  iz_0*=ngx*par->n_grid;
  iz_lo*=ngx*par->n_grid;
  iz_hi*=ngx*par->n_grid;
  iy_0*=ngx;
  iy_lo*=ngx;
  iy_hi*=ngx;

  u[0]=x0/rr; u[1]=y0/rr; u[2]=z0/rr;
  v[0]=par->grid_npot[ix_hi+iy_0+iz_0]-par->grid_npot[ix_lo+iy_0+iz_0];
  v[1]=par->grid_npot[ix_0+iy_hi+iz_0]-par->grid_npot[ix_0+iy_lo+iz_0];
  if(iz==0)
    v[2]=par->grid_npot[ix_0+iy_0+iz_hi]-par->slice_left[ix_0+iy_0];
  else if(iz==par->nz_here-1)
    v[2]=par->slice_right[ix_0+iy_0]-par->grid_npot[ix_0+iy_0+iz_lo];
  else
    v[2]=par->grid_npot[ix_0+iy_0+iz_hi]-par->grid_npot[ix_0+iy_0+iz_lo];

  return 0.5*idx*(v[0]*u[0]+v[1]*u[1]+v[2]*u[2]);
}

static void get_sources_cartesian_single(ParamCoLoRe *par,int ipop)
{
  int ii,nthr;
  int ngx=2*(par->n_grid/2+1);
  long *np_tot_thr;
  int *nsources=my_calloc(ngx*((long)(par->n_grid*par->nz_here)),sizeof(int));
#ifdef _HAVE_OMP
  nthr=omp_get_max_threads();
#else //_HAVE_OMP
  nthr=1;
#endif //_HAVE_OMP
  np_tot_thr=my_calloc(nthr,sizeof(long));

  print_info(" %d-th galaxy population\n",ipop);
  if(NodeThis==0) timer(0);
  print_info("   Poisson-sampling\n");
#ifdef _HAVE_OMP
#pragma omp parallel default(none)			\
  shared(par,np_tot_thr,IThread0,ipop,nthr,nsources)
#endif //_HAVE_OMP
  {
    long iz;
#ifdef _HAVE_OMP
    int ithr=omp_get_thread_num();
#else //_HAVE_OMP
    int ithr=0;
#endif //_HAVE_OMP
    double dx=par->l_box/par->n_grid;
    double cell_vol=dx*dx*dx;
    int ngx=2*(par->n_grid/2+1);
    unsigned int seed_thr=par->seed_rng+ithr+nthr*(ipop+par->n_srcs*IThread0);
    gsl_rng *rng_thr=init_rng(seed_thr);

#ifdef _HAVE_OMP
#pragma omp for schedule(static)
#endif //_HAVE_OMP
    for(iz=0;iz<par->nz_here;iz++) {
      int iy;
      long indexz=iz*((long)(ngx*par->n_grid));
      double z0=(iz+par->iz0_here+0.5)*dx-par->pos_obs[2];
      for(iy=0;iy<par->n_grid;iy++) {
	int ix;
	long indexy=iy*ngx;
	double y0=(iy+0.5)*dx-par->pos_obs[1];
	for(ix=0;ix<par->n_grid;ix++) {
	  int npp=0;
	  long index=ix+indexy+indexz;
	  double x0=(ix+0.5)*dx-par->pos_obs[0];
	  double r=sqrt(x0*x0+y0*y0+z0*z0);
	  if(r<par->l_box/2+20.) {
	    double ndens=get_bg(par,r,BG_NZ_SRCS,ipop);
	    if(ndens>0) {
	      double bias=get_bg(par,r,BG_BZ_SRCS,ipop);
	      double dnorm=get_bg(par,r,BG_NORM_SRCS,ipop);
	      double lambda=ndens*cell_vol*bias_model(par->grid_dens[index],bias)*dnorm;
	      npp=rng_poisson(lambda,rng_thr);
	    }
	  }

	  nsources[index]=npp;
	  np_tot_thr[ithr]+=npp;
	}
      }
    }//end omp for

    end_rng(rng_thr);
  }//end omp parallel
  if(NodeThis==0) timer(2);

  par->nsources_c_this[ipop]=0;
  for(ii=0;ii<nthr;ii++)
    par->nsources_c_this[ipop]+=np_tot_thr[ii];

  long nsources_total=0;
#ifdef _HAVE_MPI
  MPI_Allreduce(&(par->nsources_c_this[ipop]),&nsources_total,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
#else //_HAVE_MPI
  nsources_total=par->nsources_c_this[ipop];
#endif //_HAVE_MPI

  print_info("   There will be %ld objects in total \n",(long)nsources_total);
#ifdef _DEBUG
  fprintf(par->f_dbg,"Node %d has %ld particles\n",NodeThis,(long)(par->nsources_c_this[ipop]));
#endif //_DEBUG

  for(ii=nthr-1;ii>0;ii--) {
    int jj;
    long nh=0;
    for(jj=0;jj<ii;jj++)
      nh+=np_tot_thr[jj];
    np_tot_thr[ii]=nh;
  }
  np_tot_thr[0]=0;
  //np_tot_thr now contains the id of the first particle in the thread
  
  par->cats_c[ipop]=new_catalog_cartesian(par->nsources_c_this[ipop]);

  if(NodeThis==0) timer(0);
  print_info("   Assigning coordinates\n");
#ifdef _HAVE_OMP
#pragma omp parallel default(none)			\
  shared(par,IThread0,np_tot_thr,ipop,nthr,nsources)
#endif //_HAVE_OMP
  {
    long iz;
#ifdef _HAVE_OMP
    int ithr=omp_get_thread_num();
#else //_HAVE_OMP
    int ithr=0;
#endif //_HAVE_OMP
    double dx=par->l_box/par->n_grid;
    int ngx=2*(par->n_grid/2+1);
    unsigned int seed_thr=par->seed_rng+ithr+nthr*(ipop+par->n_srcs*IThread0);
    gsl_rng *rng_thr=init_rng(seed_thr);
    double factor_vel=-par->fgrowth_0/(1.5*par->hubble_0*par->OmegaM);

#ifdef _HAVE_OMP
#pragma omp for schedule(static)
#endif //_HAVE_OMP
    for(iz=0;iz<par->nz_here;iz++) {
      int iy;
      long indexz=iz*((long)(ngx*par->n_grid));
      double z0=(iz+par->iz0_here+0.5)*dx-par->pos_obs[2];
      for(iy=0;iy<par->n_grid;iy++) {
	int ix;
	long indexy=iy*ngx;
	double y0=(iy+0.5)*dx-par->pos_obs[1];
	for(ix=0;ix<par->n_grid;ix++) {
	  double x0=(ix+0.5)*dx-par->pos_obs[0];
	  long index=ix+indexy+indexz;
	  int npp=nsources[index];
	  if(npp>0) {
	    int ip;
	    double rr=sqrt(x0*x0+y0*y0+z0*z0);
	    double rvel=factor_vel*get_rvel(par,ix,iy,iz,x0,y0,z0,rr);
	    double dz_rsd=rvel*get_bg(par,rr,BG_V1,0);
	    for(ip=0;ip<npp;ip++) {
	      long pix_id_ring,pix_id_nest;
	      double pos[3];
	      long pid=np_tot_thr[ithr];

	      pos[0]=x0+dx*(rng_01(rng_thr)-0.5);
	      pos[1]=y0+dx*(rng_01(rng_thr)-0.5);
	      pos[2]=z0+dx*(rng_01(rng_thr)-0.5);
	      par->cats_c[ipop]->pos[3*pid+0]=pos[0];
	      par->cats_c[ipop]->pos[3*pid+1]=pos[1];
	      par->cats_c[ipop]->pos[3*pid+2]=pos[2];
	      par->cats_c[ipop]->dz_rsd[pid]=dz_rsd;
	      
	      vec2pix_ring(par->nside_base,pos,&pix_id_ring);
	      ring2nest(par->nside_base,pix_id_ring,&pix_id_nest);
	      par->cats_c[ipop]->ipix[pid]=pix_id_nest;
	      np_tot_thr[ithr]++;
	    }
	  }
	}
      }
    }//end omp for
    end_rng(rng_thr);
  }//end omp parallel
  if(NodeThis==0) timer(2);

  free(np_tot_thr);
  free(nsources);
}  

void get_source_positions(ParamCoLoRe *par)
{
  int ipop;

  //First, compute lensing Hessian
  print_info("*** Getting point sources\n");
  for(ipop=0;ipop<par->n_srcs;ipop++)
    get_sources_cartesian_single(par,ipop);
  print_info("\n");
}
