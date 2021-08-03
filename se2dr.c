
/*
 (rho v, u) = ( v , div(C : grad(u)) ) + (v,F)
 
 where (v,F) = (v, Mp delta_1) + (v, -div(-Ms delta_2 ))
 
 (rho v, u) = -(grad(v) , C : grad(u)) + (v, sigma.n)_ds
              + (v, Mp delta_1) + (grad(v), Ms delta_2 ) - (v, (Ms delta_2 ).n)_ds

            = -(grad(v) , C : grad(u))
              +(v, Mp delta_1 ) + (grad(v), Ms delta_2 )
              + (v, sigma.n)_ds
              - (v, (-Ms delta_2 ).n)_ds  ==> which will be dropped as we assume the source does not intersect with the boundary
*/

#include <petsc.h>
#include <petsctime.h>
#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsys.h>

#include "rupture.h"

typedef enum { TENS2D_XX=0, TENS2D_YY=1, TENS2D_XY=2 } VoigtTensor2d;

typedef struct _p_SpecFECtx *SpecFECtx;

typedef struct {
  PetscInt region;
  PetscReal lambda,mu;
  PetscReal rho;
} QPntIsotropicElastic;

struct _p_SpecFECtx {
  PetscMPIInt rank,size;
  PetscInt basisorder;
  PetscInt mx,my,mz;
  PetscInt mx_g,my_g,mz_g,nx_g,ny_g,nz_g;
  //PetscReal dx,dy,dz;
  PetscInt dim;
  PetscInt dofs;
  DM dm;
  PetscInt npe,npe_1d,ne,ne_g;
  PetscInt *element;
  PetscReal *xi1d,*w1d,*w;
  PetscReal *elbuf_coor,*elbuf_field,*elbuf_field2;
  PetscInt  *elbuf_dofs;
  PetscInt nqp;
  QPntIsotropicElastic *cell_data;
  PetscReal **dN_dxi,**dN_deta;
  PetscReal **dN_dx,**dN_dy;
  //PetscInt  source_implementation; /* DR */
  
  /* DR */
  PetscReal delta;         /* fault thickness */
  PetscReal *elbuf_field3; /* additional element buffer (will hold velocity) */
  DRVar     *dr_qp_data;   /* stores data like slip, slip-rate */
  PetscReal mu_s,mu_d,D_c; /* linear slip weakening parameters */

  SDF sdf;
  PetscReal *sigma; /* size = ne * npe * 3 */
  //PetscReal *gradu; /* required to test cg projection */
};


typedef struct {
  PetscReal xi[2];
  PetscInt  nbasis;
  PetscInt  *element_indices;
  PetscReal *element_values;
  PetscReal *buffer;
} PointwiseContext;

PetscErrorCode CreateGLLCoordsWeights(PetscInt N,PetscInt *_npoints,PetscReal **_xi,PetscReal **_w);
PetscErrorCode TabulateBasis1d_CLEGENDRE(PetscInt npoints,PetscReal xi[],PetscInt order,PetscInt *_nbasis,PetscReal ***_Ni);
void ElementEvaluateGeometry_CellWiseConstant2d(PetscInt npe,PetscReal el_coords[],
                                                PetscInt nbasis,PetscReal *detJ);

PetscErrorCode SpecFECreateQuadratureField(SpecFECtx c,PetscInt blocksize,PetscReal **_f)
{
  PetscReal      *f;
  PetscErrorCode ierr;
  
  ierr = PetscCalloc1(c->ne*c->nqp*blocksize,&f);CHKERRQ(ierr);
  *_f = f;
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFEGetElementQuadratureField(SpecFECtx c,PetscInt blocksize,PetscInt e,const PetscReal *f,PetscReal **fe)
{
  *fe = (PetscReal*)&f[e * c->nqp * blocksize];
  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureFieldDGView_PV(SpecFECtx c,PetscInt blocksize,const PetscReal field[],
                                        const char *field_name[],
                                        const char filename[])
{
  PetscErrorCode ierr;
  FILE *fp = NULL;
  PetscInt i,j,q,e,b;
  Vec coor;
  const PetscReal *LA_coor;
  PetscReal *elcoords;
  
  
  
  fp = fopen(filename,"w");
  if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open %s",filename);
  
  elcoords = c->elbuf_coor;
  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  fprintf(fp,"<?xml version=\"1.0\"?>\n");
  fprintf(fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
  fprintf(fp,"<UnstructuredGrid>\n");
  fprintf(fp,"<Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n",c->ne * c->npe,c->ne * (c->npe_1d-1)*(c->npe_1d-1));
  
  
  fprintf(fp,"<Cells>\n");
  fprintf(fp,"  <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
  {
    PetscInt cnt = 0;
    fprintf(fp,"  ");
    for (e=0; e<c->ne; e++) {
      for (j=0; j<c->npe_1d-1; j++) {
        for (i=0; i<c->npe_1d-1; i++) {
          PetscInt cell[4],d;
          
          cell[0] = i + j * (c->npe_1d);
          cell[1] = cell[0] + 1;
          cell[2] = cell[1] + c->npe_1d;
          cell[3] = cell[0] + c->npe_1d;
          for (d=0; d<4; d++) {
            cell[d] += cnt;
          }
          
          fprintf(fp,"%d %d %d %d ",cell[0],cell[1],cell[2],cell[3]);
        }
      }
      cnt += (c->npe_1d * c->npe_1d);
    }
  }
  fprintf(fp,"\n  </DataArray>\n");
  
  
  fprintf(fp,  "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  {
    PetscInt cnt = 0;
    fprintf(fp,"  ");
    for (e=0; e<c->ne; e++) {
      for (j=0; j<c->npe_1d-1; j++) {
        for (i=0; i<c->npe_1d-1; i++) {
          cnt += 4;
          fprintf(fp,"%d ",cnt);
        }
      }
      //cnt += 1;
    }
  }
  fprintf(fp,"\n  </DataArray>\n");
  
  fprintf(fp,"  <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  {
    PetscInt VTK_QUAD = 9;
    fprintf(fp,"  ");
    for (e=0; e<c->ne; e++) {
      for (j=0; j<c->npe_1d-1; j++) {
        for (i=0; i<c->npe_1d-1; i++) {
          fprintf(fp,"%d ",VTK_QUAD);
        }
      }
    }
  }
  fprintf(fp,"\n  </DataArray>\n");
  
  fprintf(fp,"</Cells>\n");
  
  /* coordinates */
  fprintf(fp,"<Points>\n");
  fprintf(fp,"  <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (e=0; e<c->ne; e++) {
    const PetscInt *elnidx = &c->element[c->npe*e];
    
    for (i=0; i<c->npe; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
    }
    
    for (i=0; i<c->npe; i++) {
      PetscReal coor[] = {0,0,0};
      coor[0] = elcoords[2*i  ];
      coor[1] = elcoords[2*i+1];
      fprintf(fp,"  %+1.4e %+1.4e %+1.4e\n",coor[0],coor[1],coor[2]);
    }
  }
  fprintf(fp,"  </DataArray>\n");
  fprintf(fp,"</Points>\n");
  
  fprintf(fp,"<PointData>\n");
  
  for (b=0; b<blocksize; b++) {
    fprintf(fp,"  <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n",field_name[b]);
    for (e=0; e<c->ne; e++) {
      PetscReal *field_e;

      ierr = SpecFEGetElementQuadratureField(c,blocksize,e,field,&field_e);CHKERRQ(ierr);
      fprintf(fp,"  ");
      for (q=0; q<c->nqp; q++) {
        fprintf(fp,"%+1.4e ",field_e[q * blocksize + b]);
      }
      fprintf(fp,"\n");
    }
    fprintf(fp,"  </DataArray>\n");
  }
  
  fprintf(fp,"</PointData>\n");
  
  fprintf(fp,"</Piece>\n");
  fprintf(fp,"</UnstructuredGrid>\n");
  fprintf(fp,"</VTKFile>\n");
  
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  fclose(fp);
  PetscFunctionReturn(0);
}

PetscErrorCode StressView_PV(SpecFECtx c,const PetscReal sigma[],const char filename[])
{
  PetscErrorCode ierr;
  const char *field_names[] = { "sigma_xx", "sigma_yy", "sigma_xy" };
  ierr = QuadratureFieldDGView_PV(c,3,sigma,field_names,filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 \int w field_q
 = \sum_q w_q field_q |J_q|
*/
PetscErrorCode DGProject_CellAssemble_ScalarMixedMassMatrix2d(
  PetscInt nbasis,
  PetscInt nqp,PetscReal w[],PetscReal dJ[],PetscReal **Ni,PetscReal field[],PetscReal m[])
{
  PetscErrorCode ierr;
  PetscInt i,q;
  
  ierr = PetscMemzero(m,nbasis*sizeof(PetscReal));CHKERRQ(ierr);
  for (q=0; q<nqp; q++) {
    for (i=0; i<nbasis; i++) {
      m[i] += w[q] * Ni[q][i] * field[q] * dJ[q];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DGProject_CellAssemble_ScalarMassMatrix2d(
  PetscInt nbasis,
  PetscReal w[],PetscReal dJ[],PetscReal m[])
{
  PetscErrorCode ierr;
  PetscInt i;
  
  ierr = PetscMemzero(m,nbasis*sizeof(PetscReal));CHKERRQ(ierr);
  for (i=0; i<nbasis; i++) {
    m[i] += w[i] * dJ[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DGProject(SpecFECtx c,PetscInt k,PetscInt blocksize,PetscReal *field)
{
  PetscErrorCode ierr;
  
  PetscInt nqp,n1d,nbasis_k,nbasis_k_1d,q,qi,qj,i,j,e;
  PetscInt npe_1d_k,npe_k;
  PetscReal *xi1d,*w;
  PetscReal *xi1d_k,*w1d_k,*w_k;
  PetscReal **N_k,**N;
  PetscReal *dJ,*dJ_k;
  PetscReal *elcoords;
  Vec coor;
  const PetscReal *LA_coor;
  PetscReal *Me,*Mf,*proj;
  PetscReal *qp_buffer;
  
  
  if (k > c->basisorder) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"SpecFECtx defines basis of degree %D. You requested to project into degree %D. Must project to degree less than or to that defined by SpecFECtx",c->basisorder,k);
  
  elcoords = c->elbuf_coor;

  nqp  = c->nqp;
  n1d  = c->npe_1d;
  xi1d = c->xi1d;
  w    = c->w;
  
  
  /* tensor product for weights */
  if (k > 0) {
    ierr = CreateGLLCoordsWeights(k,&npe_1d_k,&xi1d_k,&w1d_k);CHKERRQ(ierr);
  } else {
    npe_1d_k = 1;
    ierr = PetscCalloc1(npe_1d_k,&xi1d_k);CHKERRQ(ierr);
    xi1d_k[0] = 0.0;
    ierr = PetscCalloc1(npe_1d_k,&w1d_k);CHKERRQ(ierr);
    w1d_k[0] = 1.0;
  }
  
  npe_k = npe_1d_k * npe_1d_k;
  
  ierr = PetscCalloc1(npe_k,&w_k);CHKERRQ(ierr);
  for (j=0; j<npe_1d_k; j++) {
    for (i=0; i<npe_1d_k; i++) {
      w_k[i+j*npe_1d_k] = w1d_k[i] * w1d_k[j];
    }
  }

  
  if (k > 0) {
    ierr = TabulateBasis1d_CLEGENDRE(n1d,xi1d,k,&nbasis_k_1d,&N_k);CHKERRQ(ierr);
  } else {
    nbasis_k_1d = 1;
    ierr = PetscMalloc(sizeof(PetscReal*)*n1d,&N_k);CHKERRQ(ierr);
    for (i=0; i<n1d; i++) {
      ierr = PetscMalloc(sizeof(PetscReal)*nbasis_k_1d,&N_k[i]);CHKERRQ(ierr);
      N_k[i][0] = 1.0;
    }
  }
  
  nbasis_k = nbasis_k_1d * nbasis_k_1d;
  
  if (npe_k != nbasis_k) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Basis of projection space are inconsistent");
  
  ierr = PetscMalloc(sizeof(PetscReal*)*nqp,&N);CHKERRQ(ierr);
  for (i=0; i<nqp; i++) {
    ierr = PetscMalloc(sizeof(PetscReal)*nbasis_k,&N[i]);CHKERRQ(ierr);
  }

  for (j=0; j<nbasis_k_1d; j++) {
    for (i=0; i<nbasis_k_1d; i++) {
      
      for (qj=0; qj<n1d; qj++) {
        for (qi=0; qi<n1d; qi++) {
          
          N[qi + qj*n1d][i + j*nbasis_k_1d] = N_k[qi][i] * N_k[qj][j];
          
        }
      }
    }
  }
  
  ierr = PetscCalloc1(nqp,&dJ);CHKERRQ(ierr);
  ierr = PetscCalloc1(npe_k,&dJ_k);CHKERRQ(ierr);
  
  ierr = PetscCalloc1(npe_k,&Me);CHKERRQ(ierr);
  ierr = PetscCalloc1(npe_k,&Mf);CHKERRQ(ierr);
  ierr = PetscCalloc1(npe_k,&proj);CHKERRQ(ierr);

  ierr = PetscCalloc1(nqp,&qp_buffer);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);

  for (e=0; e<c->ne; e++) {
    PetscInt *elnidx = &c->element[c->npe*e];
    PetscReal detJ;
    PetscInt b;
    
    for (i=0; i<c->npe; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
    }
    ElementEvaluateGeometry_CellWiseConstant2d(c->npe,elcoords,c->npe_1d,&detJ);
    for (q=0; q<nqp; q++) {
      dJ[q] = detJ;
    }
    for (q=0; q<npe_k; q++) {
      dJ_k[q] = detJ;
    }

    ierr = DGProject_CellAssemble_ScalarMassMatrix2d(npe_k,w_k,dJ_k,Me);CHKERRQ(ierr);
    
    
    for (b=0; b<blocksize; b++) {
      /* pack dof into buffer */
      for (q=0; q<nqp; q++) {
        qp_buffer[q] = field[e*c->npe * blocksize + q*blocksize + b];
      }
      
      /* project */
      ierr = DGProject_CellAssemble_ScalarMixedMassMatrix2d(nbasis_k,nqp,w,dJ,N,qp_buffer,Mf);CHKERRQ(ierr);
      
      for (i=0; i<npe_k; i++) {
        proj[i] = Mf[i] / Me[i];
      }
      
      /* interpolate projection and store in buffer */
      {
        for (q=0; q<nqp; q++) {
          qp_buffer[q] = 0.0;
          for (i=0; i<npe_k; i++) {
            qp_buffer[q] += N[q][i] * proj[i];
          }
        }
      }
      
      /* unpack buffer */
      for (q=0; q<nqp; q++) {
        field[e*c->npe * blocksize + q*blocksize + b] = qp_buffer[q];
      }
    }
    
  }
  
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);

  
  ierr = PetscFree(proj);CHKERRQ(ierr);
  ierr = PetscFree(Me);CHKERRQ(ierr);
  ierr = PetscFree(Mf);CHKERRQ(ierr);
  ierr = PetscFree(qp_buffer);CHKERRQ(ierr);
  
  ierr = PetscFree(xi1d_k);CHKERRQ(ierr);
  ierr = PetscFree(w1d_k);CHKERRQ(ierr);
  ierr = PetscFree(w_k);CHKERRQ(ierr);
  
  ierr = PetscFree(dJ_k);CHKERRQ(ierr);
  ierr = PetscFree(dJ);CHKERRQ(ierr);
  for (q=0; q<n1d; q++) {
    ierr = PetscFree(N_k[q]);CHKERRQ(ierr);
  }
  ierr = PetscFree(N_k);CHKERRQ(ierr);
  for (q=0; q<nqp; q++) {
    ierr = PetscFree(N[q]);CHKERRQ(ierr);
  }
  ierr = PetscFree(N);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}



PetscErrorCode CGProject_AssembleBilinearForm_ScalarMass2d(SpecFECtx c,Vec A)
{
  PetscErrorCode ierr;
  PetscInt  e,index,q,i,nbasis;
  PetscInt  *element,*elnidx,*eldofs;
  PetscReal *elcoords,*Me,detJ;
  Vec       coor;
  const PetscReal *LA_coor;
  
  ierr = VecZeroEntries(A);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  Me       = c->elbuf_field;
  element  = c->element;
  
  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  for (e=0; e<c->ne; e++) {
    elnidx = &element[nbasis*e];
    
    for (i=0; i<nbasis; i++) {
      eldofs[i] = elnidx[i];
    }
    
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
    }
    ElementEvaluateGeometry_CellWiseConstant2d(nbasis,elcoords,c->npe_1d,&detJ);
    
    for (q=0; q<nbasis; q++) {
      PetscReal fac,Me_ii;
      
      fac = detJ * c->w[q];
      Me_ii = fac * 1.0;
      /* \int u0v0 dV */
      index = q;
      Me[index] = Me_ii;
    }
    ierr = VecSetValues(A,nbasis,eldofs,Me,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(A);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(A);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}



PetscErrorCode CGProjectNative(SpecFECtx c,PetscInt k,PetscInt blocksize,PetscReal field[],Vec *_proj)
{
  PetscInt e,i,b;
  PetscInt *eldofs;
  PetscReal *elcoords;
  PetscInt nbasis;
  PetscReal dJ;
  Vec       coor;
  const PetscReal *LA_coor;
  PetscReal *m;
  Vec M,Mf,proj;
  PetscErrorCode ierr;
  
  
  if (k > c->basisorder) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"SpecFECtx defines basis of degree %D. You requested to project into degree %D. Must project to degree less than or to that defined by SpecFECtx",c->basisorder,k);

  proj = *_proj;
  
  if (!proj) {
    Vec      x;
    PetscInt N,n,bs;
    
    ierr = DMCreateGlobalVector(c->dm,&x);CHKERRQ(ierr);
    ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
    ierr = VecGetSize(x,&N);CHKERRQ(ierr);
    ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
    
    ierr = VecCreate(PetscObjectComm((PetscObject)x),&proj);CHKERRQ(ierr);
    ierr = VecSetSizes(proj,blocksize * n/bs,blocksize * N/bs);CHKERRQ(ierr);
    ierr = VecSetBlockSize(proj,blocksize);CHKERRQ(ierr);
    ierr = VecSetFromOptions(proj);CHKERRQ(ierr);
    ierr = VecSetUp(proj);CHKERRQ(ierr);
    
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    *_proj = proj;
  }
  ierr = VecZeroEntries(proj);CHKERRQ(ierr);
  
  {
    Vec      x;
    PetscInt N,n,bs;
    
    ierr = DMCreateGlobalVector(c->dm,&x);CHKERRQ(ierr);
    ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
    ierr = VecGetSize(x,&N);CHKERRQ(ierr);
    ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
    
    ierr = VecCreate(PetscObjectComm((PetscObject)x),&M);CHKERRQ(ierr);
    ierr = VecSetSizes(M,n/bs,N/bs);CHKERRQ(ierr);
    ierr = VecSetFromOptions(M);CHKERRQ(ierr);
    ierr = VecSetUp(M);CHKERRQ(ierr);
    
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDuplicate(M,&Mf);CHKERRQ(ierr);
  }

  
  m       = c->elbuf_field;
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  
  /* project (cell-wise) into degree k, and interpolate result back into field[] */
  ierr = DGProject(c,k,blocksize,field);CHKERRQ(ierr);
  
  
  ierr = CGProject_AssembleBilinearForm_ScalarMass2d(c,M);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);

  for (b=0; b<blocksize; b++) {
    
    ierr = VecZeroEntries(Mf);CHKERRQ(ierr);
    
    for (e=0; e<c->ne; e++) {
      PetscInt *elnidx = &c->element[nbasis*e];
      PetscReal *field_e;
      
      ierr = SpecFEGetElementQuadratureField(c,blocksize,e,(const PetscReal*)field,&field_e);CHKERRQ(ierr);
      
      for (i=0; i<nbasis; i++) {
        eldofs[i] = elnidx[i];
      }
      
      for (i=0; i<nbasis; i++) {
        PetscInt nidx = elnidx[i];
        elcoords[2*i  ] = LA_coor[2*nidx  ];
        elcoords[2*i+1] = LA_coor[2*nidx+1];
      }
      ElementEvaluateGeometry_CellWiseConstant2d(nbasis,elcoords,c->npe_1d,&dJ);

      for (i=0; i<nbasis; i++) {
        m[i] = c->w[i] * field_e[blocksize * i + b] * dJ;
      }
      ierr = VecSetValues(Mf,nbasis,eldofs,m,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(Mf);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Mf);CHKERRQ(ierr);
  
    ierr = VecPointwiseDivide(Mf,Mf,M);CHKERRQ(ierr); // Mf <- Mf/M
   
    ierr = VecStrideScatter(Mf,b,proj,INSERT_VALUES);CHKERRQ(ierr);
  }
  
  ierr = VecDestroy(&M);CHKERRQ(ierr);
  ierr = VecDestroy(&Mf);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/**
 * Function to calculate weighting for the traction
*/ 
PetscErrorCode PetscTanHWeighting(PetscReal *Result, PetscReal ValueTrial, PetscReal CritValue,  PetscReal phi, PetscReal Amplitude, PetscReal Offset)
{
  PetscReal weight;

  weight = 0.5 * PetscTanhReal((PetscAbsReal(phi)-Offset) * Amplitude)  + 0.5;

  Result[0] =  CritValue * (1.0 - weight)  + ValueTrial * weight ;
  PetscFunctionReturn(0);
}

/*
 warp for dr mesh
 
 get ymax
 
 plot (exp(4*x)-1)/exp(4),x
 
 s = y / ymax
 s' = (exp(4*s)-1)/exp(4)
 
*/
PetscErrorCode warp_y_exp(SpecFECtx c,PetscReal factor)
{
  PetscInt i,N;
  PetscReal ymax = -1.0e32,s[2],sp[2];
  Vec coor;
  PetscScalar *_coor;
  PetscErrorCode ierr;
  
  DMGetCoordinates(c->dm,&coor);
  VecGetSize(coor,&N);
  N = N / 2;
  VecGetArray(coor,&_coor);
  for (i=0; i<N; i++) {
    ymax = PetscMax(ymax,_coor[2*i+1]);
  }

  for (i=0; i<N; i++) {
    s[0] = _coor[2*i+0];
    s[1] = _coor[2*i+1];

    // normalize to 1
    s[1] = s[1] / ymax;

    sp[0] = s[0];
    sp[1] = s[1];
    if (s[1] >= 0.0) {
      sp[1] = (PetscExpReal(factor * s[1]) - 1.0)/PetscExpReal(factor);
    } else {
      PetscReal _s = PetscAbsReal(s[1]);
      
      sp[1] = -(PetscExpReal(factor * _s) - 1.0)/PetscExpReal(factor);
    }

    sp[1] *= ymax;
   
    _coor[2*i+0] = sp[0];
    _coor[2*i+1] = sp[1];
  }
  
  VecRestoreArray(coor,&_coor);
  
  PetscFunctionReturn(0);
}


/* N = polynomial order */
PetscErrorCode CreateGLLCoordsWeights(PetscInt N,PetscInt *_npoints,PetscReal **_xi,PetscReal **_w)
{
  PetscInt N1;
  PetscReal *xold,*x,*w,*P;
  PetscReal eps,res;
  PetscInt i,j,k;
  PetscErrorCode ierr;
  
  
  // Truncation + 1
  N1 = N + 1;
  
  ierr = PetscMalloc(sizeof(PetscReal)*N1,&xold);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*N1,&x);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*N1,&w);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*N1*N1,&P);CHKERRQ(ierr);
  
  // Use the Chebyshev-Gauss-Lobatto nodes as the first guess
  for (i=0; i<N1; i++) {
    x[i]=PetscCosReal(PETSC_PI*i/(PetscReal)N);
  }
  
  // The Legendre Vandermonde Matrix
  for (i=0; i<N1; i++) {
    for (j=0; j<N1; j++) {
      P[i+j*N1] = 0.0;
    }
  }
  
  // Compute P_(N) using the recursion relation
  // Compute its first and second derivatives and
  // update x using the Newton-Raphson method.
  for (i=0; i<N1; i++) {
    xold[i]=2.0;
  }
  
  res = 1.0;
  eps = 1.0e-12;
  while (res > eps) {
    
    //xold=x;
    for (i=0; i<N1; i++) {
      xold[i] = x[i];
    }
    
    //P(:,1)=1;    P(:,2)=x;
    for (i=0; i<N1; i++) {
      for (j=0; j<N1; j++) {
        P[i+0*N1] = 1.0;
        P[i+1*N1] = x[i];
      }
    }
    
    //for k=2:N
    //    P(:,k+1)=( (2*k-1)*x.*P(:,k)-(k-1)*P(:,k-1) )/k;
    //end
    for (i=0; i<N1; i++) {
      for (k=1; k<N; k++) {
        P[i+(k+1)*N1] = ( (2.0*(k+1)-1.0)*x[i] * P[i+k*N1] - (k+1.0-1.0) * P[i+(k-1)*N1] ) / (PetscReal)(k+1.0);
      }
    }
    
    //x=xold-( x.*P(:,N1)-P(:,N) )./( N1*P(:,N1) );
    for (i=0; i<N1; i++) {
      x[i] = xold[i] - (x[i] * P[i+(N1-1)*N1] - P[i+(N-1)*N1]) / ( N1 * P[i+(N1-1)*N1] );
    }
    
    res = 0.0;
    for (i=0; i<N1; i++) {
      res += (x[i] - xold[i])*(x[i] - xold[i]);
    }
    res = PetscSqrtReal(res);
  }
  
  // w=2./(N*N1*P(:,N1).^2);
  for (i=0; i<N1; i++) {
    PetscReal pp = P[i+(N1-1)*N1];
    w[i] = 2.0 / (N*N1*pp*pp);
  }
  
  if (_xi) {
    /* flip order so they are ordered from -1 to 1 */
    for (i=0; i<N1/2; i++) {
      PetscReal tmp;
      
      tmp = x[i];
      x[i] = x[N1-1-i];
      x[N1-1-i] = tmp;
    }
    *_xi = x;
  } else {
    ierr = PetscFree(x);CHKERRQ(ierr);
  }
  
  if (_npoints) {
    *_npoints = N1;
  }
  if (_w) {
    *_w = w;
  } else {
    ierr = PetscFree(w);CHKERRQ(ierr);
  }
  ierr = PetscFree(xold);CHKERRQ(ierr);
  ierr = PetscFree(P);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode MatComputeConditionNumber(Mat A,PetscReal *cond)
{
  PetscReal *realpt,*complexpt,*nrmeigs;
  PetscInt rank,i;
  KSP kspV;
  PC pc;
  Vec x,y;
  PetscErrorCode ierr;
  
  ierr = MatCreateVecs(A,&y,&x);CHKERRQ(ierr);
  ierr = VecSet(y,1.0);CHKERRQ(ierr);
  
  ierr = KSPCreate(PETSC_COMM_SELF,&kspV);CHKERRQ(ierr);
  ierr = KSPSetOperators(kspV,A,A);CHKERRQ(ierr);
  ierr = KSPSetType(kspV,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(kspV,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = KSPSolve(kspV,y,x);CHKERRQ(ierr);
  
  ierr = MatGetSize(A,&rank,0);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*rank,&realpt);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*rank,&complexpt);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*rank,&nrmeigs);CHKERRQ(ierr);
  
  ierr = KSPComputeEigenvaluesExplicitly(kspV,rank,realpt,complexpt);CHKERRQ(ierr);
  for (i=0; i<rank; i++) {
    nrmeigs[i] = PetscSqrtReal( realpt[i]*realpt[i] + complexpt[i]*complexpt[i]);
  }
  ierr = PetscSortReal(rank,nrmeigs);CHKERRQ(ierr);
  
  *cond = nrmeigs[rank-1]/nrmeigs[0];
  
  ierr = PetscFree(nrmeigs);CHKERRQ(ierr);
  ierr = PetscFree(realpt);CHKERRQ(ierr);
  ierr = PetscFree(complexpt);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = KSPDestroy(&kspV);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode TabulateBasis1d_CLEGENDRE(PetscInt npoints,PetscReal xi[],PetscInt order,PetscInt *_nbasis,PetscReal ***_Ni)
{
  PetscErrorCode ierr;
  PetscReal **Ni,*xilocal,**basis_coeff, *monomials;
  PetscInt i,j,k,p;
  PetscInt nbasis,cnt;
  Mat A;
  Vec x,y;
  KSP ksp;
  PC pc;
  
  
  ierr = CreateGLLCoordsWeights(order,&nbasis,&xilocal,NULL);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscReal)*nbasis,&monomials);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscReal*)*npoints,&Ni);CHKERRQ(ierr);
  for (i=0; i<npoints; i++) {
    ierr = PetscMalloc(sizeof(PetscReal)*nbasis,&Ni[i]);CHKERRQ(ierr);
  }
  
  ierr = PetscMalloc(sizeof(PetscReal*)*nbasis,&basis_coeff);CHKERRQ(ierr);
  for (i=0; i<nbasis; i++) {
    ierr = PetscMalloc(sizeof(PetscReal)*nbasis,&basis_coeff[i]);CHKERRQ(ierr);
  }
  
  /* generate all the basis coefficients */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nbasis,nbasis,NULL,&A);CHKERRQ(ierr);
  for (k=0; k<nbasis; k++) {
    PetscReal xil,Aij;
    
    xil  = xilocal[k];
    
    cnt = 0;
    for (i=0; i<nbasis; i++) {
      Aij = PetscPowReal(xil,(PetscReal)i);
      ierr = MatSetValue(A,k,cnt,Aij,INSERT_VALUES);CHKERRQ(ierr);
      cnt++;
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  {
    PetscReal cond;
    PetscBool compute_vandermonde_condition = PETSC_FALSE;
    
    ierr = PetscOptionsGetBool(NULL,NULL,"-compute_vandermonde_condition",&compute_vandermonde_condition,NULL);CHKERRQ(ierr);
    if (compute_vandermonde_condition) {
      
      PetscPrintf(PETSC_COMM_WORLD,"Computing condition number of Vandermonde matrix\n");
      ierr = MatComputeConditionNumber(A,&cond);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"cond(V) = %1.6e \n",cond);
    }
  }
  
  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  
  ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp,"basis_");CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  
  for (k=0; k<nbasis; k++) {
    const PetscScalar *LA_x;
    
    ierr = VecZeroEntries(y);CHKERRQ(ierr);
    ierr = VecSetValue(y,k,1.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
    
    ierr = KSPSolve(ksp,y,x);CHKERRQ(ierr);
    
    ierr = VecGetArrayRead(x,&LA_x);CHKERRQ(ierr);
    for (i=0; i<nbasis; i++) {
      basis_coeff[k][i] = LA_x[i];
    }
    ierr = VecRestoreArrayRead(x,&LA_x);CHKERRQ(ierr);
  }
  
  /* evaluate basis at each xi[] */
  for (p=0; p<npoints; p++) {
    
    /* generate all monomials for point, p */
    cnt = 0;
    for (i=0; i<nbasis; i++) {
      monomials[cnt] = PetscPowReal((PetscReal)xi[p],(PetscReal)i);
      cnt++;
    }
    
    for (i=0; i<nbasis; i++) {
      Ni[p][i] = 0.0;
      
      for (j=0; j<nbasis; j++) {
        Ni[p][i] += basis_coeff[i][j] * monomials[j];
      }
      if (PetscAbsReal(Ni[p][i]) < 1.0e-12) {
        Ni[p][i] = 0.0;
      }
    }
    
    /*
     printf("p = %d (xi = %+1.4e) N = [",p,xi[p]);
     for (i=0; i<nbasis; i++) {
     printf(" %+1.4e ",Ni[p][i]);
     }
     printf("]\n");
     */
  }
  
  //for (p=0; p<npoints; p++) {
  //  ierr = PetscFree(Ni[p]);CHKERRQ(ierr);
  //}
  //ierr = PetscFree(Ni);CHKERRQ(ierr);
  *_Ni = Ni;
  *_nbasis = nbasis;
  
  ierr = PetscFree(monomials);CHKERRQ(ierr);
  for (i=0; i<nbasis; i++) {
    ierr = PetscFree(basis_coeff[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(basis_coeff);CHKERRQ(ierr);
  ierr = PetscFree(xilocal);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode TabulateBasisDerivatives1d_CLEGENDRE(PetscInt npoints,PetscReal xi[],PetscInt order,PetscInt *_nbasis,PetscReal ***_GNix)
{
  PetscErrorCode ierr;
  PetscReal **GNix,*xilocal,**basis_coeff, *monomials;
  PetscInt i,j,k,p;
  PetscInt nbasis,cnt;
  Mat A;
  Vec x,y;
  KSP ksp;
  PC pc;
  
  
  ierr = CreateGLLCoordsWeights(order,&nbasis,&xilocal,NULL);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscReal)*nbasis,&monomials);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscReal*)*npoints,&GNix);CHKERRQ(ierr);
  for (i=0; i<npoints; i++) {
    ierr = PetscMalloc(sizeof(PetscReal)*nbasis,&GNix[i]);CHKERRQ(ierr);
  }
  
  ierr = PetscMalloc(sizeof(PetscReal*)*nbasis,&basis_coeff);CHKERRQ(ierr);
  for (i=0; i<nbasis; i++) {
    ierr = PetscMalloc(sizeof(PetscReal)*nbasis,&basis_coeff[i]);CHKERRQ(ierr);
  }
  
  /* generate all the basis coefficients */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nbasis,nbasis,NULL,&A);CHKERRQ(ierr);
  for (k=0; k<nbasis; k++) {
    PetscReal xil,Aij;
    
    xil  = xilocal[k];
    
    cnt = 0;
    for (i=0; i<nbasis; i++) {
      Aij = PetscPowReal(xil,(PetscReal)i);
      ierr = MatSetValue(A,k,cnt,Aij,INSERT_VALUES);CHKERRQ(ierr);
      cnt++;
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  {
    PetscReal cond;
    PetscBool compute_vandermonde_condition = PETSC_FALSE;
    
    ierr = PetscOptionsGetBool(NULL,NULL,"-compute_vandermonde_condition",&compute_vandermonde_condition,NULL);CHKERRQ(ierr);
    if (compute_vandermonde_condition) {
      
      PetscPrintf(PETSC_COMM_WORLD,"Computing condition number of Vandermonde matrix\n");
      ierr = MatComputeConditionNumber(A,&cond);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"cond(V) = %1.6e \n",cond);
    }
  }
  
  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  
  ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp,"basis_");CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  
  for (k=0; k<nbasis; k++) {
    const PetscScalar *LA_x;
    
    ierr = VecZeroEntries(y);CHKERRQ(ierr);
    ierr = VecSetValue(y,k,1.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
    
    ierr = KSPSolve(ksp,y,x);CHKERRQ(ierr);
    
    ierr = VecGetArrayRead(x,&LA_x);CHKERRQ(ierr);
    for (i=0; i<nbasis; i++) {
      basis_coeff[k][i] = LA_x[i];
    }
    ierr = VecRestoreArrayRead(x,&LA_x);CHKERRQ(ierr);
  }
  
  /* evaluate basis at each xi[] */
  for (p=0; p<npoints; p++) {
    
    /* generate all monomials for point, p */
    cnt = 0;
    for (i=0; i<nbasis; i++) {
      PetscReal dm_dx;
      
      if (i == 0) {
        dm_dx = 0.0;
        } else {
          dm_dx = ((PetscReal)i)*PetscPowReal((PetscReal)xi[p],(PetscReal)(i-1));
        }
      
      monomials[cnt] = dm_dx;
      cnt++;
    }
    
    for (i=0; i<nbasis; i++) {
      GNix[p][i] = 0.0;
      
      for (j=0; j<nbasis; j++) {
        GNix[p][i] += basis_coeff[i][j] * monomials[j];
      }
      if (PetscAbsReal(GNix[p][i]) < 1.0e-12) {
        GNix[p][i] = 0.0;
      }
    }
    
    /*
     printf("p = %d (xi = %+1.4e) dN_dx = [",p,xi[p]);
     for (i=0; i<nbasis; i++) {
     printf(" %+1.4e ",GNix[p][i]);
     }
     printf("]\n");
     */
  }
  
  ierr = PetscFree(monomials);CHKERRQ(ierr);
  for (i=0; i<nbasis; i++) {
    ierr = PetscFree(basis_coeff[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(basis_coeff);CHKERRQ(ierr);
  ierr = PetscFree(xilocal);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  
  *_nbasis = nbasis;
  *_GNix = GNix;
  
  PetscFunctionReturn(0);
}

PetscErrorCode TabulateBasisDerivativesTensorProduct2d(PetscInt order,PetscReal ***_dN_dxi,PetscReal ***_dN_deta)
{
  PetscErrorCode ierr;
  PetscReal *xiq,**dphi_xi,**dN_dxi,**dN_deta;
  PetscInt qpoint,k,i,j,qi,qj,nqp,nbasis;
  
  ierr = CreateGLLCoordsWeights(order,&nqp,&xiq,NULL);CHKERRQ(ierr);
  ierr = TabulateBasisDerivatives1d_CLEGENDRE(nqp,xiq,order,&nbasis,&dphi_xi);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscReal*)*nqp*nqp,&dN_dxi);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal*)*nqp*nqp,&dN_deta);CHKERRQ(ierr);
  for (i=0; i<nqp*nqp; i++) {
    ierr = PetscMalloc(sizeof(PetscReal)*nbasis*nbasis,&dN_dxi[i]);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscReal)*nbasis*nbasis,&dN_deta[i]);CHKERRQ(ierr);
  }
  
  qpoint = 0;
  for (qj=0; qj<nqp; qj++) {
    for (qi=0; qi<nqp; qi++) {
      
      k = 0;
      for (j=0; j<nbasis; j++) {
        for (i=0; i<nbasis; i++) {
          PetscReal phi_xi,phi_eta;
          
          phi_xi = 0.0;
          if (qi == i) phi_xi = 1.0;
          
          phi_eta = 0.0;
          if (qj == j) phi_eta = 1.0;
          
          dN_dxi[qpoint][k]  = dphi_xi[qi][i] * phi_eta;
          dN_deta[qpoint][k] = phi_xi * dphi_xi[qj][j];
          
          k++;
        }}
      qpoint++;
    }}
  
  /* viewer */
  /*
   for (k=0; k<nqp*nqp; k++) {
   printf("qp[%d]: dNdxi  = [ ",k);
   for (j=0; j<nbasis*nbasis; j++) {
   printf(" %+1.4e ",dN_dxi[k][j]);
   } printf("]\n");
   
   printf("qp[%d]: dNdeta = [ ",k);
   for (j=0; j<nbasis*nbasis; j++) {
   printf(" %+1.4e ",dN_deta[k][j]);
   } printf("]\n");
   }
   */
  
  /* free up mempry */
  ierr = PetscFree(xiq);CHKERRQ(ierr);
  for (k=0; k<nqp; k++) {
    ierr = PetscFree(dphi_xi[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dphi_xi);CHKERRQ(ierr);
  
  if (_dN_dxi) { *_dN_dxi = dN_dxi; }
  else {
    for (k=0; k<nqp*nqp; k++) {
      ierr = PetscFree(dN_dxi[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dN_dxi);CHKERRQ(ierr);
  }
  
  if (_dN_deta) { *_dN_deta = dN_deta; }
  else {
    for (k=0; k<nqp*nqp; k++) {
      ierr = PetscFree(dN_deta[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dN_deta);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode TabulateBasisDerivativesAtPointTensorProduct2d(PetscReal xiq[],PetscInt order,PetscReal ***_dN_dxi,PetscReal ***_dN_deta)
{
  PetscErrorCode ierr;
  PetscReal **dphi_xi,**Ni_xi,**dphi_eta,**Ni_eta,**dN_dxi,**dN_deta;
  PetscInt qpoint,k,i,j,q,nqp,nbasis;
  
  nqp = 1;
  ierr = TabulateBasisDerivatives1d_CLEGENDRE(nqp,&xiq[0],order,&nbasis,&dphi_xi);CHKERRQ(ierr);
  ierr = TabulateBasis1d_CLEGENDRE(nqp,&xiq[0],order,&nbasis,&Ni_xi);CHKERRQ(ierr);
  
  ierr = TabulateBasisDerivatives1d_CLEGENDRE(nqp,&xiq[1],order,&nbasis,&dphi_eta);CHKERRQ(ierr);
  ierr = TabulateBasis1d_CLEGENDRE(nqp,&xiq[1],order,&nbasis,&Ni_eta);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscReal*)*nqp,&dN_dxi);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal*)*nqp,&dN_deta);CHKERRQ(ierr);
  for (i=0; i<nqp; i++) {
    ierr = PetscMalloc(sizeof(PetscReal)*nbasis*nbasis,&dN_dxi[i]);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscReal)*nbasis*nbasis,&dN_deta[i]);CHKERRQ(ierr);
  }
  
  qpoint = 0;
  for (q=0; q<nqp; q++) {
    
    k = 0;
    for (j=0; j<nbasis; j++) {
      for (i=0; i<nbasis; i++) {
        PetscReal phi_xi,phi_eta;
        
        phi_xi = Ni_xi[q][i];
        phi_eta = Ni_eta[q][j];
        
        dN_dxi[qpoint][k]  = dphi_xi[q][i] * phi_eta;
        dN_deta[qpoint][k] = phi_xi * dphi_eta[q][j];
        k++;
      }}
    qpoint++;
  }
  
  /* viewer */
  /*
   for (k=0; k<nqp; k++) {
   printf("qp[%d]: dNdxi  = [ ",k);
   for (j=0; j<nbasis*nbasis; j++) {
   printf(" %+1.4e ",dN_dxi[k][j]);
   } printf("]\n");
   
   printf("qp[%d]: dNdeta = [ ",k);
   for (j=0; j<nbasis*nbasis; j++) {
   printf(" %+1.4e ",dN_deta[k][j]);
   } printf("]\n");
   }
   */
  
  /* free up mempry */
  for (k=0; k<nqp; k++) {
    ierr = PetscFree(dphi_xi[k]);CHKERRQ(ierr);
    ierr = PetscFree(dphi_eta[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dphi_xi);CHKERRQ(ierr);
  ierr = PetscFree(dphi_eta);CHKERRQ(ierr);
  
  for (k=0; k<nqp; k++) {
    ierr = PetscFree(Ni_xi[k]);CHKERRQ(ierr);
    ierr = PetscFree(Ni_eta[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(Ni_xi);CHKERRQ(ierr);
  ierr = PetscFree(Ni_eta);CHKERRQ(ierr);
  
  if (_dN_dxi) { *_dN_dxi = dN_dxi; }
  else {
    for (k=0; k<nqp*nqp; k++) {
      ierr = PetscFree(dN_dxi[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dN_dxi);CHKERRQ(ierr);
  }
  
  if (_dN_deta) { *_dN_deta = dN_deta; }
  else {
    for (k=0; k<nqp*nqp; k++) {
      ierr = PetscFree(dN_deta[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dN_deta);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxCreate(SpecFECtx *c)
{
  SpecFECtx ctx;
  PetscErrorCode ierr;
  
  ierr = PetscMalloc(sizeof(struct _p_SpecFECtx),&ctx);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx,sizeof(struct _p_SpecFECtx));CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&ctx->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&ctx->size);CHKERRQ(ierr);
  *c = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxCreateENMap2d_SEQ(SpecFECtx c)
{
  PetscErrorCode ierr;
  PetscInt ni0,nj0,i,j,ei,ej,ecnt,*emap,nid;
  
  ierr = PetscMalloc(sizeof(PetscInt)*c->ne*c->npe,&c->element);CHKERRQ(ierr);
  ierr = PetscMemzero(c->element,sizeof(PetscInt)*c->ne*c->npe);CHKERRQ(ierr);
  
  ecnt = 0;
  for (ej=0; ej<c->my; ej++) {
    nj0 = ej*(c->npe_1d-1);
    
    for (ei=0; ei<c->mx; ei++) {
      ni0 = ei*(c->npe_1d-1);
      
      emap = &c->element[c->npe*ecnt];
      
      for (j=0; j<c->npe_1d; j++) {
        for (i=0; i<c->npe_1d; i++) {
          
          nid = (ni0 + i) + (nj0 + j) * c->nx_g;
          emap[i+j*c->npe_1d] = nid;
        }
      }
      
      ecnt++;
    }
  }
  
  PetscFunctionReturn(0);
}

/* Creates domain over [0,1]^d - scale later */
PetscErrorCode SpecFECtxCreateMeshCoords2d_SEQ(SpecFECtx c)
{
  PetscErrorCode ierr;
  Vec coor;
  DM cdm;
  DMDACoor2d **LA_coor2d;
  PetscInt ei,ej,i,j,ni0,nj0;
  PetscReal dx,dy,x0,y0;
  
  ierr = DMDASetUniformCoordinates(c->dm,0.0,1.0,0.0,1.0,0,0);CHKERRQ(ierr);
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(c->dm,&cdm);CHKERRQ(ierr);
  
  dx = 1.0/((PetscReal)c->mx_g);
  dy = 1.0/((PetscReal)c->my_g);
  ierr = DMDAVecGetArray(cdm,coor,&LA_coor2d);CHKERRQ(ierr);
  for (ej=0; ej<c->my; ej++) {
    
    for (ei=0; ei<c->mx; ei++) {
      x0 = 0.0 + ei*dx;
      y0 = 0.0 + ej*dy;
      
      ni0 = ei*(c->npe_1d-1);
      nj0 = ej*(c->npe_1d-1);
      
      for (j=0; j<c->npe_1d; j++) {
        for (i=0; i<c->npe_1d; i++) {
          LA_coor2d[nj0+j][ni0+i].x = 0.5*(c->xi1d[i]+1.0)*dx + x0;
          LA_coor2d[nj0+j][ni0+i].y = 0.5*(c->xi1d[j]+1.0)*dy + y0;
          
          //if ((ej==0) && (j==0)) {
          //  printf("[e %d,i %d] xc %+1.4e\n",ei,i,LA_coor2d[nj0+j][ni0+i].x*4.0e3-2.0e3);
          //}
          
        }
      }
    }
  }
  ierr = DMDAVecRestoreArray(cdm,coor,&LA_coor2d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxScaleMeshCoords(SpecFECtx c,PetscReal scale[],PetscReal shift[])
{
  PetscErrorCode ierr;
  Vec coor,lcoor;
  DM cdm;
  
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
  
  if (scale) {
    if (c->dim >= 1) ierr = VecStrideScale(coor,0,scale[0]);CHKERRQ(ierr);
    if (c->dim >= 2) ierr = VecStrideScale(coor,1,scale[1]);CHKERRQ(ierr);
    if (c->dim == 3) ierr = VecStrideScale(coor,2,scale[2]);CHKERRQ(ierr);
  }
  if (shift) {
    Vec ss;
    
    ierr = VecDuplicate(coor,&ss);CHKERRQ(ierr);
    
    if (c->dim >= 1) {
      ierr = VecZeroEntries(ss);CHKERRQ(ierr);
      ierr = VecStrideSet(ss,0,shift[0]);CHKERRQ(ierr);
      ierr = VecAXPY(coor,1.0,ss);CHKERRQ(ierr);
    }
    if (c->dim >= 2) {
      ierr = VecZeroEntries(ss);CHKERRQ(ierr);
      ierr = VecStrideSet(ss,1,shift[1]);CHKERRQ(ierr);
      ierr = VecAXPY(coor,1.0,ss);CHKERRQ(ierr);
    }
    if (c->dim >= 3) {
      ierr = VecZeroEntries(ss);CHKERRQ(ierr);
      ierr = VecStrideSet(ss,2,shift[2]);CHKERRQ(ierr);
      ierr = VecAXPY(coor,1.0,ss);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&ss);CHKERRQ(ierr);
  }
  
  ierr = DMGetCoordinateDM(c->dm,&cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(c->dm,&lcoor);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(c->dm,coor,INSERT_VALUES,lcoor);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(c->dm,coor,INSERT_VALUES,lcoor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECreateStress(SpecFECtx c,PetscReal **s)
{
  PetscErrorCode ierr;
  ierr = SpecFECreateQuadratureField(c,3,s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxCreateMesh_SEQ(SpecFECtx c,PetscInt dim,PetscInt mx,PetscInt my,PetscInt mz,PetscInt basisorder,PetscInt ndofs)
{
  PetscErrorCode ierr;
  PetscInt stencil_width,i,j;
  
  c->dim = dim;
  c->mx = mx;
  c->my = my;
  c->mz = mz;
  c->mx_g = mx;
  c->my_g = my;
  c->mz_g = mz;
  c->basisorder = basisorder;
  c->dofs = ndofs;
  
  c->nx_g = basisorder*mx + 1;
  c->ny_g = basisorder*my + 1;
  c->nz_g = basisorder*mz + 1;
  
  ierr = CreateGLLCoordsWeights(basisorder,&c->npe_1d,&c->xi1d,&c->w1d);CHKERRQ(ierr);
  
  stencil_width = 1;
  switch (dim) {
    case 2:
    c->npe = c->npe_1d * c->npe_1d;
    c->ne = mx * my;
    c->ne_g = mx * my;
    
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                        c->nx_g,c->ny_g,PETSC_DECIDE,PETSC_DECIDE,ndofs,stencil_width,NULL,NULL,&c->dm);CHKERRQ(ierr);
    ierr = DMSetUp(c->dm);CHKERRQ(ierr);
    ierr = SpecFECtxCreateENMap2d_SEQ(c);CHKERRQ(ierr);
    ierr = SpecFECtxCreateMeshCoords2d_SEQ(c);CHKERRQ(ierr);
    
    /* tensor product for weights */
    ierr = PetscMalloc(sizeof(PetscReal)*c->npe,&c->w);CHKERRQ(ierr);
    for (j=0; j<c->npe_1d; j++) {
      for (i=0; i<c->npe_1d; i++) {
        c->w[i+j*c->npe_1d] = c->w1d[i] * c->w1d[j];
      }
    }
    
    ierr = TabulateBasisDerivativesTensorProduct2d(basisorder,&c->dN_dxi,&c->dN_deta);CHKERRQ(ierr);
    ierr = TabulateBasisDerivativesTensorProduct2d(basisorder,&c->dN_dx,&c->dN_dy);CHKERRQ(ierr);
    
    break;
  }
  
  c->nqp = c->npe;
  
  ierr = PetscMalloc(sizeof(QPntIsotropicElastic)*c->ne,&c->cell_data);CHKERRQ(ierr);
  ierr = PetscMemzero(c->cell_data,sizeof(QPntIsotropicElastic)*c->ne);CHKERRQ(ierr);

  ierr = PetscMalloc1(c->ne * c->nqp,&c->dr_qp_data);CHKERRQ(ierr);
  ierr = PetscMemzero(c->dr_qp_data,sizeof(DRVar)*c->ne*c->nqp);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dim,&c->elbuf_coor);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs,&c->elbuf_field);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs,&c->elbuf_field2);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs,&c->elbuf_field3);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*c->npe*c->dofs,&c->elbuf_dofs);CHKERRQ(ierr);
  
  ierr = SpecFECreateStress(c,&c->sigma);CHKERRQ(ierr);
  //ierr = SpecFECreateQuadratureField(c,4,&c->gradu);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 Degree 4 has 5 basis in each direction
 |           |
 0--1--2--3--4
*/
PetscErrorCode SpecFECtxGetCornerBasis_MPI(SpecFECtx c,PetscInt *si,PetscInt *si_g,PetscInt *sj,PetscInt *sj_g)
{
  PetscInt gi,gj,m,n,k;
  PetscErrorCode ierr;

  ierr = DMDAGetGhostCorners(c->dm,&gi,&gj,NULL,&m,&n,NULL);CHKERRQ(ierr);
  /*printf("rank %d: gi,gj %d %d  npe %d\n",c->rank,gi,gj,c->npe_1d);*/
  for (k=0; k<m; k++) {
    if (((gi+k) % (c->npe_1d-1)) == 0) {
      *si = k;
      *si_g = gi+k;
      break;
    }
  }
  for (k=0; k<n; k++) {
    if (((gj+k) % (c->npe_1d-1)) == 0) {
      *sj = k;
      *sj_g = gj+k;
      break;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxGetLocalBoundingBox(SpecFECtx c,PetscReal gmin[],PetscReal gmax[])
{
  PetscErrorCode ierr;
  PetscInt si[]={0,0},si_g[]={0,0},m,n,ii,jj;
  const PetscReal *LA_coor;
  Vec coor;
  
  ierr = SpecFECtxGetCornerBasis_MPI(c,&si[0],&si_g[0],&si[1],&si_g[1]);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(c->dm,NULL,NULL,NULL,&m,&n,NULL);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  ii = si[0];
  jj = si[1];
  gmin[0] = LA_coor[2*(ii + jj*m)+0];
  gmin[1] = LA_coor[2*(ii + jj*m)+1];
  ii = si[0] + c->mx * c->basisorder;
  jj = si[1] + c->my * c->basisorder;
  gmax[0] = LA_coor[2*(ii + jj*m)+0];
  gmax[1] = LA_coor[2*(ii + jj*m)+1];
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxCreateENMap2d_MPI(SpecFECtx c)
{
  PetscErrorCode ierr;
  PetscInt ni0,nj0,i,j,ei,ej,ecnt,*emap,nid;
  PetscInt si,si_g,sj,sj_g,nx_local;
  
  
  ierr = PetscMalloc(sizeof(PetscInt)*c->ne*c->npe,&c->element);CHKERRQ(ierr);
  ierr = PetscMemzero(c->element,sizeof(PetscInt)*c->ne*c->npe);CHKERRQ(ierr);
  
  ierr = SpecFECtxGetCornerBasis_MPI(c,&si,&si_g,&sj,&sj_g);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(c->dm,NULL,NULL,NULL,&nx_local,NULL,NULL);CHKERRQ(ierr);
  /*printf("rank %d : %d %d x %d %d\n",c->rank,si,si_g,sj,sj_g);*/
  
  ecnt = 0;
  for (ej=0; ej<c->my; ej++) {
    nj0 = sj + ej*(c->npe_1d-1);
    
    for (ei=0; ei<c->mx; ei++) {
      ni0 = si + ei*(c->npe_1d-1);
      
      emap = &c->element[c->npe*ecnt];
      
      for (j=0; j<c->npe_1d; j++) {
        for (i=0; i<c->npe_1d; i++) {
          
          nid = (ni0 + i) + (nj0 + j) * nx_local;
          emap[i+j*c->npe_1d] = nid;
          //if (c->rank == 0) {
          //  printf("e %d : %d [max %d]\n",ecnt,nid,c->ne*c->npe);
          //}
        }
      }
      
      ecnt++;
    }
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxCreateMeshCoords2d_MPI(SpecFECtx c)
{
  PetscErrorCode ierr;
  Vec coor,gcoor;
  DM cdm;
  PetscInt ei,ej,i,j,ni0,nj0,si,si_g,sj,sj_g,gi,gj,m,n;
  PetscReal dx,dy,x0,y0;
  PetscReal *LA_coor;
  
  ierr = DMDASetUniformCoordinates(c->dm,0.0,1.0,0.0,1.0,0,0);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(c->dm,&cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);

  ierr = DMGetCoordinates(c->dm,&gcoor);CHKERRQ(ierr);
  ierr = VecZeroEntries(gcoor);CHKERRQ(ierr);
  
  ierr = SpecFECtxGetCornerBasis_MPI(c,&si,&si_g,&sj,&sj_g);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(c->dm,&gi,&gj,NULL,&m,&n,NULL);CHKERRQ(ierr);

  dx = 1.0/((PetscReal)c->mx_g);
  dy = 1.0/((PetscReal)c->my_g);
  ierr = VecGetArray(coor,&LA_coor);CHKERRQ(ierr);
  for (ej=0; ej<c->my; ej++) {
    
    for (ei=0; ei<c->mx; ei++) {
      if ( si >= m*n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Out of range-si");
      if ( sj >= m*n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Out of range-sj");

      x0 = LA_coor[2*(si + sj*m)+0] + ei*dx;
      y0 = LA_coor[2*(si + sj*m)+1] + ej*dy;

      ni0 = si + ei*(c->npe_1d-1);
      nj0 = sj + ej*(c->npe_1d-1);
      
      //printf("rank %d : (%d,%d) -> %d %d  %+1.4e %+1.4e\n",c->rank,ei,ej,ni0,nj0,x0,y0);
      
      for (j=0; j<c->npe_1d; j++) {
        for (i=0; i<c->npe_1d; i++) {
          if ( (ni0+i)+(nj0+j)*m >= m*n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Local index out of range");

          LA_coor[2*((ni0+i) + (nj0+j)*m)+0] = x0 + 0.5*(c->xi1d[i]+1.0)*dx;
          LA_coor[2*((ni0+i) + (nj0+j)*m)+1] = y0 + 0.5*(c->xi1d[j]+1.0)*dy;
          
          ierr = VecSetValueLocal(gcoor,2*((ni0+i) + (nj0+j)*m)+0,x0 + 0.5*(c->xi1d[i]+1.0)*dx,INSERT_VALUES);CHKERRQ(ierr);
          ierr = VecSetValueLocal(gcoor,2*((ni0+i) + (nj0+j)*m)+1,y0 + 0.5*(c->xi1d[j]+1.0)*dy,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecRestoreArray(coor,&LA_coor);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(gcoor);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(gcoor);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxCreateMesh_MPI(SpecFECtx c,PetscInt dim,PetscInt mx,PetscInt my,PetscInt mz,PetscInt basisorder,PetscInt ndofs)
{
  PetscErrorCode ierr;
  PetscInt stencil_width,i,j;
  DM dm_ref;
  PetscInt ranks[3];
  PetscInt r,*lx,*ly;
  const PetscInt *lx_ref,*ly_ref;
  DMDALocalInfo info;
  
  c->dim = dim;
  c->mx_g = mx;
  c->my_g = my;
  c->mz_g = mz;
  c->basisorder = basisorder;
  c->dofs = ndofs;
  
  c->nx_g = basisorder*mx + 1;
  c->ny_g = basisorder*my + 1;
  c->nz_g = basisorder*mz + 1;
  
  ierr = CreateGLLCoordsWeights(basisorder,&c->npe_1d,&c->xi1d,&c->w1d);CHKERRQ(ierr);
  
  stencil_width = 1;
  switch (dim) {
    case 2:
    c->npe = c->npe_1d * c->npe_1d;
    c->ne_g = mx * my;

    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                        c->mx_g,c->my_g,PETSC_DECIDE,PETSC_DECIDE,1,0,NULL,NULL,&dm_ref);CHKERRQ(ierr);
    ierr = DMSetUp(dm_ref);CHKERRQ(ierr);
    /*ierr = DMView(dm_ref,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

    ierr = DMDAGetInfo(dm_ref,NULL,NULL,NULL,NULL,&ranks[0],&ranks[1],NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DMDAGetOwnershipRanges(dm_ref,&lx_ref,&ly_ref,NULL);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(dm_ref,&info);CHKERRQ(ierr);
    
    c->mx = info.xm;
    c->my = info.ym;
    c->ne = c->mx * c->my;
    
    ierr = PetscMalloc1(ranks[0],&lx);CHKERRQ(ierr);
    ierr = PetscMalloc1(ranks[1],&ly);CHKERRQ(ierr);
    for (r=0; r<ranks[0]; r++) {
      lx[r] = lx_ref[r] * (c->npe_1d - 1);
    }
    lx[ranks[0]-1]++;

    /*for (r=0; r<ranks[0]; r++)  PetscPrintf(PETSC_COMM_WORLD,"npoints-i[%D] %D \n",r,lx[r]);*/
    
    for (r=0; r<ranks[1]; r++) {
      ly[r] = ly_ref[r] * (c->npe_1d - 1);
    }
    ly[ranks[1]-1]++;
    
    /*for (r=0; r<ranks[1]; r++)  PetscPrintf(PETSC_COMM_WORLD,"npoints-j[%D] %D \n",r,ly[r]);*/
    
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                        c->nx_g,c->ny_g,ranks[0],ranks[1],ndofs,stencil_width,lx,ly,&c->dm);CHKERRQ(ierr);
    ierr = DMSetUp(c->dm);CHKERRQ(ierr);
    /*ierr = DMView(c->dm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

    ierr = SpecFECtxCreateENMap2d_MPI(c);CHKERRQ(ierr);
    ierr = SpecFECtxCreateMeshCoords2d_MPI(c);CHKERRQ(ierr);
    
    /* tensor product for weights */
    ierr = PetscMalloc(sizeof(PetscReal)*c->npe,&c->w);CHKERRQ(ierr);
    for (j=0; j<c->npe_1d; j++) {
      for (i=0; i<c->npe_1d; i++) {
        c->w[i+j*c->npe_1d] = c->w1d[i] * c->w1d[j];
      }
    }
    
    ierr = TabulateBasisDerivativesTensorProduct2d(basisorder,&c->dN_dxi,&c->dN_deta);CHKERRQ(ierr);
    ierr = TabulateBasisDerivativesTensorProduct2d(basisorder,&c->dN_dx,&c->dN_dy);CHKERRQ(ierr);
    
    ierr = PetscFree(lx);CHKERRQ(ierr);
    ierr = PetscFree(ly);CHKERRQ(ierr);
    ierr = DMDestroy(&dm_ref);CHKERRQ(ierr);
    break;
  }
  
  c->nqp = c->npe;
  
  ierr = PetscMalloc(sizeof(QPntIsotropicElastic)*c->ne,&c->cell_data);CHKERRQ(ierr);
  ierr = PetscMemzero(c->cell_data,sizeof(QPntIsotropicElastic)*c->ne);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dim,&c->elbuf_coor);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs,&c->elbuf_field);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs,&c->elbuf_field2);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs,&c->elbuf_field3);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*c->npe*c->dofs,&c->elbuf_dofs);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxCreateMesh(SpecFECtx c,PetscInt dim,PetscInt mx,PetscInt my,PetscInt mz,PetscInt basisorder,PetscInt ndofs)
{
  PetscMPIInt size;
  PetscErrorCode ierr;
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = SpecFECtxCreateMesh_SEQ(c,dim,mx,my,mz,basisorder,ndofs);CHKERRQ(ierr);
  } else {
    ierr = SpecFECtxCreateMesh_MPI(c,dim,mx,my,mz,basisorder,ndofs);CHKERRQ(ierr);
  }
  ierr = DMDASetFieldName(c->dm,0,"_x");CHKERRQ(ierr);
  ierr = DMDASetFieldName(c->dm,1,"_y");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxSetConstantMaterialProperties(SpecFECtx c,PetscReal lambda,PetscReal mu,PetscReal rho)
{
  PetscInt q;
  
  for (q=0; q<c->ne; q++) {
    c->cell_data[q].lambda = lambda;
    c->cell_data[q].mu     = mu;
    c->cell_data[q].rho    = rho;
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxSetConstantMaterialProperties_Velocity(SpecFECtx c,PetscReal Vp,PetscReal Vs,PetscReal rho)
{
  PetscErrorCode ierr;
  PetscReal mu,lambda;
  
  mu = Vs * Vs * rho;
  lambda = Vp * Vp * rho - 2.0 * mu;
  ierr = SpecFECtxSetConstantMaterialProperties(c,lambda,mu,rho);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"  [material]     Vp = %1.8e\n",Vp);
  PetscPrintf(PETSC_COMM_WORLD,"  [material]     Vs = %1.8e\n",Vs);
  PetscPrintf(PETSC_COMM_WORLD,"  [material] lambda = %1.8e\n",lambda);
  PetscPrintf(PETSC_COMM_WORLD,"  [material]     mu = %1.8e\n",mu);
  PetscPrintf(PETSC_COMM_WORLD,"  [material]    rho = %1.8e\n",rho);
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxSetPerturbedMaterialProperties_Velocity(SpecFECtx c,PetscReal Vp0,PetscReal delta_Vp,PetscReal Vs0,PetscReal delta_Vs,PetscReal rho0,PetscReal delta_rho)
{
  PetscErrorCode ierr;
  Vec Vp,Vs,rho;
  PetscRandom r;
  const PetscReal *LA_Vp,*LA_Vs,*LA_rho;
  PetscInt e;
  
  ierr = VecCreate(PETSC_COMM_WORLD,&Vp);CHKERRQ(ierr);
  ierr = VecSetSizes(Vp,c->ne,c->ne_g);CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vp);CHKERRQ(ierr);
  ierr = VecDuplicate(Vp,&Vs);CHKERRQ(ierr);
  ierr = VecDuplicate(Vp,&rho);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetType(r,PETSCRAND48);CHKERRQ(ierr);

  ierr = PetscRandomSetInterval(r,Vp0-delta_Vp,Vp0+delta_Vp);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(r,1);CHKERRQ(ierr);
  ierr = PetscRandomSeed(r);CHKERRQ(ierr);
  ierr = VecSetRandom(Vp,r);CHKERRQ(ierr);

  ierr = PetscRandomSetInterval(r,Vs0-delta_Vs,Vs0+delta_Vs);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(r,2);CHKERRQ(ierr);
  ierr = PetscRandomSeed(r);CHKERRQ(ierr);
  ierr = VecSetRandom(Vs,r);CHKERRQ(ierr);
  
  ierr = PetscRandomSetInterval(r,rho0-delta_rho,rho0+delta_rho);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(r,3);CHKERRQ(ierr);
  ierr = PetscRandomSeed(r);CHKERRQ(ierr);
  ierr = VecSetRandom(rho,r);CHKERRQ(ierr);

  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);

  ierr = VecGetArrayRead(Vp,&LA_Vp);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vs,&LA_Vs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(rho,&LA_rho);CHKERRQ(ierr);
  for (e=0; e<c->ne; e++) {
    PetscReal mu,lambda;

    mu     = LA_Vs[e] * LA_Vs[e] * LA_rho[e];
    lambda = LA_Vp[e] * LA_Vp[e] * LA_rho[e] - 2.0 * mu;
    
    c->cell_data[e].lambda = lambda;
    c->cell_data[e].mu     = mu;
    c->cell_data[e].rho    = LA_rho[e];
  }
  ierr = VecRestoreArrayRead(rho,&LA_rho);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vs,&LA_Vs);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vp,&LA_Vp);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"  [material]     Vp0 = %1.8e : delta = %+1.8e\n",Vp0,delta_Vp);
  PetscPrintf(PETSC_COMM_WORLD,"  [material]     Vs0 = %1.8e : delta = %+1.8e\n",Vs0,delta_Vs);
  PetscPrintf(PETSC_COMM_WORLD,"  [material]    rho0 = %1.8e : delta = %+1.8e\n",rho0,delta_rho);
  
  ierr = VecDestroy(&rho);CHKERRQ(ierr);
  ierr = VecDestroy(&Vs);CHKERRQ(ierr);
  ierr = VecDestroy(&Vp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void ElementEvaluateGeometry_CellWiseConstant2d(PetscInt npe,PetscReal el_coords[],
                                                PetscInt nbasis,PetscReal *detJ)
{
  PetscReal J00,J11;
  PetscReal dx,dy;
  
  dx = el_coords[2*(nbasis-1)+0] - el_coords[2*0+0];
  dy = el_coords[2*(npe-1)+1]    - el_coords[2*0+1];
  
  J00 = 0.5 * dx;
  J11 = 0.5 * dy;
  
  *detJ = J00*J11;
}

void ElementEvaluateDerivatives_CellWiseConstant2d(PetscInt nqp,PetscInt npe,PetscReal el_coords[],
                                                   PetscInt nbasis,PetscReal **dN_dxi,PetscReal **dN_deta,
                                                   PetscReal **dN_dx,PetscReal **dN_dy)
{
  PetscInt k,q;
  PetscReal J00,J11,iJ00,iJ11;
  PetscReal dx,dy;
  
  dx = el_coords[2*(nbasis-1)+0] - el_coords[2*0+0];
  dy = el_coords[2*(npe-1)+1]    - el_coords[2*0+1];
  
  J00 = 0.5 * dx;
  J11 = 0.5 * dy;
  
  for (q=0; q<nqp; q++) {
    
    iJ00 = 1.0/J00;
    iJ11 = 1.0/J11;
    
    /* shape function derivatives */
    for (k=0; k<npe; k++) {
      dN_dx[q][k] = iJ00 * dN_dxi[q][k];
      dN_dy[q][k] = iJ11 * dN_deta[q][k];
    }
  }
}

/*
 Assemble rhs
 L(u) = - \int B^T D B u dV
 */
PetscErrorCode AssembleLinearForm_ElastoDynamics2d(SpecFECtx c,Vec u,Vec F)
{
  PetscErrorCode ierr;
  PetscInt  e,nqp,q,i,nbasis,ndof;
  PetscReal e_vec[3],sigma_vec[3];
  PetscInt  *element,*elnidx,*eldofs;
  PetscReal *fe,*ux,*uy,*elcoords,detJ,*field;
  Vec       coor,ul,fl;
  const PetscReal *LA_coor,*LA_u;
  QPntIsotropicElastic *celldata;
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  nqp      = c->nqp;
  ndof     = c->dofs;
  fe       = c->elbuf_field;
  element  = c->element;
  field    = c->elbuf_field2;
  
  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(c->dm,&ul);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(c->dm,u,INSERT_VALUES,ul);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(c->dm,u,INSERT_VALUES,ul);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ul,&LA_u);CHKERRQ(ierr);

  ierr = DMGetLocalVector(c->dm,&fl);CHKERRQ(ierr);
  ierr = VecZeroEntries(fl);CHKERRQ(ierr);

  ux = &field[0];
  uy = &field[nbasis];
  
  for (e=0; e<c->ne; e++) {
    /* get element -> node map */
    elnidx = &element[nbasis*e];
    
    
    /* generate dofs */
    for (i=0; i<nbasis; i++) {
      eldofs[2*i  ] = 2*elnidx[i];
      eldofs[2*i+1] = 2*elnidx[i]+1;
    }
    
    /* get element coordinates */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
    }
    
    /* get element displacements */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      ux[i] = LA_u[2*nidx  ];
      uy[i] = LA_u[2*nidx+1];
    }
    
    /* compute derivatives */
    ElementEvaluateGeometry_CellWiseConstant2d(nbasis,elcoords,c->npe_1d,&detJ);
    ElementEvaluateDerivatives_CellWiseConstant2d(nqp,nbasis,elcoords,
                                                  c->npe_1d,c->dN_dxi,c->dN_deta,
                                                  c->dN_dx,c->dN_dy);
    
    ierr = PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);CHKERRQ(ierr);
    
    /* get access to element->quadrature points */
    celldata = &c->cell_data[e];
    
    for (q=0; q<c->nqp; q++) {
      PetscReal            fac;
      PetscReal            c11,c12,c21,c22,c33,lambda_qp,mu_qp;
      PetscReal            *dNidx,*dNidy;
      
      
      dNidx = c->dN_dx[q];
      dNidy = c->dN_dy[q];
      
      /* compute strain @ quadrature point */
      /*
       e = Bu = [ d/dx  0    ][ u v ]^T
       [ 0     d/dy ]
       [ d/dy  d/dx ]
       */
      e_vec[0] = e_vec[1] = e_vec[2] = 0.0;
      for (i=0; i<nbasis; i++) {
        e_vec[0] += dNidx[i] * ux[i];
        e_vec[1] += dNidy[i] * uy[i];
        e_vec[2] += (dNidx[i] * uy[i] + dNidy[i] * ux[i]);
      }
      
      /* evaluate constitutive model */
      lambda_qp  = celldata->lambda;
      mu_qp      = celldata->mu;
      
      /*
       coeff = E_qp * (1.0 + nu_qp)/(1.0 - 2.0*nu_qp);
       c11 = coeff*(1.0 - nu_qp);
       c12 = coeff*(nu_qp);
       c21 = coeff*(nu_qp);
       c22 = coeff*(1.0 - nu_qp);
       c33 = coeff*(0.5 * (1.0 - 2.0 * nu_qp));
       */
      c11 = 2.0*mu_qp + lambda_qp;
      c12 = lambda_qp;
      c21 = lambda_qp;
      c22 = 2.0*mu_qp + lambda_qp;
      c33 = mu_qp;
      
      /* compute stress @ quadrature point */
      sigma_vec[TENS2D_XX] = c11 * e_vec[0] + c12 * e_vec[1];
      sigma_vec[TENS2D_YY] = c21 * e_vec[0] + c22 * e_vec[1];
      sigma_vec[TENS2D_XY] = c33 * e_vec[2];
      //printf("s = %1.4e %1.4e %1.4e \n",sigma_vec[0],sigma_vec[1],sigma_vec[2]);
      /*
       a(u,v) = B^T s
       = [ d/dx  0    d/dy ][ sxx syy sxy ]^T
       [ 0     d/dy d/dx ]
       */
      
      fac = detJ * c->w[q];
      
      for (i=0; i<nbasis; i++) {
        fe[2*i  ] += -fac * (dNidx[i] * sigma_vec[TENS2D_XX] + dNidy[i] * sigma_vec[TENS2D_XY]);
        fe[2*i+1] += -fac * (dNidy[i] * sigma_vec[TENS2D_YY] + dNidx[i] * sigma_vec[TENS2D_XY]);
      }
      
    }
    //ierr = VecSetValuesLocal(F,nbasis*ndof,eldofs,fe,ADD_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(fl,nbasis*ndof,eldofs,fe,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(fl);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(fl);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(c->dm,fl,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(c->dm,fl,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ul,&LA_u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(c->dm,&ul);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(c->dm,&fl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxGetDRCellData(SpecFECtx c,PetscInt e_index,DRVar **data)
{
  *data = &c->dr_qp_data[c->nqp * e_index];
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateVelocityAtPoint(SpecFECtx c,const PetscReal LA_v[],PetscReal xr[],PetscReal vr[])
{
  PetscReal      gmin[3],gmax[3],dx,dy;
  PetscInt       k,ei,ej,eid,*element,*elbasis;
  PetscReal      N[400];
  PetscErrorCode ierr;

  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Needs updating to support MPI");
  
  /* get containing element */
  ierr = DMGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
  ei = (xr[0] - gmin[0])/dx; /* todo - needs to be sub-domain gmin */
  
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
  ej = (xr[1] - gmin[1])/dy;
  
  eid = ei + ej * c->mx;
  
  /* get element -> node map */
  element = c->element;
  elbasis = &element[c->npe*eid];
  
  {
    PetscInt  nbasis,i,j;
    PetscReal **N_s1,**N_s2,xi,eta,x0,y0;
    
    x0 = gmin[0] + ei*dx; /* todo - needs to be sub-domain gmin */
    y0 = gmin[1] + ej*dy;
    
    // (xi - (-1))/2 = (x - x0)/dx
    xi = 2.0*(xr[0] - x0)/dx - 1.0;
    eta = 2.0*(xr[1] - y0)/dy - 1.0;
    
    /* compute basis */
    ierr = TabulateBasis1d_CLEGENDRE(1,&xi,c->basisorder,&nbasis,&N_s1);CHKERRQ(ierr);
    ierr = TabulateBasis1d_CLEGENDRE(1,&eta,c->basisorder,&nbasis,&N_s2);CHKERRQ(ierr);
    
    k = 0;
    for (j=0; j<c->npe_1d; j++) {
      for (i=0; i<c->npe_1d; i++) {
        N[k] = N_s1[0][i] * N_s2[0][j];
        k++;
      }
    }
    
    ierr = PetscFree(N_s1[0]);CHKERRQ(ierr);
    ierr = PetscFree(N_s1);CHKERRQ(ierr);
    ierr = PetscFree(N_s2[0]);CHKERRQ(ierr);
    ierr = PetscFree(N_s2);CHKERRQ(ierr);
  }
  
  vr[0] = vr[1] = 0.0;
  for (k=0; k<c->npe; k++) {
    PetscInt nid = elbasis[k];
    
    vr[0] += N[k] * LA_v[2*nid+0];
    vr[1] += N[k] * LA_v[2*nid+1];
  }
  
  PetscFunctionReturn(0);
}


PetscErrorCode PointLocation_v2(SpecFECtx c,const PetscReal xr[],PetscInt *_eid,PetscReal **N1,PetscReal **N2)
{
  static PetscBool beenhere = PETSC_FALSE;
  static PetscReal      gmin[3],gmax[3];
  PetscReal      dx,dy;
  PetscInt       ei,ej,eid;
  PetscInt       nbasis;
  PetscReal      **N_s1,**N_s2,xi,eta,x0,y0;
  PetscErrorCode ierr;
  
  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Needs updating to support MPI");
  
  /* get containing element */
  if (!beenhere) {
    ierr = DMGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
    beenhere = PETSC_TRUE;
  }
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
  ei = (xr[0] - gmin[0])/dx; /* todo - needs to be sub-domain gmin */
  
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
  ej = (xr[1] - gmin[1])/dy;
  
  eid = ei + ej * c->mx;
  
  x0 = gmin[0] + ei*dx; /* todo - needs to be sub-domain gmin */
  y0 = gmin[1] + ej*dy;
  
  // (xi - (-1))/2 = (x - x0)/dx
  xi = 2.0*(xr[0] - x0)/dx - 1.0;
  eta = 2.0*(xr[1] - y0)/dy - 1.0;
  
  /* compute basis */
  ierr = TabulateBasis1d_CLEGENDRE(1,&xi,c->basisorder,&nbasis,&N_s1);CHKERRQ(ierr);
  ierr = TabulateBasis1d_CLEGENDRE(1,&eta,c->basisorder,&nbasis,&N_s2);CHKERRQ(ierr);
  
  *_eid = eid;
  *N1 = N_s1[0];
  *N2 = N_s2[0];
  
  //ierr = PetscFree(N_s1[0]);CHKERRQ(ierr);
  ierr = PetscFree(N_s1);CHKERRQ(ierr);
  //ierr = PetscFree(N_s2[0]);CHKERRQ(ierr);
  ierr = PetscFree(N_s2);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


PetscErrorCode FaultSDFInit_v2(SpecFECtx c)
{
  PetscErrorCode  ierr;
  PetscInt        e,i,q,nbasis,nqp,ndof;
  Vec             coor;
  const PetscReal *LA_coor;
  PetscInt        *element,*elnidx,*eldofs;
  PetscReal       *elcoords;
  DRVar           *dr_celldata;
  PetscReal       factor;
  PetscInt        factor_i;
  void * the_sdf;
  int geometry_selection;
  int HalfNumPoints;
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  nqp      = c->nqp;
  ndof     = c->dofs;
  element  = c->element;
  
  
  factor = ((PetscReal)(c->ne)) * 0.1;
  factor_i = (PetscInt)factor;
  if (factor_i == 0) { factor_i = 1; }
  
  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  /** SDF Initialization
   * Creation,
   * Possible fill in of the BruteForce™ approach of acquiring the SDF at every quadrature point per element in the grid.
  */ 
  PetscPrintf(PETSC_COMM_WORLD,"Start SDF Init: ");
  ierr = SDFCreate(&c->sdf);CHKERRQ(ierr); // Allocate the struct object in memory (Might not be necessary as the original structure CTX has already allocated)
  geometry_selection = 1;// Setup the type of SDF used (third argument of the function): 0-> Horizontal, 1-> Tilted, 2->Sigmoid TBA
  ierr = SDFSetup(c->sdf, 2, geometry_selection);CHKERRQ(ierr);// call of the setup function.

  if (geometry_selection == 2)
  {
    ierr = PetscCalloc1(c->nqp*c->ne,&c->sdf->idxArray_ClosestFaultNode);CHKERRQ(ierr);
    ierr = PetscCalloc1(c->nqp*c->ne,&c->sdf->idxArray_SDFphi);CHKERRQ(ierr);
    ierr = PetscCalloc1(c->nqp*c->ne,&c->sdf->idxArray_DistOnFault);CHKERRQ(ierr);

    ierr = initializeZeroSetCurveFault(c->sdf);CHKERRQ(ierr);
    HalfNumPoints = ((GeometryParams) c->sdf->data)->HalfNumPoints;
  }

  the_sdf  = (void *)c->sdf;

  for (e=0; e<c->ne; e++) {
     PetscReal xcell[] = {0.0, 0.0};
     
    /* get element -> node map */
    elnidx = &element[nbasis*e];
    
    
    /* generate dofs */
    for (i=0; i<nbasis; i++) {
      eldofs[2*i  ] = 2*elnidx[i];
      eldofs[2*i+1] = 2*elnidx[i]+1;
    }
    
    /* get element coordinates */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
      
      xcell[0] += elcoords[2*i  ];
      xcell[1] += elcoords[2*i+1];
    }
    xcell[0] = xcell[0] / ((PetscReal)nbasis);
    xcell[1] = xcell[1] / ((PetscReal)nbasis);
    
    ierr = SpecFECtxGetDRCellData(c,e,&dr_celldata);CHKERRQ(ierr);
    
    for (q=0; q<c->nqp; q++) {
      PetscReal coor_qp[2];
      PetscBool modify_stress_state;
      
      coor_qp[0] = elcoords[2*q  ];
      coor_qp[1] = elcoords[2*q+1];

      // Populating the entire qp matrix with index directions of the nearest node on the fault, only for geometry type 2
      // Populating Array with the sdf phi and the projected distance onto the fault
      if (geometry_selection == 2)
      {
        find_minimum_idx(coor_qp[0], coor_qp[1], c->sdf->xList, c->sdf->fxList, HalfNumPoints*2+1, &c->sdf->idxArray_ClosestFaultNode[e*c->nqp + q]);
      
        ierr = Init_evaluate_Sigmoid_sdf(the_sdf,coor_qp, e * c->nqp + q, &c->sdf->idxArray_SDFphi[e*c->nqp + q]);CHKERRQ(ierr);
        ierr = Init_evaluate_DistOnFault_Sigmoid_sdf(the_sdf, coor_qp, e*c->nqp + q, &c->sdf->idxArray_DistOnFault[e*c->nqp + q]);CHKERRQ(ierr);
      }

      modify_stress_state = PETSC_FALSE;
      dr_celldata[q].eid[0] = -1;
      dr_celldata[q].eid[1] = -1;
      ierr = FaultSDFQuery(coor_qp,c->delta,the_sdf, e*c->nqp + q, &modify_stress_state);CHKERRQ(ierr);


      if (modify_stress_state) {
        PetscReal x_plus[2],x_minus[2];
        
        //printf("[e %d , q %d] x_qp %+1.4e , %+1.4e\n",e,q,coor_qp[0],coor_qp[1]);
        
        ierr = FaultSDFGetPlusMinusCoor(coor_qp,c->delta,the_sdf, e*c->nqp + q, x_plus,x_minus);CHKERRQ(ierr);
        
        ierr = PointLocation_v2(c,(const PetscReal*)x_plus, &dr_celldata[q].eid[0],&dr_celldata[q].N1_plus,&dr_celldata[q].N2_plus);CHKERRQ(ierr);
        ierr = PointLocation_v2(c,(const PetscReal*)x_minus,&dr_celldata[q].eid[1],&dr_celldata[q].N1_minus,&dr_celldata[q].N2_minus);CHKERRQ(ierr);
        //printf("  [e %d,q %d] x_qp -> x+ %+1.4e , %+1.4e [eid %d]\n",e,q,x_plus[0],x_plus[1],dr_celldata[q].eid[0]);
        //printf("  [e %d,q %d] x_qp -> x- %+1.4e , %+1.4e [eid %d]\n",e,q,x_minus[0],x_minus[1],dr_celldata[q].eid[1]);
      }
      
    }
    if (e%factor_i == 0) {
      printf("[Fault point location] Done element %d of %d\n",e,c->ne);
    }
  }
  printf("[Fault point location-v2] Finished\n");
  
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* Initialize quadrature points based on cell overlap condition */
PetscErrorCode FaultSDFInit_v3(SpecFECtx c)
{
  PetscErrorCode  ierr;
  PetscInt        e,i,q,nbasis,nqp,ndof;
  Vec             coor;
  const PetscReal *LA_coor;
  PetscInt        *element,*elnidx,*eldofs;
  PetscReal       *elcoords;
  DRVar           *dr_celldata;
  PetscReal       factor;
  PetscInt        factor_i;
  void * the_sdf;
  int geometry_selection;
  int HalfNumPoints;
  PetscBool modify_stress_state;  
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  nqp      = c->nqp;
  ndof     = c->dofs;
  element  = c->element;  
  
  factor = ((PetscReal)(c->ne)) * 0.1;
  factor_i = (PetscInt)factor;
  if (factor_i == 0) { factor_i = 1; }  
  
  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);  
  
  /** SDF Initialization
   * Creation,
   * Possible fill in of the BruteForce™ approach of acquiring the SDF at every quadrature point per element in the grid.
  */ 
  PetscPrintf(PETSC_COMM_WORLD,"Start SDF Init (v3): ");
  ierr = SDFCreate(&c->sdf);CHKERRQ(ierr); // Allocate the struct object in memory (Might not be necessary as the original structure CTX has already allocated)
  geometry_selection = 1;// Setup the type of SDF used (third argument of the function): 0-> Horizontal, 1-> Tilted, 2->Sigmoid TBA
  ierr = SDFSetup(c->sdf, 2, geometry_selection);CHKERRQ(ierr);// call of the setup function.  
  
  if (geometry_selection == 2)
  {
    ierr = PetscCalloc1(c->nqp*c->ne,&c->sdf->idxArray_ClosestFaultNode);CHKERRQ(ierr);
    ierr = PetscCalloc1(c->nqp*c->ne,&c->sdf->idxArray_SDFphi);CHKERRQ(ierr);
    ierr = PetscCalloc1(c->nqp*c->ne,&c->sdf->idxArray_DistOnFault);CHKERRQ(ierr);

    
    ierr = initializeZeroSetCurveFault(c->sdf);CHKERRQ(ierr);
    HalfNumPoints = ((GeometryParams) c->sdf->data)->HalfNumPoints;
  }  
  
  the_sdf  = (void *)c->sdf;  

  for (e=0; e<c->ne; e++) {
    /* get element -> node map */
    elnidx = &element[nbasis*e];    

    /* generate dofs */
    for (i=0; i<nbasis; i++) {
      eldofs[2*i  ] = 2*elnidx[i];
      eldofs[2*i+1] = 2*elnidx[i]+1;
    }    
    
    /* get element coordinates */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
    }    
    
    /* Determine if cell overlaps finite fault */
    modify_stress_state = PETSC_FALSE;
    for (i=0; i<nbasis; i++) {
      PetscReal coor_qp[2];      
      
      coor_qp[0] = elcoords[2*i  ];
      coor_qp[1] = elcoords[2*i+1]; 

      ierr = FaultSDFQuery(coor_qp,c->delta,the_sdf, e*nbasis + i, &modify_stress_state);CHKERRQ(ierr);
      if (modify_stress_state) break;
    }    
    
    ierr = SpecFECtxGetDRCellData(c,e,&dr_celldata);CHKERRQ(ierr);  

    if (geometry_selection == 2){
      for (q=0; q<c->nqp; q++) {
          PetscReal coor_qp[2];        
          
          coor_qp[0] = elcoords[2*q  ];
          coor_qp[1] = elcoords[2*q+1];        
          
          // Populating the entire qp matrix with index directions of the nearest node on the fault, only for geometry type 2
          // Populating Array with the sdf phi and the projected distance onto the fault
          
          find_minimum_idx(coor_qp[0], coor_qp[1], c->sdf->xList, c->sdf->fxList, HalfNumPoints*2+1, &c->sdf->idxArray_ClosestFaultNode[e*c->nqp + q]);
        
          ierr = Init_evaluate_Sigmoid_sdf(the_sdf,coor_qp, e * c->nqp + q, &c->sdf->idxArray_SDFphi[e*c->nqp + q]);CHKERRQ(ierr);
          ierr = Init_evaluate_DistOnFault_Sigmoid_sdf(the_sdf, coor_qp, e*c->nqp + q, &c->sdf->idxArray_DistOnFault[e*c->nqp + q]);CHKERRQ(ierr);
        }      
    }

    /* Modify all quadrature points contained within a cell overlapping the finite fault */
    if (modify_stress_state) {
      for (q=0; q<c->nqp; q++) {
        PetscReal coor_qp[2];        
        
        coor_qp[0] = elcoords[2*q  ];
        coor_qp[1] = elcoords[2*q+1];          
        
        modify_stress_state = PETSC_FALSE;
        dr_celldata[q].eid[0] = -1;
        dr_celldata[q].eid[1] = -1;        
        {
          PetscReal x_plus[2],x_minus[2];          
          ierr = FaultSDFGetPlusMinusCoor(coor_qp, c->delta, the_sdf, e*c->nqp + q, x_plus, x_minus);CHKERRQ(ierr);
          
          ierr = PointLocation_v2(c,(const PetscReal*)x_plus, &dr_celldata[q].eid[0],&dr_celldata[q].N1_plus,&dr_celldata[q].N2_plus);CHKERRQ(ierr);
          ierr = PointLocation_v2(c,(const PetscReal*)x_minus,&dr_celldata[q].eid[1],&dr_celldata[q].N1_minus,&dr_celldata[q].N2_minus);CHKERRQ(ierr);
        }
      }
    }

    if (e%factor_i == 0) {
      printf("[Fault point location] Done element %d of %d\n",e,c->ne);
    }
  }
  printf("[Fault point location-v3] Finished\n");  

  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}


/* Use tabulated basis at delta(+,-) to interpolate velocity */
PetscErrorCode FaultSDFTabulateInterpolation_v2(SpecFECtx c,const PetscReal LA_v[],DRVar *dr_celldata_q,
                                                PetscReal v_plus[],PetscReal v_minus[])
{
  PetscInt  eid_delta,*element_delta;
  PetscReal vx_d_e,vy_d_e,Ni;
  PetscInt i,ii,jj,nbasis,*element,nidx;
  
  nbasis   = c->npe;
  element  = c->element;
  
  if (!dr_celldata_q->N1_plus || !dr_celldata_q->N2_plus) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"[+] N1,N2 not allocated");
  eid_delta = dr_celldata_q->eid[0];
  element_delta = &element[nbasis*eid_delta];
  
  v_plus[0] = v_plus[1] = 0.0;
  i = 0;
  for (jj=0; jj<c->npe_1d; jj++) {
    for (ii=0; ii<c->npe_1d; ii++) {
      nidx = element_delta[i];
      
      Ni = dr_celldata_q->N1_plus[ii] * dr_celldata_q->N2_plus[jj];
      
      vx_d_e = LA_v[2*nidx  ];
      vy_d_e = LA_v[2*nidx+1];
      v_plus[0] += Ni * vx_d_e;
      v_plus[1] += Ni * vy_d_e;
      
      i++;
    }
  }

  if (!dr_celldata_q->N1_minus || !dr_celldata_q->N2_minus) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"[-] N1,N2 not allocated");
  eid_delta = dr_celldata_q->eid[1];
  element_delta = &element[nbasis*eid_delta];
  
  v_minus[0] = v_minus[1] = 0.0;
  i = 0;
  for (jj=0; jj<c->npe_1d; jj++) {
    for (ii=0; ii<c->npe_1d; ii++) {
      nidx = element_delta[i];
      
      Ni = dr_celldata_q->N1_minus[ii] * dr_celldata_q->N2_minus[jj];
      
      vx_d_e = LA_v[2*nidx  ];
      vy_d_e = LA_v[2*nidx+1];
      v_minus[0] += Ni * vx_d_e;
      v_minus[1] += Ni * vy_d_e;
      
      i++;
    }
  }
  PetscFunctionReturn(0);
}



/**%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 * AVERAGING SCHEME
 */
/**
 * Here we want to create several functions that project the stress field in a C0 element. 
 * The way it is going to be done is via identifiying repeated nodes on boundaries and corners and average the stresses
 * at repeated locations
 * After identifiying the repeated nodes in a similar manner as in Basis at location, 
 * the stress computation would be the same as in the function stress at location. Then the stress is averaged 
 * 
 * 1> Function that identifies if a point is at a boundary of corner. otw is the same as in Basisatlocation
 * 2> Use the id and qp value to calculate the basis at location from each point
 * 3> Average the stresses for the corners and the boundaries
 */

/**
 * 1. Identify element and qp id(s) for the inner, boundary, and corner cases
 */
PetscErrorCode IdentifyLocationType(SpecFECtx c, PetscReal xr[], PetscInt *_FlagRepCase, PetscInt eidVec[], PetscInt eiVec[], PetscInt ejVec[])
{
  static PetscReal      gmin[3],gmax[3];
  PetscErrorCode ierr;
  PetscReal      dx,dy,xi,eta,x0,y0;
  PetscInt       ei,ej;
  PetscInt       FlagRepCase = 1;
  

  ierr = DMGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);

  if (xr[0] < gmin[0]){exit(1);}
  if (xr[0] > gmax[0]){exit(1);}
  if (xr[1] < gmin[1]){exit(1);}
  if (xr[1] > gmax[1]){exit(1);}//abort

  /* get containing element wrt the Global bounding box*/
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
  ei = ((PetscInt)(xr[0] - gmin[0])/dx); // As a byproduct of this, I'll never have a residual of 1
  if (ei==c->mx_g){
    ei--;
  }

  dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
  ej = ((PetscInt)(xr[1] - gmin[1])/dy); 
  if (ej==c->my_g){
    ej--;
  }

  x0 = gmin[0] + ei*dx; 
  y0 = gmin[1] + ej*dy;
  /*Get containing element wrt local bounding box*/
  // (xi - (-1))/2 = (x - x0)/dx
  xi = 2.0*(xr[0] - x0)/dx - 1.0;
  eta = 2.0*(xr[1] - y0)/dy - 1.0;

  if ((xi+1.0 < 1e-10)&&(ei>0)){
    FlagRepCase = FlagRepCase * 2;
    eiVec[1] = ei;
    eiVec[3] = ei;
    ei--;
  }
  eiVec[0] = ei;


  if ((eta+1.0 < 1e-10)&&(ej>0)){
    FlagRepCase = FlagRepCase * 2;
    ejVec[2] = ej;
    ejVec[3] = ej;
    ej--;
  }
  ejVec[0] = ej;


  eidVec[0] = eiVec[0] + ejVec[0] * c->mx;

  if (FlagRepCase == 4){
    ejVec[1] = ejVec[0]; // Not -1 anymore
    eiVec[2] = eiVec[0]; // Not -1 anymore

    eidVec[1] = eiVec[1] + ejVec[1] * c->mx;
    eidVec[2] = eiVec[2] + ejVec[2] * c->mx;
    eidVec[3] = eiVec[3] + ejVec[3] * c->mx;

  } else if (FlagRepCase == 2){

    if (ejVec[1]==-1){
      eidVec[1] = eiVec[1] + ejVec[0] * c->mx;
    } else {
      eidVec[1] = eiVec[0] + ejVec[2] * c->mx;
    }

  }
  // if ((xr[0] >99.0) & (xr[0] < 101.0) ){
  //   printf(">[xi %+1.4e, %+1.4e] - Num repeated: %d > \n",xr[0],xr[1], FlagRepCase);
  //   for (int idxPrint=0; idxPrint<FlagRepCase; idxPrint++){
  //     printf("ei %d, ej: %d\n", eiVec[idxPrint], ejVec[idxPrint] );
  //   }
  //   printf("=========================\n");
  // }
  *_FlagRepCase = FlagRepCase;
   
  PetscFunctionReturn(0);
} //IdentifyLocationType: end

/**
 * 2. Compute the basis functions at a given location with given element index
 */
PetscErrorCode ComputeBasisAtLocWitheiej(SpecFECtx c, PetscReal xr[],PetscInt ei, PetscInt ej, PetscReal **_N, PetscReal ***dN1, PetscReal ***dN2){
    PetscInt       nbasis,i,j,k;
    PetscReal      **N_s1,**N_s2,xi,eta,x0,y0;
    PetscReal      gmin[3],gmax[3],dx,dy;
    PetscReal      N[400];
    PetscReal      etaVect[2];
    PetscErrorCode ierr;

    ierr = DMGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
    dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
    dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);

    x0 = gmin[0] + ei*dx; /* todo - needs to be sub-domain gmin */
    y0 = gmin[1] + ej*dy;
    
    // (xi - (-1))/2 = (x - x0)/dx
    xi = 2.0*(xr[0] - x0)/dx - 1.0;
    eta = 2.0*(xr[1] - y0)/dy - 1.0;
    
    /* compute basis */
    ierr = TabulateBasis1d_CLEGENDRE(1,&xi,c->basisorder,&nbasis,&N_s1);CHKERRQ(ierr);
    ierr = TabulateBasis1d_CLEGENDRE(1,&eta,c->basisorder,&nbasis,&N_s2);CHKERRQ(ierr);
    
    etaVect[0] = xi;
    etaVect[1] = eta;
  
    /* compute basis derivatives */
    ierr = TabulateBasisDerivativesAtPointTensorProduct2d( etaVect, c->basisorder, dN1, dN2);CHKERRQ(ierr);

    k = 0;
    for (j=0; j<c->npe_1d; j++) {
      for (i=0; i<c->npe_1d; i++) {
        N[k] = N_s1[0][i] * N_s2[0][j];
        k++;
      }
    }
    
    ierr = PetscFree(N_s1[0]);CHKERRQ(ierr);
    ierr = PetscFree(N_s1);CHKERRQ(ierr);
    ierr = PetscFree(N_s2[0]);CHKERRQ(ierr);
    ierr = PetscFree(N_s2);CHKERRQ(ierr);

    *_N = N;
    PetscFunctionReturn(0);
  }


/**
 * 3. Calculate the average of the stress for a location, and average field values at the borders of an element
 */
PetscErrorCode StressAvgElementBoundariesEval_StressAtLocation(SpecFECtx c,
                                     PetscReal xr[],  PetscReal normal[],  PetscReal tangent[],  
                                     const PetscReal *LA_coor,
                                     const PetscReal *LA_u, 
                                     const PetscReal *LA_v,
                                     PetscReal gamma, 
                                     PetscReal _sigma_vec[]){
  PetscErrorCode ierr;
  PetscInt FlagRepCase;
  PetscInt eiVec[] = {-1,-1,-1,-1}, ejVec[] = {-1,-1,-1,-1}, eidVec[] = {-1,-1,-1,-1};
  PetscReal      **dN1,**dN2;
  PetscReal      **dN_x,**dN_y;
  PetscInt       numP_alloc;
  PetscInt *elnidx, *element;
  PetscReal      *ux, *uy, *vx, *vy, *fieldU, *fieldV, *elcoords;

  PetscReal e_vec[3],edot_vec[3];
  QPntIsotropicElastic *celldata;

  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dim, &elcoords);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs, &fieldU);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs, &fieldV);CHKERRQ(ierr);

  ux = &fieldU[0];
  uy = &fieldU[c->npe];
  vx = &fieldV[0];
  vy = &fieldV[c->npe];

  
  numP_alloc = 1;

  ierr = PetscMalloc(sizeof(PetscReal*)*numP_alloc, &dN_x);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal*)*numP_alloc, &dN_y);CHKERRQ(ierr);
  for (PetscInt i=0; i<numP_alloc; i++) {
    ierr = PetscMalloc(sizeof(PetscReal)*c->npe, &dN_x[i]);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscReal)*c->npe, &dN_y[i]);CHKERRQ(ierr);
  }

  for (PetscInt d=0; d<3; d++){
    _sigma_vec[d] = 0.0; 
  }

  ierr = IdentifyLocationType(c, xr,  &FlagRepCase,  eidVec,  eiVec,  ejVec); CHKERRQ(ierr);

  element  = c->element;

  /** The averaging of a field over the repeated location occurs here*/
  for (PetscInt i_rep = 0; i_rep < FlagRepCase; i_rep ++){
    PetscReal *N;
    PetscReal ei,ej;

    ei = eiVec[i_rep];
    ej = ejVec[i_rep];

    if ((FlagRepCase == 2) && (i_rep == 1)){
      if (ej==-1){
        ei = eiVec[1];
        ej = ejVec[0];
      } else {
        ei = eiVec[0];
        ej = ejVec[2];
      }
    }

    ierr = ComputeBasisAtLocWitheiej(c, xr, ei, ej, &N, &dN1, &dN2);CHKERRQ(ierr);
    
    elnidx = &element[c->npe*eidVec[i_rep]];
    celldata = &c->cell_data[eidVec[i_rep]];

    for (PetscInt i = 0; i < c->npe; i++){
      PetscInt nidx = elnidx[i];
      /* get element coordinates */
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
      /** get element displacement & velocities*/ 
      ux[i] = LA_u[2*nidx  ];
      uy[i] = LA_u[2*nidx+1];
      
      vx[i] = LA_v[2*nidx  ];
      vy[i] = LA_v[2*nidx+1];
    }
    ElementEvaluateDerivatives_CellWiseConstant2d(numP_alloc,c->npe,elcoords,
                                                  c->npe_1d,dN1,dN2,
                                                  dN_x,dN_y);

    PetscInt q = 0;
    {
      PetscReal c11,c12,c21,c22,c33,lambda_qp,mu_qp;
      PetscReal *dNx,*dNy;
      PetscReal sigma_vec[]={0.0, 0.0, 0.0};

      dNx = dN_x[q];
      dNy = dN_y[q];

      /* compute strain @ an arbitrary location */
      /*
       e = Bu = [ d/dx  0    ][ u v ]^T
       [ 0     d/dy ]
       [ d/dy  d/dx ]
       */
      e_vec[0] = e_vec[1] = e_vec[2] = 0.0;
      for (PetscInt i=0; i<c->npe; i++) 
      {
        e_vec[0] += dNx[i] * ux[i];
        e_vec[1] += dNy[i] * uy[i];
        e_vec[2] += (dNx[i] * uy[i] + dNy[i] * ux[i]);
      }
      edot_vec[0] = edot_vec[1] = edot_vec[2] = 0.0;
      for (PetscInt i=0; i<c->npe; i++) 
      {
        edot_vec[0] += dNx[i] * vx[i];
        edot_vec[1] += dNy[i] * vy[i];
        edot_vec[2] += (dNx[i] * vy[i] + dNy[i] * vx[i]);
      }
     
      /* evaluate constitutive model */
      lambda_qp = celldata->lambda;
      mu_qp     = celldata->mu;
    
      /*
      coeff = E_qp * (1.0 + nu_qp)/(1.0 - 2.0*nu_qp);
      c11 = coeff*(1.0 - nu_qp);
      c12 = coeff*(nu_qp);
      c21 = coeff*(nu_qp);
      c22 = coeff*(1.0 - nu_qp);
      c33 = coeff*(0.5 * (1.0 - 2.0 * nu_qp));
      */
      c11 = 2.0 * mu_qp + lambda_qp;
      c12 = lambda_qp;
      c21 = lambda_qp;
      c22 = 2.0 * mu_qp + lambda_qp;
      c33 = mu_qp;
      
      /* compute stress @ quadrature point */
      sigma_vec[TENS2D_XX] = c11 * e_vec[0] + c12 * e_vec[1];
      sigma_vec[TENS2D_YY] = c21 * e_vec[0] + c22 * e_vec[1];
      sigma_vec[TENS2D_XY] = c33 * e_vec[2];
      
      {
        PetscReal factor = 1.0;
        
        c11 = factor * (2.0 * mu_qp + lambda_qp) * gamma;
        c12 = factor * (lambda_qp) * gamma;
        c21 = factor * (lambda_qp) * gamma;
        c22 = factor * (2.0 * mu_qp + lambda_qp) * gamma;
        c33 = factor * (mu_qp) * gamma;
      }
      
      /* compute stress @ quadrature point */
      sigma_vec[TENS2D_XX] += c11 * edot_vec[0] + c12 * edot_vec[1];
      sigma_vec[TENS2D_YY] += c21 * edot_vec[0] + c22 * edot_vec[1];
      sigma_vec[TENS2D_XY] += c33 * edot_vec[2];

      for (PetscInt d=0; d<3; d++){
        _sigma_vec[d] += sigma_vec[d] / ((PetscReal) (FlagRepCase)); 
      }
    }    

  }

  /** Free memory */
  ierr = PetscFree(elcoords);CHKERRQ(ierr);
  ierr = PetscFree(fieldU);CHKERRQ(ierr);
  ierr = PetscFree(fieldV);CHKERRQ(ierr);

  for (PetscInt i=0; i<numP_alloc; i++) {
    ierr = PetscFree(dN_x[i]);CHKERRQ(ierr);
    ierr = PetscFree(dN_y[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dN_x);CHKERRQ(ierr);
  ierr = PetscFree(dN_y);CHKERRQ(ierr);


  for (PetscInt i=0; i<numP_alloc; i++) {
      ierr = PetscFree(dN1[i]);CHKERRQ(ierr);
      ierr = PetscFree(dN2[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dN1);CHKERRQ(ierr);
  ierr = PetscFree(dN2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/**
 * AVERAGING SCHEME END
 * %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 */

/**%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 * Save the slip in a file
*/

/**  Create the header of the file containing the slip
  and slip rate
*/
PetscErrorCode SaveTheReceiversHeader(){
  FILE *fpFetch;
  fpFetch = fopen("./SlipAtReceiver.txt","a");
  fprintf(fpFetch,"Time\tele\tq\tSlip\tSlipRate\tx\ty\tFaultX\tFaultY\n"); 
  fclose(fpFetch);
  PetscFunctionReturn(0);
} 
/**Function to populate the  file containing the slip and slip rate*/
PetscErrorCode SaveTheReceivers(PetscReal Time, PetscInt e, PetscInt q, 
                                PetscReal Slip, PetscReal SlipRate,
                                PetscReal x[2], PetscReal FaultX, PetscReal FaultY){
  
  const int Size = 4;
  PetscReal ReceiversLocX[] = {0, 2000., 4000., 6000., 8000.};

  for (int i = 0; i < Size; i++){
    if(PetscAbsReal(FaultX - ReceiversLocX[i]) < 1.e0){
      FILE *fpFetch;
      fpFetch = fopen("./SlipAtReceiver.txt","a");
      fprintf(fpFetch,"%+1.4e\t%d\t%d\t%+1.4e\t%+1.4e\t%+1.4e\t%+1.4e\t%+1.4e\t%+1.4e\n",
              Time, e, q, 
              Slip, SlipRate,
              x[0], x[1], FaultX, FaultY); 
      fclose(fpFetch);
    }
  }
  PetscFunctionReturn(0);
} 
/**
 * Save the slip in a file
 * %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/

PetscErrorCode GlobalToLocalChangeOfBasis(PetscReal n[],PetscReal t[], PetscReal sigma_trial[]){
  PetscReal Sig[3];
  PetscErrorCode ierr;

  Sig[0]=sigma_trial[TENS2D_XX];
  Sig[1]=sigma_trial[TENS2D_YY];
  Sig[2]=sigma_trial[TENS2D_XY];
  
  sigma_trial[TENS2D_XX] = Sig[TENS2D_XX]*(t[0]*t[0]) + Sig[TENS2D_YY]*t[1]*t[1] + Sig[TENS2D_XY]*(t[0]*t[1] + t[1]*t[0]);
  sigma_trial[TENS2D_YY] = Sig[TENS2D_XX]*(n[0]*n[0]) + Sig[TENS2D_YY]*n[1]*n[1] + Sig[TENS2D_XY]*(n[0]*n[1] + n[1]*n[0]);
  sigma_trial[TENS2D_XY] = Sig[TENS2D_XX]*(t[0]*n[0]) + Sig[TENS2D_YY]*t[1]*n[1] + Sig[TENS2D_XY]*(t[0]*n[1] + t[1]*n[0]);
  PetscFunctionReturn(0);
}

PetscErrorCode Local2GlobalChangeOfBasis(PetscReal n[],PetscReal t[], PetscReal sigma_trial[]){
  PetscReal Sig[3];
  PetscErrorCode ierr;

  Sig[0]=sigma_trial[TENS2D_XX];
  Sig[1]=sigma_trial[TENS2D_YY];
  Sig[2]=sigma_trial[TENS2D_XY];
  
  sigma_trial[TENS2D_XX] = Sig[TENS2D_XX]*(t[0]*t[0]) + Sig[TENS2D_YY]*(n[0]*n[0])+ Sig[TENS2D_XY]*(t[0]*n[0] + t[0]*n[0]);
  sigma_trial[TENS2D_YY] = Sig[TENS2D_XX]*(t[1]*t[1]) + Sig[TENS2D_YY]*(n[1]*n[1])+ Sig[TENS2D_XY]*(t[1]*n[1] + t[1]*n[1]);
  sigma_trial[TENS2D_XY] = Sig[TENS2D_XX]*(t[0]*t[1]) + Sig[TENS2D_YY]*(n[0]*n[1])+ Sig[TENS2D_XY]*(t[0]*n[1] + t[1]*n[0]);
  PetscFunctionReturn(0);
}

PetscErrorCode VoigtTensorContract_ai_Tij_bj(PetscReal a[],PetscReal t[2][2],PetscReal b[],PetscReal *r)
{
  PetscReal s=0;
  PetscInt i,j;
  
  for (i=0; i<2; i++) {
    for (j=0; j<2; j++) {
      s += a[i] * t[i][j] * b[j];
    }
  }
  *r = s;
  PetscFunctionReturn(0);
}

PetscErrorCode VoigtTensorConvert(PetscReal T[],PetscReal t[2][2])
{
  t[0][0] = T[TENS2D_XX];
  t[0][1] = T[TENS2D_XY];
  t[1][0] = T[TENS2D_XY];
  t[1][1] = T[TENS2D_YY];
  PetscFunctionReturn(0);
}

PetscErrorCode TensorConvertToVoigt(PetscReal t[2][2],PetscReal T[])
{
  T[TENS2D_XX] = t[0][0];
  T[TENS2D_XY] = t[0][1];
  T[TENS2D_YY] = t[1][1];
  if (fabs(t[1][0] - t[0][1]) > 1.0e-12) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot convert non-symmetric tensor into Voigt format");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TensorZeroEntries(PetscReal t[2][2])
{
  t[0][0] = 0;
  t[0][1] = 0;
  t[1][0] = 0;
  t[1][1] = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode TensorScale(PetscReal y[2][2],PetscReal a)
{
  y[0][0] = a*y[0][0];
  y[0][1] = a*y[0][1];
  y[1][0] = a*y[1][0];
  y[1][1] = a*y[1][1];
  PetscFunctionReturn(0);
}

PetscErrorCode TensorAXPY(PetscReal y[2][2],PetscReal a,PetscReal x[2][2])
{
  y[0][0] += a*x[0][0];
  y[0][1] += a*x[0][1];
  y[1][0] += a*x[1][0];
  y[1][1] += a*x[1][1];
  PetscFunctionReturn(0);
}


PetscErrorCode VectorContract_ai_bj(PetscReal a[],PetscReal b[],PetscReal t[2][2])
{
  PetscInt i,j;
  
  for (i=0; i<2; i++) {
    for (j=0; j<2; j++) {
      t[i][j] = a[i] * b[j];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VectorContractAdd_ai_bj(PetscReal a[],PetscReal b[],PetscReal t[2][2])
{
  PetscInt i,j;
  
  for (i=0; i<2; i++) {
    for (j=0; j<2; j++) {
      t[i][j] += a[i] * b[j];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VectorContractAbsAdd_ai_bj(PetscReal a[],PetscReal b[],PetscReal t[2][2])
{
  PetscInt i,j;
  
  for (i=0; i<2; i++) {
    for (j=0; j<2; j++) {
      t[i][j] += fabs(a[i] * b[j]);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TensorRtAR(PetscReal R[2][2],PetscReal T[2][2],PetscReal Tr[2][2])
{
  PetscInt i,j,k,l;
  
  // Tr[i][j] = Rt[i][k]T[k][l]R[l][j] = Rt[k][i]T[k][l]R[l][j]
  for (i=0; i<2; i++) {
    for (j=0; j<2; j++) {
      Tr[i][j] = 0.0;
      for (k=0; k<2; k++) {
        for (l=0; l<2; l++) {
          Tr[i][j] += R[k][i] * T[k][l] * R[l][j];
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TensorTransform(PetscReal e1[],PetscReal e2[],PetscReal T[2][2],PetscReal Tr[2][2])
{
  PetscReal R[2][2];
  
  R[0][0] = e1[0]; R[0][1] = e2[0];
  R[1][0] = e1[1]; R[1][1] = e2[1];
  TensorRtAR(R,T,Tr);
  PetscFunctionReturn(0);
}

PetscErrorCode TensorInverseTransform(PetscReal e1[],PetscReal e2[],PetscReal T[2][2],PetscReal Tr[2][2])
{
  PetscReal R[2][2],iR[2][2],det;
  
  R[0][0] = e1[0]; R[0][1] = e2[0];
  R[1][0] = e1[1]; R[1][1] = e2[1];
  det = R[0][0] * R[1][1] - R[0][1] * R[1][0];
  iR[0][0] =  R[1][1]/det;
  iR[0][1] = -R[0][1]/det;
  iR[1][0] = -R[1][0]/det;
  iR[1][1] =  R[0][0]/det;
  TensorRtAR(iR,T,Tr);
  PetscFunctionReturn(0);
}

PetscErrorCode AssembleLinearForm_ElastoDynamics_StressGlut2d(SpecFECtx c,Vec u,Vec v,PetscReal dt,PetscReal time,PetscReal gamma,Vec F, PetscInt step)
{
  PetscErrorCode ierr;
  PetscInt  e,nqp,q,i,nbasis,ndof;
  PetscInt  *element,*elnidx,*eldofs;
  PetscReal *fe,*ux,*uy,*vx,*vy,*elcoords,detJ,*fieldU,*fieldV;
  Vec       coor,ul,vl,fl;
  const PetscReal *LA_coor,*LA_u,*LA_v;
  QPntIsotropicElastic *celldata;
  DRVar                *dr_celldata;
  void *the_sdf;
  Vec proj= NULL;
  const PetscReal *_proj_sigma;

  PetscInt OrderLimiter = 4;

  PetscReal *sigma_tilde,*sigma_tilde2;
  PetscBool *cell_flag;

  PetscReal sigma_n_0 = 40.0 * 1.0e6;
  PetscReal sigma_t_0 = 20.0 * 1.0e6;
  PetscReal sigma_n_1 = 40.0 * 1.0e6;
  PetscReal sigma_t_1 = 20.0 * 1.0e6;
  
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  nqp      = c->nqp;
  ndof     = c->dofs;
  fe       = c->elbuf_field;
  element  = c->element;
  fieldU   = c->elbuf_field2;
  fieldV   = c->elbuf_field3;
  the_sdf  = (void *) c->sdf; 
 
  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(c->dm,&ul);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(c->dm,u,INSERT_VALUES,ul);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(c->dm,u,INSERT_VALUES,ul);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ul,&LA_u);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(c->dm,&vl);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(c->dm,v,INSERT_VALUES,vl);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(c->dm,v,INSERT_VALUES,vl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vl,&LA_v);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(c->dm,&fl);CHKERRQ(ierr);
  ierr = VecZeroEntries(fl);CHKERRQ(ierr);
  
  ux = &fieldU[0];
  uy = &fieldU[nbasis];
  
  vx = &fieldV[0];
  vy = &fieldV[nbasis];
  
  PetscCalloc1(c->ne,&cell_flag);
  SpecFECreateQuadratureField(c,3,&sigma_tilde);
  SpecFECreateQuadratureField(c,3,&sigma_tilde2);


  for (e=0; e<c->ne; e++) {
    ierr = SpecFECtxGetDRCellData(c,e,&dr_celldata);CHKERRQ(ierr);
  }

  /* compute and stress stress */
  for (e=0; e<c->ne; e++) {
    PetscBool inside_fault_region = PETSC_FALSE;
    
    /* get element -> node map */
    elnidx = &element[nbasis*e];
    
    /* generate dofs */
    for (i=0; i<nbasis; i++) {
      eldofs[2*i  ] = 2*elnidx[i];
      eldofs[2*i+1] = 2*elnidx[i]+1;
    }
    
    /* get element coordinates */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
    }

    cell_flag[e] = PETSC_FALSE;
    for (i=0; i<nbasis; i++) {
      ierr = FaultSDFQuery(&elcoords[2*i], c->delta, the_sdf, e*nbasis + i, &inside_fault_region);CHKERRQ(ierr);
      if (inside_fault_region) {
        cell_flag[e] = PETSC_TRUE;
        break;
      }
    }
    
    /* get element displacements & velocities */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      ux[i] = LA_u[2*nidx  ];
      uy[i] = LA_u[2*nidx+1];
      
      vx[i] = LA_v[2*nidx  ];
      vy[i] = LA_v[2*nidx+1];
    }
    
    /* compute derivatives */
    ElementEvaluateGeometry_CellWiseConstant2d(nbasis,elcoords,c->npe_1d,&detJ);
    ElementEvaluateDerivatives_CellWiseConstant2d(nqp,nbasis,elcoords,
                                                  c->npe_1d,c->dN_dxi,c->dN_deta,
                                                  c->dN_dx,c->dN_dy);
    
    /* get access to element->quadrature points */
    celldata = &c->cell_data[e];
    ierr = SpecFECtxGetDRCellData(c,e,&dr_celldata);CHKERRQ(ierr);
    
    
    for (q=0; q<c->nqp; q++) {
      PetscReal c11,c12,c21,c22,c33,lambda_qp,mu_qp;
      PetscReal *dNidx,*dNidy;
      PetscReal e_vec[]={0,0,0},edot_vec[]={0,0,0},sigma_vec[]={0,0,0};
      
      dNidx = c->dN_dx[q];
      dNidy = c->dN_dy[q];
      
      /* compute strain @ quadrature point */
      /*
       e = Bu = [ d/dx  0    ][ u v ]^T
                [ 0     d/dy ]
                [ d/dy  d/dx ]
       */
      for (i=0; i<nbasis; i++) {
        e_vec[0] += dNidx[i] * ux[i];
        e_vec[1] += dNidy[i] * uy[i];
        e_vec[2] += (dNidx[i] * uy[i] + dNidy[i] * ux[i]);
      }
      
      for (i=0; i<nbasis; i++) {
        edot_vec[0] += dNidx[i] * vx[i];
        edot_vec[1] += dNidy[i] * vy[i];
        edot_vec[2] += (dNidx[i] * vy[i] + dNidy[i] * vx[i]);
      }
      
      /* evaluate constitutive model */
      lambda_qp = celldata->lambda;
      mu_qp     = celldata->mu;
      
      /*
       coeff = E_qp * (1.0 + nu_qp)/(1.0 - 2.0*nu_qp);
       c11 = coeff*(1.0 - nu_qp);
       c12 = coeff*(nu_qp);
       c21 = coeff*(nu_qp);
       c22 = coeff*(1.0 - nu_qp);
       c33 = coeff*(0.5 * (1.0 - 2.0 * nu_qp));
       */
      c11 = 2.0 * mu_qp + lambda_qp;
      c12 = lambda_qp;
      c21 = lambda_qp;
      c22 = 2.0 * mu_qp + lambda_qp;
      c33 = mu_qp;
      
      /* compute stress @ quadrature point */
      sigma_vec[TENS2D_XX] = c11 * e_vec[0] + c12 * e_vec[1];
      sigma_vec[TENS2D_YY] = c21 * e_vec[0] + c22 * e_vec[1];
      sigma_vec[TENS2D_XY] = c33 * e_vec[2];
      
      /*
       From 
       Day and Ely "Effect of a Shallow Weak Zone on Fault Rupture: Numerical Simulation of Scale-Model Experiments",
       BSSA, 2002
       
       alpha = cp
       beta = cs
       volumetric terms; rho (cp^2 - 2 cs^2) gamma [div(v)]
       shear terms; rho cs^2 gamma [v_{i,j} + v_{j,i}]
       */
      
      {
        PetscReal factor = 1.0;
        
        c11 = factor * (2.0 * mu_qp + lambda_qp) * gamma;
        c12 = factor * (lambda_qp) * gamma;
        c21 = factor * (lambda_qp) * gamma;
        c22 = factor * (2.0 * mu_qp + lambda_qp) * gamma;
        c33 = factor * (mu_qp) * gamma;
      }
      
      /* compute stress @ quadrature point */
      sigma_vec[TENS2D_XX] += c11 * edot_vec[0] + c12 * edot_vec[1];
      sigma_vec[TENS2D_YY] += c21 * edot_vec[0] + c22 * edot_vec[1];
      sigma_vec[TENS2D_XY] += c33 * edot_vec[2];


      c->sigma[e*(c->npe * 3) + q*3 + TENS2D_XX] = sigma_vec[TENS2D_XX];
      c->sigma[e*(c->npe * 3) + q*3 + TENS2D_YY] = sigma_vec[TENS2D_YY];
      c->sigma[e*(c->npe * 3) + q*3 + TENS2D_XY] = sigma_vec[TENS2D_XY];

    } // Loop of over the qp: End
  } // Loop over the elements: End


  // Here would be the projection after updated with elastic increments as it has already been stored on c->sigma.
  // The function CGProjectNative already takes care of the separation of each component of the stress, the decision is to take either the rewritten 
  // c->sigma or the proj vector that should be destroyed later
  if(c->basisorder > OrderLimiter){
    ierr = CGProjectNative(c, c->basisorder, 3, c->sigma, &proj);CHKERRQ(ierr);
    ierr = VecGetArrayRead(proj,&_proj_sigma);CHKERRQ(ierr);
  }

  for (e=0; e<c->ne; e++) {
    
    /* get element -> node map */
    elnidx = &element[nbasis*e];
    
    /* generate dofs */
    for (i=0; i<nbasis; i++) {
      eldofs[2*i  ] = 2*elnidx[i];
      eldofs[2*i+1] = 2*elnidx[i]+1;
    }
    
    /* get element coordinates */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
    }
    
    /* get element displacements & velocities */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      ux[i] = LA_u[2*nidx  ];
      uy[i] = LA_u[2*nidx+1];
      
      vx[i] = LA_v[2*nidx  ];
      vy[i] = LA_v[2*nidx+1];
    }
    
    /* compute derivatives */
    ElementEvaluateGeometry_CellWiseConstant2d(nbasis,elcoords,c->npe_1d,&detJ);
    ElementEvaluateDerivatives_CellWiseConstant2d(nqp,nbasis,elcoords,
                                                  c->npe_1d,c->dN_dxi,c->dN_deta,
                                                  c->dN_dx,c->dN_dy);
    
    // ierr = PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);CHKERRQ(ierr);
    
    /* get access to element->quadrature points */
    celldata = &c->cell_data[e];
    ierr = SpecFECtxGetDRCellData(c,e,&dr_celldata);CHKERRQ(ierr);
    

    for (q=0; q<c->nqp; q++) {
      PetscReal normal[2],tangent[2];
      // PetscReal fac;
      PetscReal mu_qp;
      PetscReal *dNidx,*dNidy;
      PetscReal coor_qp[2];
      PetscBool inside_fault_region;
      PetscReal DistOnFault; /**Project coords onto fault to get friction in the case of tilting */
      PetscReal e_vec[3],sigma_vec[3],sigma_trial[3],gradu[4];

      dNidx = c->dN_dx[q];
      dNidy = c->dN_dy[q];
      
      /* compute strain @ quadrature point */
      /*
       e = Bu = [ d/dx  0    ][ u v ]^T
       [ 0     d/dy ]
       [ d/dy  d/dx ]
       */
      e_vec[0] = e_vec[1] = e_vec[2] = 0.0;
      for (i=0; i<nbasis; i++) {
        e_vec[0] += dNidx[i] * ux[i];
        e_vec[1] += dNidy[i] * uy[i];
        e_vec[2] += (dNidx[i] * uy[i] + dNidy[i] * ux[i]);
      }

      gradu[0] = gradu[1] = gradu[2] = gradu[3] = 0.0;
      for (i=0; i<nbasis; i++) {
        gradu[0] += dNidx[i] * ux[i];
        gradu[1] += dNidy[i] * ux[i];
        gradu[2] += dNidx[i] * uy[i];
        gradu[3] += dNidy[i] * uy[i];
      }
      
      /*
       // test 3
      {
        PetscReal *ge;
        int b;
        ierr = SpecFEGetElementQuadratureField(c,4,e,(const PetscReal*)c->gradu,&ge);CHKERRQ(ierr);
        for (b=0; b<4; b++) { ge[4*q + b] = gradu[b]; }
      }
      */
       
      coor_qp[0] = elcoords[2*q  ];
      coor_qp[1] = elcoords[2*q+1];

      mu_qp      = celldata->mu;
      
      //Only DG scheme
      sigma_vec[TENS2D_XX] = c->sigma[e*(c->npe * 3) + q*3 + TENS2D_XX];
      sigma_vec[TENS2D_YY] = c->sigma[e*(c->npe * 3) + q*3 + TENS2D_YY];
      sigma_vec[TENS2D_XY] = c->sigma[e*(c->npe * 3) + q*3 + TENS2D_XY];

      
      // sigma = \sum_nbasis N_i(quadrature_point) sigma_i
      // Since this is a spectral element method we dont need to 
      // interpolate from basis to quadrature points, rather we
      // can simply assign a qp value based on the basis index.
      if(c->basisorder > OrderLimiter){
        sigma_vec[0] = sigma_vec[1] = sigma_vec[2] = 0;
        {
          PetscInt nidx = elnidx[q]; // NOTE: index via quad point index rather than basis function index
          sigma_vec[TENS2D_XX] = _proj_sigma[3*nidx + TENS2D_XX]; 
          sigma_vec[TENS2D_YY] = _proj_sigma[3*nidx + TENS2D_YY]; 
          sigma_vec[TENS2D_XY] = _proj_sigma[3*nidx + TENS2D_XY]; 
        }
      }

      // if ((fabs(c->sigma[e*(c->npe * 3) + q*3 + TENS2D_XX]) > 0.0) || (fabs(sigma_vec[TENS2D_XX]) > 0.0))
      // {
      //   printf("[%d, %d] >>Buffer: sigma proj %+1.4e, SigmaVec %+1.4e,\n",e,q,c->sigma[e*(c->npe * 3) + q*3 + TENS2D_XX], sigma_vec[TENS2D_XX]);
      // }

      sigma_trial[TENS2D_XX] = sigma_vec[TENS2D_XX];
      sigma_trial[TENS2D_YY] = sigma_vec[TENS2D_YY];
      sigma_trial[TENS2D_XY] = sigma_vec[TENS2D_XY];

      inside_fault_region = PETSC_FALSE;

 
      ierr = FaultSDFQuery(coor_qp, c->delta, the_sdf, e*c->nqp + q, &inside_fault_region);CHKERRQ(ierr);


      DistOnFault = 0.0;
      ierr = evaluate_DistOnFault_sdf(the_sdf, coor_qp, e*c->nqp + q, &DistOnFault);CHKERRQ(ierr);

      ierr = FaultSDFNormal(coor_qp,the_sdf, e*c->nqp + q, normal);CHKERRQ(ierr);
      ierr = FaultSDFTangent(coor_qp,the_sdf, e*c->nqp + q, tangent);CHKERRQ(ierr);
      ierr = GlobalToLocalChangeOfBasis(normal, tangent, sigma_trial);CHKERRQ(ierr);

      
      /* add the initial stress state on fault */
      if (fabs(DistOnFault) < 1.5*1.0e3 && cell_flag[e]) {
        sigma_trial[TENS2D_XY] += sigma_t_1;
        sigma_trial[TENS2D_YY] += (-sigma_n_1); /* negative in compression */
      } else {
        sigma_trial[TENS2D_XY] += sigma_t_0;
        sigma_trial[TENS2D_YY] += (-sigma_n_0); /* negative in compression */
      }


      {
        PetscReal *_sigma_store = &sigma_tilde[e*(c->npe * 3) + q*3];
        _sigma_store[TENS2D_XX] = sigma_trial[TENS2D_XX];
        _sigma_store[TENS2D_YY] = sigma_trial[TENS2D_YY];
        _sigma_store[TENS2D_XY] = sigma_trial[TENS2D_XY];
        ierr = Local2GlobalChangeOfBasis(normal, tangent, _sigma_store);CHKERRQ(ierr);
      }


      //printf("Type, %d,\n", c->sdf->type);
      //printf("[%+1.4e, %+1.4e, DistOnFault: %+1.4e ],\n", coor_qp[0], coor_qp[1],, DistOnFault);
      
      /* Make stress glut corrections here */
      if (cell_flag[e]) {
        PetscReal x_plus[2],x_minus[2],v_plus[2],v_minus[2];
        PetscReal Vplus,Vminus,slip,slip_rate;
        PetscReal sigma_n,sigma_t,phi_p;
        PetscReal tau,mu_s,mu_d,mu_friction,T, ttau;
        //printf(">>[e %d , q %d] x_qp %+1.4e , %+1.4e\n",e,q,coor_qp[0],coor_qp[1]);
        ierr = evaluate_sdf(the_sdf,coor_qp, e*c->nqp + q, &phi_p);CHKERRQ(ierr);
        
        ierr = FaultSDFGetPlusMinusCoor(coor_qp,c->delta,the_sdf, e*c->nqp + q, x_plus,x_minus);CHKERRQ(ierr);
        ierr = FaultSDFTabulateInterpolation_v2(c,LA_v,&dr_celldata[q],v_plus,v_minus);CHKERRQ(ierr);
      
        // PetscReal RtgraduR = gradu[0]*normal[0]*tangent[0] + gradu[1]*normal[0]*tangent[1] + 
        //                     gradu[2]*normal[1]*tangent[0] + gradu[3]*normal[1]*tangent[1];

         /* ================================================================ */
        ierr = FaultSDFTabulateInterpolation_v2(c,LA_v,&dr_celldata[q],v_plus,v_minus);CHKERRQ(ierr);



        /* Resolve velocities at delta(+,-) onto fault */
        {
          PetscReal mag_vdotn;
          
          mag_vdotn =  (v_plus[0] * normal[0] +  v_plus[1] * normal[1]);
          v_plus[0] = v_plus[0] - mag_vdotn * normal[0];
          v_plus[1] = v_plus[1] - mag_vdotn * normal[1];
          
          mag_vdotn =  (v_minus[0] * normal[0] +  v_minus[1] * normal[1]);
          v_minus[0] = v_minus[0] - mag_vdotn * normal[0];
          v_minus[1] = v_minus[1] - mag_vdotn * normal[1];
        }
        
        Vplus  = v_plus[0];
        Vminus = v_minus[0];

        slip_rate = Vplus - Vminus;      
        slip = dr_celldata[q].slip;


        sigma_n = sigma_trial[TENS2D_YY];
        sigma_t = sigma_trial[TENS2D_XY];

        dr_celldata[q].mu = 0;
        if (sigma_n < 0) { /* only consider inelastic corrections if in compression */
          PetscReal L = 250.0;
          PetscReal V = 2000.0;
          PetscReal mu_f;
          
          T = sqrt(sigma_t * sigma_t);
          
          /* Hard code friction */
          mu_s = 0.5;
          mu_d = 0.25;

          /** Friction wrt the distance on the fault */
          mu_f = mu_s - (mu_s - mu_d) * (V * time - fabs(DistOnFault)) / L;
          mu_friction = PetscMax(mu_d,mu_f); 
          
          dr_celldata[q].mu = mu_friction;
          
          tau = -mu_friction * sigma_n;
          if (tau < 0) {
            printf("-mu sigma_n < 0 error\n");
            exit(1);
          }
          
          if ((T > tau) || (dr_celldata[q].sliding == PETSC_TRUE)) {
            
            /**Antiparallel condition between slip rate and critical shear */
            ttau = tau;
            if ( sigma_t < 0.0) //slip_rate=v(+)-v(-) defined following Dalguer
            {
              ttau = -tau;
            } 
            sigma_trial[TENS2D_XY] = ttau;

            /**Self-similar crack - Smoothing for p > 1 */
            if(c->basisorder > OrderLimiter)
            {
              ierr = PetscTanHWeighting( &sigma_t,  sigma_t, ttau, phi_p , (4.*c->basisorder)/c->delta,  0.65*c->delta); CHKERRQ(ierr);
              sigma_trial[TENS2D_XY] = sigma_t;
            }

            //printf("  sigma_xy %+1.8e\n",sigma_vec[TENS2D_XY]);
            dr_celldata[q].sliding = PETSC_TRUE;
          } else {
            slip_rate = 0.0;
          }
          
          
          // Error checking / verification that consistency conditions are approximately satisfied
          /*
           if (T > fabs(tau)) {
           if (fabs(sigma_trial[TENS2D_XY]) - fabs(tau) > 1e-7) {
           printf("  [1] |T_t| - mu |T_n| %+1.12e < = ?\n",fabs(sigma_trial[TENS2D_XY]) - fabs(tau));
           }
           {
           double a = slip_rate * fabs(sigma_trial[TENS2D_XY]) - fabs(slip_rate) * sigma_trial[TENS2D_XY];
           
           if (fabs(a) > 1.0e-10) {
           printf("  [3] dot s |T_t| - |dot s| T_t %+1.12e = 0 ?\n",slip_rate * fabs(sigma_trial[TENS2D_XY]) - fabs(slip_rate) * sigma_trial[TENS2D_XY]);
           }
           }
           }
           */
        } // Compresive sigma_n
      } // Inside Fault Region END


      {
        PetscReal *_sigma_store = &sigma_tilde2[e*(c->npe * 3) + q*3];
        _sigma_store[TENS2D_XX] = sigma_trial[TENS2D_XX];
        _sigma_store[TENS2D_YY] = sigma_trial[TENS2D_YY];
        _sigma_store[TENS2D_XY] = sigma_trial[TENS2D_XY];
        ierr = Local2GlobalChangeOfBasis(normal, tangent, _sigma_store);CHKERRQ(ierr);
      }


      /* add the initial stress state on fault */
      if (fabs(DistOnFault) < 1.5*1.0e3 && cell_flag[e]) {
        sigma_trial[TENS2D_XY] -= sigma_t_1;
        sigma_trial[TENS2D_YY] -= (-sigma_n_1); /* negative in compression */
      } else {
        sigma_trial[TENS2D_XY] -= sigma_t_0;
        sigma_trial[TENS2D_YY] -= (-sigma_n_0); /* negative in compression */
      } 
      
      ierr = Local2GlobalChangeOfBasis(normal, tangent, sigma_trial);CHKERRQ(ierr);

      /* These components weren't modified in the horizontal fault case - but they might be in general */
      sigma_vec[TENS2D_XX] = sigma_trial[TENS2D_XX];
      sigma_vec[TENS2D_YY] = sigma_trial[TENS2D_YY];
      /* This component was modified in the horizontal fault case - it's likely it might also be modified in the general case */
      sigma_vec[TENS2D_XY] = sigma_trial[TENS2D_XY];

      c->sigma[e*(c->npe * 3) + q*3 + TENS2D_XX] = sigma_vec[TENS2D_XX];
      c->sigma[e*(c->npe * 3) + q*3 + TENS2D_YY] = sigma_vec[TENS2D_YY];
      c->sigma[e*(c->npe * 3) + q*3 + TENS2D_XY] = sigma_vec[TENS2D_XY];
    } // Loop over the quadrature points: END
  } // Loop over the elements: END

  //ierr = CGProjectNative(c, c->basisorder, 3, c->sigma, &proj);CHKERRQ(ierr);
  //ierr = VecGetArrayRead(proj,&_proj_sigma);CHKERRQ(ierr);


  // if (step%10 == 0) {
  //   char prefix[PETSC_MAX_PATH_LEN];
  //   ierr = DGProject(c, c->basisorder, 3, sigma_tilde);CHKERRQ(ierr);
  //   ierr = DGProject(c, c->basisorder, 3, sigma_tilde2);CHKERRQ(ierr);
    
  //   ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"step-%.4D-sigma_tilde_APre.vtu",step);CHKERRQ(ierr);
  //   ierr = StressView_PV(c,(const PetscReal*)sigma_tilde,prefix);CHKERRQ(ierr);

  //   ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"step-%.4D-sigma_tilde_BPost.vtu",step);CHKERRQ(ierr);
  //   ierr = StressView_PV(c,(const PetscReal*)sigma_tilde2,prefix);CHKERRQ(ierr);
  // }

  for (e=0; e<c->ne; e++) {
    ierr = PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);CHKERRQ(ierr);

    /* get element -> node map */
    elnidx = &element[nbasis*e];
    
    /* generate dofs */
    for (i=0; i<nbasis; i++) {
      eldofs[2*i  ] = 2*elnidx[i];
      eldofs[2*i+1] = 2*elnidx[i]+1;
    }
    
    /* get element coordinates */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
    }
    ElementEvaluateGeometry_CellWiseConstant2d(nbasis,elcoords,c->npe_1d,&detJ);
    ElementEvaluateDerivatives_CellWiseConstant2d(nqp,nbasis,elcoords,
                                                  c->npe_1d,c->dN_dxi,c->dN_deta,
                                                  c->dN_dx,c->dN_dy);
    for (q=0; q<c->nqp; q++) {
      PetscReal fac;
      PetscReal *dNidx,*dNidy;
      dNidx = c->dN_dx[q];
      dNidy = c->dN_dy[q];
      PetscReal sigma_vec[3];

      sigma_vec[TENS2D_XX] = c->sigma[e*(c->npe * 3) + q*3 + TENS2D_XX];
      sigma_vec[TENS2D_YY] = c->sigma[e*(c->npe * 3) + q*3 + TENS2D_YY];
      sigma_vec[TENS2D_XY] = c->sigma[e*(c->npe * 3) + q*3 + TENS2D_XY];

      // sigma_vec[0] = sigma_vec[1] = sigma_vec[2] = 0;
      // {
      //   PetscInt nidx = elnidx[q]; // NOTE: index via quad point index rather than basis function index
      //   sigma_vec[TENS2D_XX] = _proj_sigma[3*nidx + TENS2D_XX]; 
      //   sigma_vec[TENS2D_YY] = _proj_sigma[3*nidx + TENS2D_YY]; 
      //   sigma_vec[TENS2D_XY] = _proj_sigma[3*nidx + TENS2D_XY]; 
      // }

      fac = detJ * c->w[q];
      for (i=0; i<nbasis; i++) {
        fe[2*i  ] += -fac * (dNidx[i] * sigma_vec[TENS2D_XX] + dNidy[i] * sigma_vec[TENS2D_XY]);
        fe[2*i+1] += -fac * (dNidy[i] * sigma_vec[TENS2D_YY] + dNidx[i] * sigma_vec[TENS2D_XY]);
      }
      
    }
    ierr = VecSetValues(fl,nbasis*ndof,eldofs,fe,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(fl);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(fl);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(c->dm,fl,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(c->dm,fl,ADD_VALUES,F);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(vl,&LA_v);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(c->dm,&vl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ul,&LA_u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(c->dm,&ul);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(c->dm,&fl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  if(c->basisorder > OrderLimiter){
    ierr = VecRestoreArrayRead(proj,&_proj_sigma);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&proj);CHKERRQ(ierr);

  ierr = PetscFree(sigma_tilde2);CHKERRQ(ierr);
  ierr = PetscFree(sigma_tilde);CHKERRQ(ierr);
  ierr = PetscFree(cell_flag);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode Update_StressGlut2d(SpecFECtx c,Vec u,Vec v,PetscReal dt)
{
  PetscErrorCode ierr;
  PetscInt  e,nqp,q,i,nbasis,ndof;
  PetscReal e_vec[3],gradu[4],gradv[4],gradv_q[9*9][4];
  PetscInt  *element,*elnidx,*eldofs;
  PetscReal *ux,*uy,*vx,*vy,*elcoords,detJ,*fieldU,*fieldV;
  Vec       coor,ul,vl;
  const PetscReal *LA_coor,*LA_u,*LA_v;
  QPntIsotropicElastic *celldata;
  DRVar                *dr_celldata;
  
  static PetscBool beenhere = PETSC_FALSE;
  static PetscReal gmin[3],gmax[3];
  PetscReal dx,dy;
  
  if (!beenhere) {
    ierr = DMGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
    beenhere = PETSC_TRUE;
  }
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
  
  
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  nqp      = c->nqp;
  ndof     = c->dofs;
  element  = c->element;
  fieldU   = c->elbuf_field2;
  fieldV   = c->elbuf_field3;
  
  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(c->dm,&ul);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(c->dm,u,INSERT_VALUES,ul);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(c->dm,u,INSERT_VALUES,ul);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ul,&LA_u);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(c->dm,&vl);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(c->dm,v,INSERT_VALUES,vl);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(c->dm,v,INSERT_VALUES,vl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vl,&LA_v);CHKERRQ(ierr);
  
  ux = &fieldU[0];
  uy = &fieldU[nbasis];
  
  vx = &fieldV[0];
  vy = &fieldV[nbasis];
  
  for (e=0; e<c->ne; e++) {
    PetscReal x_cell[] = {0,0};
    
    /* get element -> node map */
    elnidx = &element[nbasis*e];
    
    /* generate dofs */
    for (i=0; i<nbasis; i++) {
      eldofs[2*i  ] = 2*elnidx[i];
      eldofs[2*i+1] = 2*elnidx[i]+1;
    }
    
    /* get element coordinates */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
      x_cell[0] += elcoords[2*i  ];
      x_cell[1] += elcoords[2*i+1];
    }
    x_cell[0] = x_cell[0] / ((PetscReal)nbasis);
    x_cell[1] = x_cell[1] / ((PetscReal)nbasis);
    
    /* get element displacements & velocities */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      ux[i] = LA_u[2*nidx  ];
      uy[i] = LA_u[2*nidx+1];
      
      vx[i] = LA_v[2*nidx  ];
      vy[i] = LA_v[2*nidx+1];
    }
    
    /* compute derivatives */
    ElementEvaluateGeometry_CellWiseConstant2d(nbasis,elcoords,c->npe_1d,&detJ);
    ElementEvaluateDerivatives_CellWiseConstant2d(nqp,nbasis,elcoords,
                                                  c->npe_1d,c->dN_dxi,c->dN_deta,
                                                  c->dN_dx,c->dN_dy);
    
    /* get access to element->quadrature points */
    celldata = &c->cell_data[e];
    
    ierr = SpecFECtxGetDRCellData(c,e,&dr_celldata);CHKERRQ(ierr);
    
    for (q=0; q<c->nqp; q++) {
      PetscReal *dNidx,*dNidy;
      
      dNidx = c->dN_dx[q];
      dNidy = c->dN_dy[q];
      
      gradv[0] = gradv[1] = gradv[2] = gradv[3] = 0.0;
      for (i=0; i<nbasis; i++) {
        gradv[0] += dNidx[i] * vx[i];
        gradv[1] += dNidy[i] * vx[i];
        gradv[2] += dNidx[i] * vy[i];
        gradv[3] += dNidy[i] * vy[i];
      }
      gradv_q[q][0] = gradv[0];
      gradv_q[q][1] = gradv[1];
      gradv_q[q][2] = gradv[2];
      gradv_q[q][3] = gradv[3];
    }
    
    
    for (q=0; q<c->nqp; q++) {
      PetscReal *dNidx,*dNidy;
      PetscReal coor_qp[2];
      PetscBool inside_fault_region;
      
      dNidx = c->dN_dx[q];
      dNidy = c->dN_dy[q];
      
      /* compute strain @ quadrature point */
      /*
       e = Bu = [ d/dx  0    ][ u v ]^T
       [ 0     d/dy ]
       [ d/dy  d/dx ]
       */
      e_vec[0] = e_vec[1] = e_vec[2] = 0.0;
      for (i=0; i<nbasis; i++) {
        e_vec[0] += dNidx[i] * ux[i];
        e_vec[1] += dNidy[i] * uy[i];
        e_vec[2] += (dNidx[i] * uy[i] + dNidy[i] * ux[i]);
      }
      
      
      coor_qp[0] = elcoords[2*q  ];
      coor_qp[1] = elcoords[2*q+1];
      
      inside_fault_region = PETSC_FALSE;

      ierr = FaultSDFQuery(coor_qp,c->delta,(void *)c->sdf, e*c->nqp + q, &inside_fault_region);CHKERRQ(ierr);

      /* Make stress glut corrections here */
      if (inside_fault_region) {
        PetscReal x_plus[2],x_minus[2],plus[2],minus[2];
        PetscReal normal[2],tangent[2],Uplus,Uminus,Vplus,Vminus,slip,slip_rate,phi_p;
        
        ierr = evaluate_sdf((void *)c->sdf,coor_qp, e*c->nqp + q, &phi_p);CHKERRQ(ierr);
        
        ierr = FaultSDFGetPlusMinusCoor(coor_qp,c->delta,(void *)c->sdf, e*c->nqp + q, x_plus,x_minus);CHKERRQ(ierr);
        
        /* ================================================================ */
        ierr = FaultSDFTabulateInterpolation_v2(c,LA_v,&dr_celldata[q],plus,minus);CHKERRQ(ierr);
        ierr = FaultSDFNormal(coor_qp,(void *)c->sdf, e*c->nqp + q, normal);CHKERRQ(ierr);
        ierr = FaultSDFTangent(coor_qp,(void *)c->sdf, e*c->nqp + q, tangent);CHKERRQ(ierr);
        
        /* Resolve velocities at delta(+,-) onto fault */
        {
          PetscReal mag_vdotn;
          
          mag_vdotn = plus[0] * normal[0] +  plus[1] * normal[1];
          plus[0] = plus[0] - mag_vdotn * normal[0];
          plus[1] = plus[1] - mag_vdotn * normal[1];
          
          mag_vdotn = minus[0] * normal[0] +  minus[1] * normal[1];
          minus[0] = minus[0] - mag_vdotn * normal[0];
          minus[1] = minus[1] - mag_vdotn * normal[1];
        }

        Vplus  = plus[0];
        Vminus = minus[0];
        // slip_rate = Vplus - Vminus;
				slip_rate = ( plus[0] - minus[0] ) * tangent[0] + ( plus[1] - minus[1] ) * tangent[1] ;	

        
        
        /* ================================================================ */
        ierr = FaultSDFTabulateInterpolation_v2(c,LA_u,&dr_celldata[q],plus,minus);CHKERRQ(ierr);
        ierr = FaultSDFNormal(coor_qp,(void *)c->sdf, e*c->nqp + q, normal);CHKERRQ(ierr);
        ierr = FaultSDFTangent(coor_qp,(void *)c->sdf, e*c->nqp + q, tangent);CHKERRQ(ierr);
        
        /* Resolve displacement at delta(+,-) onto fault */
        {
          PetscReal mag_vdotn;
          
          mag_vdotn = plus[0] * normal[0] +  plus[1] * normal[1];
          plus[0] = plus[0] - mag_vdotn * normal[0];
          plus[1] = plus[1] - mag_vdotn * normal[1];
          
          mag_vdotn = minus[0] * normal[0] +  minus[1] * normal[1];
          minus[0] = minus[0] - mag_vdotn * normal[0];
          minus[1] = minus[1] - mag_vdotn * normal[1];
        }
        
        Uplus  = plus[0];
        Uminus = minus[0];
        //slip = Uplus - Uminus;
        slip= ( plus[0] - minus[0] ) * tangent[0] + ( plus[1] - minus[1] ) * tangent[1] ;	
        /* ================================================================ */
        
        
        if (dr_celldata[q].sliding) {
          /*
          dr_celldata[q].slip_rate = slip_rate;
          dr_celldata[q].slip      += slip_rate * dt;

          dr_celldata[q].slip_rate  = (slip - dr_celldata[q].slip)/dt;
          dr_celldata[q].slip       = slip;
          */

          dr_celldata[q].slip_rate = slip_rate;        
          dr_celldata[q].slip      = slip;
        }
        
      }
      
    }
  }
  
  ierr = VecRestoreArrayRead(vl,&LA_v);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(c->dm,&vl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ul,&LA_u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(c->dm,&ul);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


PetscErrorCode AssembleBilinearForm_Mass2d(SpecFECtx c,Vec A)
{
  PetscErrorCode ierr;
  PetscInt  e,index,q,i,nbasis,ndof;
  PetscInt  *element,*elnidx,*eldofs;
  PetscReal *elcoords,*Me,detJ;
  Vec       coor;
  const PetscReal *LA_coor;
  QPntIsotropicElastic *celldata;
  
  ierr = VecZeroEntries(A);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  ndof     = c->dofs;
  Me       = c->elbuf_field;
  element  = c->element;
  
  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  for (e=0; e<c->ne; e++) {
    /* get element -> node map */
    elnidx = &element[nbasis*e];
    
    /* generate dofs */
    for (i=0; i<nbasis; i++) {
      
      eldofs[2*i  ] = 2*elnidx[i];
      eldofs[2*i+1] = 2*elnidx[i]+1;
    }
    
    /* get element coordinates */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
    }
    
    ElementEvaluateGeometry_CellWiseConstant2d(nbasis,elcoords,c->npe_1d,&detJ);
    
    /* get access to element->quadrature points */
    celldata = &c->cell_data[e];

    for (q=0; q<nbasis; q++) {
      PetscReal            fac,Me_ii;
      
      fac = detJ * c->w[q];
      
      Me_ii = fac * (celldata->rho);
      
      /* \int u0v0 dV */
      index = 2*q;
      Me[index] = Me_ii;
      
      /* \int u1v1 dV */
      index = 2*q + 1;
      Me[index] = Me_ii;
    }
    ierr = VecSetValuesLocal(A,nbasis*ndof,eldofs,Me,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(A);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(A);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ElastoDynamicsConvertLame2Velocity(PetscReal rho,PetscReal mu,PetscReal lambda,PetscReal *Vs,PetscReal *Vp)
{
  if (Vs) { *Vs = PetscSqrtReal(mu/rho); }
  if (Vp) { *Vp = PetscSqrtReal( (lambda + 2.0*mu)/rho); }
  
  PetscFunctionReturn(0);
}

PetscErrorCode ElastoDynamicsComputeTimeStep_2d(SpecFECtx ctx,PetscReal *_dt)
{
  PetscInt e,q,order;
  PetscReal dt_min,dt_min_g,polynomial_fac;
  //QPntIsotropicElastic *qpdata;
  PetscReal gmin[3],gmax[3],min_el_r,dx,dy;
  PetscErrorCode ierr;
  QPntIsotropicElastic *celldata;
  
  *_dt = PETSC_MAX_REAL;
  dt_min = PETSC_MAX_REAL;
  
  order = ctx->basisorder;
  polynomial_fac = 1.0 / (2.0 * (PetscReal)order + 1.0);
  
  ierr = DMGetBoundingBox(ctx->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)ctx->mx_g);
  dy = (gmax[1] - gmin[1])/((PetscReal)ctx->my_g);
  
  min_el_r = dx;
  min_el_r = PetscMin(min_el_r,dy);
  
  /* find smallest dx across the element in local coordinates */
  {
    PetscInt  n;
    PetscReal sep2min,sep2;
    
    sep2min = 1.0e32;
    for (n=0; n<ctx->npe_1d-1; n++) {
      sep2 = PetscAbsReal(ctx->xi1d[n+1] - ctx->xi1d[n]);
      /*printf(" xi %+1.4e [n] : xi %+1.4e [n+1] : delta_xi %+1.6e\n",ctx->xi1d[n],ctx->xi1d[n+1],sep2); */
      if (sep2 < sep2min) {
        sep2min = sep2;
      }
    }
    
    polynomial_fac = 1.0;
    min_el_r = min_el_r * ( sep2min / 2.0 ); /* the factor 2.0 here is associated with the size of the element in the local coordinate system xi \in [-1,+1] */
  }
  
  

  
  for (e=0; e<ctx->ne; e++) {
    PetscReal max_el_Vp,value;
    
    /* get max Vp for element */
    max_el_Vp = PETSC_MIN_REAL;
    
    /* get access to element->quadrature points */
    celldata = &ctx->cell_data[e];
    
    for (q=0; q<ctx->nqp; q++) {
      PetscReal qp_rho,qp_mu,qp_lambda,qp_Vp;
      
      qp_rho    = celldata->rho;
      qp_mu     = celldata->mu;
      qp_lambda = celldata->lambda;
      
      ierr = ElastoDynamicsConvertLame2Velocity(qp_rho,qp_mu,qp_lambda,0,&qp_Vp);CHKERRQ(ierr);
      
      max_el_Vp = PetscMax(max_el_Vp,qp_Vp);
    }
    
    value = polynomial_fac * 1.0 * min_el_r / max_el_Vp;
    
    dt_min = PetscMin(dt_min,value);
  }
  ierr = MPI_Allreduce(&dt_min,&dt_min_g,1,MPIU_REAL,MPIU_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
  
  *_dt = dt_min_g;
  
  PetscFunctionReturn(0);
}

PetscErrorCode RecordUV(SpecFECtx c,PetscReal time,PetscReal xr[],Vec u,Vec v)
{
  FILE *fp = NULL;
  PetscReal gmin[3],gmax[3],dx,dy,sep2min,sep2;
  const PetscReal *LA_u,*LA_v,*LA_c;
  Vec coor;
  static PetscBool beenhere = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt ei,ej,n,nid,eid,*element,*elbasis;
  static char filename[PETSC_MAX_PATH_LEN];
  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Needs updating to support MPI");
  if (!beenhere) {
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"receiverCP-%Dx%D-p%D.dat",c->mx_g,c->my_g,c->basisorder);CHKERRQ(ierr);
  }
  
  ierr = DMGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
  ei = (xr[0] - gmin[0])/dx; /* todo - needs to be sub-domain gmin */
  
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
  ej = (xr[1] - gmin[1])/dy; /* todo - needs to be sub-domain gmin */
  
  eid = ei + ej * c->mx;
  
  /* get element -> node map */
  element = c->element;
  elbasis = &element[c->npe*eid];
  
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_c);CHKERRQ(ierr);
  
  // find closest //
  sep2min = 1.0e32;
  nid = -1;
  for (n=0; n<c->npe; n++) {
    sep2  = (xr[0]-LA_c[2*elbasis[n]])*(xr[0]-LA_c[2*elbasis[n]]);
    sep2 += (xr[1]-LA_c[2*elbasis[n]+1])*(xr[1]-LA_c[2*elbasis[n]+1]);
    if (sep2 < sep2min) {
      nid = elbasis[n];
      sep2min = sep2;
    }
  }
  
  if (!beenhere) {
    fp = fopen(filename,"w");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",filename);
    fprintf(fp,"# SpecFECtx meta data\n");
    fprintf(fp,"#   mx %d : my %d : basis order %d\n",c->mx_g,c->my_g,c->basisorder);
    fprintf(fp,"# Receiver meta data\n");
    fprintf(fp,"#   + receiver location: x,y %+1.8e %+1.8e\n",xr[0],xr[1]);
    fprintf(fp,"#   + takes displ/velo from basis nearest to requested receiver location\n");
    fprintf(fp,"#   + receiver location: x,y %+1.8e %+1.8e --mapped to nearest node --> %+1.8e %+1.8e\n",xr[0],xr[1],LA_c[2*nid],LA_c[2*nid+1]);
    fprintf(fp,"# Time series header\n");
    fprintf(fp,"#   time ux uy vx vy\n");
    beenhere = PETSC_TRUE;
  } else {
    fp = fopen(filename,"a");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",filename);
  }
  
  ierr = VecGetArrayRead(u,&LA_u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(v,&LA_v);CHKERRQ(ierr);
  
  fprintf(fp,"%1.4e %+1.8e %+1.8e %+1.8e %+1.8e\n",time,LA_u[2*nid],LA_u[2*nid+1],LA_v[2*nid],LA_v[2*nid+1]);
  
  ierr = VecRestoreArrayRead(v,&LA_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(u,&LA_u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&LA_c);CHKERRQ(ierr);
  
  fclose(fp);
  
  PetscFunctionReturn(0);
}

PetscErrorCode RecordUV_interp(SpecFECtx c,PetscReal time,PetscReal xr[],Vec u,Vec v)
{
  FILE *fp = NULL;
  PetscReal gmin[3],gmax[3],dx,dy,ur[2],vr[2];
  const PetscReal *LA_u,*LA_v;
  static PetscBool beenhere = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt k,ei,ej,eid,*element,*elbasis;
  static PetscReal N[400];
  static char filename[PETSC_MAX_PATH_LEN];
  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Needs updating to support MPI");
  if (!beenhere) {
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"receiver-%Dx%D-p%D.dat",c->mx_g,c->my_g,c->basisorder);CHKERRQ(ierr);
  }
  
  if (!beenhere) {
    fp = fopen(filename,"w");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",filename);
    fprintf(fp,"# SpecFECtx meta data\n");
    fprintf(fp,"#   mx %d : my %d : basis order %d\n",c->mx_g,c->my_g,c->basisorder);
    fprintf(fp,"# Receiver meta data\n");
    fprintf(fp,"#   + receiver location: x,y %+1.8e %+1.8e\n",xr[0],xr[1]);
    fprintf(fp,"#   + records displ/velo at requested receiver location through interpolating the FE solution\n");
    fprintf(fp,"# Time series header\n");
    fprintf(fp,"#   time ux uy vx vy\n");
  } else {
    fp = fopen(filename,"a");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",filename);
  }
  
  /* get containing element */
  ierr = DMGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
  ei = (xr[0] - gmin[0])/dx; /* todo - needs to be sub-domain gmin */
  
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
  ej = (xr[1] - gmin[1])/dy;
  
  eid = ei + ej * c->mx;
  
  /* get element -> node map */
  element = c->element;
  elbasis = &element[c->npe*eid];
  
  if (!beenhere) {
    PetscInt nbasis,i,j;
    PetscReal **N_s1,**N_s2,xri[2],xi,eta,x0,y0;
    const PetscReal *LA_c;
    Vec coor;
    
    /* compute xi,eta */
    ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
    
    x0 = gmin[0] + ei*dx; /* todo - needs to be sub-domain gmin */
    y0 = gmin[1] + ej*dy;
    
    // (xi - (-1))/2 = (x - x0)/dx
    xi = 2.0*(xr[0] - x0)/dx - 1.0;
    eta = 2.0*(xr[1] - y0)/dy - 1.0;
    
    /* compute basis */
    ierr = TabulateBasis1d_CLEGENDRE(1,&xi,c->basisorder,&nbasis,&N_s1);CHKERRQ(ierr);
    ierr = TabulateBasis1d_CLEGENDRE(1,&eta,c->basisorder,&nbasis,&N_s2);CHKERRQ(ierr);
    
    k = 0;
    for (j=0; j<c->npe_1d; j++) {
      for (i=0; i<c->npe_1d; i++) {
        N[k] = N_s1[0][i] * N_s2[0][j];
        k++;
      }
    }
    
    ierr = VecGetArrayRead(coor,&LA_c);CHKERRQ(ierr);
    
    xri[0] = xri[1] = 0.0;
    for (k=0; k<c->npe; k++) {
      PetscInt nid = elbasis[k];
      
      xri[0] += N[k] * LA_c[2*nid+0];
      xri[1] += N[k] * LA_c[2*nid+1];
    }
    
    
    PetscPrintf(PETSC_COMM_SELF,"# receiver location: x,y %+1.8e %+1.8e -- interpolated coordinate --> %+1.8e %+1.8e\n",xr[0],xr[1],xri[0],xri[1]);
    
    ierr = VecRestoreArrayRead(coor,&LA_c);CHKERRQ(ierr);
    ierr = PetscFree(N_s1[0]);CHKERRQ(ierr);
    ierr = PetscFree(N_s1);CHKERRQ(ierr);
    ierr = PetscFree(N_s2[0]);CHKERRQ(ierr);
    ierr = PetscFree(N_s2);CHKERRQ(ierr);
  }
  
  
  ierr = VecGetArrayRead(u,&LA_u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(v,&LA_v);CHKERRQ(ierr);
  
  ur[0] = ur[1] = vr[0] = vr[1] = 0.0;
  for (k=0; k<c->npe; k++) {
    PetscInt nid = elbasis[k];
    
    ur[0] += N[k] * LA_u[2*nid+0];
    ur[1] += N[k] * LA_u[2*nid+1];
    
    vr[0] += N[k] * LA_v[2*nid+0];
    vr[1] += N[k] * LA_v[2*nid+1];
  }
  
  fprintf(fp,"%1.4e %+1.8e %+1.8e %+1.8e %+1.8e\n",time,ur[0],ur[1],vr[0],vr[1]);
  
  ierr = VecRestoreArrayRead(v,&LA_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(u,&LA_u);CHKERRQ(ierr);
  
  beenhere = PETSC_TRUE;
  fclose(fp);
  
  PetscFunctionReturn(0);
}

PetscErrorCode RecordUVA_MultipleStations_NearestGLL_SEQ(SpecFECtx c,PetscReal time,PetscInt nr,PetscReal xr[],Vec u,Vec v,Vec a)
{
  FILE             *fp = NULL;
  const PetscReal  *LA_u,*LA_v,*LA_a;
  static PetscBool beenhere = PETSC_FALSE;
  static char      filename[PETSC_MAX_PATH_LEN];
  static PetscInt  *nid_list = NULL;
  PetscInt         r;
  PetscErrorCode   ierr;

  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Supports sequential only");
  if (!beenhere) {
    const PetscReal *LA_c;
    Vec coor;
    PetscReal gmin[3],gmax[3],dx,dy,sep2min,sep2;
    PetscInt ei,ej,n,nid,eid,*element,*elbasis;
    
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"closestqpsource-receiverCP-uva-%Dx%D-p%D.dat",c->mx_g,c->my_g,c->basisorder);CHKERRQ(ierr);
    ierr = PetscMalloc1(nr,&nid_list);CHKERRQ(ierr);
    
    ierr = DMGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
    ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coor,&LA_c);CHKERRQ(ierr);
    
    for (r=0; r<nr; r++) {
      
      if (xr[2*r+0] < gmin[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, x-coordinate (%+1.4e) < min(domain).x (%+1.4e)",r,xr[2*r+0],gmin[0]);
      if (xr[2*r+1] < gmin[1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, y-coordinate (%+1.4e) < min(domain).y (%+1.4e)",r,xr[2*r+1],gmin[1]);
      if (xr[2*r+0] > gmax[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, x-coordinate (%+1.4e) > max(domain).x (%+1.4e)",r,xr[2*r+0],gmax[0]);
      if (xr[2*r+1] > gmax[1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, y-coordinate (%+1.4e) > max(domain).y (%+1.4e)",r,xr[2*r+1],gmax[1]);
      
      dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
      ei = (xr[2*r+0] - gmin[0])/dx;
      if (ei == c->mx_g) ei--;
      
      dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
      ej = (xr[2*r+1] - gmin[1])/dy;
      if (ej == c->my_g) ej--;
      
      eid = ei + ej * c->mx_g;
    
      /* get element -> node map */
      element = c->element;
      elbasis = &element[c->npe*eid];
    
      // find closest //
      sep2min = 1.0e32;
      nid = -1;
      for (n=0; n<c->npe; n++) {
        sep2  = (xr[2*r+0]-LA_c[2*elbasis[n]])*(xr[2*r+0]-LA_c[2*elbasis[n]]);
        sep2 += (xr[2*r+1]-LA_c[2*elbasis[n]+1])*(xr[2*r+1]-LA_c[2*elbasis[n]+1]);
        if (sep2 < sep2min) {
          nid = elbasis[n];
          sep2min = sep2;
        }
      }
      nid_list[r] = nid;
    }
  
    fp = fopen(filename,"w");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",filename);
    fprintf(fp,"# SpecFECtx meta data\n");
    fprintf(fp,"#   mx %d : my %d : basis order %d\n",c->mx_g,c->my_g,c->basisorder);
    fprintf(fp,"# Receiver meta data\n");
    fprintf(fp,"#   + number receiver locations: %d\n",nr);
    fprintf(fp,"#   + takes displ/velo/accel from basis nearest to requested receiver location\n");
    for (r=0; r<nr; r++) {
      fprintf(fp,"#   + receiver location [%d]: x,y %+1.8e %+1.8e\n",r,xr[2*r+0],xr[2*r+1]);
      fprintf(fp,"#   +   mapped to nearest node --> %+1.8e %+1.8e\n",LA_c[2*nid_list[r]],LA_c[2*nid_list[r]+1]);
    }
    fprintf(fp,"# Time series header <field>(<column index>)\n");
    fprintf(fp,"#   time(1)\n");
    for (r=0; r<nr; r++) {
      PetscInt offset = 1 + r*6; /* 1 is for time */
      
      fprintf(fp,"#     ux(%d) uy(%d) vx(%d) vy(%d) ax(%d) ay(%d) -> station [%d]\n",offset+1,offset+2,offset+3,offset+4,offset+5,offset+6,r);
    }
    ierr = VecRestoreArrayRead(coor,&LA_c);CHKERRQ(ierr);
    beenhere = PETSC_TRUE;
  } else {
    fp = fopen(filename,"a");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",filename);
  }
  
  ierr = VecGetArrayRead(u,&LA_u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(v,&LA_v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(a,&LA_a);CHKERRQ(ierr);
  
  fprintf(fp,"%1.4e",time);
  for (r=0; r<nr; r++) {
    fprintf(fp," %+1.8e %+1.8e %+1.8e %+1.8e %+1.8e %+1.8e",LA_u[2*nid_list[r]],LA_u[2*nid_list[r]+1],LA_v[2*nid_list[r]],LA_v[2*nid_list[r]+1],LA_a[2*nid_list[r]],LA_a[2*nid_list[r]+1]);
  }
  fprintf(fp,"\n");
  
  ierr = VecRestoreArrayRead(a,&LA_a);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(v,&LA_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(u,&LA_u);CHKERRQ(ierr);
  
  fclose(fp);
  
  PetscFunctionReturn(0);
}

PetscErrorCode RecordUVA_MultipleStations_NearestGLL_MPI(SpecFECtx c,PetscReal time,PetscInt nr,PetscReal xr[],Vec u,Vec v,Vec a)
{
  FILE             *fp = NULL;
  const PetscReal  *LA_u,*LA_v,*LA_a,*LA_c;
  static PetscBool beenhere = PETSC_FALSE;
  static char      filename[PETSC_MAX_PATH_LEN];
  static PetscInt  *nid_list = NULL;
  static PetscInt  *eid_list = NULL;
  static PetscInt  *gll_list = NULL;
  static PetscInt  nr_local = 0;
  PetscInt         r,k;
  Vec              lu,lv,la,coor;
  PetscErrorCode   ierr;
  
  
  if (!beenhere) {
    PetscReal       gmin[3],gmax[3],gmin_domain[3],gmax_domain[3],dx,dy,sep2min,sep2;
    PetscInt        ei,ej,n,nid,gllid,eid,*element,*elbasis;
    
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"closestqpsource-receiverCP-uva-%Dx%D-p%D-rank%d.dat",c->mx_g,c->my_g,c->basisorder,(int)c->rank);CHKERRQ(ierr);
    ierr = PetscMalloc1(nr,&nid_list);CHKERRQ(ierr);
    ierr = PetscMalloc1(nr,&eid_list);CHKERRQ(ierr);
    ierr = PetscMalloc1(nr,&gll_list);CHKERRQ(ierr);
    for (r=0; r<nr; r++) {
      nid_list[r] = -1;
      eid_list[r] = -1;
      gll_list[r] = -1;
    }

    ierr = DMGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
    ierr = SpecFECtxGetLocalBoundingBox(c,gmin_domain,gmax_domain);CHKERRQ(ierr);
    
    ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coor,&LA_c);CHKERRQ(ierr);
    
    for (r=0; r<nr; r++) {
      int count,recv_count;
      PetscBool receiver_found = PETSC_TRUE;
      int rank,rank_min_g;
      
      if (xr[2*r+0] < gmin[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, x-coordinate (%+1.4e) < min(domain).x (%+1.4e)",r,xr[2*r+0],gmin[0]);
      if (xr[2*r+1] < gmin[1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, y-coordinate (%+1.4e) < min(domain).y (%+1.4e)",r,xr[2*r+1],gmin[1]);
      if (xr[2*r+0] > gmax[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, x-coordinate (%+1.4e) > max(domain).x (%+1.4e)",r,xr[2*r+0],gmax[0]);
      if (xr[2*r+1] > gmax[1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, y-coordinate (%+1.4e) > max(domain).y (%+1.4e)",r,xr[2*r+1],gmax[1]);
      
      if (xr[2*r+0] < gmin_domain[0]) receiver_found = PETSC_FALSE;
      if (xr[2*r+1] < gmin_domain[1]) receiver_found = PETSC_FALSE;
      if (xr[2*r+0] > gmax_domain[0]) receiver_found = PETSC_FALSE;
      if (xr[2*r+1] > gmax_domain[1]) receiver_found = PETSC_FALSE;
      
      dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
      ei = (xr[2*r+0] - gmin_domain[0])/dx;
      if (ei == c->mx) ei--;
      
      dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
      ej = (xr[2*r+1] - gmin_domain[1])/dy;
      if (ej == c->my) ej--;
      
      if (ei < 0) receiver_found = PETSC_FALSE;
      if (ej < 0) receiver_found = PETSC_FALSE;
      
      if (ei > c->mx) receiver_found = PETSC_FALSE;
      if (ej > c->my) receiver_found = PETSC_FALSE;
      
      nid = -1;
      gllid = -1;
      if (receiver_found) {
        eid = ei + ej * c->mx;
        
        /* get element -> node map */
        element = c->element;
        elbasis = &element[c->npe*eid];
        
        // find closest //
        sep2min = 1.0e32;
        for (n=0; n<c->npe; n++) {
          sep2  = (xr[2*r+0]-LA_c[2*elbasis[n]])*(xr[2*r+0]-LA_c[2*elbasis[n]]);
          sep2 += (xr[2*r+1]-LA_c[2*elbasis[n]+1])*(xr[2*r+1]-LA_c[2*elbasis[n]+1]);
          if (sep2 < sep2min) {
            nid = elbasis[n];
            gllid = n;
            sep2min = sep2;
          }
        }
      }
      
      /* check for duplicates */
      count = 0;
      if (receiver_found) {
        count = 1;
      }
      ierr = MPI_Allreduce(&count,&recv_count,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
      
      if (recv_count == 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"A receiver was defined but no rank claimed it");
      
      if (recv_count > 1) {
        /* resolve duplicates */
        
        rank = (int)c->rank;
        if (!receiver_found) {
          rank = (int)c->size;
        }
        ierr = MPI_Allreduce(&rank,&rank_min_g,1,MPI_INT,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
        if (rank == rank_min_g) {
          PetscPrintf(PETSC_COMM_SELF,"[RecordUVA]  + Multiple ranks located receiver (%+1.4e,%+1.4e) - rank %d claiming ownership\n",xr[2*r+0],xr[2*r+1],rank_min_g);
        }
        
        /* mark non-owning ranks as not claiming source */
        if (rank != rank_min_g) {
          receiver_found = PETSC_FALSE;
        }
      }

      if (receiver_found) {
        nid_list[r] = nid;
        eid_list[r] = eid;
        gll_list[r] = gllid;
        nr_local++;
      }
    }
    
    fp = fopen(filename,"w");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",filename);
    fprintf(fp,"# SpecFECtx meta data\n");
    fprintf(fp,"#   mx %d : my %d : basis order %d\n",c->mx_g,c->my_g,c->basisorder);
    fprintf(fp,"# Receiver meta data\n");
    fprintf(fp,"#   + number receiver locations: %d\n",nr);
    fprintf(fp,"#   + number receiver locations <local>: %d\n",nr_local);
    fprintf(fp,"#   + takes displ/velo/accel from basis nearest to requested receiver location\n");
    for (r=0; r<nr; r++) {
      if (nid_list[r] == -1) { continue; }
      fprintf(fp,"#   + receiver location [%d]: x,y %+1.8e %+1.8e\n",r,xr[2*r+0],xr[2*r+1]);
      fprintf(fp,"#   +   mapped to nearest node --> %+1.8e %+1.8e\n",LA_c[2*nid_list[r]],LA_c[2*nid_list[r]+1]);
    }

    if (nr_local != 0) {
      PetscInt count = 0;

      fprintf(fp,"# Time series header <field>(<column index>)\n");
      fprintf(fp,"#   time(1)\n");

    
      for (r=0; r<nr; r++) {
        PetscInt offset;
        
        if (nid_list[r] == -1) { continue; }

        offset = 1 + count*7; /* 1 is for time */
        
        fprintf(fp,"#     ux(%d) uy(%d) vx(%d) vy(%d) ax(%d) ay(%d) curl(v) (%d)-> station [%d]\n",offset+1,offset+2,offset+3,offset+4,offset+5,offset+6,offset+7,r);
        count++;
      }
    } else {
      fprintf(fp,"# <note> No receivers found on this sub-domain\n");
      fprintf(fp,"# <note> This file will remain empty\n");
    }

    ierr = VecRestoreArrayRead(coor,&LA_c);CHKERRQ(ierr);
    
    fclose(fp);
    fp = NULL;
  }
  
  if (!beenhere) {
    char metafname[PETSC_MAX_PATH_LEN];
    FILE *fp_meta = NULL;
    int *owned,*owned_g;
    
    ierr = PetscSNPrintf(metafname,PETSC_MAX_PATH_LEN-1,"closestqpsource-receiverCP-uva-%Dx%D-p%D.mpimeta",c->mx_g,c->my_g,c->basisorder);CHKERRQ(ierr);
    
    ierr = PetscMalloc1(nr,&owned);CHKERRQ(ierr);
    ierr = PetscMalloc1(nr,&owned_g);CHKERRQ(ierr);
    for (r=0; r<nr; r++) {
      owned[r] = -1;
      if (nid_list[r] != -1) { owned[r] = (int)c->rank; }
    }
    ierr = MPI_Allreduce(owned,owned_g,nr,MPI_INT,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
    
    if (c->rank == 0) {
      fp_meta = fopen(metafname,"w");
      if (!fp_meta) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",metafname);
    
      fprintf(fp_meta,"# SpecFECtx parallel/MPI meta data\n");
      fprintf(fp_meta,"#   mx %d : my %d : basis order %d\n",c->mx_g,c->my_g,c->basisorder);
      fprintf(fp_meta,"# Receiver meta data\n");
      fprintf(fp_meta,"#   + number receiver locations: %d\n",nr);
      for (r=0; r<nr; r++) {
        fprintf(fp_meta,"#   + receiver [%d]: mapped to MPI rank %d\n",r,owned_g[r]);
      }
      
      fclose(fp_meta);
    }
    
    ierr = PetscFree(owned_g);CHKERRQ(ierr);
    ierr = PetscFree(owned);CHKERRQ(ierr);
  }
  
  beenhere = PETSC_TRUE;

  ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_c);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(c->dm,&lu);CHKERRQ(ierr);
  ierr = DMGetLocalVector(c->dm,&lv);CHKERRQ(ierr);
  ierr = DMGetLocalVector(c->dm,&la);CHKERRQ(ierr);
  
  ierr = DMGlobalToLocalBegin(c->dm,u,INSERT_VALUES,lu);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(c->dm,u,INSERT_VALUES,lu);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(c->dm,v,INSERT_VALUES,lv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(c->dm,v,INSERT_VALUES,lv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(c->dm,a,INSERT_VALUES,la);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(c->dm,a,INSERT_VALUES,la);CHKERRQ(ierr);
  
  ierr = VecGetArrayRead(lu,&LA_u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(lv,&LA_v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(la,&LA_a);CHKERRQ(ierr);
  
  if (nr_local != 0) {
    fp = fopen(filename,"a");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",filename);
  
    fprintf(fp,"%1.4e",time);
  
    for (r=0; r<nr; r++) {
      PetscInt eidx,gllidx;
      PetscReal *elcoor,*elvelocity;
      PetscReal dvxdy,dvydx,curl;
      PetscReal *grad_N_xi[2];
      PetscReal *grad_N_x[2];
      
      if (eid_list[r] == -1) { continue; }
      
      /* write the components of u,v,a */
      fprintf(fp," %+1.8e %+1.8e %+1.8e %+1.8e %+1.8e %+1.8e",LA_u[2*nid_list[r]],LA_u[2*nid_list[r]+1],LA_v[2*nid_list[r]],LA_v[2*nid_list[r]+1],LA_a[2*nid_list[r]],LA_a[2*nid_list[r]+1]);
      
      /* compute and write the k^th component of the curl(v) */
      eidx   = eid_list[r];
      gllidx = gll_list[r];
      
      grad_N_xi[0] = c->dN_dxi[gllidx];
      grad_N_xi[1] = c->dN_deta[gllidx];
      
      grad_N_x[0] = c->dN_dx[gllidx];
      grad_N_x[1] = c->dN_dy[gllidx];
      
      elcoor = c->elbuf_field;
      elvelocity = c->elbuf_field2;
      
      for (k=0; k<c->npe; k++) {
        PetscInt basisid = c->element[c->npe*eidx + k];
        
        elcoor[2*k+0] = LA_c[2*basisid + 0];
        elcoor[2*k+1] = LA_c[2*basisid + 1];
        
        elvelocity[2*k+0] = LA_v[2*basisid + 0];
        elvelocity[2*k+1] = LA_v[2*basisid + 1];
      }
      
      ElementEvaluateDerivatives_CellWiseConstant2d(1,c->npe,elcoor,c->npe_1d,&grad_N_xi[0],&grad_N_xi[1],&grad_N_x[0],&grad_N_x[1]);
      
      dvxdy = 0.0;
      dvydx = 0.0;
      for (k=0; k<c->npe; k++) {
        PetscReal vx,vy;
        
        vx = elvelocity[2*k+0];
        vy = elvelocity[2*k+1];
        
        dvxdy += grad_N_x[1][k] * vx;
        dvydx += grad_N_x[0][k] * vy;
      }
      curl = dvydx - dvxdy;
      fprintf(fp," %+1.8e",curl);
    }
    
    fprintf(fp,"\n");

    if (fp) {
      fclose(fp);
      fp = NULL;
    }
  }
  
  ierr = VecRestoreArrayRead(a,&LA_a);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(v,&LA_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(u,&LA_u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&LA_c);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(c->dm,&la);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(c->dm,&lv);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(c->dm,&lu);CHKERRQ(ierr);

  
  PetscFunctionReturn(0);
}

PetscErrorCode RecordUVA_MultipleStations_NearestGLL(SpecFECtx c,PetscReal time,PetscInt nr,PetscReal xr[],Vec u,Vec v,Vec a)
{
  PetscErrorCode ierr;
  
  //if (c->size == 1) {
  //ierr = RecordUVA_MultipleStations_NearestGLL_SEQ(c,time,nr,xr,u,v,a);CHKERRQ(ierr);
  //} else {
  ierr = RecordUVA_MultipleStations_NearestGLL_MPI(c,time,nr,xr,u,v,a);CHKERRQ(ierr);
  //}
  PetscFunctionReturn(0);
}

PetscErrorCode RecordDRVar_MultipleStations_NearestGLL_SEQ(SpecFECtx c,PetscReal time,PetscInt nr,PetscReal xr[])
{
  FILE             *fp = NULL;
  static PetscBool beenhere = PETSC_FALSE;
  static char      filename[PETSC_MAX_PATH_LEN];
  static PetscInt  *qid_list = NULL;
  static PetscInt  *nid_list = NULL;
  static PetscInt  *eid_list = NULL;
  PetscInt         r;
  PetscErrorCode   ierr;
  
  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Supports sequential only");
  if (!beenhere) {
    const PetscReal *LA_c;
    Vec coor;
    PetscReal gmin[3],gmax[3],dx,dy,sep2min,sep2;
    PetscInt ei,ej,n,qid,nid,eid,*element,*elbasis;
    
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"receiverCP-dr-%Dx%D-p%D.dat",c->mx_g,c->my_g,c->basisorder);CHKERRQ(ierr);
    ierr = PetscMalloc1(nr,&qid_list);CHKERRQ(ierr);
    ierr = PetscMalloc1(nr,&nid_list);CHKERRQ(ierr);
    ierr = PetscMalloc1(nr,&eid_list);CHKERRQ(ierr);
    
    ierr = DMGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
    ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coor,&LA_c);CHKERRQ(ierr);
    
    for (r=0; r<nr; r++) {
      
      if (xr[2*r+0] < gmin[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, x-coordinate (%+1.4e) < min(domain).x (%+1.4e)",r,xr[2*r+0],gmin[0]);
      if (xr[2*r+1] < gmin[1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, y-coordinate (%+1.4e) < min(domain).y (%+1.4e)",r,xr[2*r+1],gmin[1]);
      if (xr[2*r+0] > gmax[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, x-coordinate (%+1.4e) > max(domain).x (%+1.4e)",r,xr[2*r+0],gmax[0]);
      if (xr[2*r+1] > gmax[1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, y-coordinate (%+1.4e) > max(domain).y (%+1.4e)",r,xr[2*r+1],gmax[1]);
      
      dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
      ei = (xr[2*r+0] - gmin[0])/dx;
      if (ei == c->mx_g) ei--;
      
      dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
      ej = (xr[2*r+1] - gmin[1])/dy;
      if (ej == c->my_g) ej--;
      
      eid = ei + ej * c->mx_g;
      eid_list[r] = eid;
      
      /* get element -> node map */
      element = c->element;
      elbasis = &element[c->npe*eid];
      
      // find closest //
      sep2min = 1.0e32;
      nid = -1;
      qid = -1;
      for (n=0; n<c->npe; n++) {
        sep2  = (xr[2*r+0]-LA_c[2*elbasis[n]])*(xr[2*r+0]-LA_c[2*elbasis[n]]);
        sep2 += (xr[2*r+1]-LA_c[2*elbasis[n]+1])*(xr[2*r+1]-LA_c[2*elbasis[n]+1]);
        if (sep2 < sep2min) {
          nid = elbasis[n];
          qid = n;
          sep2min = sep2;
        }
      }
      nid_list[r] = nid;
      qid_list[r] = qid;
    }
    
    fp = fopen(filename,"w");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",filename);
    fprintf(fp,"# SpecFECtx meta data\n");
    fprintf(fp,"#   mx %d : my %d : basis order %d\n",c->mx_g,c->my_g,c->basisorder);
    fprintf(fp,"# Receiver meta data\n");
    fprintf(fp,"#   + number receiver locations: %d\n",nr);
    fprintf(fp,"#   + takes DR variables from quadrature point nearest to requested receiver location\n");
    for (r=0; r<nr; r++) {
      fprintf(fp,"#   + receiver location [%d]: x,y %+1.8e %+1.8e\n",r,xr[2*r+0],xr[2*r+1]);
      fprintf(fp,"#   +   mapped to nearest node --> %+1.8e %+1.8e\n",LA_c[2*nid_list[r]],LA_c[2*nid_list[r]+1]);
      fprintf(fp,"#   +   mapped to nearest element/quad-point --> %d %d\n",eid_list[r],qid_list[r]);
    }
    fprintf(fp,"# Time series header <field>(<column index>)\n");
    fprintf(fp,"#   time(1)\n");
    for (r=0; r<nr; r++) {
      PetscInt offset = 1 + r*4; /* 1 is for time */
      
      fprintf(fp,"#     slip(%d) sliprate(%d) mu(%d) sliding(%d) -> station [%d]\n",offset+1,offset+2,offset+3,offset+4,r);
    }
    ierr = VecRestoreArrayRead(coor,&LA_c);CHKERRQ(ierr);
    beenhere = PETSC_TRUE;
  } else {
    fp = fopen(filename,"a");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",filename);
  }
  
  fprintf(fp,"%1.4e",time);
  for (r=0; r<nr; r++) {
    DRVar *cell_data;
    PetscInt e_index,q_index;
    
    e_index = eid_list[r];
    q_index = qid_list[r];
    ierr = SpecFECtxGetDRCellData(c,e_index,&cell_data);CHKERRQ(ierr);
    
    fprintf(fp," %+1.8e %+1.8e %+1.8e %+1.2e",cell_data[q_index].slip,cell_data[q_index].slip_rate,cell_data[q_index].mu,(double)cell_data[q_index].sliding);
  }
  fprintf(fp,"\n");
  
  fclose(fp);
  
  PetscFunctionReturn(0);
}

PetscErrorCode SE2WaveViewer_JSON(SpecFECtx ctx,PetscInt step,PetscReal time,
                                  const char data_description[],PetscInt len,const char *fieldname[],const Vec field[],
                                  const char pbin[],const char jfilename[])
{
  PetscErrorCode ierr;
  char str[PETSC_MAX_PATH_LEN];
  FILE *fp = NULL;
  PetscMPIInt commrank;
  PetscInt k;
  
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&commrank);CHKERRQ(ierr);
  if (commrank != 0) PetscFunctionReturn(0);
  
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"%s",jfilename);CHKERRQ(ierr);
  fp = fopen(str,"w");
  if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",str);
  
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"{\"se2wave\":{");CHKERRQ(ierr); fprintf(fp,"%s\n",str);
  
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"time\": %1.12e",time);CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"step\": %D",step);CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"spatial_dimension\": %D",ctx->dim);CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"mx\": %D",ctx->mx_g);CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"my\": %D",ctx->my_g);CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"nx\": %D",ctx->nx_g);CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"ny\": %D",ctx->ny_g);CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"basis_degree\": %D",ctx->basisorder);CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"fields\": [ ");CHKERRQ(ierr); fprintf(fp,"%s",str);
  for (k=0; k<len; k++) {
    ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"\"null\"");CHKERRQ(ierr);
    if (field[k]) {
      ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"\"%s\"",fieldname[k]);CHKERRQ(ierr);
    }
    
    if (k != (len-1)) { fprintf(fp,"%s, ",str);
    } else { fprintf(fp,"%s",str); }
  }
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1," ]");CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  
  //ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"datafile\": \"%s\"",pbin);CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"data\":{");CHKERRQ(ierr); fprintf(fp,"%s\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"    \"description\": \"%s\"",data_description);CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"    \"fields\": [ ");CHKERRQ(ierr); fprintf(fp,"%s",str);
  for (k=0; k<len; k++) {
    ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"\"null\"");CHKERRQ(ierr);
    if (field[k]) {
      ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"\"%s\"",fieldname[k]);CHKERRQ(ierr);
    }
    
    if (k != (len-1)) { fprintf(fp,"%s, ",str);
    } else { fprintf(fp,"%s",str); }
  }
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1," ]");CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  
  
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"    \"writer\": \"petsc_binary\"");CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"    \"type\": \"Vec\"");CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"    \"filename\": \"%s\"",pbin);CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  /* petsc always writes binary in big endian ordering (even on a small endian machine) */
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"    \"endian\": \"big\"");CHKERRQ(ierr); fprintf(fp,"%s\n",str);
  /*
   #ifdef WORDSIZE_BIGENDIAN
   ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"    \"endian\": \"big\"");CHKERRQ(ierr); fprintf(fp,"%s\n",str);
   #else
   ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"    \"endian\": \"little\"");CHKERRQ(ierr); fprintf(fp,"%s\n",str);
   #endif
   */
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  }");CHKERRQ(ierr); fprintf(fp,"%s,\n",str);
  
  
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"  \"version\": [1,0,0]");CHKERRQ(ierr); fprintf(fp,"%s\n",str);
  ierr = PetscSNPrintf(str,PETSC_MAX_PATH_LEN-1,"}}");CHKERRQ(ierr); fprintf(fp,"%s\n",str);
  
  fclose(fp);
  
  PetscFunctionReturn(0);
}

PetscErrorCode SE2WaveCoordinateViewerViewer(SpecFECtx ctx,PetscInt step,PetscReal time,const char prefix[])
{
  PetscErrorCode ierr;
  PetscViewer vu;
  Vec coor = NULL;
  char fname[PETSC_MAX_PATH_LEN];
  
  ierr = DMGetCoordinates(ctx->dm,&coor);CHKERRQ(ierr);
  if (!coor) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must have a valid coordinate vector");
  
  ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s_coor.pbin",prefix);CHKERRQ(ierr);
  
  {
    char jname[PETSC_MAX_PATH_LEN];
    Vec input[] = {NULL};
    const char *fieldname[] = { "coor" };
    
    input[0] = coor;
    ierr = PetscSNPrintf(jname,PETSC_MAX_PATH_LEN-1,"%s_coor.json",prefix);CHKERRQ(ierr);
    ierr = SE2WaveViewer_JSON(ctx,step,time,"coordinates",1,fieldname,input,fname,jname);CHKERRQ(ierr);
  }
  
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fname,FILE_MODE_WRITE,&vu);CHKERRQ(ierr);
  
  ierr = PetscViewerBinaryWrite(vu,(void*)&ctx->mx_g,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(vu,(void*)&ctx->my_g,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(vu,(void*)&ctx->nx_g,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(vu,(void*)&ctx->ny_g,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  
  ierr = VecView(coor,vu);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vu);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode SE2WaveWaveFieldViewer(SpecFECtx ctx,PetscInt step,PetscReal time,Vec u,Vec v,const char prefix[])
{
  PetscErrorCode ierr;
  PetscViewer vu;
  char fname[PETSC_MAX_PATH_LEN];
  
  if (!u && !v) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"At least one of the displacement or velocity vectors must be non-NULL");
  
  ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s_wavefield.pbin",prefix);CHKERRQ(ierr);
  
  {
    char jname[PETSC_MAX_PATH_LEN];
    Vec input[] = {NULL,NULL};
    const char *fieldname[] = { "u", "v" };
    input[0] = u;
    input[1] = v;
    
    ierr = PetscSNPrintf(jname,PETSC_MAX_PATH_LEN-1,"%s_wavefield.json",prefix);CHKERRQ(ierr);
    ierr = SE2WaveViewer_JSON(ctx,step,time,"wavefield",2,fieldname,input,fname,jname);CHKERRQ(ierr);
  }
  
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fname,FILE_MODE_WRITE,&vu);CHKERRQ(ierr);
  
  ierr = PetscViewerBinaryWrite(vu,(void*)&ctx->mx_g,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(vu,(void*)&ctx->my_g,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(vu,(void*)&ctx->nx_g,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(vu,(void*)&ctx->ny_g,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  
  ierr = PetscViewerBinaryWrite(vu,(void*)&step,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(vu,(void*)&time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  
  if (u) { ierr = VecView(u,vu);CHKERRQ(ierr); }
  if (v) { ierr = VecView(v,vu);CHKERRQ(ierr); }
  ierr = PetscViewerDestroy(&vu);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
/**
 * Function to calculate the new KV timestep following Galvez (2014) eq 27.
 * dT_KV = (sqrt(1+(eta*eta)/(dT*dT))-eta/dT)
*/ 
void GetStableTimeStep(double dT, double eta, double * dT_KV)
{
  dT_KV[0] = (sqrt(1 + (eta/dT) * (eta/dT)) - eta / dT)*dT;
} 


PetscErrorCode se2dr_demo(PetscInt mx,PetscInt my)
{
  PetscErrorCode ierr;
  SpecFECtx ctx;
  PetscInt p,k,nt,of;
  PetscViewer viewer;
  Vec u,v,a,f,g,Md;
  PetscReal time,dt,time_max;
  PetscInt nrecv;
  PetscReal *xr_list;
  PetscBool dump_ic_src_vts = PETSC_FALSE;
  PetscBool ignore_receiver_output = PETSC_FALSE;
  PetscReal nrm,max,min,dx,dy;
  PetscReal gamma;
  char vts_fname[PETSC_MAX_PATH_LEN];
  
  
  ierr = PetscOptionsGetBool(NULL,NULL,"-dump_ic_src",&dump_ic_src_vts,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-ignore_receiver_output",&ignore_receiver_output,NULL);CHKERRQ(ierr);
  
  /*
    Create the structured mesh for the spectral element method.
   The default mesh is defined over the domain [0,1]^2.
  */
  ierr = SpecFECtxCreate(&ctx);CHKERRQ(ierr);
  p = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-bdegree",&p,NULL);CHKERRQ(ierr);
  ierr = SpecFECtxCreateMesh(ctx,2,mx,my,PETSC_DECIDE,p,2);CHKERRQ(ierr);

  /*
   Define your domain by shifting and scaling the default [0,1]^2 domain
  */
  {
    PetscReal alpha = 10.0e3;
    PetscReal scale[] = {  2.0*alpha, 2.0*alpha };
    PetscReal shift[] = { -1.0*alpha,-1.0*alpha };
    
    ierr = SpecFECtxScaleMeshCoords(ctx,scale,shift);CHKERRQ(ierr);
  }

  /* 
   Specify fault dimensions
  */
  {
    PetscReal gmin[3],gmax[3];
    
    ierr = DMGetBoundingBox(ctx->dm,gmin,gmax);CHKERRQ(ierr);
    dx = (gmax[0] - gmin[0])/((PetscReal)ctx->mx_g);
    dy = (gmax[1] - gmin[1])/((PetscReal)ctx->my_g);
  }
  PetscPrintf(PETSC_COMM_WORLD,"[se2dr] cell sizes: dx = %1.4e, dy = %1.4e\n",dx,dy);
  
  
  ctx->delta = 25.0;
  ierr = PetscOptionsGetReal(NULL,NULL,"-delta",&ctx->delta,NULL);CHKERRQ(ierr);
  {
    PetscBool found;
    PetscReal delta_factor = 0.0;
    
    found = PETSC_FALSE;
    ierr = PetscOptionsGetReal(NULL,NULL,"-delta_cell_factor",&delta_factor,&found);CHKERRQ(ierr);
    if (found) {
      ctx->delta = dy * delta_factor;
    }
  }
  PetscPrintf(PETSC_COMM_WORLD,"[se2dr] using fault delta = %1.4e\n",ctx->delta);
  PetscPrintf(PETSC_COMM_WORLD,"[se2dr] elements across fault = %1.4e\n",2.0 * ctx->delta/dy);
  
  ierr = FaultSDFInit_v3(ctx);CHKERRQ(ierr);
  
  /*
   Specify the material properties for the domain.
   This function sets constant material properties in every cell.
   More general methods can be easily added.
  */
  ierr = SpecFECtxSetConstantMaterialProperties_Velocity(ctx,4000.0 ,4000.0/sqrt(3.0), 2500.0);CHKERRQ(ierr); // vp,vs,rho
  /* Linear slip weakening parameters */
  ctx->mu_s = 0.677;
  ctx->mu_d = 0.525;
  ctx->D_c = 0.40;

  
  ierr = DMCreateGlobalVector(ctx->dm,&u);CHKERRQ(ierr); ierr = PetscObjectSetName((PetscObject)u,"disp");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ctx->dm,&v);CHKERRQ(ierr); ierr = PetscObjectSetName((PetscObject)v,"velo");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ctx->dm,&a);CHKERRQ(ierr); ierr = PetscObjectSetName((PetscObject)a,"accl");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ctx->dm,&f);CHKERRQ(ierr); ierr = PetscObjectSetName((PetscObject)f,"f");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ctx->dm,&g);CHKERRQ(ierr); ierr = PetscObjectSetName((PetscObject)g,"g");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ctx->dm,&Md);CHKERRQ(ierr);
  
  ierr = VecZeroEntries(u);CHKERRQ(ierr);
  
  /*
   Write out the mesh and intial values for the displacement, velocity and acceleration (u,v,a)
  */
  if (dump_ic_src_vts) {
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"uva.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(u,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  ierr = AssembleBilinearForm_Mass2d(ctx,Md);CHKERRQ(ierr);
  
  /*
   Define the location of the receivers
  */
  nrecv = 7;
  ierr = PetscMalloc1(nrecv*2,&xr_list);CHKERRQ(ierr);
  xr_list[0] = 0.0;
  xr_list[1] = ctx->delta;

  xr_list[2] = 2000.0;
  xr_list[3] = ctx->delta-1;

  xr_list[4] = 4000.0;
  xr_list[5] = ctx->delta-1;

  xr_list[6] = 6000.0;
  xr_list[7] = ctx->delta-1;

  xr_list[8] = 8000.0;
  xr_list[9] = ctx->delta-1;

  xr_list[10] = 10.0e3;
  xr_list[11] = ctx->delta;

  xr_list[12] = 1.0e3;
  xr_list[13] = 2.0*ctx->delta;


  /* Initialize time loop */
  k = 0;
  time = 0.0;
  
  time_max = 0.4;
  ierr = PetscOptionsGetReal(NULL,NULL,"-tmax",&time_max,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2dr] Requested time period: %1.4e\n",time_max);
  
  ierr = ElastoDynamicsComputeTimeStep_2d(ctx,&dt);CHKERRQ(ierr);
  
  /** */
  PetscPrintf(PETSC_COMM_WORLD,"[Nico] estimated dt: %1.4e\n",dt);
  /** Artificial viscosity - KV time stepping */
  gamma  = 0.6*dt;
  GetStableTimeStep(dt, gamma, &dt);
  PetscPrintf(PETSC_COMM_WORLD,"[Nico] New estimated dt: %1.4e\n",dt);
  /** */

  dt = dt * 0.5;
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2dr] Using time step size: %1.4e\n",dt);
  
  nt = 1000000;
  nt = (PetscInt)(time_max / dt ) + 4;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2dr] Estimated number of time steps: %D\n",nt);
  
  of = 5000;
  ierr = PetscOptionsGetInt(NULL,NULL,"-of",&of,NULL);CHKERRQ(ierr);
  
  ierr = SE2WaveCoordinateViewerViewer(ctx,0,0.0,"default_mesh");CHKERRQ(ierr);
  {
    char prefix[PETSC_MAX_PATH_LEN];
    
    ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"step-%.4D",0);CHKERRQ(ierr);
    ierr = SE2WaveWaveFieldViewer(ctx,k,time,u,v,prefix);CHKERRQ(ierr);
  }
  
  if (k%of == 0) {
    ierr = PetscSNPrintf(vts_fname,PETSC_MAX_PATH_LEN-1,"step-%.4D.vts",0);CHKERRQ(ierr);
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,vts_fname,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(u,viewer);CHKERRQ(ierr);
    ierr = VecView(v,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  /* Perform time stepping */
  for (k=1; k<=nt; k++) {
    
    time = time + dt;
    
    ierr = VecAXPY(u,dt,v);CHKERRQ(ierr); /* u_{n+1} = u_{n} + dt.v_{n} */
    
    ierr = VecAXPY(u,0.5*dt*dt,a);CHKERRQ(ierr); /* u_{n+1} = u_{n+1} + 0.5.dt^2.a_{n} */
    
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v' = v_{n} + 0.5.dt.a_{n} */
    
    /* Compute f = -F^{int}( u_{n+1} ) */
    
    ierr = AssembleLinearForm_ElastoDynamics_StressGlut2d(ctx,u,v,dt,time, gamma,f,k);CHKERRQ(ierr);

    /* Update force; F^{ext}_{n+1} = f + S(t_{n+1}) g(x) */
    ierr = VecAXPY(f,1.0,g);CHKERRQ(ierr);
    
    /* "Solve"; a_{n+1} = M^{-1} f */
    ierr = VecPointwiseDivide(a,f,Md);CHKERRQ(ierr);
    
    /* Update velocity */
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v_{n+1} = v' + 0.5.dt.a_{n+1} */
    
    /* Update slip-rate & slip */
    ierr = Update_StressGlut2d(ctx,u,v,dt);CHKERRQ(ierr);

    
    if (k%10 == 0) {
      PetscPrintf(PETSC_COMM_WORLD,"[step %9D] time = %1.4e : dt = %1.4e \n",k,time,dt);
      VecNorm(u,NORM_2,&nrm);
      VecMin(u,0,&min);
      VecMax(u,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [displacement] max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
      VecNorm(v,NORM_2,&nrm);
      VecMin(v,0,&min);
      VecMax(v,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [velocity]     max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
    }

    
    
    /*
      Write out the u,v,a values at each receiver
    */
    if (!ignore_receiver_output) {
      ierr = RecordUVA_MultipleStations_NearestGLL(ctx,time,nrecv,xr_list,u,v,a);CHKERRQ(ierr);
      ierr = RecordDRVar_MultipleStations_NearestGLL_SEQ(ctx,time,nrecv,xr_list);CHKERRQ(ierr);
    }
    
    if (k%of == 0) {
      ierr = PetscSNPrintf(vts_fname,PETSC_MAX_PATH_LEN-1,"step-%.4D.vts",k);CHKERRQ(ierr);
      ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,vts_fname,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(u,viewer);CHKERRQ(ierr);
      ierr = VecView(v,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    if (k%of == 0) {
      char prefix[PETSC_MAX_PATH_LEN];
      ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"step-%.4D",k);CHKERRQ(ierr);
      ierr = SE2WaveWaveFieldViewer(ctx,k,time,u,v,prefix);CHKERRQ(ierr);
    }
    if (k%of == 0) {
      char prefix[PETSC_MAX_PATH_LEN];
      ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"step-%.4D-sigma.vtu",k);CHKERRQ(ierr);
      ierr = StressView_PV(ctx,(const PetscReal*)ctx->sigma,prefix);CHKERRQ(ierr);
      
      /*
      Vec proj = NULL, v[3];
      PetscInt b;
      DM dm1;
      ierr = CGProjectNative(ctx,2,3,ctx->sigma,&proj);CHKERRQ(ierr);

      ierr = DMDACreateCompatibleDMDA(ctx->dm,1,&dm1);CHKERRQ(ierr);
        
        for (b=0; b<3; b++) { DMCreateGlobalVector(dm1,&v[b]); }
          PetscObjectSetName((PetscObject)v[0],"sxx");
          PetscObjectSetName((PetscObject)v[1],"syy");
          PetscObjectSetName((PetscObject)v[2],"sxy");
          ierr = PetscSNPrintf(vts_fname,PETSC_MAX_PATH_LEN-1,"step-%.4D_sigmaProj.vts",k);CHKERRQ(ierr);
          ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,vts_fname,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
          for (b=0; b<3; b++) {
            ierr = VecStrideGather(proj,b,v[b],INSERT_VALUES);CHKERRQ(ierr);
            ierr = VecView(v[b],viewer);CHKERRQ(ierr);
          }
          ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
          ierr = DMDestroy(&dm1);CHKERRQ(ierr);
          for (b=0; b<3; b++) {
            ierr = VecDestroy(&v[b]);CHKERRQ(ierr);
          }
          ierr = VecDestroy(&proj);CHKERRQ(ierr);
      */
     
    
      /*
      // test 1
      ierr = DGProject(ctx,1,3,ctx->sigma);CHKERRQ(ierr);
      ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"step-%.4D-sigma_p.vtu",k);CHKERRQ(ierr);
      ierr = StressView_PV(ctx,(const PetscReal*)ctx->sigma,prefix);CHKERRQ(ierr);
      */
      
      /*
       // test 2
      {
      PetscReal *_u;
      PetscReal *uqp;
      PetscInt e,i,b;
      
      SpecFECreateQuadratureField(ctx,2,&uqp);
      VecGetArray(u,&_u);
      for (e=0; e<ctx->ne; e++) {
        PetscInt *elnidx = &ctx->element[ctx->npe*e];
        for (i=0; i<ctx->npe; i++) {
          for (b=0; b<2; b++) {
            uqp[e*ctx->npe * 2 + i*2 + b] = _u[2*elnidx[i] + b];
          }
        }
      }
      VecRestoreArray(u,&_u);
      ierr = DGProject(ctx,1,2,uqp);CHKERRQ(ierr);
      {
        const char *field_names[] = { "ux", "uy" };
        ierr = QuadratureFieldDGView_PV(ctx,2,uqp,field_names,"uproj.vtu");CHKERRQ(ierr);
      }
      }
       
      */

      /*
       // test 3
      {
        Vec proj = NULL;
        DM dm1;
        Vec v[4];
        int b;
        
        ierr = CGProjectNative(ctx,1,4,ctx->gradu,&proj);CHKERRQ(ierr);
        
        ierr = DMDACreateCompatibleDMDA(ctx->dm,1,&dm1);CHKERRQ(ierr);
        
        for (b=0; b<4; b++) { DMCreateGlobalVector(dm1,&v[b]); }
        PetscObjectSetName((PetscObject)v[0],"uxx");
        PetscObjectSetName((PetscObject)v[1],"uxy");
        PetscObjectSetName((PetscObject)v[2],"uyx");
        PetscObjectSetName((PetscObject)v[3],"uyy");
        ierr = PetscSNPrintf(vts_fname,PETSC_MAX_PATH_LEN-1,"step-%.4D_gradu_cg.vts",k);CHKERRQ(ierr);
        ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,vts_fname,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
        for (b=0; b<4; b++) {
          ierr = VecStrideGather(proj,b,v[b],INSERT_VALUES);CHKERRQ(ierr);
          ierr = VecView(v[b],viewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        ierr = DMDestroy(&dm1);CHKERRQ(ierr);
        for (b=0; b<4; b++) {
          ierr = VecDestroy(&v[b]);CHKERRQ(ierr);
        }
        ierr = VecDestroy(&proj);CHKERRQ(ierr);
      }
      */
      
      //
       // test 4
      // {
      //   Vec proj = NULL;
      //   DM dm1;
      //   Vec v[3];
      //   int b;
        
      //   ierr = CGProjectNative(ctx,0,3,ctx->sigma,&proj);CHKERRQ(ierr);
        
      //   ierr = DMDACreateCompatibleDMDA(ctx->dm,1,&dm1);CHKERRQ(ierr);
        
      //   DMCreateGlobalVector(dm1,&v[0]); PetscObjectSetName((PetscObject)v[0],"sxx");
      //   DMCreateGlobalVector(dm1,&v[1]); PetscObjectSetName((PetscObject)v[1],"syy");
      //   DMCreateGlobalVector(dm1,&v[2]); PetscObjectSetName((PetscObject)v[2],"sxy");
      //   ierr = PetscSNPrintf(vts_fname,PETSC_MAX_PATH_LEN-1,"step-%.4D_cg.vts",k);CHKERRQ(ierr);
      //   ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,vts_fname,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      //   for (b=0; b<3; b++) {
      //     ierr = VecStrideGather(proj,b,v[b],INSERT_VALUES);CHKERRQ(ierr);
      //     ierr = VecView(v[b],viewer);CHKERRQ(ierr);
      //   }
      //   ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      //   ierr = DMDestroy(&dm1);CHKERRQ(ierr);
      //   for (b=0; b<3; b++) {
      //     ierr = VecDestroy(&v[b]);CHKERRQ(ierr);
      //   }
      //   ierr = VecDestroy(&proj);CHKERRQ(ierr);
      // }
      //
    }
    
    if (time >= time_max) {
      break;
    }
  }
  PetscPrintf(PETSC_COMM_WORLD,"[step %9D] time = %1.4e : dt = %1.4e \n",k,time,dt);
  VecNorm(u,NORM_2,&nrm);
  VecMin(u,0,&min);
  VecMax(u,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [displacement] max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
  VecNorm(v,NORM_2,&nrm);
  VecMin(v,0,&min);
  VecMax(v,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [velocity]     max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
  
  /* plot last snapshot */
  ierr = PetscSNPrintf(vts_fname,PETSC_MAX_PATH_LEN-1,"step-%.4D.vts",k);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,vts_fname,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = VecView(v,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  //ierr = PetscSNPrintf(vts_fname,PETSC_MAX_PATH_LEN-1,"step-%.4D-sigma.vtu",k);CHKERRQ(ierr);
  //ierr = StressView_PV(ctx,(const PetscReal*)ctx->sigma,vts_fname);CHKERRQ(ierr);
  
  
  ierr = PetscFree(xr_list);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&a);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = VecDestroy(&Md);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       mx,my;
  PetscMPIInt    size;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  
  mx = my = 8;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,NULL);CHKERRQ(ierr);
  my = mx;
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&my,NULL);CHKERRQ(ierr);

  ierr = se2dr_demo(mx,my);CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return(ierr);
}
