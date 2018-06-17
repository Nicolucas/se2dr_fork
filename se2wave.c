
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
  PetscInt  source_implementation;
};

typedef struct _p_SeismicSource *SeismicSource;
typedef struct _p_SeismicSTF *SeismicSTF;

struct _p_SeismicSTF {
  char name[PETSC_MAX_PATH_LEN];
  void *data;
  PetscErrorCode (*evaluate)(SeismicSTF,PetscReal,PetscReal *);
  PetscErrorCode (*destroy)(SeismicSTF);
};

typedef enum { SOURCE_TYPE_FORCE = 0 , SOURCE_TYPE_MOMENT } SeismicSourceType;
typedef enum { SOURCE_IMPL_POINTWISE = 0 , SOURCE_IMPL_NEAREST_QPOINT, SOURCE_IMPL_SPLINE, SOURCE_IMPL_DEFAULT } SeismicSourceImplementationType;

struct _p_SeismicSource {
  SpecFECtx sfem;
  PetscReal coor[2],xi[2];
  PetscInt  element_index,closest_gll;
  PetscMPIInt rank;
  SeismicSourceType type;
  SeismicSourceImplementationType implementation_type;
  PetscReal *values;
  void *data;
  PetscErrorCode (*setup)(SeismicSource);
  PetscErrorCode (*add_values)(SeismicSource,PetscReal,Vec);
  PetscErrorCode (*destroy)(SeismicSource);
  PetscBool issetup;
};

typedef struct {
  PetscReal xi[2];
  PetscInt  nbasis;
  PetscInt  *element_indices;
  PetscReal *element_values;
  PetscReal *buffer;
} PointwiseContext;

typedef struct {
  Vec g;
} SplineContext;

/* prototypes */
PetscErrorCode SeismicSourceSetup_Moment_Pointwise(SeismicSource s);
PetscErrorCode SeismicSourceAddValues_Pointwise(SeismicSource s,PetscReal stf,Vec f);
PetscErrorCode SeismicSourceDestroy_Pointwise(SeismicSource s);


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
    nrmeigs[i] = sqrt( realpt[i]*realpt[i] + complexpt[i]*complexpt[i]);
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
      Aij = pow(xil,i);
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
      monomials[cnt] = pow((double)xi[p],(double)i);
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
      Aij = pow(xil,i);
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
          dm_dx = ((double)i)*pow((double)xi[p],(double)(i-1));
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
  ctx->source_implementation = -1;
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
  
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dim,&c->elbuf_coor);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs,&c->elbuf_field);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs,&c->elbuf_field2);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*c->npe*c->dofs,&c->elbuf_dofs);CHKERRQ(ierr);
  
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
  PetscInt si[2],si_g[2],m,n,ii,jj;
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

void ElementEvaluateGeometry_CellWiseConstant2d(PetscInt npe,PetscReal el_coords[],
                                                PetscInt nbasis,PetscReal *detJ)
{
  PetscInt i,j;
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
  PetscInt  e,npe,nqp,q,i,nbasis,ndof;
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

PetscErrorCode AssembleBilinearForm_Mass2d(SpecFECtx c,Vec A)
{
  PetscErrorCode ierr;
  PetscInt  e,npe,index,q,i,j,nbasis,ndof;
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

PetscErrorCode AssembleLinearForm_ElastoDynamicsMomentDirac2d_NearestInternalQP(SpecFECtx c,PetscInt nsources,PetscReal xs[],PetscReal moment[],Vec F)
{
  PetscErrorCode ierr;
  PetscInt  e,nel,npe;
  PetscInt  i,k;
  PetscInt  nbasis,ndof,nb;
  PetscInt  *element;
  PetscReal   *fe,*elcoords;
  PetscInt    *elnidx,*eldofs;
  PetscInt    *eowner_source,*closest_qp;
  PetscInt    ii,jj,ni,nj,nid;
  PetscReal   dx,dy,gmin[3],gmax[3];
  const PetscReal   *LA_coor;
  Vec         coor;
  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Needs updating to support MPI");
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  ndof     = c->dofs;
  fe       = c->elbuf_field;
  element  = c->element;
  
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscInt)*nsources,&eowner_source);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*nsources,&closest_qp);CHKERRQ(ierr);
  
  /* locate cell containing source */
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
  for (k=0; k<nsources; k++) {
    ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx ); /* todo - needs to be sub-domain gmin */
    jj = (PetscInt)( ( xs[2*k+1] - gmin[1] )/dy );
    
    if (ii == c->mx) ii--;
    if (jj == c->my) jj--;
    
    if (ii < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: x < gmin[0]");
    if (jj < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: y < gmin[1]");
    
    if (ii > c->mx_g) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: x > gmax[0]");
    if (jj > c->my_g) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: y > gmax[1]");
    
    eowner_source[k] = ii + jj * c->mx;
    printf("source[%d] (%+1.4e,%+1.4e) --> element %d \n",k,xs[2*k],xs[2*k+1],eowner_source[k]);
  }
  
  /* locate closest quadrature point */
  for (k=0; k<nsources; k++) {
    PetscReal sep2,sep2_min = PETSC_MAX_REAL;
    PetscInt  min_qp,_ni,_nj;
    
    e = eowner_source[k];
    
    /* get element -> node map */
    elnidx = &element[nbasis*e];
    
    /* get element coordinates */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
    }
    
    //    for (nj=0; nj<c->npe_1d; nj++) {
    //      for (ni=0; ni<c->npe_1d; ni++) {
    
    for (nj=1; nj<c->npe_1d-1; nj++) {
      for (ni=1; ni<c->npe_1d-1; ni++) {
        nid = ni + nj * c->npe_1d;
        
        sep2 = (elcoords[2*nid]-xs[2*k])*(elcoords[2*nid]-xs[2*k]) + (elcoords[2*nid+1]-xs[2*k+1])*(elcoords[2*nid+1]-xs[2*k+1]);
        if (sep2 < sep2_min) {
          sep2_min = sep2;
          min_qp = nid;
          _ni = ni;
          _nj = nj;
        }
      }
    }
    closest_qp[k] = min_qp;
    //closest_qp[k] = 3+3*7;
    printf("source[%d] --> qp %d xi: (%+1.4e,%+1.4e) [%d,%d] \n",k,closest_qp[k],c->xi1d[_ni],c->xi1d[_nj],_ni,_nj);
    printf("source[%d] --> qp %d x:  (%+1.4e,%+1.4e)\n",k,closest_qp[k],elcoords[2*min_qp],elcoords[2*min_qp+1]);
  }
  
  for (k=0; k<nsources; k++) {
    PetscReal *moment_k;
    PetscReal *dN_dxi_q,*dN_deta_q;
    PetscReal *dN_dx_q,*dN_dy_q;
    
    moment_k = &moment[4*k];
    dN_dxi_q   = c->dN_dxi[closest_qp[k]];
    dN_deta_q  = c->dN_deta[closest_qp[k]];
    
    dN_dx_q    = c->dN_dx[closest_qp[k]];
    dN_dy_q    = c->dN_dy[closest_qp[k]];
    
    e = eowner_source[k];
    
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
    
    ierr = PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);CHKERRQ(ierr);
    
    ElementEvaluateDerivatives_CellWiseConstant2d(1,nbasis,elcoords,
                                                  c->npe_1d,&dN_dxi_q,&dN_deta_q,&dN_dx_q,&dN_dy_q);
    
    /* compute moment contribution @ source */
    for (i=0; i<nbasis; i++) {
      fe[2*i  ] += (moment_k[0]*dN_dx_q[i] + moment_k[1]*dN_dy_q[i]);
      fe[2*i+1] += (moment_k[2]*dN_dx_q[i] + moment_k[3]*dN_dy_q[i]);
      //printf("fe[%d] %+1.4e %+1.4e\n",i,fe[2*i],fe[2*i+1]);
    }
    ierr = VecSetValues(F,nbasis*ndof,eldofs,fe,ADD_VALUES);CHKERRQ(ierr);
    
    
    /* torque calculation */
#if 0
    {
      
      PetscReal net_torque = 0.0, torque;
      PetscReal Svec[2],Fvec[2],Fvec_normal[2],Fvec_tangent[2],nrmS,nrmF,costheta,theta,unit[2],centroid[2];
      
      ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx ); /* todo - needs to be sub-domain gmin */
      jj = (PetscInt)( ( xs[2*k+1] - gmin[1] )/dy );
      
      if (ii == c->mx) ii--;
      if (jj == c->my) jj--;
      
      //centroid[0] = gmin[0] + dx * ii + 0.5*dx;
      //centroid[1] = gmin[1] + dy * jj + 0.5*dy;
      
      centroid[0] = elcoords[2*closest_qp[0]];
      centroid[1] = elcoords[2*closest_qp[0]+1];
      
      for (nj=0; nj<c->npe_1d; nj++) {
        for (ni=0; ni<c->npe_1d; ni++) {
          nid = ni + nj * c->npe_1d;
          
          
          /* vector point from cell center to GLL point */
          Svec[0] = elcoords[2*nid]   - centroid[0];
          Svec[1] = elcoords[2*nid+1] - centroid[1];
          
          Fvec[0] = fe[2*nid];
          Fvec[1] = fe[2*nid+1];
          
          nrmS = sqrt(Svec[0]*Svec[0] + Svec[1]*Svec[1]); //printf("nrmS = %1.4e\n",nrmS);
          nrmF = sqrt(Fvec[0]*Fvec[0] + Fvec[1]*Fvec[1]); //printf("nrmF = %1.4e\n",nrmF);
          
          if (nrmF < 1.0e-16) continue;
          
          printf("point[%d][%d]\n",ni,nj);
          
          printf("  c0 %1.4e %1.4e\n", centroid[0],centroid[1]);
          printf("  Svec = %1.4e %1.4e\n",Svec[0],Svec[1]);
          printf("  nrmS = %1.4e\n",nrmS);
          
          costheta = (Svec[0]*Fvec[0] + Svec[1]*Fvec[1])/(nrmS * nrmF); //printf("cos(theta) = %1.4e\n",costheta);
          
          unit[0] = Svec[0]/nrmS;
          unit[1] = Svec[1]/nrmS; //printf("unitS = %1.4e %1.4e\n",unit[0],unit[1]);
          
          // F = F_n + F_t //
          Fvec_tangent[0] = unit[0] * nrmF * costheta;
          Fvec_tangent[1] = unit[1] * nrmF * costheta;
          
          Fvec_normal[0] = Fvec[0] - Fvec_tangent[0];
          Fvec_normal[1] = Fvec[1] - Fvec_tangent[1];
          
          printf("  F = %1.4e %1.4e\n",Fvec[0],Fvec[1]);
          printf("  F_tang = %1.4e %1.4e\n",Fvec_tangent[0],Fvec_tangent[1]);
          printf("  F_norm = %1.4e %1.4e\n",Fvec_normal[0],Fvec_normal[1]);
          
          //torque = Fvec[0]*normal[0] + Fvec[1]*normal[1]; printf("point[%d][%d] torque = %1.4e\n",ni,nj,torque);
          torque = Svec[0] * Fvec_normal[1] - Svec[1] * Fvec_normal[0]; printf("  torque = %1.4e\n",torque);
          
          
          net_torque += torque;
        }
      }
      printf("Net torque over element = %1.4e\n",net_torque);
    }
#endif
    
  }
  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  ierr = PetscFree(eowner_source);CHKERRQ(ierr);
  ierr = PetscFree(closest_qp);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode AssembleLinearForm_ElastoDynamicsMomentDirac2d(SpecFECtx c,PetscInt nsources,PetscReal xs[],PetscReal moment[],Vec F)
{
  PetscErrorCode ierr;
  PetscInt  e,nel,npe;
  PetscInt  i,k;
  PetscInt  nbasis,ndof,nb;
  PetscInt  *element;
  PetscReal   *fe,*elcoords;
  PetscInt    *elnidx,*eldofs;
  PetscInt    *eowner_source;
  PetscInt    ii,jj,ni,nj,nid;
  PetscReal   dx,dy,gmin[3],gmax[3];
  const PetscReal   *LA_coor;
  Vec         coor;
  PetscReal   **dN_dxi,**dN_deta;
  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Needs updating to support MPI");
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  ndof     = c->dofs;
  fe       = c->elbuf_field;
  element  = c->element;
  
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscInt)*nsources,&eowner_source);CHKERRQ(ierr);
  
  /* locate cell containing source */
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
  for (k=0; k<nsources; k++) {
    ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx ); /* todo - needs to be sub-domain gmin */
    jj = (PetscInt)( ( xs[2*k+1] - gmin[1] )/dy );
    
    if (ii == c->mx) ii--;
    if (jj == c->my) jj--;
    
    if (ii < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: x < gmin[0]");
    if (jj < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: y < gmin[1]");
    
    if (ii > c->mx_g) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: x > gmax[0]");
    if (jj > c->my_g) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: y > gmax[1]");
    
    eowner_source[k] = ii + jj * c->mx;
    printf("source[%d] (%+1.4e,%+1.4e) --> element %d \n",k,xs[2*k],xs[2*k+1],eowner_source[k]);
  }
  
  
  
  for (k=0; k<nsources; k++) {
    PetscReal *moment_k;
    PetscReal *dN_dxi_q,*dN_deta_q;
    PetscReal *dN_dx_q,*dN_dy_q;
    PetscReal xi_source[2],cell_min[2];
    
    
    e = eowner_source[k];
    
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
    
    /* Get source local coordinates */
    ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx ); /* todo - needs to be sub-domain gmin */
    jj = (PetscInt)( ( xs[2*k+1] - gmin[1] )/dy );
    
    if (ii == c->mx) ii--;
    if (jj == c->my) jj--;
    
    cell_min[0] = gmin[0] + ii * dx;
    cell_min[1] = gmin[1] + jj * dy;
    
    xi_source[0] = 2.0 * (xs[2*k+0] - cell_min[0])/dx - 1.0;
    xi_source[1] = 2.0 * (xs[2*k+1] - cell_min[1])/dy - 1.0;
    
    if (PetscAbsReal(xi_source[0]) < 1.0e-12) xi_source[0] = 0.0;
    if (PetscAbsReal(xi_source[1]) < 1.0e-12) xi_source[1] = 0.0;
    
    ierr = TabulateBasisDerivativesAtPointTensorProduct2d(xi_source,c->basisorder,&dN_dxi,&dN_deta);CHKERRQ(ierr);
    printf("source[%d] --> xi (%+1.4e,%+1.4e)\n",k,xi_source[0],xi_source[1]);
    
    
    
    moment_k = &moment[4*k];
    dN_dxi_q   = dN_dxi[0];
    dN_deta_q  = dN_deta[0];
    
    dN_dx_q    = c->dN_dx[0];
    dN_dy_q    = c->dN_dy[0];
    
    ierr = PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);CHKERRQ(ierr);
    
    ElementEvaluateDerivatives_CellWiseConstant2d(1,nbasis,elcoords,
                                                  c->npe_1d,&dN_dxi_q,&dN_deta_q,&dN_dx_q,&dN_dy_q);
    
    /* compute moment contribution @ source */
    for (i=0; i<nbasis; i++) {
      fe[2*i  ] += (moment_k[0]*dN_dx_q[i] + moment_k[1]*dN_dy_q[i]);
      fe[2*i+1] += (moment_k[2]*dN_dx_q[i] + moment_k[3]*dN_dy_q[i]);
      //printf("fe[%d] %+1.4e %+1.4e\n",i,fe[2*i],fe[2*i+1]);
    }
    ierr = VecSetValues(F,nbasis*ndof,eldofs,fe,ADD_VALUES);CHKERRQ(ierr);
    
    
    /* torque calculation */
#if 0
    {
      
      PetscReal net_torque = 0.0, torque;
      PetscReal Svec[2],Fvec[2],Fvec_normal[2],Fvec_tangent[2],nrmS,nrmF,costheta,theta,unit[2],centroid[2];
      
      ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx ); /* todo - needs to be sub-domain gmin */
      jj = (PetscInt)( ( xs[2*k+1] - gmin[1] )/dy );
      
      if (ii == c->mx) ii--;
      if (jj == c->my) jj--;
      
      //centroid[0] = gmin[0] + dx * ii + 0.5*dx;
      //centroid[1] = gmin[1] + dy * jj + 0.5*dy;
      
      centroid[0] = xs[2*0+0];
      centroid[1] = xs[2*0+1];
      
      for (nj=0; nj<c->npe_1d; nj++) {
        for (ni=0; ni<c->npe_1d; ni++) {
          nid = ni + nj * c->npe_1d;
          
          
          /* vector point from cell center to GLL point */
          Svec[0] = elcoords[2*nid]   - centroid[0];
          Svec[1] = elcoords[2*nid+1] - centroid[1];
          
          Fvec[0] = fe[2*nid];
          Fvec[1] = fe[2*nid+1];
          
          nrmS = sqrt(Svec[0]*Svec[0] + Svec[1]*Svec[1]); //printf("nrmS = %1.4e\n",nrmS);
          nrmF = sqrt(Fvec[0]*Fvec[0] + Fvec[1]*Fvec[1]); //printf("nrmF = %1.4e\n",nrmF);
          
          if (nrmF < 1.0e-16) continue;
          
          printf("point[%d][%d]\n",ni,nj);
          
          printf("  c0 %1.4e %1.4e\n", centroid[0],centroid[1]);
          printf("  Svec = %1.4e %1.4e\n",Svec[0],Svec[1]);
          printf("  nrmS = %1.4e\n",nrmS);
          
          costheta = (Svec[0]*Fvec[0] + Svec[1]*Fvec[1])/(nrmS * nrmF); //printf("cos(theta) = %1.4e\n",costheta);
          
          unit[0] = Svec[0]/nrmS;
          unit[1] = Svec[1]/nrmS; //printf("unitS = %1.4e %1.4e\n",unit[0],unit[1]);
          
          // F = F_n + F_t //
          Fvec_tangent[0] = unit[0] * nrmF * costheta;
          Fvec_tangent[1] = unit[1] * nrmF * costheta;
          
          Fvec_normal[0] = Fvec[0] - Fvec_tangent[0];
          Fvec_normal[1] = Fvec[1] - Fvec_tangent[1];
          
          printf("  F = %1.4e %1.4e\n",Fvec[0],Fvec[1]);
          printf("  F_tang = %1.4e %1.4e\n",Fvec_tangent[0],Fvec_tangent[1]);
          printf("  F_norm = %1.4e %1.4e\n",Fvec_normal[0],Fvec_normal[1]);
          
          //torque = Fvec[0]*normal[0] + Fvec[1]*normal[1]; printf("point[%d][%d] torque = %1.4e\n",ni,nj,torque);
          torque = Svec[0] * Fvec_normal[1] - Svec[1] * Fvec_normal[0]; printf("  torque = %1.4e\n",torque);
          
          
          net_torque += torque;
        }
      }
      printf("Net torque over element = %1.4e\n",net_torque);
    }
#endif
    
    for (k=0; k<1; k++) {
      ierr = PetscFree(dN_dxi[k]);CHKERRQ(ierr);
      ierr = PetscFree(dN_deta[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dN_dxi);CHKERRQ(ierr);
    ierr = PetscFree(dN_deta);CHKERRQ(ierr);
    
  }
  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);
  
  ierr = PetscFree(eowner_source);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  
  
  PetscFunctionReturn(0);
}

PetscErrorCode AssembleLinearForm_ElastoDynamicsMomentDirac2d_Kernel_CSpline(SpecFECtx c,PetscInt nsources,PetscReal xs[],PetscReal moment[],Vec F)
{
  PetscErrorCode ierr;
  PetscInt  e,nel,npe;
  PetscInt  i,k,q;
  PetscInt  nbasis,ndof,nb;
  PetscInt  *element;
  PetscReal   *fe,*elcoords,int_w = 0.0;
  PetscInt    *elnidx,*eldofs;
  const PetscReal   *LA_coor;
  Vec         coor;
  PetscReal gmin[3],gmax[3],dx,dy,ds,kernel_h;
  PetscBool flg;
  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Needs updating to support MPI");
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
  ds = dx;
  if (dy > dx) ds = dy;
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  ndof     = c->dofs;
  fe       = c->elbuf_field;
  element  = c->element;
  
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  if (c->basisorder <= 3) {
    kernel_h = 1.5 * (0.5*ds);
  } else {
    PetscReal pfac;
    
    //kernel_h = 0.75 * (0.5*ds);
    //printf("kernel_h = %1.4e \n",kernel_h);
    
    //pfac = 2.0/( 2.0*(c->basisorder - 3.0) + 1.0 );
    //kernel_h = pfac * 0.75 * (0.5*ds);
    
    pfac = 3.0/( c->basisorder - 2.0 );
    kernel_h = pfac * (0.5*ds);
    
    
    printf("kernel_h = %1.4e <with pfac>\n",kernel_h);
  }
  ierr = PetscOptionsGetReal(NULL,NULL,"-sm_h",&kernel_h,&flg);CHKERRQ(ierr);
  if (flg) {
    printf("kernel_h = %1.4e <from options>\n",kernel_h);
  }
  
  for (k=0; k<nsources; k++) {
    PetscReal *moment_k;
    PetscReal *dN_dx_q,*dN_dy_q;
    PetscReal detJ;
    
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
      
      moment_k = &moment[4*k];
      
      ierr = PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);CHKERRQ(ierr);
      
      ElementEvaluateDerivatives_CellWiseConstant2d(c->nqp,nbasis,elcoords,
                                                    c->npe_1d,c->dN_dxi,c->dN_deta,c->dN_dx,c->dN_dy);
      
      ElementEvaluateGeometry_CellWiseConstant2d(nbasis,elcoords,c->npe_1d,&detJ);
      
      for (q=0; q<c->nqp; q++) {
        PetscReal smooth_dirac,arg_q,sep,*x_q,*xs_k;
        
        dN_dx_q    = c->dN_dx[q];
        dN_dy_q    = c->dN_dy[q];
        
        xs_k = &xs[2*k];
        
        /* get physical coordinates of quadrature point */
        x_q = &elcoords[2*q];
        
        /* evaluate cubic spline at each quadrature point */
        sep =  (x_q[0] - xs_k[0]) * (x_q[0] - xs_k[0]);
        sep += (x_q[1] - xs_k[1]) * (x_q[1] - xs_k[1]);
        arg_q = PetscSqrtReal(sep) / kernel_h;
        
        if (arg_q < 1.0) {
          smooth_dirac = 0.25 * (2.0 - arg_q)*(2.0 - arg_q)*(2.0 - arg_q) - (1.0 - arg_q)*(1.0 - arg_q)*(1.0 - arg_q);
        } else if (arg_q < 2.0) {
          smooth_dirac = 0.25 * (2.0 - arg_q)*(2.0 - arg_q)*(2.0 - arg_q);
        } else {
          smooth_dirac = 0.0;
        }
        smooth_dirac = 10.0/(7.0*M_PI) * smooth_dirac * (1.0/(kernel_h*kernel_h));
        
        /* compute moment contribution @ source */
        for (i=0; i<nbasis; i++) {
          fe[2*i  ] += c->w[q] * detJ * (moment_k[0]*dN_dx_q[i] + moment_k[1]*dN_dy_q[i]) * smooth_dirac;
          fe[2*i+1] += c->w[q] * detJ * (moment_k[2]*dN_dx_q[i] + moment_k[3]*dN_dy_q[i]) * smooth_dirac;
        }
        
        int_w += c->w[q] * detJ *smooth_dirac;
      }
      ierr = VecSetValues(F,nbasis*ndof,eldofs,fe,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  //ierr = VecScale(F,1.0/int_w);CHKERRQ(ierr);
  
  printf("\\int W = %1.14e \n",int_w);
  
  PetscFunctionReturn(0);
}

PetscErrorCode AssembleLinearForm_ElastoDynamicsMomentDirac2d_P0(SpecFECtx c,PetscInt nsources,PetscReal xs[],PetscReal moment[],Vec F)
{
  PetscErrorCode ierr;
  PetscInt  e,nel,npe;
  PetscInt  i,k,q;
  PetscInt  nbasis,ndof,nb;
  PetscInt  *element;
  PetscReal   *fe,*elcoords;
  PetscInt    *elnidx,*eldofs;
  PetscInt    *eowner_source;
  PetscInt    ii,jj,ni,nj,nid;
  PetscReal   dx,dy,gmin[3],gmax[3];
  const PetscReal   *LA_coor;
  Vec         coor;
  PetscReal   **dN_dxi,**dN_deta;
  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Needs updating to support MPI");
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  ndof     = c->dofs;
  fe       = c->elbuf_field;
  element  = c->element;
  
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscInt)*nsources,&eowner_source);CHKERRQ(ierr);
  
  /* locate cell containing source */
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
  for (k=0; k<nsources; k++) {
    ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx ); /* todo - needs to be sub-domain gmin */
    jj = (PetscInt)( ( xs[2*k+1] - gmin[1] )/dy );
    
    if (ii == c->mx) ii--;
    if (jj == c->my) jj--;
    
    if (ii < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: x < gmin[0]");
    if (jj < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: y < gmin[1]");
    
    if (ii > c->mx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: x > gmax[0]");
    if (jj > c->my) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"source: y > gmax[1]");
    
    eowner_source[k] = ii + jj * c->mx;
    printf("source[%d] (%+1.4e,%+1.4e) --> element %d \n",k,xs[2*k],xs[2*k+1],eowner_source[k]);
  }
  
  for (k=0; k<nsources; k++) {
    PetscReal *moment_k;
    PetscReal *dN_dxi_q,*dN_deta_q;
    PetscReal *dN_dx_q,*dN_dy_q;
    PetscReal xi_source[2],cell_min[2];
    PetscReal detJ;
    
    
    e = eowner_source[k];
    
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
    
    /* Get source local coordinates */
    ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx ); /* todo - needs to be sub-domain gmin */
    jj = (PetscInt)( ( xs[2*k+1] - gmin[1] )/dy );
    
    if (ii == c->mx) ii--;
    if (jj == c->my) jj--;
    
    cell_min[0] = gmin[0] + ii * dx; /* todo - needs to be sub-domain gmin */
    cell_min[1] = gmin[1] + jj * dy;
    
    xi_source[0] = 0.0;
    xi_source[1] = 0.0;
    
    ierr = TabulateBasisDerivativesAtPointTensorProduct2d(xi_source,c->basisorder,&dN_dxi,&dN_deta);CHKERRQ(ierr);
    printf("source[%d] --> xi (%+1.4e,%+1.4e)\n",k,xi_source[0],xi_source[1]);
    
    moment_k = &moment[4*k];
    dN_dxi_q   = dN_dxi[0];
    dN_deta_q  = dN_deta[0];
    
    dN_dx_q    = c->dN_dx[0];
    dN_dy_q    = c->dN_dy[0];
    
    ierr = PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);CHKERRQ(ierr);
    
    ElementEvaluateDerivatives_CellWiseConstant2d(1,nbasis,elcoords,
                                                  c->npe_1d,&dN_dxi_q,&dN_deta_q,&dN_dx_q,&dN_dy_q);
    ElementEvaluateGeometry_CellWiseConstant2d(nbasis,elcoords,c->npe_1d,&detJ);
    
#if 0
    /* compute moment contribution @ source */
    for (i=0; i<nbasis; i++) {
      fe[2*i  ] += c->w[i]*(moment_k[0]*dN_dx_q[i] + moment_k[1]*dN_dy_q[i])*4.0;///(dx*dy);
      fe[2*i+1] += c->w[i]*(moment_k[2]*dN_dx_q[i] + moment_k[3]*dN_dy_q[i])*4.0;///(dx*dy);
                                                                                 //printf("fe[%d] %+1.4e %+1.4e\n",i,fe[2*i],fe[2*i+1]);
    }
#endif
    
    for (q=0; q<c->nqp; q++) {
      
      /* compute moment contribution @ source */
      for (i=0; i<nbasis; i++) {
        fe[2*i  ] += c->w[q] * detJ * (moment_k[0]*dN_dx_q[i] + moment_k[1]*dN_dy_q[i]) / (dx * dy);
        fe[2*i+1] += c->w[q] * detJ * (moment_k[2]*dN_dx_q[i] + moment_k[3]*dN_dy_q[i]) / (dx * dy);
      }
    }
    ierr = VecSetValues(F,nbasis*ndof,eldofs,fe,ADD_VALUES);CHKERRQ(ierr);
    
    for (k=0; k<1; k++) {
      ierr = PetscFree(dN_dxi[k]);CHKERRQ(ierr);
      ierr = PetscFree(dN_deta[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dN_dxi);CHKERRQ(ierr);
    ierr = PetscFree(dN_deta);CHKERRQ(ierr);
    
  }
  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);
  
  ierr = PetscFree(eowner_source);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ElastoDynamicsSetSourceImplementation(SpecFECtx c,PetscInt ii)
{
  PetscInt itype = 0;
  
  
  itype = ii;
  
  switch (itype) {
    case 0:
    PetscPrintf(PETSC_COMM_WORLD,"  [source implementation]: pointwise evaluation\n");
    break;
    case 1:
    PetscPrintf(PETSC_COMM_WORLD,"  [source implementation]: closest internal basis\n");
    break;
    case 2:
    PetscPrintf(PETSC_COMM_WORLD,"  [source implementation]: cspline kernel\n");
    break;
    case 3:
    PetscPrintf(PETSC_COMM_WORLD,"  [source implementation]: P0\n");
    break;
    
    default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Valid choices are 0,1,2");
    break;
  }
  c->source_implementation = itype;
  
  PetscFunctionReturn(0);
}

PetscErrorCode ElastoDynamicsSourceSetup(SpecFECtx ctx,PetscReal source_coor[],PetscReal moment[],Vec g)
{
  PetscErrorCode ierr;
  PetscInt itype = -1;
  PetscBool found = PETSC_FALSE;
  
  ierr = PetscOptionsGetInt(NULL,NULL,"-source_impl",&itype,&found);CHKERRQ(ierr);
  if (found) {
    PetscPrintf(PETSC_COMM_WORLD,"  [source implementation]: processed from command line args\n");
    ierr = ElastoDynamicsSetSourceImplementation(ctx,itype);CHKERRQ(ierr);
  }
  
  switch (ctx->source_implementation) {
    case -1:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Source implementation not yet set. Valid choices are 0,1,2");
    break;
    case 0:
    ierr = AssembleLinearForm_ElastoDynamicsMomentDirac2d(ctx,1,source_coor,moment,g);CHKERRQ(ierr);
    break;
    case 1:
    ierr = AssembleLinearForm_ElastoDynamicsMomentDirac2d_NearestInternalQP(ctx,1,source_coor,moment,g);CHKERRQ(ierr);
    break;
    case 2:
    ierr = AssembleLinearForm_ElastoDynamicsMomentDirac2d_Kernel_CSpline(ctx,1,source_coor,moment,g);CHKERRQ(ierr);
    break;
    case 3:
    ierr = AssembleLinearForm_ElastoDynamicsMomentDirac2d_P0(ctx,1,source_coor,moment,g);CHKERRQ(ierr);
    break;
    
    default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Valid choices are 0,1,2");
    break;
  }
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
  QPntIsotropicElastic *qpdata;
  PetscReal gmin[3],gmax[3],min_el_r,dx,dy;
  PetscErrorCode ierr;
  QPntIsotropicElastic *celldata;
  
  *_dt = PETSC_MAX_REAL;
  dt_min = PETSC_MAX_REAL;
  
  order = ctx->basisorder;
  polynomial_fac = 1.0 / (2.0 * (PetscReal)order + 1.0);
  
  ierr = DMDAGetBoundingBox(ctx->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)ctx->mx_g);
  dy = (gmax[1] - gmin[1])/((PetscReal)ctx->my_g);
  
  min_el_r = dx;
  min_el_r = PetscMin(min_el_r,dy);
  
  
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
    
    value = polynomial_fac * 2.0 * min_el_r / max_el_Vp;
    
    dt_min = PetscMin(dt_min,value);
  }
  ierr = MPI_Allreduce(&dt_min,&dt_min_g,1,MPIU_SCALAR,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
  
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
  PetscInt ni,nj,ei,ej,n,nid,eid,*element,*elbasis;
  static char filename[PETSC_MAX_PATH_LEN];
  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Needs updating to support MPI");
  if (!beenhere) {
    switch (c->source_implementation) {
      case -1:
      sprintf(filename,"defaultsource-receiverCP-%dx%d-p%d.dat",c->mx_g,c->my_g,c->basisorder);
      break;
      case 0:
      sprintf(filename,"deltasource-receiverCP-%dx%d-p%d.dat",c->mx_g,c->my_g,c->basisorder);
      break;
      case 1:
      sprintf(filename,"closestqpsource-receiverCP-%dx%d-p%d.dat",c->mx_g,c->my_g,c->basisorder);
      break;
      case 2:
      sprintf(filename,"csplinesource-receiverCP-%dx%d-p%d.dat",c->mx_g,c->my_g,c->basisorder);
      break;
      case 3:
      sprintf(filename,"p0source-receiverCP-%dx%d-p%d.dat",c->mx_g,c->my_g,c->basisorder);
      break;
      default:
      break;
    }
  }
  
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
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
    fprintf(fp,"#   source implementation %d\n",c->source_implementation);
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
  PetscInt k,ei,ej,nid,eid,*element,*elbasis;
  static PetscReal N[400];
  static char filename[PETSC_MAX_PATH_LEN];
  
  if (c->size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Needs updating to support MPI");
  if (!beenhere) {
    switch (c->source_implementation) {
      case -1:
      sprintf(filename,"defaultsource-receiver-%dx%d-p%d.dat",c->mx_g,c->my_g,c->basisorder);
      break;
      case 0:
      sprintf(filename,"deltasource-receiver-%dx%d-p%d.dat",c->mx_g,c->my_g,c->basisorder);
      break;
      case 1:
      sprintf(filename,"closestqpsource-receiver-%dx%d-p%d.dat",c->mx_g,c->my_g,c->basisorder);
      break;
      case 2:
      sprintf(filename,"csplinesource-receiver-%dx%d-p%d.dat",c->mx_g,c->my_g,c->basisorder);
      break;
      case 3:
      sprintf(filename,"p0source-receiver-%dx%d-p%d.dat",c->mx_g,c->my_g,c->basisorder);
      break;
      default:
      break;
    }
  }
  
  if (!beenhere) {
    fp = fopen(filename,"w");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file \"%s\"",filename);
    fprintf(fp,"# SpecFECtx meta data\n");
    fprintf(fp,"#   mx %d : my %d : basis order %d\n",c->mx_g,c->my_g,c->basisorder);
    fprintf(fp,"#   source implementation %d\n",c->source_implementation);
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
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
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
    
    
    printf("# receiver location: x,y %+1.8e %+1.8e -- interpolated coordinate --> %+1.8e %+1.8e\n",xr[0],xr[1],xri[0],xri[1]);
    
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
    PetscInt ni,nj,ei,ej,n,nid,eid,*element,*elbasis;
    
    sprintf(filename,"closestqpsource-receiverCP-uva-%dx%d-p%d.dat",c->mx_g,c->my_g,c->basisorder);
    ierr = PetscMalloc1(nr,&nid_list);CHKERRQ(ierr);
    
    ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
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
  const PetscReal  *LA_u,*LA_v,*LA_a;
  static PetscBool beenhere = PETSC_FALSE;
  static char      filename[PETSC_MAX_PATH_LEN];
  static PetscInt  *nid_list = NULL;
  static PetscInt  nr_local = 0;
  PetscInt         r;
  Vec              lu,lv,la;
  PetscErrorCode   ierr;
  
  
  if (!beenhere) {
    const PetscReal *LA_c;
    Vec             coor;
    PetscReal       gmin[3],gmax[3],gmin_domain[3],gmax_domain[3],dx,dy,sep2min,sep2;
    PetscInt        ni,nj,ei,ej,n,nid,eid,*element,*elbasis;
    
    sprintf(filename,"closestqpsource-receiverCP-uva-%dx%d-p%d-rank%d.dat",c->mx_g,c->my_g,c->basisorder,c->rank);
    ierr = PetscMalloc1(nr,&nid_list);CHKERRQ(ierr);
    for (r=0; r<nr; r++) {
      nid_list[r] = -1;
    }
    
    ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
    ierr = SpecFECtxGetLocalBoundingBox(c,gmin_domain,gmax_domain);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(c->dm,&coor);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coor,&LA_c);CHKERRQ(ierr);
    
    for (r=0; r<nr; r++) {
      
      if (xr[2*r+0] < gmin[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, x-coordinate (%+1.4e) < min(domain).x (%+1.4e)",r,xr[2*r+0],gmin[0]);
      if (xr[2*r+1] < gmin[1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, y-coordinate (%+1.4e) < min(domain).y (%+1.4e)",r,xr[2*r+1],gmin[1]);
      if (xr[2*r+0] > gmax[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, x-coordinate (%+1.4e) > max(domain).x (%+1.4e)",r,xr[2*r+0],gmax[0]);
      if (xr[2*r+1] > gmax[1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Receiver %D, y-coordinate (%+1.4e) > max(domain).y (%+1.4e)",r,xr[2*r+1],gmax[1]);
      
      if (xr[2*r+0] < gmin_domain[0]) continue;
      if (xr[2*r+1] < gmin_domain[1]) continue;
      if (xr[2*r+0] > gmax_domain[0]) continue;
      if (xr[2*r+1] > gmax_domain[1]) continue;
      
      dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
      ei = (xr[2*r+0] - gmin_domain[0])/dx;
      if (ei == c->mx) ei--;
      
      dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
      ej = (xr[2*r+1] - gmin_domain[1])/dy;
      if (ej == c->my) ej--;
      
      if (ei < 0) continue;
      if (ej < 0) continue;
      
      if (ei > c->mx) continue;
      if (ej > c->my) continue;
      
      
      eid = ei + ej * c->mx;
      
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
      nr_local++;
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

        offset = 1 + count*6; /* 1 is for time */
        
        fprintf(fp,"#     ux(%d) uy(%d) vx(%d) vy(%d) ax(%d) ay(%d) -> station [%d]\n",offset+1,offset+2,offset+3,offset+4,offset+5,offset+6,r);
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
    
    PetscSNPrintf(metafname,PETSC_MAX_PATH_LEN-1,"closestqpsource-receiverCP-uva-%dx%d-p%d.mpimeta",c->mx_g,c->my_g,c->basisorder,c->size);
    
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
      if (nid_list[r] == -1) { continue; }
      fprintf(fp," %+1.8e %+1.8e %+1.8e %+1.8e %+1.8e %+1.8e",LA_u[2*nid_list[r]],LA_u[2*nid_list[r]+1],LA_v[2*nid_list[r]],LA_v[2*nid_list[r]+1],LA_a[2*nid_list[r]],LA_a[2*nid_list[r]+1]);
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

  ierr = DMRestoreLocalVector(c->dm,&la);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(c->dm,&lv);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(c->dm,&lu);CHKERRQ(ierr);

  
  PetscFunctionReturn(0);
}

PetscErrorCode RecordUVA_MultipleStations_NearestGLL(SpecFECtx c,PetscReal time,PetscInt nr,PetscReal xr[],Vec u,Vec v,Vec a)
{
  PetscErrorCode ierr;
  
  if (c->size == 1) {
    ierr = RecordUVA_MultipleStations_NearestGLL_SEQ(c,time,nr,xr,u,v,a);CHKERRQ(ierr);
  } else {
    ierr = RecordUVA_MultipleStations_NearestGLL_MPI(c,time,nr,xr,u,v,a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSourceCreate(SpecFECtx c,SeismicSourceType type,SeismicSourceImplementationType itype,PetscReal coor[],PetscReal values[],SeismicSource *s)
{
  SeismicSource src;
  PetscReal gmin[2],gmax[2],gmin_domain[2],gmax_domain[2],cell_min[2];
  PetscReal dx,dy,xi_source[2];
  PetscBool source_found;
  PetscInt eowner_source;
  PetscMPIInt size,rank,rank_owner;
  PetscInt ii,jj;
  int source_count,count,r,r_min_g;
  PetscErrorCode ierr;
  
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  /* locate cell containing source */
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx_g);
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my_g);
  
  if (coor[0] < gmin[0]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Source x-coordinate (%+1.4e) < min(domain).x (%+1.4e)",coor[0],gmin[0]);
  if (coor[1] < gmin[1]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Source y-coordinate (%+1.4e) < min(domain).y (%+1.4e)",coor[1],gmin[1]);
  if (coor[0] > gmax[0]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Source x-coordinate (%+1.4e) > max(domain).x (%+1.4e)",coor[0],gmax[0]);
  if (coor[1] > gmax[1]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Source y-coordinate (%+1.4e) > max(domain).y (%+1.4e)",coor[1],gmax[1]);

  ierr = SpecFECtxGetLocalBoundingBox(c,gmin_domain,gmax_domain);CHKERRQ(ierr);
  /*PetscPrintf(PETSC_COMM_SELF,"rank %d : x[%+1.4e,%+1.4e] - y[%+1.4e,%+1.4e]\n",c->rank,gmin_domain[0],gmax_domain[0],gmin_domain[1],gmax_domain[1]);*/
  
  source_found = PETSC_TRUE;
  eowner_source = -1;
  ii = -1;
  jj = -1;
  
  if (coor[0] < gmin_domain[0]) source_found = PETSC_FALSE;
  if (coor[1] < gmin_domain[1]) source_found = PETSC_FALSE;
  if (coor[0] > gmax_domain[0]) source_found = PETSC_FALSE;
  if (coor[1] > gmax_domain[1]) source_found = PETSC_FALSE;
  
  if (source_found != PETSC_FALSE) { /* skip if already determined point is outside of the sub-domain */
    
    ii = (PetscInt)( ( coor[0] - gmin_domain[0] )/dx );
    jj = (PetscInt)( ( coor[1] - gmin_domain[1] )/dy );
    /*printf("  rank %d ii jj %d %d \n",c->rank,ii,jj);*/
    
    if (ii == c->mx) ii--;
    if (jj == c->my) jj--;
    
    if (ii < 0) source_found = PETSC_FALSE;
    if (jj < 0) source_found = PETSC_FALSE;
    
    if (ii >= c->mx) source_found = PETSC_FALSE;
    if (jj >= c->my) source_found = PETSC_FALSE;
    
    if (source_found) {
      eowner_source = ii + jj * c->mx;
    }
  }
  if (source_found) {
    PetscPrintf(PETSC_COMM_SELF,"[SeismicSource] source (%+1.4e,%+1.4e) --> element %d | rank %d\n",coor[0],coor[1],eowner_source,rank);
  }
  
  /* check for duplicates */
  count = 0;
  if (source_found) {
    count = 1;
  }
  ierr = MPI_Allreduce(&count,&source_count,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);

  if (source_count == 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"A source was defined but no rank claimed it");
  
  if (source_count > 1) {
    /* resolve duplicates */

    r = (int)rank;
    if (!source_found) {
      r = (int)c->size;
    }
    ierr = MPI_Allreduce(&r,&r_min_g,1,MPI_INT,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
    if ( r == r_min_g) {
      PetscPrintf(PETSC_COMM_SELF,"[SeismicSource]  + Multiple ranks located source (%+1.4e,%+1.4e) - rank %d claiming ownership\n",coor[0],coor[1],r_min_g);
      PetscPrintf(PETSC_COMM_SELF,"[SeismicSource]  + Multiple ranks located source **** WARNING: This may produce results which slightly differ from a sequential run. ****\n");
      PetscPrintf(PETSC_COMM_SELF,"[SeismicSource]  + Multiple ranks located source **** WARNING: To remove solution variations, suggest you slightly shift the source location. ****\n");
    }
    
    /* mark non-owning ranks as not claiming source */
    if (r != r_min_g) {
      source_found = PETSC_FALSE;
    }
  }
  
  if (!source_found) {
    *s = NULL;
    PetscFunctionReturn(0);
  }
  
  /* get local coordinates */
  cell_min[0] = gmin_domain[0] + ii * dx;
  cell_min[1] = gmin_domain[1] + jj * dy;
  
  xi_source[0] = 2.0 * (coor[0] - cell_min[0])/dx - 1.0;
  xi_source[1] = 2.0 * (coor[1] - cell_min[1])/dy - 1.0;
  
  if (PetscAbsReal(xi_source[0]) < 1.0e-12) xi_source[0] = 0.0;
  if (PetscAbsReal(xi_source[1]) < 1.0e-12) xi_source[1] = 0.0;
  
  ierr = PetscMalloc1(1,&src);CHKERRQ(ierr);
  ierr = PetscMemzero(src,sizeof(struct _p_SeismicSource));CHKERRQ(ierr);
  
  src->issetup = PETSC_FALSE;
  src->sfem = c;
  src->rank = rank;
  src->element_index = eowner_source;
  src->coor[0] = coor[0];
  src->coor[1] = coor[1];
  src->type = type;
  src->implementation_type = itype;
  if (itype == SOURCE_IMPL_DEFAULT) {
    src->implementation_type = SOURCE_IMPL_NEAREST_QPOINT;
  }
  src->xi[0] = xi_source[0];
  src->xi[1] = xi_source[1];
  
  /* locate closest quadrature point */
  src->closest_gll = -1;
  if (itype == SOURCE_IMPL_NEAREST_QPOINT) {
    PetscReal sep2,sep2_min = PETSC_MAX_REAL;
    PetscInt  e,i,min_qp = -1,ni,nj,_ni,_nj,closest_qp,nid;
    PetscReal *elcoords;
    const PetscInt *element,*elnidx;
    PetscInt nbasis;
    Vec coordinates;
    const PetscReal *LA_coor;
    
    ierr = DMGetCoordinatesLocal(c->dm,&coordinates);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates,&LA_coor);CHKERRQ(ierr);
    
    elcoords = c->elbuf_coor;
    nbasis   = c->npe;
    element  = (const PetscInt*)c->element;
    
    e = eowner_source;
    
    /* get element -> node map */
    elnidx = (const PetscInt*)&element[nbasis*e];
    
    /* get element coordinates */
    for (i=0; i<nbasis; i++) {
      PetscInt nidx = elnidx[i];
      elcoords[2*i  ] = LA_coor[2*nidx  ];
      elcoords[2*i+1] = LA_coor[2*nidx+1];
    }
    
    for (nj=1; nj<c->npe_1d-1; nj++) {
      for (ni=1; ni<c->npe_1d-1; ni++) {
        nid = ni + nj * c->npe_1d;
        
        sep2 = (elcoords[2*nid]-coor[0])*(elcoords[2*nid]-coor[0]) + (elcoords[2*nid+1]-coor[1])*(elcoords[2*nid+1]-coor[1]);
        if (sep2 < sep2_min) {
          sep2_min = sep2;
          min_qp = nid;
          _ni = ni;
          _nj = nj;
        }
      }
    }
    ierr = VecRestoreArrayRead(coordinates,&LA_coor);CHKERRQ(ierr);
    closest_qp = min_qp;
    src->closest_gll = closest_qp;
    PetscPrintf(PETSC_COMM_SELF,"[SeismicSource] source --> closest_gll %d xi: (%+1.4e,%+1.4e) [%d,%d] \n",closest_qp,c->xi1d[_ni],c->xi1d[_nj],_ni,_nj);
    PetscPrintf(PETSC_COMM_SELF,"[SeismicSource] source --> closest_gll %d x:  (%+1.4e,%+1.4e)\n",closest_qp,elcoords[2*min_qp],elcoords[2*min_qp+1]);
  }
  
  
  switch (type) {
    case SOURCE_TYPE_FORCE:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Source type \"force\" not supported");
    ierr = PetscMalloc1(2,&src->values);CHKERRQ(ierr);
    ierr = PetscMemzero(src->values,sizeof(PetscReal)*2);CHKERRQ(ierr);
    ierr = PetscMemcpy(src->values,values,sizeof(PetscReal)*2);CHKERRQ(ierr);
    break;
    
    case SOURCE_TYPE_MOMENT:
    ierr = PetscMalloc1(4,&src->values);CHKERRQ(ierr);
    ierr = PetscMemzero(src->values,sizeof(PetscReal)*4);CHKERRQ(ierr);
    ierr = PetscMemcpy(src->values,values,sizeof(PetscReal)*4);CHKERRQ(ierr);
    break;
    
    default:
    break;
  }
  
  
  switch (src->implementation_type) {
    
    case SOURCE_IMPL_POINTWISE:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Source implementation type \"pointwise\" not supported");
    if (type == SOURCE_TYPE_FORCE) {
    } else if (type == SOURCE_TYPE_MOMENT) {
      src->setup = SeismicSourceSetup_Moment_Pointwise;
    }
    src->add_values = SeismicSourceAddValues_Pointwise;
    src->destroy    = SeismicSourceDestroy_Pointwise;
    break;
    
    case SOURCE_IMPL_NEAREST_QPOINT:
    if (type == SOURCE_TYPE_FORCE) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Source implementation type \"pointwise\" for type \"force\" not supported");
    } else if (type == SOURCE_TYPE_MOMENT) {
      src->setup = SeismicSourceSetup_Moment_Pointwise;
    }
    src->add_values = SeismicSourceAddValues_Pointwise;
    src->destroy    = SeismicSourceDestroy_Pointwise;
    break;
    
    case SOURCE_IMPL_SPLINE:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Source implementation type \"spline\" not supported");
    break;
    
    case SOURCE_IMPL_DEFAULT:
    break;
    
    default:
    break;
  }
  
  *s = src;
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSourceSetup(SeismicSource src)
{
  PetscErrorCode ierr;
  
  if (!src) PetscFunctionReturn(0);
  
  if (src->issetup) PetscFunctionReturn(0);
  if (src->setup) {
    ierr = src->setup(src);CHKERRQ(ierr);
    src->issetup = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSourceDestroy(SeismicSource *src)
{
  PetscErrorCode ierr;
  SeismicSource s;
  
  if (!src) PetscFunctionReturn(0);
  s = *src;
  if (!s) PetscFunctionReturn(0);
  ierr = PetscFree(s->values);CHKERRQ(ierr);
  if (s->destroy) {
    ierr = s->destroy(s);CHKERRQ(ierr);
  }
  ierr = PetscFree(s);CHKERRQ(ierr);
  *src = NULL;
  
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSourceAddValues_Pointwise(SeismicSource s,PetscReal stf,Vec f)
{
  PetscErrorCode ierr;
  PointwiseContext *pw = (PointwiseContext*)s->data;
  PetscInt k;
  
  for (k=0; k<2*pw->nbasis; k++) {
    pw->buffer[k] = stf * pw->element_values[k];
  }
  /*
  for (k=0; k<pw->nbasis; k++) {
    printf("buffer[%4d] %+1.12e %+1.12e\n",k,pw->buffer[2*k],pw->buffer[2*k+1]);
  }
  */
  ierr = VecSetValuesLocal(f,pw->nbasis*2,pw->element_indices,pw->buffer,ADD_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSourceDestroy_Pointwise(SeismicSource s)
{
  PointwiseContext *ctx;
  PetscErrorCode ierr;
  
  ctx = (PointwiseContext*)s->data;
  ierr = PetscFree(ctx->element_indices);CHKERRQ(ierr);
  ierr = PetscFree(ctx->element_values);CHKERRQ(ierr);
  ierr = PetscFree(ctx->buffer);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  s->data = NULL;
  
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSourceSetup_Pointwise(SeismicSource s, PointwiseContext **pw)
{
  PointwiseContext *ctx;
  PetscInt i;
  Vec coordinates;
  const PetscReal *LA_coor;
  PetscInt e;
  const PetscInt *elnidx;
  PetscErrorCode ierr;
  
  if (s->issetup) {
    *pw = (PointwiseContext*)s->data;
    PetscFunctionReturn(0);
  }
  
  ierr = PetscMalloc1(1,&ctx);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx,sizeof(PointwiseContext));CHKERRQ(ierr);
  s->data = (void*)ctx;
  
  if (s->implementation_type == SOURCE_IMPL_NEAREST_QPOINT) {
    PetscInt qi,qj;
    
    qj = s->closest_gll/s->sfem->npe_1d;
    qi = s->closest_gll - qj * s->sfem->npe_1d;
    
    ctx->xi[0] = s->sfem->xi1d[qi];
    ctx->xi[1] = s->sfem->xi1d[qj];
  } else if (s->implementation_type == SOURCE_IMPL_POINTWISE) {
    ctx->xi[0] = s->xi[0];
    ctx->xi[1] = s->xi[1];
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only valid for SOURCE_IMPL_POINTWISE or SOURCE_IMPL_NEAREST_QPOINT");
  
  ctx->nbasis = s->sfem->npe;
  ierr = PetscMalloc1(ctx->nbasis*2,&ctx->element_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->nbasis*2,&ctx->element_values);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->nbasis*2,&ctx->buffer);CHKERRQ(ierr);
  
  ierr = PetscMemzero(ctx->element_indices,sizeof(PetscInt)*ctx->nbasis*2);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx->element_values,sizeof(PetscReal)*ctx->nbasis*2);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx->buffer,sizeof(PetscReal)*ctx->nbasis*2);CHKERRQ(ierr);
  
  e = s->element_index;
  
  /* get element -> node map */
  elnidx = (const PetscInt*)&s->sfem->element[ctx->nbasis*e];
  
  /* generate dofs */
  for (i=0; i<ctx->nbasis; i++) {
    ctx->element_indices[2*i  ] = 2*elnidx[i];
    ctx->element_indices[2*i+1] = 2*elnidx[i]+1;
    //printf("src %d  %d\n",ctx->element_indices[2*i  ],ctx->element_indices[2*i+1]);
  }
  
  *pw = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSourceSetup_Moment_Pointwise(SeismicSource s)
{
  PointwiseContext *ctx;
  PetscInt i;
  PetscReal *moment_k;
  PetscReal **dN_dxi,**dN_deta;
  PetscReal *dN_dxi_q,*dN_deta_q;
  PetscReal *dN_dx_q,*dN_dy_q;
  Vec coordinates;
  const PetscReal *LA_coor;
  PetscReal *elcoords;
  PetscErrorCode ierr;
  
  ierr = SeismicSourceSetup_Pointwise(s,&ctx);CHKERRQ(ierr);
  
  /* get element coordinates */
  ierr = DMGetCoordinatesLocal(s->sfem->dm,&coordinates);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates,&LA_coor);CHKERRQ(ierr);
  elcoords = s->sfem->elbuf_coor;
  
  for (i=0; i<2*ctx->nbasis; i++) {
    PetscInt nidx = ctx->element_indices[i];
    
    elcoords[i] = LA_coor[nidx];
  }
  ierr = VecRestoreArrayRead(coordinates,&LA_coor);CHKERRQ(ierr);
  
  ierr = TabulateBasisDerivativesAtPointTensorProduct2d(ctx->xi,s->sfem->basisorder,&dN_dxi,&dN_deta);CHKERRQ(ierr);
  
  moment_k  = s->values;
  dN_dxi_q  = dN_dxi[0];
  dN_deta_q = dN_deta[0];
  
  /*
   for (i=0; i<ctx->nbasis; i++) {
   printf("GN_xi %d : %+1.12e\n",i,dN_dxi_q[i]);
   }
   for (i=0; i<ctx->nbasis; i++) {
   printf("GN_eta %d : %+1.12e\n",i,dN_deta_q[i]);
   }
   */
  
  dN_dx_q   = s->sfem->dN_dx[0];
  dN_dy_q   = s->sfem->dN_dy[0];
  
  ElementEvaluateDerivatives_CellWiseConstant2d(1,ctx->nbasis,elcoords,
                                                s->sfem->npe_1d,&dN_dxi_q,&dN_deta_q,&dN_dx_q,&dN_dy_q);
  
  /* compute moment contribution @ source */
  for (i=0; i<ctx->nbasis; i++) {
    ctx->element_values[2*i  ] = (moment_k[0]*dN_dx_q[i] + moment_k[1]*dN_dy_q[i]);
    ctx->element_values[2*i+1] = (moment_k[2]*dN_dx_q[i] + moment_k[3]*dN_dy_q[i]);
  }
  
  for (i=0; i<1; i++) {
    ierr = PetscFree(dN_dxi[i]);CHKERRQ(ierr);
    ierr = PetscFree(dN_deta[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dN_dxi);CHKERRQ(ierr);
  ierr = PetscFree(dN_deta);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSTFCreate(const char name[],SeismicSTF *s)
{
  PetscErrorCode ierr;
  SeismicSTF stf;
  
  ierr = PetscMalloc1(1,&stf);CHKERRQ(ierr);
  ierr = PetscSNPrintf(stf->name,PETSC_MAX_PATH_LEN-1,"%s",name);CHKERRQ(ierr);
  stf->data = NULL;
  stf->evaluate = NULL;
  stf->destroy = NULL;
  *s = stf;
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSTFEvaluate(SeismicSTF stf,PetscReal time,PetscReal *value)
{
  PetscErrorCode ierr;
  if (!stf->evaluate) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"A non-NULL source-time function evaluator must be provided");
  ierr = stf->evaluate(stf,time,value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSTFDestroy(SeismicSTF *s)
{
  PetscErrorCode ierr;
  SeismicSTF stf;
  
  if (!s) PetscFunctionReturn(0);
  stf = *s;
  if (!stf) PetscFunctionReturn(0);
  if (stf->destroy) {
    ierr = stf->destroy(stf);CHKERRQ(ierr);
  }
  stf->data = NULL;
  ierr = PetscFree(stf);CHKERRQ(ierr);
  *s = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSourceEvaluate(PetscReal time,PetscInt nsources,SeismicSource s[],SeismicSTF stf[],Vec f)
{
  PetscErrorCode ierr;
  PetscInt p;
  PetscReal value;
  
  ierr = VecZeroEntries(f);CHKERRQ(ierr);
  for (p=0; p<nsources; p++) {
    if (!s[p]) continue;
    
    if (!s[p]->issetup) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Seismic source not setup. Must call SeismicSourceSetup()");
    
    if (!stf) {
      if (s[p]) {
        value = 1.0;
        ierr = s[p]->add_values(s[p],value,f);CHKERRQ(ierr);
      }
    } else {
      if (stf[p] && s[p]) {
        ierr = stf[p]->evaluate(stf[p],time,&value);CHKERRQ(ierr);
        ierr = s[p]->add_values(s[p],value,f);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Ricker wavelet source-time function implementation */
typedef struct {
  PetscReal t0,freq,amp;
} SeismicSTF_Ricker;

PetscErrorCode SeismicSTFEvaluate_Ricker(SeismicSTF stf,PetscReal time,PetscReal *psi)
{
  SeismicSTF_Ricker *ctx = (SeismicSTF_Ricker*)stf->data;
  PetscReal arg,arg2,a,b;
  
  arg = PETSC_PI * ctx->freq * (time-ctx->t0);
  arg2 = arg * arg;
  a = 1.0 - 2.0 * arg2;
  b = PetscExpReal(-arg2);
  *psi = ctx->amp * a * b;
  
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSTFDestroy_Ricker(SeismicSTF stf)
{
  PetscErrorCode ierr;
  SeismicSTF_Ricker *ctx = (SeismicSTF_Ricker*)stf->data;
  
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  stf->data = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSTFCreate_Ricker(PetscReal t0,PetscReal freq,PetscReal amp,SeismicSTF *s)
{
  PetscErrorCode ierr;
  SeismicSTF stf;
  SeismicSTF_Ricker *ricker;
  
  ierr = SeismicSTFCreate("ricker",&stf);CHKERRQ(ierr);
  ierr = PetscMalloc1(1,&ricker);CHKERRQ(ierr);
  ricker->t0 = t0;
  ricker->freq = freq;
  ricker->amp = amp;
  stf->data = (void*)ricker;
  stf->evaluate = SeismicSTFEvaluate_Ricker;
  stf->destroy = SeismicSTFDestroy_Ricker;
  
  *s = stf;
  PetscFunctionReturn(0);
}

/* Gaussian source-time function implementation */
typedef struct {
  PetscReal t0,omega,amp;
} SeismicSTF_Gaussian;

PetscErrorCode SeismicSTFEvaluate_Gaussian(SeismicSTF stf,PetscReal time,PetscReal *psi)
{
  SeismicSTF_Gaussian *ctx = (SeismicSTF_Gaussian*)stf->data;
  PetscReal arg;
  
  arg = ctx->omega * ctx->omega * (time - ctx->t0) * (time - ctx->t0) * 0.5;
  *psi = ctx->amp * PetscExpReal( - arg );
  
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSTFDestroy_Gaussian(SeismicSTF stf)
{
  PetscErrorCode ierr;
  SeismicSTF_Gaussian *ctx = (SeismicSTF_Gaussian*)stf->data;
  
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  stf->data = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode SeismicSTFCreate_Gaussian(PetscReal t0,PetscReal omega,PetscReal amp,SeismicSTF *s)
{
  PetscErrorCode ierr;
  SeismicSTF stf;
  SeismicSTF_Gaussian *gaussian;
  
  ierr = SeismicSTFCreate("gaussian",&stf);CHKERRQ(ierr);
  ierr = PetscMalloc1(1,&gaussian);CHKERRQ(ierr);
  gaussian->t0    = t0;
  gaussian->omega = omega;
  gaussian->amp   = amp;
  stf->data = (void*)gaussian;
  stf->evaluate = SeismicSTFEvaluate_Gaussian;
  stf->destroy = SeismicSTFDestroy_Gaussian;
  
  *s = stf;
  PetscFunctionReturn(0);
}

PetscErrorCode specfem(PetscInt mx,PetscInt my)
{
  PetscErrorCode ierr;
  SpecFECtx ctx;
  PetscInt p,k,nt,of;
  PetscViewer viewer;
  Vec u,v,a,f,g,Md;
  PetscReal time,dt,stf,time_max;
  PetscReal stf_exp_T;
  
  ierr = SpecFECtxCreate(&ctx);CHKERRQ(ierr);
  p = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-border",&p,NULL);CHKERRQ(ierr);
  ierr = SpecFECtxCreateMesh(ctx,2,mx,my,PETSC_DECIDE,p,2);CHKERRQ(ierr);
  
  {
    PetscReal scale[] = { 30.0e3, 17.0e3 };
    PetscReal shift[] = { -15.0e3, -17.0e3 };
    
    ierr = SpecFECtxScaleMeshCoords(ctx,scale,shift);CHKERRQ(ierr);
  }
  
  ierr = SpecFECtxSetConstantMaterialProperties_Velocity(ctx,4000.0,2000.0,2600.0);CHKERRQ(ierr); // vp,vs,rho
  
  ierr = DMDASetFieldName(ctx->dm,0,"_x");CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx->dm,1,"_y");CHKERRQ(ierr);
  
  DMCreateGlobalVector(ctx->dm,&u); PetscObjectSetName((PetscObject)u,"disp");
  DMCreateGlobalVector(ctx->dm,&v); PetscObjectSetName((PetscObject)v,"velo");
  DMCreateGlobalVector(ctx->dm,&a); PetscObjectSetName((PetscObject)a,"accl");
  DMCreateGlobalVector(ctx->dm,&f);
  DMCreateGlobalVector(ctx->dm,&g);
  DMCreateGlobalVector(ctx->dm,&Md);
  
  ierr = VecZeroEntries(u);CHKERRQ(ierr);
  
  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"uva.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  /* Test basis and basis derivs */
  {
    PetscReal *xi,**GNx;
    PetscInt nbasis,npoints;
    
    //ierr = CreateGLLCoordsWeights(p,&npoints,&xi,NULL);CHKERRQ(ierr);
    //ierr = TabulateBasis1d_CLEGENDRE(p+1,xi,p);CHKERRQ(ierr);
    //ierr = TabulateBasisDerivatives1d_CLEGENDRE(npoints,xi,p,&nbasis,&GNx);CHKERRQ(ierr);
  }
  
  ierr = AssembleBilinearForm_Mass2d(ctx,Md);CHKERRQ(ierr);
  
  {
    PetscReal moment[] = { 0.0, 1.0e8, 1.0e8, 0.0 };
    PetscReal source_coor[] = { 0.0, -2.0e3 };
    
    ierr = AssembleLinearForm_ElastoDynamicsMomentDirac2d_NearestInternalQP(ctx,1,source_coor,moment,g);CHKERRQ(ierr);
  }
  
  
  
  k = 0;
  time = 0.0;
  
  time_max = 1.0;
  ierr = PetscOptionsGetReal(NULL,NULL,"-tmax",&time_max,NULL);CHKERRQ(ierr);
  
  ierr = ElastoDynamicsComputeTimeStep_2d(ctx,&dt);CHKERRQ(ierr);
  dt = dt * 0.3;
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);
  
  nt = 1000;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);CHKERRQ(ierr);
  
  of = 50;
  ierr = PetscOptionsGetInt(NULL,NULL,"-of",&of,NULL);CHKERRQ(ierr);
  
  stf_exp_T = 0.1;
  ierr = PetscOptionsGetReal(NULL,NULL,"-stf_exp_T",&stf_exp_T,NULL);CHKERRQ(ierr);
  
  
  /* Perform time stepping */
  for (k=1; k<=nt; k++) {
    PetscReal nrm,max,min;
    
    time = time + dt;
    
    PetscPrintf(PETSC_COMM_WORLD,"[step %9D] time = %1.4e : dt = %1.4e \n",k,time,dt);
    
    ierr = VecAXPY(u,dt,v);CHKERRQ(ierr); /* u_{n+1} = u_{n} + dt.v_{n} */
    
    ierr = VecAXPY(u,0.5*dt*dt,a);CHKERRQ(ierr); /* u_{n+1} = u_{n+1} + 0.5.dt^2.a_{n} */
    
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v' = v_{n} + 0.5.dt.a_{n} */
    
    //
    /* Evaluate source time function, S(t_{n+1}) */
    stf = 1.0;
    {
      PetscReal arg;
      
      // moment-time history
      arg = time / stf_exp_T;
      stf = 1.0 - (1.0 + arg) * PetscExpReal(-arg);
    }
    //
    
    /* Compute f = -F^{int}( u_{n+1} ) */
    ierr = AssembleLinearForm_ElastoDynamics2d(ctx,u,f);CHKERRQ(ierr);
    
    /* Update force; F^{ext}_{n+1} = f + S(t_{n+1}) g(x) */
    ierr = VecAXPY(f,stf,g);CHKERRQ(ierr);
    
    /* "Solve"; a_{n+1} = M^{-1} f */
    ierr = VecPointwiseDivide(a,f,Md);CHKERRQ(ierr);
    
    /* Update velocity */
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v_{n+1} = v' + 0.5.dt.a_{n+1} */
    
    VecNorm(u,NORM_2,&nrm);
    VecMin(u,0,&min);
    VecMax(u,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [displacement] max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
    VecNorm(v,NORM_2,&nrm);
    VecMin(v,0,&min);
    VecMax(v,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [velocity]     max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
    
    {
      PetscReal xr[] = { 0.0, 0.0 };
      ierr = RecordUV(ctx,time,xr,u,v);CHKERRQ(ierr);
    }
    
    if (k%of == 0) {
      char name[PETSC_MAX_PATH_LEN];
      
      PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"step-%.4d.vts",k);
      ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,name,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(u,viewer);CHKERRQ(ierr);
      ierr = VecView(v,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    
    if (time >= time_max) {
      break;
    }
  }
  
  /* plot last snapshot */
  {
    char name[PETSC_MAX_PATH_LEN];
    
    PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"step-%.4d.vts",k);
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,name,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(u,viewer);CHKERRQ(ierr);
    ierr = VecView(v,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&a);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = VecDestroy(&Md);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  
  
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateRickerWavelet(PetscReal time,PetscReal t0,PetscReal freq,PetscReal amp,PetscReal *psi)
{
  PetscReal arg,arg2,a,b;
  arg = M_PI * freq * (time-t0);
  arg2 = arg * arg;
  a = 1.0 - 2.0 * arg2;
  b = PetscExpReal(-arg2);
  *psi = amp * a * b;
  PetscFunctionReturn(0);
}

PetscErrorCode specfem_ex2(PetscInt mx,PetscInt my)
{
  PetscErrorCode ierr;
  SpecFECtx ctx;
  PetscInt p,k,nt,of;
  PetscViewer viewer;
  Vec u,v,a,f,g,Md;
  PetscReal time,dt,stf,time_max;
  PetscReal stf_exp_T;
  
  ierr = SpecFECtxCreate(&ctx);CHKERRQ(ierr);
  p = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-border",&p,NULL);CHKERRQ(ierr);
  ierr = SpecFECtxCreateMesh(ctx,2,mx,my,PETSC_DECIDE,p,2);CHKERRQ(ierr);
  
  {
    PetscReal scale[] = { 4.0e3, 2.0e3 };
    PetscReal shift[] = { 0.0, -2.0e3 };
    
    ierr = SpecFECtxScaleMeshCoords(ctx,scale,shift);CHKERRQ(ierr);
  }
  
  ierr = SpecFECtxSetConstantMaterialProperties_Velocity(ctx,3200.0,1847.5,2000.0);CHKERRQ(ierr); // vp,vs,rho
  
  ierr = DMDASetFieldName(ctx->dm,0,"_x");CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx->dm,1,"_y");CHKERRQ(ierr);
  
  DMCreateGlobalVector(ctx->dm,&u); PetscObjectSetName((PetscObject)u,"disp");
  DMCreateGlobalVector(ctx->dm,&v); PetscObjectSetName((PetscObject)v,"velo");
  DMCreateGlobalVector(ctx->dm,&a); PetscObjectSetName((PetscObject)a,"accl");
  DMCreateGlobalVector(ctx->dm,&f); PetscObjectSetName((PetscObject)f,"f");
  DMCreateGlobalVector(ctx->dm,&g); PetscObjectSetName((PetscObject)g,"g");
  DMCreateGlobalVector(ctx->dm,&Md);
  
  ierr = VecZeroEntries(u);CHKERRQ(ierr);
  
  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"uva.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  ierr = AssembleBilinearForm_Mass2d(ctx,Md);CHKERRQ(ierr);
  
  {
    PetscReal moment[] = { 0.0, 1.0e1, 1.0e1, 0.0 };
    PetscReal source_coor[] = { 1500.0, -200.0 };
    
    //ierr = AssembleLinearForm_ElastoDynamicsMomentDirac2d_NearestInternalQP(ctx,1,source_coor,moment,g);CHKERRQ(ierr);
    //ierr = AssembleLinearForm_ElastoDynamicsMomentDirac2d(ctx,1,source_coor,moment,g);CHKERRQ(ierr);
    ierr = AssembleLinearForm_ElastoDynamicsMomentDirac2d_Kernel_CSpline(ctx,1,source_coor,moment,g);CHKERRQ(ierr);
  }
  
  //ierr = VecView(g,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"f.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(g,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  k = 0;
  time = 0.0;
  
  time_max = 4.0;
  ierr = PetscOptionsGetReal(NULL,NULL,"-tmax",&time_max,NULL);CHKERRQ(ierr);
  
  ierr = ElastoDynamicsComputeTimeStep_2d(ctx,&dt);CHKERRQ(ierr);
  dt = dt * 0.2;
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);
  
  nt = 10;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);CHKERRQ(ierr);
  
  of = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-of",&of,NULL);CHKERRQ(ierr);
  
  stf_exp_T = 0.1;
  ierr = PetscOptionsGetReal(NULL,NULL,"-stf_exp_T",&stf_exp_T,NULL);CHKERRQ(ierr);
  
  /* Perform time stepping */
  for (k=1; k<=nt; k++) {
    PetscReal nrm,max,min;
    
    time = time + dt;
    
    PetscPrintf(PETSC_COMM_WORLD,"[step %9D] time = %1.4e : dt = %1.4e \n",k,time,dt);
    
    ierr = VecAXPY(u,dt,v);CHKERRQ(ierr); /* u_{n+1} = u_{n} + dt.v_{n} */
    
    ierr = VecAXPY(u,0.5*dt*dt,a);CHKERRQ(ierr); /* u_{n+1} = u_{n+1} + 0.5.dt^2.a_{n} */
    
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v' = v_{n} + 0.5.dt.a_{n} */
    
    /* Evaluate source time function, S(t_{n+1}) */
    stf = 1.0;
    {
      PetscReal arg;
      
      // moment-time history
      ierr = EvaluateRickerWavelet(time,0.15,10.0,1.0,&stf);CHKERRQ(ierr);
      
      // moment-time history
      arg = time / stf_exp_T;
      stf = 1.0 - (1.0 + arg) * PetscExpReal(-arg);
    }
    //stf = time;
    printf("STF(%1.4e) = %+1.4e\n",time,stf);
    
    /* Compute f = -F^{int}( u_{n+1} ) */
    ierr = AssembleLinearForm_ElastoDynamics2d(ctx,u,f);CHKERRQ(ierr);
    
    /* Update force; F^{ext}_{n+1} = f + S(t_{n+1}) g(x) */
    ierr = VecAXPY(f,stf,g);CHKERRQ(ierr);
    
    /* "Solve"; a_{n+1} = M^{-1} f */
    ierr = VecPointwiseDivide(a,f,Md);CHKERRQ(ierr);
    
    /* Update velocity */
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v_{n+1} = v' + 0.5.dt.a_{n+1} */
    
    VecNorm(u,NORM_2,&nrm);
    VecMin(u,0,&min);
    VecMax(u,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [displacement] max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
    VecNorm(v,NORM_2,&nrm);
    VecMin(v,0,&min);
    VecMax(v,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [velocity]     max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
    
    {
      PetscReal xr[] = { 2200.0, 0.0 };
      ierr = RecordUV(ctx,time,xr,u,v);CHKERRQ(ierr);
    }
    
    if (k%of == 0) {
      char name[PETSC_MAX_PATH_LEN];
      
      PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"step-%.4d.vts",k);
      ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,name,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(u,viewer);CHKERRQ(ierr);
      ierr = VecView(v,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    
    if (time >= time_max) {
      break;
    }
  }
  
  /* plot last snapshot */
  {
    char name[PETSC_MAX_PATH_LEN];
    
    PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"step-%.4d.vts",k);
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,name,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(u,viewer);CHKERRQ(ierr);
    ierr = VecView(v,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&a);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = VecDestroy(&Md);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


PetscErrorCode specfem_gare6(PetscInt mx,PetscInt my)
{
  PetscErrorCode ierr;
  SpecFECtx ctx;
  PetscInt p,k,nt,of;
  PetscViewer viewer;
  Vec u,v,a,f,g,Md;
  PetscReal time,dt,stf,time_max;
  PetscReal stf_exp_T;
  
  ierr = SpecFECtxCreate(&ctx);CHKERRQ(ierr);
  p = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-border",&p,NULL);CHKERRQ(ierr);
  ierr = SpecFECtxCreateMesh(ctx,2,mx,my,PETSC_DECIDE,p,2);CHKERRQ(ierr);
  
  {
    PetscReal scale[] = { 4.0e3, 2.0e3 };
    PetscReal shift[] = { -2.0e3, 0.0 };
    
    ierr = SpecFECtxScaleMeshCoords(ctx,scale,shift);CHKERRQ(ierr);
  }
  
  ierr = SpecFECtxSetConstantMaterialProperties_Velocity(ctx,4746.3670317412243 ,2740.2554625435928, 1000.0);CHKERRQ(ierr); // vp,vs,rho
  
  ierr = DMDASetFieldName(ctx->dm,0,"_x");CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx->dm,1,"_y");CHKERRQ(ierr);
  
  DMCreateGlobalVector(ctx->dm,&u); PetscObjectSetName((PetscObject)u,"disp");
  DMCreateGlobalVector(ctx->dm,&v); PetscObjectSetName((PetscObject)v,"velo");
  DMCreateGlobalVector(ctx->dm,&a); PetscObjectSetName((PetscObject)a,"accl");
  DMCreateGlobalVector(ctx->dm,&f); PetscObjectSetName((PetscObject)f,"f");
  DMCreateGlobalVector(ctx->dm,&g); PetscObjectSetName((PetscObject)g,"g");
  DMCreateGlobalVector(ctx->dm,&Md);
  
  ierr = VecZeroEntries(u);CHKERRQ(ierr);
  
  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"uva.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  ierr = AssembleBilinearForm_Mass2d(ctx,Md);CHKERRQ(ierr);
  
  ierr = ElastoDynamicsSetSourceImplementation(ctx,0);CHKERRQ(ierr);
  {
    PetscReal moment[] = { 0.0, 0.0, 0.0, 0.0 };
    PetscReal source_coor[] = { 0.0, 500.0 };
    PetscReal M;
    
    M = 1000.0; /* gar6more input usings M/rho = 1 */
    moment[0] = moment[3] = M; /* p-source <explosive> */
    //moment[1] = moment[2] = M; /* s-source <double-couple> */
    
    ierr = ElastoDynamicsSourceSetup(ctx,source_coor,moment,g);CHKERRQ(ierr);
  }
  
  //ierr = VecView(g,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"f.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(g,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  k = 0;
  time = 0.0;
  
  time_max = 0.6;
  ierr = PetscOptionsGetReal(NULL,NULL,"-tmax",&time_max,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2wave] Requested time period: %1.4e\n",time_max);
  
  ierr = ElastoDynamicsComputeTimeStep_2d(ctx,&dt);CHKERRQ(ierr);
  dt = dt * 0.2;
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2wave] Using time step size: %1.4e\n",dt);
  
  nt = 1000000;
  nt = (PetscInt)(time_max / dt ) + 4;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2wave] Estimated number of time steps: %D\n",nt);
  
  of = 5000;
  ierr = PetscOptionsGetInt(NULL,NULL,"-of",&of,NULL);CHKERRQ(ierr);
  
  stf_exp_T = 0.1;
  ierr = PetscOptionsGetReal(NULL,NULL,"-stf_exp_T",&stf_exp_T,NULL);CHKERRQ(ierr);
  
  /* Perform time stepping */
  for (k=1; k<=nt; k++) {
    
    time = time + dt;
    
    ierr = VecAXPY(u,dt,v);CHKERRQ(ierr); /* u_{n+1} = u_{n} + dt.v_{n} */
    
    ierr = VecAXPY(u,0.5*dt*dt,a);CHKERRQ(ierr); /* u_{n+1} = u_{n+1} + 0.5.dt^2.a_{n} */
    
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v' = v_{n} + 0.5.dt.a_{n} */
    
    /* Evaluate source time function, S(t_{n+1}) */
    stf = 1.0;
    {
      PetscReal arg;
      
      // moment-time history
      //ierr = EvaluateRickerWavelet(time,0.08,14.5,1.0,&stf);CHKERRQ(ierr);
      ierr = EvaluateRickerWavelet(time,0.15,8.0,1.0,&stf);CHKERRQ(ierr);
      
      // moment-time history
      //arg = time / stf_exp_T;
      //stf = 1.0 - (1.0 + arg) * PetscExpReal(-arg);
    }
    //stf = time;
    
    /* Compute f = -F^{int}( u_{n+1} ) */
    ierr = AssembleLinearForm_ElastoDynamics2d(ctx,u,f);CHKERRQ(ierr);
    
    /* Update force; F^{ext}_{n+1} = f + S(t_{n+1}) g(x) */
    ierr = VecAXPY(f,stf,g);CHKERRQ(ierr);
    
    /* "Solve"; a_{n+1} = M^{-1} f */
    ierr = VecPointwiseDivide(a,f,Md);CHKERRQ(ierr);
    
    /* Update velocity */
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v_{n+1} = v' + 0.5.dt.a_{n+1} */
    
    if (k%100 == 0) {
      PetscReal nrm,max,min;
      
      PetscPrintf(PETSC_COMM_WORLD,"[step %9D] time = %1.4e : dt = %1.4e \n",k,time,dt);
      printf("  STF(%1.4e) = %+1.4e\n",time,stf);
      VecNorm(u,NORM_2,&nrm);
      VecMin(u,0,&min);
      VecMax(u,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [displacement] max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
      VecNorm(v,NORM_2,&nrm);
      VecMin(v,0,&min);
      VecMax(v,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [velocity]     max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
    }
    {
      //PetscReal xr[] = { 10.0, 10.0 };
      PetscReal xr[] = { 100.0, 600.0 };
      ierr = RecordUV(ctx,time,xr,u,v);CHKERRQ(ierr);
      ierr = RecordUV_interp(ctx,time,xr,u,v);CHKERRQ(ierr);
    }
    
    if (k%of == 0) {
      char name[PETSC_MAX_PATH_LEN];
      
      PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"step-%.4d.vts",k);
      ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,name,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(u,viewer);CHKERRQ(ierr);
      ierr = VecView(v,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    
    if (time >= time_max) {
      break;
    }
  }
  {
    PetscReal nrm,max,min;
    
    PetscPrintf(PETSC_COMM_WORLD,"[step %9D] time = %1.4e : dt = %1.4e \n",k,time,dt);
    printf("  STF(%1.4e) = %+1.4e\n",time,stf);
    VecNorm(u,NORM_2,&nrm);
    VecMin(u,0,&min);
    VecMax(u,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [displacement] max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
    VecNorm(v,NORM_2,&nrm);
    VecMin(v,0,&min);
    VecMax(v,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [velocity]     max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
  }
  
  /* plot last snapshot */
  {
    char name[PETSC_MAX_PATH_LEN];
    
    PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"step-%.4d.vts",k);
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,name,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(u,viewer);CHKERRQ(ierr);
    ierr = VecView(v,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&a);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = VecDestroy(&Md);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode specfem_gare6_ex2(PetscInt mx,PetscInt my)
{
  PetscErrorCode ierr;
  SpecFECtx ctx;
  PetscInt p,k,nt,of;
  PetscViewer viewer;
  Vec u,v,a,f,g,Md;
  PetscReal time,dt,time_max;
  PetscBool psource=PETSC_TRUE,ssource=PETSC_FALSE;
  SeismicSource src;
  SeismicSTF stf;
  
  ierr = SpecFECtxCreate(&ctx);CHKERRQ(ierr);
  p = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-border",&p,NULL);CHKERRQ(ierr);
  ierr = SpecFECtxCreateMesh(ctx,2,mx,my,PETSC_DECIDE,p,2);CHKERRQ(ierr);
  
  {
    PetscReal alpha = 2.0e3;
    
    //PetscReal scale[] = {  4.0e3, 4.0e3 };
    //PetscReal shift[] = { -2.0e3,-2.0e3 };
    
    PetscReal scale[] = {  alpha, alpha };
    PetscReal shift[] = { -alpha/2.0,-alpha/2.0 };
    
    ierr = SpecFECtxScaleMeshCoords(ctx,scale,shift);CHKERRQ(ierr);
  }
  
  ierr = SpecFECtxSetConstantMaterialProperties_Velocity(ctx,4746.3670317412243 ,2740.2554625435928, 1000.0);CHKERRQ(ierr); // vp,vs,rho
  
  ierr = DMDASetFieldName(ctx->dm,0,"_x");CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx->dm,1,"_y");CHKERRQ(ierr);
  
  DMCreateGlobalVector(ctx->dm,&u); PetscObjectSetName((PetscObject)u,"disp");
  DMCreateGlobalVector(ctx->dm,&v); PetscObjectSetName((PetscObject)v,"velo");
  DMCreateGlobalVector(ctx->dm,&a); PetscObjectSetName((PetscObject)a,"accl");
  DMCreateGlobalVector(ctx->dm,&f); PetscObjectSetName((PetscObject)f,"f");
  DMCreateGlobalVector(ctx->dm,&g); PetscObjectSetName((PetscObject)g,"g");
  DMCreateGlobalVector(ctx->dm,&Md);
  
  ierr = VecZeroEntries(u);CHKERRQ(ierr);
  
  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"uva.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  ierr = AssembleBilinearForm_Mass2d(ctx,Md);CHKERRQ(ierr);
  
  ierr = ElastoDynamicsSetSourceImplementation(ctx,1);CHKERRQ(ierr);
  
  /* define a single source */
  {
    PetscReal moment[] = { 0.0, 0.0, 0.0, 0.0 };
    PetscReal source_coor[] = { 0.0, 100.0 };
    PetscReal M;
    
    ierr = PetscOptionsGetBool(NULL,NULL,"-p_source",&psource,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-s_source",&ssource,NULL);CHKERRQ(ierr);
    
    M = 1000.0; /* gar6more input usings M/rho = 1 */
    if (psource) {
      moment[0] = moment[3] = M; /* p-source <explosive> */
    }
    if (ssource) {
      moment[1] = moment[2] = M; /* s-source <double-couple> */
      moment[1] = -M;
      moment[2] = M;
    }
    PetscPrintf(PETSC_COMM_WORLD,"Moment: [ %+1.2e , %+1.2e ; %+1.2e , %+1.2e ]\n",moment[0],moment[1],moment[2],moment[3]);

    ierr = SeismicSourceCreate(ctx,SOURCE_TYPE_MOMENT,SOURCE_IMPL_NEAREST_QPOINT,source_coor,moment,&src);CHKERRQ(ierr);
    ierr = SeismicSourceSetup(src);CHKERRQ(ierr);
    
    /* dummy evaluation just for testing */
    ierr = SeismicSourceEvaluate(0.0,1,&src,NULL,g);CHKERRQ(ierr);
  }
  
  //ierr = VecView(g,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"f.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(g,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  /* define a single source time function */
  ierr = SeismicSTFCreate_Ricker(0.15,12.0,1.0,&stf);CHKERRQ(ierr);
  
  k = 0;
  time = 0.0;
  
  time_max = 0.4;
  ierr = PetscOptionsGetReal(NULL,NULL,"-tmax",&time_max,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2wave] Requested time period: %1.4e\n",time_max);
  
  ierr = ElastoDynamicsComputeTimeStep_2d(ctx,&dt);CHKERRQ(ierr);
  dt = dt * 0.2;
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2wave] Using time step size: %1.4e\n",dt);
  
  nt = 1000000;
  nt = (PetscInt)(time_max / dt ) + 4;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2wave] Estimated number of time steps: %D\n",nt);
  
  of = 5000;
  ierr = PetscOptionsGetInt(NULL,NULL,"-of",&of,NULL);CHKERRQ(ierr);
  
  /* Perform time stepping */
  for (k=1; k<=nt; k++) {
    
    time = time + dt;
    
    ierr = VecAXPY(u,dt,v);CHKERRQ(ierr); /* u_{n+1} = u_{n} + dt.v_{n} */
    
    ierr = VecAXPY(u,0.5*dt*dt,a);CHKERRQ(ierr); /* u_{n+1} = u_{n+1} + 0.5.dt^2.a_{n} */
    
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v' = v_{n} + 0.5.dt.a_{n} */
    
    /* Evaluate source time function, S(t_{n+1}) */
    ierr = SeismicSourceEvaluate(time,1,&src,&stf,g);CHKERRQ(ierr);
    
    /* Compute f = -F^{int}( u_{n+1} ) */
    ierr = AssembleLinearForm_ElastoDynamics2d(ctx,u,f);CHKERRQ(ierr);
    
    /* Update force; F^{ext}_{n+1} = f + S(t_{n+1}) g(x) */
    ierr = VecAXPY(f,1.0,g);CHKERRQ(ierr);
    
    /* "Solve"; a_{n+1} = M^{-1} f */
    ierr = VecPointwiseDivide(a,f,Md);CHKERRQ(ierr);
    
    /* Update velocity */
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v_{n+1} = v' + 0.5.dt.a_{n+1} */
    
    if (k%100 == 0) {
      PetscReal nrm,max,min;
      
      PetscPrintf(PETSC_COMM_WORLD,"[step %9D] time = %1.4e : dt = %1.4e \n",k,time,dt);
      VecNorm(u,NORM_2,&nrm);
      VecMin(u,0,&min);
      VecMax(u,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [displacement] max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
      VecNorm(v,NORM_2,&nrm);
      VecMin(v,0,&min);
      VecMax(v,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [velocity]     max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
    }
    {
      PetscReal xr[] = { 100.0, 200.0 };
      PetscReal xr_list[] = { 100.0, 200.0, 150.0, 200.0 };
      
      ierr = RecordUV(ctx,time,xr,u,v);CHKERRQ(ierr);
      ierr = RecordUV_interp(ctx,time,xr,u,v);CHKERRQ(ierr);

      ierr = RecordUVA_MultipleStations_NearestGLL(ctx,time,2,xr_list,u,v,a);CHKERRQ(ierr);
    }
    
    if (k%of == 0) {
      char name[PETSC_MAX_PATH_LEN];
      
      PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"step-%.4d.vts",k);
      ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,name,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(u,viewer);CHKERRQ(ierr);
      ierr = VecView(v,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    
    if (time >= time_max) {
      break;
    }
  }
  {
    PetscReal nrm,max,min;
    
    PetscPrintf(PETSC_COMM_WORLD,"[step %9D] time = %1.4e : dt = %1.4e \n",k,time,dt);
    VecNorm(u,NORM_2,&nrm);
    VecMin(u,0,&min);
    VecMax(u,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [displacement] max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
    VecNorm(v,NORM_2,&nrm);
    VecMin(v,0,&min);
    VecMax(v,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [velocity]     max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
  }
  
  /* plot last snapshot */
  {
    char name[PETSC_MAX_PATH_LEN];
    
    PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"step-%.4d.vts",k);
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,name,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(u,viewer);CHKERRQ(ierr);
    ierr = VecView(v,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  ierr = SeismicSourceDestroy(&src);CHKERRQ(ierr);
  ierr = SeismicSTFDestroy(&stf);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&a);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = VecDestroy(&Md);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode se2wave_demo(PetscInt mx,PetscInt my)
{
  PetscErrorCode ierr;
  SpecFECtx ctx;
  PetscInt p,k,nt,of;
  PetscViewer viewer;
  Vec u,v,a,f,g,Md;
  PetscReal time,dt,time_max;
  PetscBool psource=PETSC_TRUE,ssource=PETSC_FALSE;
  PetscInt s,nsources;
  SeismicSource *src;
  SeismicSTF *stf;
  PetscInt nrecv;
  PetscReal *xr_list;
  
  
  /*
    Create the structured mesh for the spectral element method.
   The default mesh is defined over the domain [0,1]^2.
  */
  ierr = SpecFECtxCreate(&ctx);CHKERRQ(ierr);
  p = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-border",&p,NULL);CHKERRQ(ierr);
  ierr = SpecFECtxCreateMesh(ctx,2,mx,my,PETSC_DECIDE,p,2);CHKERRQ(ierr);

  /*
   Define your domain by shifting and scaling the default [0,1]^2 domain
  */
  {
    PetscReal alpha = 2.0e3;
    
    //PetscReal scale[] = {  4.0e3, 4.0e3 };
    //PetscReal shift[] = { -2.0e3,-2.0e3 };
    
    PetscReal scale[] = {  alpha, alpha };
    PetscReal shift[] = { -alpha/2.0,-alpha/2.0 };
    
    ierr = SpecFECtxScaleMeshCoords(ctx,scale,shift);CHKERRQ(ierr);
  }

  /*
   Specify the material properties for the domain.
   This function sets constant material properties in every cell.
   More general methods can be easily added.
  */
  ierr = SpecFECtxSetConstantMaterialProperties_Velocity(ctx,4746.3670317412243 ,2740.2554625435928, 1000.0);CHKERRQ(ierr); // vp,vs,rho
  
  ierr = DMDASetFieldName(ctx->dm,0,"_x");CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx->dm,1,"_y");CHKERRQ(ierr);
  
  DMCreateGlobalVector(ctx->dm,&u); PetscObjectSetName((PetscObject)u,"disp");
  DMCreateGlobalVector(ctx->dm,&v); PetscObjectSetName((PetscObject)v,"velo");
  DMCreateGlobalVector(ctx->dm,&a); PetscObjectSetName((PetscObject)a,"accl");
  DMCreateGlobalVector(ctx->dm,&f); PetscObjectSetName((PetscObject)f,"f");
  DMCreateGlobalVector(ctx->dm,&g); PetscObjectSetName((PetscObject)g,"g");
  DMCreateGlobalVector(ctx->dm,&Md);
  
  ierr = VecZeroEntries(u);CHKERRQ(ierr);
  
  /*
   Write out the mesh and intial values for the displacement, velocity and acceleration (u,v,a)
  */
  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"uva.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  ierr = AssembleBilinearForm_Mass2d(ctx,Md);CHKERRQ(ierr);
  
  /* Define two sources */
  
  nsources = 2;
  ierr = PetscMalloc1(nsources,&src);CHKERRQ(ierr);
  for (s=0; s<nsources; s++) {
    src[s] = NULL;
  }
  
  /* configure source 1 */
  {
    PetscReal moment[] = { 0.0, 0.0, 0.0, 0.0 }; /* initialize moment tensor to 0 */
    PetscReal source_coor[] = { 0.1, 100.1 }; /* define source location */
    PetscReal M;
    
    M = 1000.0; /* gar6more input usings M/rho = 1 */
    //if (psource) {
      moment[0] = moment[3] = M; /* p-source <explosive> */
    //}
    PetscPrintf(PETSC_COMM_WORLD,"Moment: [ %+1.2e , %+1.2e ; %+1.2e , %+1.2e ]\n",moment[0],moment[1],moment[2],moment[3]);
    
    ierr = SeismicSourceCreate(ctx,SOURCE_TYPE_MOMENT,SOURCE_IMPL_NEAREST_QPOINT,source_coor,moment,&src[0]);CHKERRQ(ierr); /* note the index of src[] here */
  }

  /* configure source 2 */
  {
    PetscReal moment[] = { 0.0, 0.0, 0.0, 0.0 }; /* initialize moment tensor to 0 */
    PetscReal source_coor[] = { 400.1, 400.1 }; /* define source location */
    PetscReal M;
    
    M = 1000.0; /* gar6more input usings M/rho = 1 */
    //if (ssource) {
      moment[1] = moment[2] = M; /* s-source <double-couple> */
      moment[1] = -M;
      moment[2] = M;
    //}
    PetscPrintf(PETSC_COMM_WORLD,"Moment: [ %+1.2e , %+1.2e ; %+1.2e , %+1.2e ]\n",moment[0],moment[1],moment[2],moment[3]);
    
    ierr = SeismicSourceCreate(ctx,SOURCE_TYPE_MOMENT,SOURCE_IMPL_NEAREST_QPOINT,source_coor,moment,&src[1]);CHKERRQ(ierr); /* note the index of src[] here */
  }
  
  /* setup sources */
  for (s=0; s<nsources; s++) {
    ierr = SeismicSourceSetup(src[s]);CHKERRQ(ierr);
  }

  /*
   Write out the representation of the sources using STF() = 1.0 for all sources.
   Requires a dummy evaluation.
  */
  ierr = SeismicSourceEvaluate(0.0,nsources,src,NULL,g);CHKERRQ(ierr);

  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"f.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(g,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  /* 
   Define two source time functions
  */
  ierr = PetscMalloc1(nsources,&stf);CHKERRQ(ierr);
  for (s=0; s<nsources; s++) {
    stf[s] = NULL;
  }
  
  /* configure stf for source 1 */
  if (src[0]) { ierr = SeismicSTFCreate_Ricker(0.15,12.0,1.0,&stf[0]);CHKERRQ(ierr); /* note the index of stf[] here */ }

  /* configure stf for source 2 */
  if (src[1]) { ierr = SeismicSTFCreate_Ricker(0.25,4.0,1.0,&stf[1]);CHKERRQ(ierr); /* note the index of stf[] here */ }

  /*
   Define the location of the receivers
  */
  nrecv = 3;
  ierr = PetscMalloc1(nrecv*2,&xr_list);CHKERRQ(ierr);
  xr_list[0] = 100.0; /* x-coordinate for receiver 1 */
  xr_list[1] = 200.0; /* y-coordinate for receiver 1 */

  xr_list[2] = 150.0;
  xr_list[3] = 200.0;

  xr_list[4] = 180.0;
  xr_list[5] = 200.0;


  /* Initialize time loop */
  k = 0;
  time = 0.0;
  
  time_max = 0.4;
  ierr = PetscOptionsGetReal(NULL,NULL,"-tmax",&time_max,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2wave] Requested time period: %1.4e\n",time_max);
  
  ierr = ElastoDynamicsComputeTimeStep_2d(ctx,&dt);CHKERRQ(ierr);
  dt = dt * 0.2;
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2wave] Using time step size: %1.4e\n",dt);
  
  nt = 1000000;
  nt = (PetscInt)(time_max / dt ) + 4;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[se2wave] Estimated number of time steps: %D\n",nt);
  
  of = 5000;
  ierr = PetscOptionsGetInt(NULL,NULL,"-of",&of,NULL);CHKERRQ(ierr);
  
  /* Perform time stepping */
  for (k=1; k<=nt; k++) {
    
    time = time + dt;
    
    ierr = VecAXPY(u,dt,v);CHKERRQ(ierr); /* u_{n+1} = u_{n} + dt.v_{n} */
    
    ierr = VecAXPY(u,0.5*dt*dt,a);CHKERRQ(ierr); /* u_{n+1} = u_{n+1} + 0.5.dt^2.a_{n} */
    
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v' = v_{n} + 0.5.dt.a_{n} */
    
    /* Evaluate source time function, S(t_{n+1}) */
    ierr = SeismicSourceEvaluate(time,nsources,src,stf,g);CHKERRQ(ierr);
    
    /* Compute f = -F^{int}( u_{n+1} ) */
    ierr = AssembleLinearForm_ElastoDynamics2d(ctx,u,f);CHKERRQ(ierr);
    
    /* Update force; F^{ext}_{n+1} = f + S(t_{n+1}) g(x) */
    ierr = VecAXPY(f,1.0,g);CHKERRQ(ierr);
    
    /* "Solve"; a_{n+1} = M^{-1} f */
    ierr = VecPointwiseDivide(a,f,Md);CHKERRQ(ierr);
    
    /* Update velocity */
    ierr = VecAXPY(v,0.5*dt,a);CHKERRQ(ierr); /* v_{n+1} = v' + 0.5.dt.a_{n+1} */
    
    if (k%100 == 0) {
      PetscReal nrm,max,min;
      
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
    ierr = RecordUVA_MultipleStations_NearestGLL(ctx,time,nrecv,xr_list,u,v,a);CHKERRQ(ierr);
    
    if (k%of == 0) {
      char name[PETSC_MAX_PATH_LEN];
      
      PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"step-%.4d.vts",k);
      ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,name,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(u,viewer);CHKERRQ(ierr);
      ierr = VecView(v,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    
    if (time >= time_max) {
      break;
    }
  }
  {
    PetscReal nrm,max,min;
    
    PetscPrintf(PETSC_COMM_WORLD,"[step %9D] time = %1.4e : dt = %1.4e \n",k,time,dt);
    VecNorm(u,NORM_2,&nrm);
    VecMin(u,0,&min);
    VecMax(u,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [displacement] max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
    VecNorm(v,NORM_2,&nrm);
    VecMin(v,0,&min);
    VecMax(v,0,&max); PetscPrintf(PETSC_COMM_WORLD,"  [velocity]     max = %+1.4e : min = %+1.4e : l2 = %+1.4e \n",max,min,nrm);
  }
  
  /* plot last snapshot */
  {
    char name[PETSC_MAX_PATH_LEN];
    
    PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"step-%.4d.vts",k);
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,name,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(u,viewer);CHKERRQ(ierr);
    ierr = VecView(v,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  for (s=0; s<nsources; s++) {
    ierr = SeismicSourceDestroy(&src[s]);CHKERRQ(ierr);
    ierr = SeismicSTFDestroy(&stf[s]);CHKERRQ(ierr);
  }
  ierr = PetscFree(src);CHKERRQ(ierr);
  ierr = PetscFree(stf);CHKERRQ(ierr);
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
  
  //ierr = specfem(mx,my);CHKERRQ(ierr);
  //ierr = specfem_ex2(mx,my);CHKERRQ(ierr); // comparison with sem2dpack
  //ierr = specfem_gare6(mx,my);CHKERRQ(ierr); // comparison with gare6more
  //ierr = specfem_gare6_ex2(mx,my);CHKERRQ(ierr); // comparison with gare6more
  
  ierr = se2wave_demo(mx,my);CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return 0;
}
