
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
  PetscInt basisorder;
  PetscInt mx,my,mz,nx,ny,nz;
  //PetscReal dx,dy,dz;
  PetscInt dim;
  PetscInt dofs;
  DM dm;
  PetscInt npe,npe_1d,ne;
  PetscInt *element;
  PetscReal *xi1d,*w1d,*w;
  PetscReal *elbuf_coor,*elbuf_field,*elbuf_field2;
  PetscInt  *elbuf_dofs;
  PetscInt nqp;
  QPntIsotropicElastic *qp_data;
  PetscReal **dN_dxi,**dN_deta;
  PetscReal **dN_dx,**dN_dy;
  PetscInt  source_implementation;
};


/* N = polynomial order */
PetscErrorCode CreateGLLCoordsWeights(PetscInt N,PetscInt *_npoints,PetscReal **_xi,PetscReal **_w)
{
  PetscInt N1;
  PetscReal *xold,*x,*w,*P;
  PetscReal eps,res;
  PetscInt i,j,k;
  
  
  // Truncation + 1
  N1 = N + 1;

  PetscMalloc(sizeof(PetscReal)*N1,&xold);
  PetscMalloc(sizeof(PetscReal)*N1,&x);
  PetscMalloc(sizeof(PetscReal)*N1,&w);
  PetscMalloc(sizeof(PetscReal)*N1*N1,&P);
  
  // Use the Chebyshev-Gauss-Lobatto nodes as the first guess
  for (i=0; i<N1; i++) {
    x[i]=cos(M_PI*i/(PetscReal)N);
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
        P[i+(k+1)*N1] = ( (2.0*(k+1)-1.0)*x[i] * P[i+k*N1] - (k+1.0-1.0) * P[i+(k-1)*N1] ) / (double)(k+1.0);
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
    PetscFree(x);
  }

  if (_npoints) {
    *_npoints = N1;
  }
  if (_w) {
    *_w = w;
  } else {
    PetscFree(w);
  }
  PetscFree(xold);
  PetscFree(P);

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
  
  
  CreateGLLCoordsWeights(order,&nbasis,&xilocal,NULL);
  
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
    
    PetscOptionsGetBool(NULL,NULL,"-compute_vandermonde_condition",&compute_vandermonde_condition,0);
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
  
  
  CreateGLLCoordsWeights(order,&nbasis,&xilocal,NULL);
  
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
    
    PetscOptionsGetBool(NULL,NULL,"-compute_vandermonde_condition",&compute_vandermonde_condition,0);
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
  PetscFree(xiq);
  for (k=0; k<nqp; k++) {
    PetscFree(dphi_xi[k]);
  }
  PetscFree(dphi_xi);
  
  if (_dN_dxi) { *_dN_dxi = dN_dxi; }
  else {
    for (k=0; k<nqp*nqp; k++) {
      PetscFree(dN_dxi[k]);
    }
    PetscFree(dN_dxi);
  }

  if (_dN_deta) { *_dN_deta = dN_deta; }
  else {
    for (k=0; k<nqp*nqp; k++) {
      PetscFree(dN_deta[k]);
    }
    PetscFree(dN_deta);
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
    PetscFree(dphi_xi[k]);
    PetscFree(dphi_eta[k]);
  }
  PetscFree(dphi_xi);
  PetscFree(dphi_eta);
  
  for (k=0; k<nqp; k++) {
    PetscFree(Ni_xi[k]);
    PetscFree(Ni_eta[k]);
  }
  PetscFree(Ni_xi);
  PetscFree(Ni_eta);
  
  if (_dN_dxi) { *_dN_dxi = dN_dxi; }
  else {
    for (k=0; k<nqp*nqp; k++) {
      PetscFree(dN_dxi[k]);
    }
    PetscFree(dN_dxi);
  }
  
  if (_dN_deta) { *_dN_deta = dN_deta; }
  else {
    for (k=0; k<nqp*nqp; k++) {
      PetscFree(dN_deta[k]);
    }
    PetscFree(dN_deta);
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxCreate(SpecFECtx *c)
{
  SpecFECtx ctx;
  
  PetscMalloc(sizeof(struct _p_SpecFECtx),&ctx);
  PetscMemzero(ctx,sizeof(struct _p_SpecFECtx));
  ctx->source_implementation = -1;
  *c = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxCreateENMap2d(SpecFECtx c)
{
  PetscErrorCode ierr;
  PetscInt ni0,nj0,i,j,ei,ej,ecnt,*emap,nid;
  
  PetscMalloc(sizeof(PetscInt)*c->ne*c->npe,&c->element);
  PetscMemzero(c->element,sizeof(PetscInt)*c->ne*c->npe);

  ecnt = 0;
  for (ej=0; ej<c->my; ej++) {
    nj0 = ej*(c->npe_1d-1);

    for (ei=0; ei<c->mx; ei++) {
      ni0 = ei*(c->npe_1d-1);
    
      emap = &c->element[c->npe*ecnt];
      
      for (j=0; j<c->npe_1d; j++) {
        for (i=0; i<c->npe_1d; i++) {
        
          nid = (ni0 + i) + (nj0 + j) * c->nx;
          emap[i+j*c->npe_1d] = nid;
        }
      }
     
      ecnt++;
    }
  }
  
  PetscFunctionReturn(0);
}

/* Creates domain over [0,1]^d - scale later */
PetscErrorCode SpecFECtxCreateMeshCoords2d(SpecFECtx c)
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
  
  dx = 1.0/((PetscReal)c->mx);
  dy = 1.0/((PetscReal)c->my);
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
  Vec coor;
  
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
  
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxCreateMesh(SpecFECtx c,PetscInt dim,PetscInt mx,PetscInt my,PetscInt mz,PetscInt basisorder,PetscInt ndofs)
{
  PetscErrorCode ierr;
  PetscInt stencil_width,i,j;

  c->dim = dim;
  c->mx = mx;
  c->my = my;
  c->mz = mz;
  c->basisorder = basisorder;
  c->dofs = ndofs;
  
  c->nx = basisorder*mx + 1;
  c->ny = basisorder*my + 1;
  c->nz = basisorder*mz + 1;

  ierr = CreateGLLCoordsWeights(basisorder,&c->npe_1d,&c->xi1d,&c->w1d);CHKERRQ(ierr);
  
  stencil_width = 1;
  switch (dim) {
    case 2:
      c->npe = c->npe_1d * c->npe_1d;
      c->ne = mx * my;

      ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                          c->nx,c->ny,PETSC_DECIDE,PETSC_DECIDE,ndofs,stencil_width,NULL,NULL,&c->dm);CHKERRQ(ierr);
      ierr = DMSetUp(c->dm);CHKERRQ(ierr);
      ierr = SpecFECtxCreateENMap2d(c);CHKERRQ(ierr);
      ierr = SpecFECtxCreateMeshCoords2d(c);CHKERRQ(ierr);

      /* tensor product for weights */
      PetscMalloc(sizeof(PetscReal)*c->npe,&c->w);
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
  
  PetscMalloc(sizeof(QPntIsotropicElastic)*c->nqp*c->ne,&c->qp_data);
  PetscMemzero(c->qp_data,sizeof(QPntIsotropicElastic)*c->nqp*c->ne);
  
  PetscMalloc(sizeof(PetscReal)*c->npe*c->dim,&c->elbuf_coor);
  PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs,&c->elbuf_field);
  PetscMalloc(sizeof(PetscReal)*c->npe*c->dofs,&c->elbuf_field2);
  PetscMalloc(sizeof(PetscInt)*c->npe*c->dofs,&c->elbuf_dofs);
  
  PetscFunctionReturn(0);
}

PetscErrorCode SpecFECtxSetConstantMaterialProperties(SpecFECtx c,PetscReal lambda,PetscReal mu,PetscReal rho)
{
  PetscInt q;
  
  for (q=0; q<c->nqp*c->ne; q++) {
    c->qp_data[q].lambda = lambda;
    c->qp_data[q].mu     = mu;
    c->qp_data[q].rho    = rho;
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
  Vec       coor;
  const PetscReal *LA_coor,*LA_u;
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  nqp      = c->nqp;
  ndof     = c->dofs;
  fe       = c->elbuf_field;
  element  = c->element;
  field    = c->elbuf_field2;
  
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);

  ierr = VecGetArrayRead(u,&LA_u);CHKERRQ(ierr);
  
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
    
    PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);
    
    for (q=0; q<c->nqp; q++) {
      PetscReal            fac;
      PetscReal            c11,c12,c21,c22,c33,lambda_qp,mu_qp;
      QPntIsotropicElastic *qpdata;
      PetscReal            *dNidx,*dNidy;
      
      
      /* get access to element->quadrature points */
      qpdata = &c->qp_data[e*c->nqp];
      
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
      lambda_qp  = qpdata[q].lambda;
      mu_qp      = qpdata[q].mu;
      
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
    ierr = VecSetValues(F,nbasis*ndof,eldofs,fe,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(u,&LA_u);CHKERRQ(ierr);
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
  
  ierr = VecZeroEntries(A);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  ndof     = c->dofs;
  Me       = c->elbuf_field;
  element  = c->element;
  
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
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
    
    for (q=0; q<nbasis; q++) {
      PetscReal            fac,Me_ii;
      QPntIsotropicElastic *qpdata;
      
      /* get access to element->quadrature points */
      qpdata = &c->qp_data[e*c->nqp];
      
      fac = detJ * c->w[q];

      Me_ii = fac * (qpdata[q].rho);
      
      /* \int u0v0 dV */
      index = 2*q;
      Me[index] = Me_ii;
      
      /* \int u1v1 dV */
      index = 2*q + 1;
      Me[index] = Me_ii;
    }
    ierr = VecSetValues(A,nbasis*ndof,eldofs,Me,ADD_VALUES);CHKERRQ(ierr);
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
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  ndof     = c->dofs;
  fe       = c->elbuf_field;
  element  = c->element;
  
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  PetscMalloc(sizeof(PetscInt)*nsources,&eowner_source);
  PetscMalloc(sizeof(PetscInt)*nsources,&closest_qp);

  /* locate cell containing source */
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx);
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my);
  for (k=0; k<nsources; k++) {
    ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx );
    jj = (PetscInt)( ( xs[2*k+1] - gmin[1] )/dy );
    
    if (ii == c->mx) ii--;
    if (jj == c->my) jj--;
    
    if (ii < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: x < gmin[0]");
    if (jj < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: y < gmin[1]");
    
    if (ii > c->mx) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: x > gmax[0]");
    if (jj > c->my) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: y > gmax[1]");
    
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
    
    PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);
    
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

      ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx );
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
  
  PetscFree(eowner_source);
  PetscFree(closest_qp);
  
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
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  ndof     = c->dofs;
  fe       = c->elbuf_field;
  element  = c->element;
  
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  PetscMalloc(sizeof(PetscInt)*nsources,&eowner_source);
  
  /* locate cell containing source */
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx);
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my);
  for (k=0; k<nsources; k++) {
    ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx );
    jj = (PetscInt)( ( xs[2*k+1] - gmin[1] )/dy );
    
    if (ii == c->mx) ii--;
    if (jj == c->my) jj--;
    
    if (ii < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: x < gmin[0]");
    if (jj < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: y < gmin[1]");
    
    if (ii > c->mx) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: x > gmax[0]");
    if (jj > c->my) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: y > gmax[1]");
    
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
    ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx );
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
    
    PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);
    
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
      
      ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx );
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
      PetscFree(dN_dxi[k]);
      PetscFree(dN_deta[k]);
    }
    PetscFree(dN_dxi);
    PetscFree(dN_deta);
  
  }
  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);
  
  PetscFree(eowner_source);
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
  
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx);
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my);
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
  PetscOptionsGetReal(NULL,NULL,"-sm_h",&kernel_h,&flg);
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

      PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);
      
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
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  
  eldofs   = c->elbuf_dofs;
  elcoords = c->elbuf_coor;
  nbasis   = c->npe;
  ndof     = c->dofs;
  fe       = c->elbuf_field;
  element  = c->element;
  
  ierr = DMGetCoordinates(c->dm,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&LA_coor);CHKERRQ(ierr);
  
  PetscMalloc(sizeof(PetscInt)*nsources,&eowner_source);
  
  /* locate cell containing source */
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx);
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my);
  for (k=0; k<nsources; k++) {
    ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx );
    jj = (PetscInt)( ( xs[2*k+1] - gmin[1] )/dy );
    
    if (ii == c->mx) ii--;
    if (jj == c->my) jj--;
    
    if (ii < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: x < gmin[0]");
    if (jj < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: y < gmin[1]");
    
    if (ii > c->mx) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: x > gmax[0]");
    if (jj > c->my) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"source: y > gmax[1]");
    
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
    ii = (PetscInt)( ( xs[2*k+0] - gmin[0] )/dx );
    jj = (PetscInt)( ( xs[2*k+1] - gmin[1] )/dy );
    
    if (ii == c->mx) ii--;
    if (jj == c->my) jj--;
    
    cell_min[0] = gmin[0] + ii * dx;
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
    
    PetscMemzero(fe,sizeof(PetscReal)*nbasis*ndof);
    
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
      PetscFree(dN_dxi[k]);
      PetscFree(dN_deta[k]);
    }
    PetscFree(dN_dxi);
    PetscFree(dN_deta);
    
  }
  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);
  
  PetscFree(eowner_source);
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
  
  PetscOptionsGetInt(NULL,NULL,"-source_impl",&itype,&found);
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
  PetscReal dt_min,polynomial_fac;
  QPntIsotropicElastic *qpdata;
  PetscReal gmin[3],gmax[3],min_el_r,dx,dy;
  PetscErrorCode ierr;
  
  *_dt = PETSC_MAX_REAL;
  dt_min = PETSC_MAX_REAL;
  
  order = ctx->basisorder;
  polynomial_fac = 1.0 / (2.0 * (PetscReal)order + 1.0);
  
  ierr = DMDAGetBoundingBox(ctx->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)ctx->mx);
  dy = (gmax[1] - gmin[1])/((PetscReal)ctx->my);

  min_el_r = dx;
  min_el_r = PetscMin(min_el_r,dy);

  
  for (e=0; e<ctx->ne; e++) {
    PetscReal max_el_Vp,value;
    
    /* get max Vp for element */
    max_el_Vp = PETSC_MIN_REAL;
    
    /* get access to element->quadrature points */
    qpdata = &ctx->qp_data[e*ctx->nqp];
    
    for (q=0; q<ctx->nqp; q++) {
      PetscReal qp_rho,qp_mu,qp_lambda,qp_Vp;
      
      qp_rho    = qpdata[q].rho;
      qp_mu     = qpdata[q].mu;
      qp_lambda = qpdata[q].lambda;
      
      ierr = ElastoDynamicsConvertLame2Velocity(qp_rho,qp_mu,qp_lambda,0,&qp_Vp);CHKERRQ(ierr);
      
      max_el_Vp = PetscMax(max_el_Vp,qp_Vp);
    }
    
    value = polynomial_fac * 2.0 * min_el_r / max_el_Vp;
    
    dt_min = PetscMin(dt_min,value);
  }
  
  *_dt = dt_min;
  
  PetscFunctionReturn(0);
}

PetscErrorCode RecordUV(SpecFECtx c,PetscReal time,PetscReal xr[],Vec u,Vec v)
{
  FILE *fp;
  PetscReal gmin[3],gmax[3],dx,dy,sep2min,sep2;
  const PetscReal *LA_u,*LA_v,*LA_c;
  Vec coor;
  static PetscBool beenhere = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt ni,nj,ei,ej,n,nid,eid,*element,*elbasis;
  static char filename[PETSC_MAX_PATH_LEN];

  if (!beenhere) {
    switch (c->source_implementation) {
      case -1:
        sprintf(filename,"defaultsource-receiverCP-%dx%d-p%d.dat",c->mx,c->my,c->basisorder);
        break;
      case 0:
        sprintf(filename,"deltasource-receiverCP-%dx%d-p%d.dat",c->mx,c->my,c->basisorder);
        break;
      case 1:
        sprintf(filename,"closestqpsource-receiverCP-%dx%d-p%d.dat",c->mx,c->my,c->basisorder);
        break;
      case 2:
        sprintf(filename,"csplinesource-receiverCP-%dx%d-p%d.dat",c->mx,c->my,c->basisorder);
        break;
      case 3:
        sprintf(filename,"p0source-receiverCP-%dx%d-p%d.dat",c->mx,c->my,c->basisorder);
        break;
      default:
        break;
    }
  }
  
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx);
  ei = (xr[0] - gmin[0])/dx;

  dy = (gmax[1] - gmin[1])/((PetscReal)c->my);
  ej = (xr[1] - gmin[1])/dy;

  eid = ei + ej * c->mx;
  
  /* get element -> node map */
  element = c->element;
  elbasis = &element[c->npe*eid];
  
  DMGetCoordinates(c->dm,&coor);
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
    fprintf(fp,"# SpecFECtx meta data\n");
    fprintf(fp,"#   mx %d : my %d : basis order %d\n",c->mx,c->my,c->basisorder);
    fprintf(fp,"#   source implementation %d\n",c->source_implementation);
    fprintf(fp,"# Reciever meta data\n");
    fprintf(fp,"#   + receiver location: x,y %+1.8e %+1.8e\n",xr[0],xr[1]);
    fprintf(fp,"#   + takes displ/velo from basis nearest to requested reciever location\n");
    fprintf(fp,"#   + receiver location: x,y %+1.8e %+1.8e --mapped to nearest node --> %+1.8e %+1.8e\n",xr[0],xr[1],LA_c[2*nid],LA_c[2*nid+1]);
    fprintf(fp,"# Time series header\n");
    fprintf(fp,"#   time ux uy vx vy\n");
    beenhere = PETSC_TRUE;
  } else {
    fp = fopen(filename,"a");
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
  FILE *fp;
  PetscReal gmin[3],gmax[3],dx,dy,ur[2],vr[2];
  const PetscReal *LA_u,*LA_v;
  static PetscBool beenhere = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt k,ei,ej,nid,eid,*element,*elbasis;
  static PetscReal N[400];
  static char filename[PETSC_MAX_PATH_LEN];
  
  if (!beenhere) {
    switch (c->source_implementation) {
      case -1:
        sprintf(filename,"defaultsource-receiver-%dx%d-p%d.dat",c->mx,c->my,c->basisorder);
        break;
      case 0:
        sprintf(filename,"deltasource-receiver-%dx%d-p%d.dat",c->mx,c->my,c->basisorder);
        break;
      case 1:
        sprintf(filename,"closestqpsource-receiver-%dx%d-p%d.dat",c->mx,c->my,c->basisorder);
        break;
      case 2:
        sprintf(filename,"csplinesource-receiver-%dx%d-p%d.dat",c->mx,c->my,c->basisorder);
        break;
      case 3:
        sprintf(filename,"p0source-receiver-%dx%d-p%d.dat",c->mx,c->my,c->basisorder);
        break;
      default:
        break;
    }
  }
  
  if (!beenhere) {
    fp = fopen(filename,"w");
    fprintf(fp,"# SpecFECtx meta data\n");
    fprintf(fp,"#   mx %d : my %d : basis order %d\n",c->mx,c->my,c->basisorder);
    fprintf(fp,"#   source implementation %d\n",c->source_implementation);
    fprintf(fp,"# Reciever meta data\n");
    fprintf(fp,"#   + receiver location: x,y %+1.8e %+1.8e\n",xr[0],xr[1]);
    fprintf(fp,"#   + records displ/velo at requested reciever location through interpolating the FE solution\n");
    fprintf(fp,"# Time series header\n");
    fprintf(fp,"#   time ux uy vx vy\n");
  } else {
    fp = fopen(filename,"a");
  }

  /* get containing element */
  ierr = DMDAGetBoundingBox(c->dm,gmin,gmax);CHKERRQ(ierr);
  dx = (gmax[0] - gmin[0])/((PetscReal)c->mx);
  ei = (xr[0] - gmin[0])/dx;
  
  dy = (gmax[1] - gmin[1])/((PetscReal)c->my);
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
    
    x0 = gmin[0] + ei*dx;
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

/*
 INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN ENGINEERING 
 Int. J. Numer. Meth. Engng. 45, 1139–1164 (1999)
 THE SPECTRAL ELEMENT METHOD FOR ELASTIC WAVE EQUATIONS—APPLICATION TO 2-D AND 3-D SEISMIC PROBLEMS
 DIMITRI KOMATITSCH, JEAN-PIERRE VILOTTE, ROSSANA VAI, JOSE M.CASTILLO-COVARRUBIAS AND FRANCISCOJ.SANCHEZ-SESMA
 
 stf[0] S(t_{n})
 stf[1] S(t_{n+1})
*/
PetscErrorCode TSExplicitNewmark(Vec u,Vec v,Vec a,Vec Md,Vec f,PetscReal stf[],PetscReal dt,PetscReal alpha,PetscReal beta,PetscReal gamma)
{
  Vec vi,dv,f_alpha;
  
  VecDuplicate(u,&vi);
  VecDuplicate(u,&dv);
  VecDuplicate(u,&f_alpha);
  
  /* save old velocity */
  VecCopy(v,vi);
  
  /* rhs */
  VecCopy(f,f_alpha);
  VecScale(f_alpha,alpha*stf[1]);
  if (alpha < 1.0) {
    VecAXPY(f_alpha,(1-alpha)*stf[0],f);
  }

  /* solve */
  VecScale(f_alpha,dt);
  VecPointwiseDivide(dv,f_alpha,Md);
  
  /* update velocity */
  VecAXPY(v,1.0,dv); // v_{n+1} = dv + v_{n}

  /* update displacement */
  VecAXPY(u,dt*(1.0 - beta/gamma),vi);
  VecAXPY(u,dt*(beta/gamma),v);
  VecAXPY(u,dt*dt*(0.5 - beta/gamma),a);
  
  /* update acceleration */
  VecScale(a,1.0 - 1.0/gamma);
  VecAXPY(a,1.0/(gamma*dt),dv);
  
  VecDestroy(&vi);
  VecDestroy(&dv);
  VecDestroy(&f_alpha);
  
  PetscFunctionReturn(0);
}

#if 0
/*
 COMPUTER METHODS IN APPLIED MECHANICS AND ENGINEERING
 CMAME 115 (1994), 223-252
 How to render second order accurate time-stepping algorithms into fourth
 order accurate while retaining the stability and conservation properties
 N. TARNOW and J.C. SIMO
*/
PetscErrorCode TSExplicitNewmarkFourth(Vec u,Vec v,Vec a,Vec Md,Vec f,PetscReal stf[],PetscReal time,PetscReal dt,PetscReal alpha,PetscReal beta,PetscReal gamma)
{
  PetscReal theta;
  PetscErrorCode ierr;
  const PetscReal oo3 = 1.0/3.0;
  PetscReal theta = oo3 * ( 2.0 + pow(2.0,-oo3) + pow(2.0,oo3) );
  PetscReal dt_sub0,dt_sub1,dt_sub2,stf_sub[2];
  
  dt_sub0 = theta * dt;
  stf_sub[0] = stf[0]; /* t = t */
  stf_sub[1] = 0;      /* t = t + dt_sub0 */
  ierr = TSExplicitNewmark(u,v,a,Md,f,stf_sub,dt_sub0,alpha,beta,gamma);CHKERRQ(ierr);

  dt_sub1 = (1.0 - 2.0*theta) * dt;
  stf_sub[0] = stf[0]; /* t = t + dt_sub0 */
  stf_sub[1] = 0;      /* t = t + dt_sub0 + dt_sub1 */
  ierr = TSExplicitNewmark(u,v,a,Md,f,stf_sub,dt_sub1,alpha,beta,gamma);CHKERRQ(ierr);

  dt_sub2 = theta * dt;
  stf_sub[0] = stf[0]; /* t = t + dt_sub0 + dt_sub1 */
  stf_sub[1] = 0;      /* t = t + dt_sub0 + dt_sub1 + dt_sub2 */
  ierr = TSExplicitNewmark(u,v,a,Md,f,stf_sub,dt_sub2,alpha,beta,gamma);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
#endif

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
  PetscOptionsGetInt(NULL,NULL,"-border",&p,NULL);
  ierr = SpecFECtxCreateMesh(ctx,2,mx,my,PETSC_DECIDE,p,2);CHKERRQ(ierr);

  {
    PetscReal scale[] = { 30.0e3, 17.0e3 };
    PetscReal shift[] = { -15.0e3, -17.0e3 };
    
    ierr = SpecFECtxScaleMeshCoords(ctx,scale,shift);CHKERRQ(ierr);
  }
  
  ierr = SpecFECtxSetConstantMaterialProperties_Velocity(ctx,4000.0,2000.0,2600.0);CHKERRQ(ierr); // vp,vs,rho
  
  DMDASetFieldName(ctx->dm,0,"_x");
  DMDASetFieldName(ctx->dm,1,"_y");

  DMCreateGlobalVector(ctx->dm,&u); PetscObjectSetName((PetscObject)u,"disp");
  DMCreateGlobalVector(ctx->dm,&v); PetscObjectSetName((PetscObject)v,"velo");
  DMCreateGlobalVector(ctx->dm,&a); PetscObjectSetName((PetscObject)a,"accl");
  DMCreateGlobalVector(ctx->dm,&f);
  DMCreateGlobalVector(ctx->dm,&g);
  DMCreateGlobalVector(ctx->dm,&Md);

  VecZeroEntries(u);
  
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
  PetscOptionsGetReal(NULL,NULL,"-tmax",&time_max,NULL);

  ierr = ElastoDynamicsComputeTimeStep_2d(ctx,&dt);CHKERRQ(ierr);
  dt = dt * 0.3;
  PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);

  nt = 1000;
  PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);
  
  of = 50;
  PetscOptionsGetInt(NULL,NULL,"-of",&of,NULL);
  
  stf_exp_T = 0.1;
  PetscOptionsGetReal(NULL,NULL,"-stf_exp_T",&stf_exp_T,NULL);
  
  
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
      stf = 1.0 - (1.0 + arg) * exp(-arg);
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
  
  VecDestroy(&u);
  VecDestroy(&v);
  VecDestroy(&a);
  VecDestroy(&f);
  VecDestroy(&Md);
  VecDestroy(&g);

  
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateRickerWavelet(PetscReal time,PetscReal t0,PetscReal freq,PetscReal amp,PetscReal *psi)
{
  PetscReal arg,arg2,a,b;
  arg = M_PI * freq * (time-t0);
  arg2 = arg * arg;
  a = 1.0 - 2.0 * arg2;
  b = exp(-arg2);
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
  PetscOptionsGetInt(NULL,NULL,"-border",&p,NULL);
  ierr = SpecFECtxCreateMesh(ctx,2,mx,my,PETSC_DECIDE,p,2);CHKERRQ(ierr);
  
  {
    PetscReal scale[] = { 4.0e3, 2.0e3 };
    PetscReal shift[] = { 0.0, -2.0e3 };
    
    ierr = SpecFECtxScaleMeshCoords(ctx,scale,shift);CHKERRQ(ierr);
  }
  
  ierr = SpecFECtxSetConstantMaterialProperties_Velocity(ctx,3200.0,1847.5,2000.0);CHKERRQ(ierr); // vp,vs,rho
  
  DMDASetFieldName(ctx->dm,0,"_x");
  DMDASetFieldName(ctx->dm,1,"_y");
  
  DMCreateGlobalVector(ctx->dm,&u); PetscObjectSetName((PetscObject)u,"disp");
  DMCreateGlobalVector(ctx->dm,&v); PetscObjectSetName((PetscObject)v,"velo");
  DMCreateGlobalVector(ctx->dm,&a); PetscObjectSetName((PetscObject)a,"accl");
  DMCreateGlobalVector(ctx->dm,&f); PetscObjectSetName((PetscObject)f,"f");
  DMCreateGlobalVector(ctx->dm,&g); PetscObjectSetName((PetscObject)g,"g");
  DMCreateGlobalVector(ctx->dm,&Md);
  
  VecZeroEntries(u);
  
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
  PetscOptionsGetReal(NULL,NULL,"-tmax",&time_max,NULL);
  
  ierr = ElastoDynamicsComputeTimeStep_2d(ctx,&dt);CHKERRQ(ierr);
  dt = dt * 0.2;
  PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);
  
  nt = 10;
  PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);
  
  of = 2;
  PetscOptionsGetInt(NULL,NULL,"-of",&of,NULL);
  
  stf_exp_T = 0.1;
  PetscOptionsGetReal(NULL,NULL,"-stf_exp_T",&stf_exp_T,NULL);
  
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
      stf = 1.0 - (1.0 + arg) * exp(-arg);
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
  
  VecDestroy(&u);
  VecDestroy(&v);
  VecDestroy(&a);
  VecDestroy(&f);
  VecDestroy(&Md);
  VecDestroy(&g);
  
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
  PetscOptionsGetInt(NULL,NULL,"-border",&p,NULL);
  ierr = SpecFECtxCreateMesh(ctx,2,mx,my,PETSC_DECIDE,p,2);CHKERRQ(ierr);
  
  {
    PetscReal scale[] = { 4.0e3, 2.0e3 };
    PetscReal shift[] = { -2.0e3, 0.0 };
    
    ierr = SpecFECtxScaleMeshCoords(ctx,scale,shift);CHKERRQ(ierr);
  }
  
  ierr = SpecFECtxSetConstantMaterialProperties_Velocity(ctx,4746.3670317412243 ,2740.2554625435928, 1000.0);CHKERRQ(ierr); // vp,vs,rho
  
  DMDASetFieldName(ctx->dm,0,"_x");
  DMDASetFieldName(ctx->dm,1,"_y");
  
  DMCreateGlobalVector(ctx->dm,&u); PetscObjectSetName((PetscObject)u,"disp");
  DMCreateGlobalVector(ctx->dm,&v); PetscObjectSetName((PetscObject)v,"velo");
  DMCreateGlobalVector(ctx->dm,&a); PetscObjectSetName((PetscObject)a,"accl");
  DMCreateGlobalVector(ctx->dm,&f); PetscObjectSetName((PetscObject)f,"f");
  DMCreateGlobalVector(ctx->dm,&g); PetscObjectSetName((PetscObject)g,"g");
  DMCreateGlobalVector(ctx->dm,&Md);
  
  VecZeroEntries(u);
  
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
  PetscOptionsGetReal(NULL,NULL,"-tmax",&time_max,NULL);
  PetscPrintf(PETSC_COMM_WORLD,"[spec2d] Requested time period: %1.4e\n",time_max);
  
  ierr = ElastoDynamicsComputeTimeStep_2d(ctx,&dt);CHKERRQ(ierr);
  dt = dt * 0.2;
  PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);
  PetscPrintf(PETSC_COMM_WORLD,"[spec2d] Using time step size: %1.4e\n",dt);
  
  nt = 1000000;
  nt = (PetscInt)(time_max / dt ) + 4;
  PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);
  PetscPrintf(PETSC_COMM_WORLD,"[spec2d] Estimated number of time steps: %D\n",nt);
  
  of = 5000;
  PetscOptionsGetInt(NULL,NULL,"-of",&of,NULL);
  
  stf_exp_T = 0.1;
  PetscOptionsGetReal(NULL,NULL,"-stf_exp_T",&stf_exp_T,NULL);
  
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
      //stf = 1.0 - (1.0 + arg) * exp(-arg);
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
  
  VecDestroy(&u);
  VecDestroy(&v);
  VecDestroy(&a);
  VecDestroy(&f);
  VecDestroy(&Md);
  VecDestroy(&g);
  
  PetscFunctionReturn(0);
}

PetscErrorCode specfem_gare6_ex2(PetscInt mx,PetscInt my)
{
  PetscErrorCode ierr;
  SpecFECtx ctx;
  PetscInt p,k,nt,of;
  PetscViewer viewer;
  Vec u,v,a,f,g,Md;
  PetscReal time,dt,stf,time_max;
  PetscReal stf_exp_T;
  PetscBool psource=PETSC_TRUE,ssource=PETSC_FALSE;
  
  ierr = SpecFECtxCreate(&ctx);CHKERRQ(ierr);
  p = 2;
  PetscOptionsGetInt(NULL,NULL,"-border",&p,NULL);
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
  
  DMDASetFieldName(ctx->dm,0,"_x");
  DMDASetFieldName(ctx->dm,1,"_y");
  
  DMCreateGlobalVector(ctx->dm,&u); PetscObjectSetName((PetscObject)u,"disp");
  DMCreateGlobalVector(ctx->dm,&v); PetscObjectSetName((PetscObject)v,"velo");
  DMCreateGlobalVector(ctx->dm,&a); PetscObjectSetName((PetscObject)a,"accl");
  DMCreateGlobalVector(ctx->dm,&f); PetscObjectSetName((PetscObject)f,"f");
  DMCreateGlobalVector(ctx->dm,&g); PetscObjectSetName((PetscObject)g,"g");
  DMCreateGlobalVector(ctx->dm,&Md);
  
  VecZeroEntries(u);
  
  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"uva.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  ierr = AssembleBilinearForm_Mass2d(ctx,Md);CHKERRQ(ierr);
  
  ierr = ElastoDynamicsSetSourceImplementation(ctx,0);CHKERRQ(ierr);
  {
    PetscReal moment[] = { 0.0, 0.0, 0.0, 0.0 };
    PetscReal source_coor[] = { 0.0, 100.0 };
    PetscReal M;
    
    PetscOptionsGetBool(NULL,NULL,"-p_source",&psource,NULL);
    PetscOptionsGetBool(NULL,NULL,"-s_source",&ssource,NULL);
    
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
    ierr = ElastoDynamicsSourceSetup(ctx,source_coor,moment,g);CHKERRQ(ierr);
  }
  
  //ierr = VecView(g,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"f.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(g,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  k = 0;
  time = 0.0;
  
  time_max = 0.4;
  PetscOptionsGetReal(NULL,NULL,"-tmax",&time_max,NULL);
  PetscPrintf(PETSC_COMM_WORLD,"[spec2d] Requested time period: %1.4e\n",time_max);
  
  ierr = ElastoDynamicsComputeTimeStep_2d(ctx,&dt);CHKERRQ(ierr);
  dt = dt * 0.2;
  PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);
  PetscPrintf(PETSC_COMM_WORLD,"[spec2d] Using time step size: %1.4e\n",dt);
  
  nt = 1000000;
  nt = (PetscInt)(time_max / dt ) + 4;
  PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);
  PetscPrintf(PETSC_COMM_WORLD,"[spec2d] Estimated number of time steps: %D\n",nt);
  
  of = 5000;
  PetscOptionsGetInt(NULL,NULL,"-of",&of,NULL);
  
  stf_exp_T = 0.1;
  PetscOptionsGetReal(NULL,NULL,"-stf_exp_T",&stf_exp_T,NULL);
  
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
      ierr = EvaluateRickerWavelet(time,0.15,12.0,1.0,&stf);CHKERRQ(ierr);
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
      PetscReal xr[] = { 100.0, 200.0 };
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
  
  VecDestroy(&u);
  VecDestroy(&v);
  VecDestroy(&a);
  VecDestroy(&f);
  VecDestroy(&Md);
  VecDestroy(&g);
  
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       mx,my;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);CHKERRQ(ierr);
  
  mx = my = 8;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,NULL);CHKERRQ(ierr);
  my = mx;
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&my,NULL);CHKERRQ(ierr);
  
  //ierr = specfem(mx,my);CHKERRQ(ierr);
  //ierr = specfem_ex2(mx,my);CHKERRQ(ierr); // comparison with sem2dpack
  //ierr = specfem_gare6(mx,my);CHKERRQ(ierr); // comparison with gare6more
  ierr = specfem_gare6_ex2(mx,my);CHKERRQ(ierr); // comparison with gare6more
  
  ierr = PetscFinalize();
  return 0;
}
