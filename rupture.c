
#include <petsc.h>

void evaluate_sdf(void *ctx,PetscReal coor[],PetscReal *phi)
{
  *phi = coor[1];
}

void evaluate_grad_sdf(void *ctx,PetscReal coor[],PetscReal grad[])
{
  grad[0] = 0;
  grad[1] = 1.0;
}

PetscErrorCode FaultSDFQuery(PetscReal coor[],PetscReal delta,void *ctx,PetscBool *inside)
{
  PetscReal phi;
  
  *inside = PETSC_FALSE;
  
  if (coor[0] < -10.0e3) { PetscFunctionReturn(0); }
  if (coor[0] >  10.0e3) { PetscFunctionReturn(0); }
  
  evaluate_sdf(ctx,coor,&phi);
  if (PetscAbsReal(phi) > delta) { PetscFunctionReturn(0); }
  *inside = PETSC_TRUE;
  
  PetscFunctionReturn(0);
}

PetscErrorCode FaultSDFNormal(PetscReal coor[],void *ctx,PetscReal n[])
{
  PetscReal gradphi[2],mag;
  
  evaluate_grad_sdf(ctx,coor,gradphi);
  mag = sqrt(gradphi[0]*gradphi[0] + gradphi[1]*gradphi[1]);
  n[0] = gradphi[0] / mag;
  n[1] = gradphi[1] / mag;
  
  PetscFunctionReturn(0);
}

PetscErrorCode FaultSDFTangent(PetscReal coor[],void *ctx,PetscReal t[])
{
  PetscReal gradphi[2],mag;
  
  evaluate_grad_sdf(ctx,coor,gradphi);
  mag = sqrt(gradphi[0]*gradphi[0] + gradphi[1]*gradphi[1]);
  t[0] = -gradphi[1] / mag;
  t[1] =  gradphi[0] / mag;
  
  PetscFunctionReturn(0);
}

PetscErrorCode FaultSDFGetPlusMinusCoor(PetscReal coor[],PetscReal delta,void *ctx,
                                        PetscReal x_plus[],PetscReal x_minus[])
{
  PetscReal phi,normal[2];
  PetscErrorCode ierr;
  
  ierr = FaultSDFNormal(coor,ctx,normal);CHKERRQ(ierr);
  evaluate_sdf(ctx,coor,&phi);
  
  /* project to phi = 0 */
  x_plus[0] = coor[0] - phi * normal[0];
  x_plus[1] = coor[1] - phi * normal[1];
  
#if 1
  /* project to sdf(x) = delta */
  x_plus[0] = x_plus[0] + delta * normal[0];
  x_plus[1] = x_plus[1] + delta * normal[1];
  
  /* project to sdf(x) = -delta */
  x_minus[0] = x_plus[0] - 2.0 * delta * normal[0];
  x_minus[1] = x_plus[1] - 2.0 * delta * normal[1];
#endif

#if 0
  /* project to sdf(x) = delta */
  x_plus[0] = x_plus[0] + 1.5*delta * normal[0];
  x_plus[1] = x_plus[1] + 1.5*delta * normal[1];
  
  /* project to sdf(x) = -delta */
  x_minus[0] = x_plus[0] - 2.0 * 1.5*delta * normal[0];
  x_minus[1] = x_plus[1] - 2.0 * 1.5*delta * normal[1];
#endif

#if 0
  /* project to sdf(x) = +phi */
  x_plus[0] = x_plus[0] + fabs(phi) * normal[0];
  x_plus[1] = x_plus[1] + fabs(phi) * normal[1];
  
  /* project to sdf(x) = +phi */
  x_minus[0] = x_plus[0] - 2.0 * fabs(phi) * normal[0];
  x_minus[1] = x_plus[1] - 2.0 * fabs(phi) * normal[1];
#endif
  
  PetscFunctionReturn(0);
}

/**
 * Slip-Weakening (SW)
 *
 * Shear stress is a decreasing function of slip S up to some distance d_c,
 * beyond which a constant stress is prescribed.
 * Linear Slip Weakening in the form introduced by Andrews (1976):
 * if s < D_c: Tau = \sigma_n * (\mu_s - (\mu_s - \mu_d) * s / D_c)
 * Otherwise:  Tau = \sigma_n * \mu_d
 */

void FricSW(double *Fric, double mu_s, double mu_d, double D_c, double Slip)
{
  if (Slip < D_c) {
    *Fric = mu_s - (mu_s - mu_d) * (Slip / D_c);
  } else {
    *Fric = mu_d;
  }
  
}
void EvalSlipWeakening(double *Tau, double sigma_n, double mu_s, double mu_d, double D_c,double Slip)
{
  double Fric;
  FricSW(&Fric, mu_s, mu_d, D_c, Slip);
  *Tau = sigma_n * Fric;
}



