
#include <petsc.h>
#include <math.h>
#include <rupture.h>


/**===============Tilting Function==============*/

/** 00. Default function, sdf geometry and gradient for a horizontal fault*/
void horizontal_sdf(double coor[], struct GeometryParams GeoParamList, double *phi)
{
*phi = coor[1];
}
void horizontal_grad_sdf(double coor[], struct GeometryParams GeoParamList, double grad[])
{
  grad[0] = 0.0;
  grad[1] = 1.0;
}

/** 01. Counterclock-wise Tilted Function: sdf geometry and gradient*/
void tilted_sdf(double coor[], struct GeometryParams GeoParamList, double *phi)
{
  *phi = -sin(GeoParamList.angle* M_PI/180.0) * coor[0] + cos(GeoParamList.angle* M_PI/180.0) * coor[1];
}

void tilted_grad_sdf(double coor[], struct GeometryParams GeoParamList, double grad[])
{
  grad[0] = -sin(GeoParamList.angle* M_PI/180.0);
  grad[1] = cos(GeoParamList.angle* M_PI/180.0);
}


/** Definition of function to pointer for SDF */
void (*sdf_func[])(double coor[], struct GeometryParams GeoParamList, double *phi) =
  {horizontal_sdf, tilted_sdf};
void (*sdf_grad_func[])(double coor[], struct GeometryParams GeoParamList, double grad[]) =
  {horizontal_grad_sdf, tilted_grad_sdf};

/**
typedef void (*SDFFunc)(double*, struct GeometryParams, double*);

SDFFunc mystuff[] = { horiz_sdf, tilted_sdf };
*/


void evaluate_sdf(void *ctx,PetscReal coor[],PetscReal *phi)
{
  struct GeometryParams GeomParam = {0}; //Initialized to null
  GeomParam.angle = CONST_FAULT_ANGLE_DEG;
  (*sdf_func[1])(coor, GeomParam,  phi);
}
void evaluate_grad_sdf(void *ctx,PetscReal coor[],PetscReal grad[])
{
  struct GeometryParams GeomParam = {0}; //Initialized to null
  GeomParam.angle = CONST_FAULT_ANGLE_DEG;
  (*sdf_grad_func[1])(coor, GeomParam,  grad);
}

/**=============================================*/
/** Mohr transform function */
void MohrTranformSymmetricRot(PetscReal RotAngleDeg, PetscReal *s_xx, PetscReal *s_yy,PetscReal *s_xy)
{
  PetscReal RotAngle = RotAngleDeg * M_PI/180.0;
  PetscReal _s_xx = *s_xx;
  PetscReal _s_xy = *s_xy;
  PetscReal _s_yy = *s_yy;

  *s_xx = _s_xx * cos(RotAngle)*cos(RotAngle) + _s_yy * sin(RotAngle)*sin(RotAngle) + (_s_xy)* sin(2.0 * RotAngle);
  *s_xy = (_s_yy - _s_xx) * cos(RotAngle) * sin(RotAngle) + (_s_xy)* cos(2.0 * RotAngle);
  *s_yy = _s_xx * sin(RotAngle)*sin(RotAngle) + _s_yy * cos(RotAngle)*cos(RotAngle) - (_s_xy)* sin(2.0 * RotAngle);
}
/** Get distance from a coordinate projected onto a tilted (placed here to leave it in a single spot)*/
void DistOnTiltedFault(PetscReal coor[], PetscReal *DistOnFault)
{
  *DistOnFault = cos(CONST_FAULT_ANGLE_DEG* M_PI/180.0) * coor[0] + sin(CONST_FAULT_ANGLE_DEG* M_PI/180.0) * coor[1];
}
/**=============================================*/
/**
void evaluate_sdf(void *ctx,PetscReal coor[],PetscReal *phi)
{
  *phi = coor[1];
}

void evaluate_grad_sdf(void *ctx,PetscReal coor[],PetscReal grad[])
{
  grad[0] = 0;
  grad[1] = 1.0;
}
*/
PetscErrorCode FaultSDFQuery(PetscReal coor[],PetscReal delta,void *ctx,PetscBool *inside)
{
  PetscReal phi;
  PetscReal DistOnFault;
  PetscReal normal[2];
  
  *inside = PETSC_FALSE;
 
  evaluate_sdf(ctx,coor,&phi);
  DistOnTiltedFault(coor, &DistOnFault);

  if (PetscAbsReal(DistOnFault) >  10.0e3) { PetscFunctionReturn(0); }
  
  if (PetscAbsReal(phi) > delta) { PetscFunctionReturn(0); }
  *inside = PETSC_TRUE;
  
  PetscFunctionReturn(0);
}

/** Commented to add a version that considers tilted geometry
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
}*/

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



