
#include <petsc.h>
#include <math.h>
#include <rupture.h>

/* ==== SDF API ==== */
PetscErrorCode SDFCreate(SDF *_s)
{
    SDF s;
    
    s = malloc(sizeof(struct _SDF));
    memset(s,0,sizeof(struct _SDF));
    *_s = s;
    PetscFunctionReturn(0);
}

PetscErrorCode SDFDestroy(SDF *_s)
{
    SDF s;
    
    if (!_s) PetscFunctionReturn(0);;
    s = *_s;
    if (!s) PetscFunctionReturn(0);;
    free(s);
    *_s = NULL;
    PetscFunctionReturn(0);
}

PetscErrorCode GeoParamsCreate(GeometryParams *_g)
{
    GeometryParams g;
    
    g = malloc(sizeof(struct _GeometryParams));
    memset(g,0,sizeof(struct _GeometryParams));
    *_g = g;
    PetscFunctionReturn(0);
}
PetscErrorCode GeoParamsDestroy(GeometryParams *_g)
{
    GeometryParams g;
    
    if (!_g) PetscFunctionReturn(0);;
    g = *_g;
    if (!g) PetscFunctionReturn(0);;
    free(g);
    *_g = NULL;
    PetscFunctionReturn(0);
}


PetscErrorCode SDFEvaluate(SDF s,double c[],double *phi)
{
    if (!s->evaluate) {
        printf("Error[SDFEvaluate]: SDF evaluator not set - must call SDFSetup() first\n");
        exit(1);
    }
    s->evaluate(s->data, c, phi);
    PetscFunctionReturn(0);
}

PetscErrorCode SDFEvaluateGradient(SDF s,double c[],double g[])
{
    if (!s->evaluate_gradient) {
        printf("Error[SDFEvaluateGradient]: SDF gradient valuator not set - must call SDFSetup() first\n");
        exit(1);
    }
    s->evaluate_gradient(s->data, c, g);

    PetscFunctionReturn(0);
}

void SDFEvaluateNormal(double c[],SDF s,double n[])
{
    if (!s->evaluate_normal) {
        printf("Error[SDFEvaluateNormal]: Normal vector evaluator not set - must call SDFSetup() first\n");
        exit(1);
    }
    s->evaluate_normal(c,s,n);
    PetscFunctionReturn(0);
}

void SDFEvaluateTangent(double c[],SDF s,double t[])
{
    if (!s->evaluate_tangent) {
        printf("Error[SDFEvaluateTangent]: Tangent vector evaluator not set - must call SDFSetup() first\n");
        exit(1);
    }
    s->evaluate_tangent(c,s,t);
    PetscFunctionReturn(0);
}

PetscErrorCode EvaluateDistOnFault(SDF s,double c[],double * distVal)
{
    if (!s->evaluate_DistOnFault) {
        printf("Error[EvaluateDistOnFault]: Distance on fault function not set  - must call SDFSetup() first\n");
        exit(1);
    }
    s->evaluate_DistOnFault(s->data, c, distVal);
    PetscFunctionReturn(0);
}

/**===============Tilting Function==============*/

/** 00. Default function, sdf geometry and gradient for a horizontal fault*/
void horizontal_sdf( void * ctx, double coor[], double *phi)
{
  *phi = coor[1];
}

void horizontal_grad_sdf(void * ctx, double coor[], double grad[])
{
  grad[0] = 0.0;
  grad[1] = 1.0;
}

/** Get distance from a coordinate projected onto a tilted (placed here to leave it in a single spot)*/
void Horizontal_DistOnFault(void * ctx, double coor[], double *DistOnFault)
{
  *DistOnFault = coor[0];
}

/** 01. Counterclock-wise Tilted Function: sdf geometry and gradient*/
void tilted_sdf(void * ctx,  double coor[], double *phi)
{
  GeometryParams GeoParamList = (GeometryParams) ctx;
  //printf("%f\n",GeoParamList->angle);
  *phi = -sin(GeoParamList->angle* M_PI/180.0) * coor[0] + cos(GeoParamList->angle* M_PI/180.0) * coor[1];
}

void tilted_grad_sdf(void * ctx, double coor[], double grad[])
{
  GeometryParams GeoParamList = (GeometryParams) ctx;
  grad[0] = -sin(GeoParamList->angle * M_PI/180.0);
  grad[1] = cos(GeoParamList->angle * M_PI/180.0);
}

/** Get distance from a coordinate projected onto a tilted (placed here to leave it in a single spot)*/
void Tilted_DistOnFault(void *ctx, double coor[],  double *DistOnFault)
{
  double Fault_angle_deg = *(double*) ctx;
  *DistOnFault = cos(Fault_angle_deg * M_PI/180.0) * coor[0] + sin(Fault_angle_deg* M_PI/180.0) * coor[1];
}

/**===============Tilting Function==============*/

PetscErrorCode SDFSetup(SDF s,int dim,int type)
{ 
  PetscErrorCode ierr;
  switch (dim) {
    case 2:
        switch (type) {
            // Horizontal Fault
            case 0:
                printf("Horizontal Fault\n");
                
                s->evaluate             = horizontal_sdf;
                printf("... Evaluated sdf\n");
                s->evaluate_gradient    = horizontal_grad_sdf;
                printf("... Evaluated sdf gradient\n");
                s->evaluate_DistOnFault = Horizontal_DistOnFault;
                printf("... Evaluated distance on fault function\n");
                s->evaluate_normal      = FaultSDFNormal;
                printf("... Evaluated Normal\n");
                s->evaluate_tangent     = FaultSDFTangent;
                printf("... Evaluated Tangent\n");

                s->data = NULL;
                printf("Data (Null) evaluated\n");
                
                
                break;
            
            // Tilted Fault
            case 1:
                {
                GeometryParams g;
                ierr = GeoParamsCreate(&g);CHKERRQ(ierr);

                printf("Tilted Fault ");
                g->angle = CONST_FAULT_ANGLE_DEG;
                printf("(angle %f deg)\n",g->angle);
                s->data = (void *) g;
                printf("... Evaluated data (Tilting angle)\n");
                
                s->evaluate             = tilted_sdf;
                printf("... Evaluated sdf\n");
                s->evaluate_gradient    = tilted_grad_sdf;
                printf("... Evaluated sdf gradient\n");
                s->evaluate_DistOnFault = Tilted_DistOnFault;
                printf("... Evaluated distance on fault function\n");
                s->evaluate_normal      = FaultSDFNormal;
                printf("... Evaluated Normal\n");
                s->evaluate_tangent     = FaultSDFTangent;
                printf("... Evaluated Tangent\n");

                //ierr = GeoParamsDestroy(&g);CHKERRQ(ierr);
                }
                break;
            
            default:
                printf("Error[SDFSetup]: No support for dim = 2. SDF type = %d\n",type);
                exit(1);
                break;
            }
            break;
      
    default:
        printf("Error[SDFSetup]: No support for dimension = %d. dim must be 2\n",dim);
        exit(1);
        break;
    }
    s->dim = dim;
    s->type = type;
    PetscFunctionReturn(0);
}

PetscErrorCode evaluate_sdf(void *ctx, PetscReal coor[],PetscReal *phi)
{
  PetscErrorCode ierr;
  SDF s = (SDF) ctx;
  ierr = SDFEvaluate(s, coor, phi);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
PetscErrorCode evaluate_grad_sdf(void *ctx,PetscReal coor[],PetscReal grad[])
{
  PetscErrorCode ierr;
  SDF s = (SDF) ctx;
  ierr = SDFEvaluateGradient(s, coor, grad);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode evaluate_DistOnFault_sdf(void *ctx, PetscReal coor[], double *DistOnFault)
{
  PetscErrorCode ierr;
  SDF s = (SDF) ctx;
  ierr = EvaluateDistOnFault(s, coor, DistOnFault);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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

/**=============================================*/

PetscErrorCode FaultSDFQuery(PetscReal coor[],PetscReal delta,void *ctx,PetscBool *inside)
{
  PetscReal phi;
  PetscReal DistOnFault;
  PetscReal normal[2];
  PetscErrorCode ierr;
  
  *inside = PETSC_FALSE;
  ierr = evaluate_sdf(ctx,coor,&phi);CHKERRQ(ierr);
  ierr = evaluate_DistOnFault_sdf(ctx, coor, &DistOnFault);CHKERRQ(ierr);

  if (PetscAbsReal(DistOnFault) >  10.0e3) { PetscFunctionReturn(0); }
  
  if (PetscAbsReal(phi) > delta) { PetscFunctionReturn(0); }
  *inside = PETSC_TRUE;
  
  PetscFunctionReturn(0);
}



PetscErrorCode FaultSDFNormal(PetscReal coor[],void *ctx,PetscReal n[])
{
  PetscReal gradphi[2],mag;
  PetscErrorCode ierr;
  ierr = evaluate_grad_sdf(ctx,coor,gradphi);CHKERRQ(ierr);
  mag = sqrt(gradphi[0]*gradphi[0] + gradphi[1]*gradphi[1]);
  n[0] = gradphi[0] / mag;
  n[1] = gradphi[1] / mag;
  
  PetscFunctionReturn(0);
}

PetscErrorCode FaultSDFTangent(PetscReal coor[],void *ctx,PetscReal t[])
{
  PetscReal gradphi[2],mag;
  PetscErrorCode ierr;
  ierr = evaluate_grad_sdf(ctx,coor,gradphi);CHKERRQ(ierr);
  mag = sqrt(gradphi[0]*gradphi[0] + gradphi[1]*gradphi[1]);
  t[0] = -gradphi[1] / mag;
  t[1] =  gradphi[0] / mag;
  
  PetscFunctionReturn(0);
}

PetscErrorCode FaultSDFGetPlusMinusCoor(PetscReal coor[],PetscReal delta, void *ctx,
                                        PetscReal x_plus[],PetscReal x_minus[])
{
  PetscReal phi,normal[2];
  PetscErrorCode ierr;
  
  ierr = FaultSDFNormal(coor,ctx,normal);CHKERRQ(ierr);
  ierr = evaluate_sdf(ctx,coor,&phi);CHKERRQ(ierr);
  
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



