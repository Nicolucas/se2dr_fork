
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
    s->evaluate((void *) s, c, phi);
    PetscFunctionReturn(0);
}

PetscErrorCode SDFEvaluateGradient(SDF s,double c[],double g[])
{
    if (!s->evaluate_gradient) {
        printf("Error[SDFEvaluateGradient]: SDF gradient valuator not set - must call SDFSetup() first\n");
        exit(1);
    }
    s->evaluate_gradient((void *) s, c, g);

    PetscFunctionReturn(0);
}

void SDFEvaluateNormal(double c[],SDF s,double n[])
{
    if (!s->evaluate_normal) {
        printf("Error[SDFEvaluateNormal]: Normal vector evaluator not set - must call SDFSetup() first\n");
        exit(1);
    }
    s->evaluate_normal(c,s,n);
}

void SDFEvaluateTangent(double c[],SDF s,double t[])
{
    if (!s->evaluate_tangent) {
        printf("Error[SDFEvaluateTangent]: Tangent vector evaluator not set - must call SDFSetup() first\n");
        exit(1);
    }
    s->evaluate_tangent(c,s,t);
}

PetscErrorCode EvaluateDistOnFault(SDF s,double c[],double * distVal)
{
    if (!s->evaluate_DistOnFault) {
        printf("Error[EvaluateDistOnFault]: Distance on fault function not set  - must call SDFSetup() first\n");
        exit(1);
    }
    s->evaluate_DistOnFault(s, c, distVal);
    PetscFunctionReturn(0);
}

/**=============== Geometry Functions ==============*/

/** 00. Default function, sdf geometry and gradient for a horizontal fault*/
void horizontal_sdf(void * ctx, double coor[], double *phi)
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
  SDF s = (SDF) ctx;
  GeometryParams GeoParamList = (GeometryParams) s->data;
  //printf("%f\n",GeoParamList->angle);
  *phi = -sin(GeoParamList->angle* M_PI/180.0) * coor[0] + cos(GeoParamList->angle* M_PI/180.0) * coor[1];
}

void tilted_grad_sdf(void * ctx, double coor[], double grad[])
{
  SDF s = (SDF) ctx;
  GeometryParams GeoParamList = (GeometryParams) s->data;
  grad[0] = -sin(GeoParamList->angle * M_PI/180.0);
  grad[1] = cos(GeoParamList->angle * M_PI/180.0);
}

/** Get distance from a coordinate projected onto a tilted (placed here to leave it in a single spot)*/
void Tilted_DistOnFault(void *ctx, double coor[],  double *DistOnFault)
{
  SDF s = (SDF) ctx;
  GeometryParams GeoParamList = (GeometryParams) s->data;

  *DistOnFault = cos(GeoParamList->angle * M_PI/180.0) * coor[0] + sin(GeoParamList->angle* M_PI/180.0) * coor[1];
}

/**=============== Sigmoid Function ==============*/

// Application of a sigmoid function dependant on a parameter k for values on the interval (-1,0) 

void Sigmoid_Function_map(double x, double *fx, double k, double amp)
{
  *fx = amp * (x - x * k) / (k - fabs(x) * 2.0 * k + 1.0);
}

void DerSigmoid_Function_map(double x, double *fx, double k, double amp)
{
  *fx = amp * (1 - k * k) / ((k - fabs(x) * 2.0 * k + 1.0)*(k - fabs(x) * 2.0 * k + 1.0));
}

// Standard distance function using as input parameters the coordinates of two points
void DistanceFunction(double x0, double y0, double x1, double y1, double * distVal)
{
  *distVal = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
}

// Function to evaluate a function in a discrete equidistant list of 2*HalfNumPoints +1 values of x. We highlight that the origin (0) and the limits
// x0, x1 are also evaluated.
void DiscretizeFunction(double x[], double fx[], void* ctx)
{
  int idx;
  double xstepsize;
  double x0;
  double x1; 
  int HalfNumPoints; 
  double amp;
  double k;

  GeometryParams g = (GeometryParams) ctx;
  HalfNumPoints = g->HalfNumPoints;
  x0 = g->xorigin;
  x1 = g->xend;
  amp = g->amp;
  k = g->k;


  xstepsize = (x1 - x0)/(2.0 * HalfNumPoints);

  for (idx = 0; idx <= 2 * HalfNumPoints; idx++)
  {
    x[idx] = xstepsize * idx + x0;
    Sigmoid_Function_map(x[idx],  &fx[idx], k, amp);
  }
}


void find_minimum_idx(double coorx, double coory, double x[], double fx[], int NumPointsDiscreteCurve, int *idxCurve)
{
  double minimum,DistVal;
  int idx = -1;
  minimum = 1.0e200;

  for (int i=0; i < NumPointsDiscreteCurve; i++)
  {
    DistanceFunction(coorx, coory, x[i], fx[i], &DistVal);
    if (DistVal < minimum)
    {
      minimum = DistVal;
      idx = i;
    }
  }
  *idxCurve = idx;
}

PetscErrorCode initializeZeroSetCurveFault(SDF s)
{
  PetscErrorCode ierr;
  GeometryParams g = (GeometryParams) s->data;

  ierr = PetscCalloc1(2 * g->HalfNumPoints + 1,&s->xList);CHKERRQ(ierr);
  ierr = PetscCalloc1(2 * g->HalfNumPoints + 1,&s->fxList);CHKERRQ(ierr);

  DiscretizeFunction(s->xList, s->fxList, s->data);

  PetscFunctionReturn(0);
}

/** 02. Sigmoid Function: sdf geometry and gradient*/

void sigmoid_sdf(void * ctx,  double coor[], double *phi)
{
  double MagnitudePhi;
  double yValue;

  SDF s = (SDF) ctx;
  DistanceFunction(coor[0], coor[1], s->xList[s->curve_idx_carrier], s->fxList[s->curve_idx_carrier], &MagnitudePhi);

  
  if (s->fxList[s->curve_idx_carrier] > coor[1])
  {
    *phi = -MagnitudePhi;
  }
  else
  {
    *phi = MagnitudePhi;
  }
}

void sigmoid_grad_sdf(void * ctx, double coor[], double grad[])
{
  double fPrimeX;
  double amp;
  double k;
  double mag;

  SDF s = (SDF) ctx;
  GeometryParams g = (GeometryParams) s->data;

  amp = g->amp;
  k = g->k;

  DerSigmoid_Function_map(s->xList[s->curve_idx_carrier], &fPrimeX, k, amp);

  mag = sqrt(1.0 + fPrimeX * fPrimeX);

  grad[0] =  -fPrimeX / mag;
  grad[1] =  1.0 / mag;
}

void getAngleFromDerivative(void* ctx, double *angle)
{
  double fPrimeX;
  double amp;
  double k;

  SDF s = (SDF) ctx;
  GeometryParams g = (GeometryParams) s->data;

  amp = g->amp;
  k = g->k;

  DerSigmoid_Function_map(s->xList[s->curve_idx_carrier], &fPrimeX, k, amp);
  *angle = atan(fPrimeX)* 180.0 / M_PI;
}

void Sigmoid_DistOnFault(void *ctx, double coor[],  double *DistOnFault)
{
  int idx;
  SDF s = (SDF) ctx;
  GeometryParams g = (GeometryParams) s->data;
  *DistOnFault = 0.0;

  //printf("%d\t",s->curve_idx_carrier);
  if (s->curve_idx_carrier > g->HalfNumPoints)
  {
    for (idx = g->HalfNumPoints; idx < s->curve_idx_carrier ; idx++)
    { double LocalDist = 0.0;
      DistanceFunction(s->xList[idx], s->fxList[idx], s->xList[idx+1], s->fxList[idx+1], &LocalDist);
      *DistOnFault += LocalDist;
    }
  }
  else if (s->curve_idx_carrier < g->HalfNumPoints)
  {
    for (idx = s->curve_idx_carrier; idx < g->HalfNumPoints ; idx++)
    { double LocalDist = 0.0;
      DistanceFunction(s->xList[idx], s->fxList[idx], s->xList[idx+1], s->fxList[idx+1], &LocalDist);
      *DistOnFault += LocalDist;
    }
  }
  else
  {
    *DistOnFault = 0.0;
  }  
}
/**=============== Sigmoid ==============*/


/**===============Geometry Functions==============*/
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
                }
                break;

            case 2:
                {
                GeometryParams g;
                ierr = GeoParamsCreate(&g);CHKERRQ(ierr);
                printf("Sigmoid Fault \n");
                g->HalfNumPoints = 3000; 
                g->k             = -0.0002;
                g->amp           = 2.0;
                g->xorigin       = -1.0e4;
                g->xend          = 1.0e4;

                s->data = (void *) g;
                printf("... Evaluated data (Sigmoid parameters)\n");
                
                s->evaluate             = sigmoid_sdf;
                printf("... Evaluated sdf\n");
                s->evaluate_gradient    = sigmoid_grad_sdf;
                printf("... Evaluated sdf gradient\n");
                s->evaluate_DistOnFault = Sigmoid_DistOnFault;
                printf("... Evaluated distance on fault function\n");
                s->evaluate_normal      = FaultSDFNormal;
                printf("... Evaluated Normal\n");
                s->evaluate_tangent     = FaultSDFTangent;
                printf("... Evaluated Tangent\n");
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
PetscErrorCode evaluate_grad_sdf(void *ctx, PetscReal coor[], PetscReal grad[])
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
  PetscReal phi = 1.0e32;
  PetscReal DistOnFault = 1.0e32;
  PetscReal normal[2];
  PetscErrorCode ierr;
  
  *inside = PETSC_FALSE;
  ierr = evaluate_sdf(ctx,coor,&phi);CHKERRQ(ierr);
  ierr = evaluate_DistOnFault_sdf(ctx, coor, &DistOnFault);CHKERRQ(ierr);
  //printf("DistOnFault, %+1.4e, Phi: %+1.4e \n", DistOnFault, phi);

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
  t[0] =  gradphi[1] / mag;
  t[1] = -gradphi[0] / mag;
  
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



