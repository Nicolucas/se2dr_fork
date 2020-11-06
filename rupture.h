
#ifndef __rupture_h__
#define __rupture_h__

/* slip */
/* slip rate */
/* initial shear stress */
/* initial normal stress */
#define __RUPTURE_HEADER__ \
  PetscReal slip; \
  PetscReal slip_rate; \
  PetscReal sigma_tau_0; \
  PetscReal sigma_n_0; \
  PetscBool sliding; \

typedef struct {
  __RUPTURE_HEADER__;
  PetscInt eid[2];
  //PetscReal Ni_0[9*9];
  //PetscReal Ni_1[9*9];
  PetscReal *N1_plus,*N2_plus;
  PetscReal *N1_minus,*N2_minus;
  PetscReal mu;
} DRVar;

typedef struct {
  __RUPTURE_HEADER__;
  PetscReal theta; /* state variable required by R&S */
} DRVarState;

typedef struct _SDF *SDF;
typedef struct _GeometryParams *GeometryParams;

struct _GeometryParams {
  double angle;
  double radius;
};

struct _SDF {
  int  type;
  int dim;
  void *data;
  void (*evaluate)(void *, double*, double*);
  void (*evaluate_gradient)(void *, double*,double*);
  PetscErrorCode (*evaluate_normal)(double*, void *, double*);
  PetscErrorCode (*evaluate_tangent)(double*, void *,double*);
  void (*evaluate_DistOnFault)(void *, double*, double*);
};

PetscErrorCode SDFCreate(SDF*);
PetscErrorCode SDFDestroy(SDF *_s);
PetscErrorCode SDFSetup(SDF,int,int);

PetscErrorCode GeoParamsCreate(GeometryParams *_g);
PetscErrorCode GeoParamsDestroy(GeometryParams *_g);

void SDFEvaluate(SDF s,double c[],double *phi);
void SDFEvaluateGradient(SDF s,double c[],double g[]);
void SDFEvaluateNormal(double c[],SDF s,double n[]);
void SDFEvaluateTangent(double c[],SDF s,double t[]);
void EvaluateDistOnFault(SDF s,double c[],double *distVal);


void horizontal_sdf(void * ctx, double coor[],  double *phi);
void horizontal_grad_sdf(void * ctx, double coor[], double grad[]);
void Horizontal_DistOnFault(void * ctx, double coor[], double *DistOnFault);

void tilted_sdf(void * ctx, double coor[], double *phi);
void tilted_grad_sdf(void * ctx, double coor[], double grad[]);
void Tilted_DistOnFault(void * ctx, double coor[], double *DistOnFault);




void evaluate_sdf(void *ctx,PetscReal coor[],PetscReal *phi);
void evaluate_grad_sdf(void *ctx,PetscReal coor[],PetscReal grad[]);
void evaluate_DistOnFault_sdf(void *ctx, PetscReal coor[], double *DistOnFault);
void EvalSlipWeakening(double *Tau, double sigma_n, double mu_s, double mu_d, double D_c,double Slip);


void FricSW(double *Fric, double mu_s, double mu_d, double D_c, double Slip);
void MohrTranformSymmetricRot(PetscReal RotAngleDeg, PetscReal *s_xx, PetscReal *s_yy,PetscReal *s_xy);

PetscErrorCode FaultSDFQuery(PetscReal coor[],PetscReal delta,void *ctx,PetscBool *inside);
PetscErrorCode FaultSDFNormal(PetscReal coor[],void *ctx,PetscReal n[]);
PetscErrorCode FaultSDFTangent(PetscReal coor[],void *ctx,PetscReal t[]);
PetscErrorCode FaultSDFGetPlusMinusCoor(PetscReal coor[],PetscReal delta,void *ctx,
                                        PetscReal x_plus[],PetscReal x_minus[]);

#define CONST_FAULT_ANGLE_DEG 45.0

#endif
