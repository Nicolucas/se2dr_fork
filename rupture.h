
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

  //For the sigmoid
  int HalfNumPoints;
  double k;
  double amp;
  double xorigin;
  double xend;
};

struct _SDF {
  int  type;
  int dim;
  void *data;
  void (*evaluate)(void *, double*, int, double*);
  void (*evaluate_gradient)(void *, double*, int, double*);
  PetscErrorCode (*evaluate_normal)(double*, void *, int, double*);
  PetscErrorCode (*evaluate_tangent)(double*, void *, int, double*);
  void (*evaluate_DistOnFault)(void *, double*, int, double*);

  int *idxArray_ClosestFaultNode;
  double *idxArray_SDFphi;
  double *idxArray_DistOnFault;
  double *xList;
  double *fxList;
};

PetscErrorCode SDFCreate(SDF*);
PetscErrorCode SDFDestroy(SDF *_s);
PetscErrorCode SDFSetup(SDF,int,int);

PetscErrorCode GeoParamsCreate(GeometryParams *_g);
PetscErrorCode GeoParamsDestroy(GeometryParams *_g);

PetscErrorCode SDFEvaluate(SDF s,double c[], int qpIdx, double *phi);
PetscErrorCode SDFEvaluateGradient(SDF s,double c[], int qpIdx, double g[]);
void SDFEvaluateNormal(double c[],SDF s, int qpIdx, double n[]);
void SDFEvaluateTangent(double c[],SDF s, int qpIdx, double t[]);
PetscErrorCode EvaluateDistOnFault(SDF s,double c[], int qpIdx, double *distVal);


void horizontal_sdf(void * ctx, double coor[],  int qpIdx, double *phi);
void horizontal_grad_sdf(void * ctx, double coor[], int qpIdx, double grad[]);
void Horizontal_DistOnFault(void * ctx, double coor[], int qpIdx, double *DistOnFault);

void tilted_sdf(void * ctx, double coor[], int qpIdx, double *phi);
void tilted_grad_sdf(void * ctx, double coor[], int qpIdx, double grad[]);
void Tilted_DistOnFault(void * ctx, double coor[], int qpIdx, double *DistOnFault);

//--- Sigmoid functions
void Sigmoid_Function_map(double x, double *fx, double k, double amp);
void DerSigmoid_Function_map(double x, double *fx, double k, double amp);
void DistanceFunction(double x0, double y0, double x1, double y1, double * distVal);
void DiscretizeFunction(double x[], double fx[], void* ctx);
void find_minimum_idx(double coorx, double coory, double x[], double fx[], int NumPointsDiscreteCurve, int *idxCurve);

PetscErrorCode initializeZeroSetCurveFault(SDF s);
PetscErrorCode Init_evaluate_Sigmoid_sdf(void *ctx, PetscReal coor[], PetscInt qpIdx, PetscReal *phi);
PetscErrorCode Init_evaluate_DistOnFault_Sigmoid_sdf(void *ctx, PetscReal coor[], PetscInt qpIdx, double *DistOnFault);


void sigmoid_sdf(void * ctx,  double coor[], int qpIdx, double *phi);
void sigmoid_grad_sdf(void * ctx, double coor[], int qpIdx, double grad[]);
void Sigmoid_DistOnFault(void *ctx, double coor[], int qpIdx, double *DistOnFault);
void getAngleFromDerivative(void* ctx, int qpIdx, double *angle);

//--- Sigmoid functions End

PetscErrorCode evaluate_sdf(void *ctx,PetscReal coor[], PetscInt qpIdx, PetscReal *phi);
PetscErrorCode evaluate_grad_sdf(void *ctx,PetscReal coor[], PetscInt qpIdx, PetscReal grad[]);
PetscErrorCode evaluate_DistOnFault_sdf(void *ctx, PetscReal coor[], PetscInt qpIdx, double *DistOnFault);
void EvalSlipWeakening(double *Tau, double sigma_n, double mu_s, double mu_d, double D_c,double Slip);


void FricSW(double *Fric, double mu_s, double mu_d, double D_c, double Slip);
void MohrTranformSymmetricRot(PetscReal RotAngleDeg, PetscReal *s_xx, PetscReal *s_yy,PetscReal *s_xy);

PetscErrorCode FaultSDFQuery(PetscReal coor[],PetscReal delta,void *ctx, PetscInt qpIdx, PetscBool *inside);
PetscErrorCode FaultSDFNormal(PetscReal coor[],void *ctx, PetscInt qpIdx, PetscReal n[]);
PetscErrorCode FaultSDFTangent(PetscReal coor[],void *ctx, PetscInt qpIdx, PetscReal t[]);
PetscErrorCode FaultSDFGetPlusMinusCoor(PetscReal coor[],PetscReal delta,void *ctx, PetscInt qpIdx, 
                                        PetscReal x_plus[],PetscReal x_minus[]);


#endif
