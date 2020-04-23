
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


void evaluate_sdf(void *ctx,PetscReal coor[],PetscReal *phi);
void FricSW(double *Fric, double mu_s, double mu_d, double D_c, double Slip);

PetscErrorCode FaultSDFQuery(PetscReal coor[],PetscReal delta,void *ctx,PetscBool *inside);
PetscErrorCode FaultSDFNormal(PetscReal coor[],void *ctx,PetscReal n[]);
PetscErrorCode FaultSDFTangent(PetscReal coor[],void *ctx,PetscReal t[]);
PetscErrorCode FaultSDFGetPlusMinusCoor(PetscReal coor[],PetscReal delta,void *ctx,
                                        PetscReal x_plus[],PetscReal x_minus[]);

#endif
