export PETSC_DIR=petsc-3.12.5
export PETSC_ARCH=arch-linux-c-debug
export LD_LIBRARY_PATH=${PETSC_DIR}/${PETSC_ARCH}/lib
export OMP_NUM_THREADS=32

cd ../se2wave
make clean
make all 

now=$(date)
echo ">> $now"

fname=SimulationTestName
export STORAGE_VAR_NAME=Path-to-Folder/$fname
export SE2DR_RUN="path-to-se2dr.app"

mkdir -p $STORAGE_VAR_NAME
echo ">> $STORAGE_VAR_NAME"


# Options to run se2dr: 
##        geometry_type: 
###            0 -> Horizontal fault
###            1 -> tilted fault 
###            2 -> sigmoid fault
##        model_type: 
###            0 -> Kostrov model
###            1 -> TPV3 model (slip weakening friction law)
##        tau_correction:
###            0 -> Volumetric yielding criterion
###            1 -> Interface yielding criterion
##        delta_cell_factor: 
###            factor for the fault zone half thickness delta relative to the element width 
##        delta:
###            alternative to delta_cell_factor. fault zone half thickness
##        mx, my: 
###            element dimensions
##        bdegree:
###            polynomial degree of the quadrilateral element
##        tmax:
###            max simulation time
##        of:
###            Output frequency
##        angle:
###            Tilting angle for the tilted fault geometry (1)
##        blend_a_factor:
###            blending parameter a
##        blend_phio_factor:
###            blending parameter phio
##        Large_Domain:
###            boolean to use a larger domain (for TVP3 simulation)




##-angle -blend_a_factor -blend_phio_factor -geometry_type -model_type -tau_correction
##-mx -my -tmax -bdegree -delta_cell_factor -of -model_type -tau_correction

#######################################################################################
### 1. Kos_T0_P3_025x025_A12phi65_Delta1.001_4s C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 800 -my 800 -tmax 4.0 -bdegree 3 -delta_cell_factor 1.001 -of 10 -model_type 0 -geometry_type 1 -tau_correction 1)

### 2. Kos_T20_P3_025x025_A12phi65_Delta2.5_4s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 800 -my 800 -tmax 4.0 -bdegree 3 -delta_cell_factor 2.5 -of 10 -model_type 0 -geometry_type 1 -angle 20 -tau_correction 1)

### 2.1 Kos_T20_P3_025x025_A12phi65_Delta1.001_4s C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 800 -my 800 -tmax 4.0 -bdegree 3 -delta_cell_factor 1.001 -of 10 -model_type 0 -geometry_type 1 -angle 20  -tau_correction 1)

### 3. Kos_Sig_P3_025x025_A12phi65_Delta2.5_4s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 800 -my 800 -tmax 4.0 -bdegree 3 -delta_cell_factor 2.5 -of 10 -model_type 0 -geometry_type 2 -tau_correction 1)

### 4. TPV_T0_P3_025x025_A12phi65_Delta1.001_3s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 800 -my 800 -tmax 3.0 -bdegree 3 -delta_cell_factor 1.001 -of 10 -model_type 1 -tau_correction 1)

### 5.0 TPV_T20_P3_025x025_A12phi65_Delta1.0_2s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 800 -my 800 -tmax 2.0 -bdegree 3 -delta_cell_factor 1.0 -of 10 -model_type 1 -angle 20.0 -tau_correction 1)

### 5.1 TPV_T20_P3_025x025_A12phi65_Delta1.43_2s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 800 -my 800 -tmax 2.0 -bdegree 3 -delta_cell_factor 1.43 -of 10 -model_type 1 -angle 20.0 -tau_correction 1)

### 5.2 TPV_T20_P3_025x025_A12phi65_Delta3.00_2s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 800 -my 800 -tmax 2.0 -bdegree 3 -delta_cell_factor 3.0 -of 10 -model_type 1 -angle 20.0 -tau_correction 1)

### 5.3 TPV_T20_P3_025x025_A12phi65_Delta4.00_2s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 800 -my 800 -tmax 2.0 -bdegree 3 -delta_cell_factor 4.0 -of 10 -model_type 1 -angle 20.0 -tau_correction 1)

### 6.0 TPV_T0_P3_050x050_A12phi65_Delta1.001_3s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 2400 -my 2400 -tmax 3.0 -bdegree 3 -delta_cell_factor 1.001 -of 5 -model_type 1 -blend_a_factor 18.0 -Large_Domain 1 -tau_correction 1)

### 6.1 TPV_T0_P3_050x050_A18phi65_Delta1.001_3s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 1200 -my 1200 -tmax 3.0 -bdegree 3 -delta_cell_factor 1.001 -of 5 -model_type 1 -blend_a_factor 18.0 -Large_Domain 1 -tau_correction 1)

### 6.2 TPV_T0_P3_100x100_A18phi65_Delta1.001_3s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 600 -my 600 -tmax 3.0 -bdegree 3 -delta_cell_factor 1.001 -of 5 -model_type 1 -blend_a_factor 18.0 -Large_Domain 1 -tau_correction 1)

### 6.3 TPV_T0_P2_025x025_A18phi65_Delta1.001_3s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 2400 -my 2400 -tmax 3.0 -bdegree 2 -delta_cell_factor 1.001 -of 5 -model_type 1 -blend_a_factor 18.0 -Large_Domain 1 -tau_correction 1)

### 6.4 TPV_T0_P2_050x050_A18phi65_Delta1.001_3s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 1200 -my 1200 -tmax 3.0 -bdegree 2 -delta_cell_factor 1.001 -of 5 -model_type 1 -blend_a_factor 18.0 -Large_Domain 1 -tau_correction 1)

### 6.5 TPV_T0_P2_100x100_A18phi65_Delta1.001_3s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 600 -my 600 -tmax 3.0 -bdegree 2 -delta_cell_factor 1.001 -of 5 -model_type 1 -blend_a_factor 18.0 -Large_Domain 1 -tau_correction 1)

### 6.6 TPV_T0_P1_025x025_A18phi65_Delta1.001_3s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 800 -my 800 -tmax 3.0 -bdegree 1 -delta_cell_factor 1.001 -of 1 -model_type 1 -blend_a_factor 18.0 -tau_correction 1)

### 6.7 TPV_T0_P1_050x050_A18phi65_Delta1.001_3s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 400 -my 400 -tmax 3.0 -bdegree 1 -delta_cell_factor 1.001 -of 1 -model_type 1 -blend_a_factor 18.0 -tau_correction 1)

### 6.8 TPV_T0_P1_100x100_A18phi65_Delta1.001_3s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 200 -my 200 -tmax 3.0 -bdegree 1 -delta_cell_factor 1.001 -of 1 -model_type 1 -blend_a_factor 18.0 -tau_correction 1)

### 7.0 TPV_T0_P3_025x025_A12phi65_Delt1.001_7s_NC
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 2400 -my 2400 -tmax 7.0 -bdegree 3 -delta_cell_factor 1.001 -of 10 -model_type 1 -Large_Domain 1)

### 7.1 T0_P3_025x025_A12phi65_Delt1.001_7s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 2400 -my 2400 -tmax 7.0 -bdegree 3 -delta_cell_factor 1.001 -of 10 -model_type 1 -Large_Domain 1 -tau_correction 1)

### 8.0 TPV_T0_P3_025x025_A18phi65_Delt1.001_7s_C
(cd $STORAGE_VAR_NAME && $SE2DR_RUN -mx 2400 -my 2400 -tmax 7.0 -bdegree 3 -delta_cell_factor 1.001 -of 10 -model_type 1 -blend_a_factor 18.0 -Large_Domain 1 -tau_correction 1)



##################################################################################

echo ">> $STORAGE_VAR_NAME"
echo ">> $now"
