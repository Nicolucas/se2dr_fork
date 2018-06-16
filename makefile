
include ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/petscvariables
include ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/petscrules

CFLAGS = -g -O0
INC = -I. -I${PETSC_DIR}/include ${PETSC_CC_INCLUDES}
LIB = ${PETSC_LIB}

SRC =   se2wave.c

OBJ = $(SRC:.c=.o)
APP = $(APP:.c=.app)

all:
	-@${MAKE} se2wave.app


%.app: %.o
	-@echo "----- Linking $*.app -----"
	${PCC} ${CFLAGS} -o $*.app $*.o ${INC} ${LIB}

%.o: %.c
	-@echo "---- Compiling $*.c -----"
	${PCC} ${CFLAGS} -c $*.c ${INC}

clean_all:
	-@rm -rf *.o
	-@rm -rf *.app
	-@rm -rf *.dSYM

clean_output:
	-@rm -rf *.vtu *.vts *.dat

