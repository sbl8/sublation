//go:build amd64

#include "textflag.h"

// vectorAddASM performs vectorized addition using AVX2
// func vectorAddASM(a, b, result []float32)
TEXT ·vectorAddASM(SB), NOSPLIT, $0-72
	MOVQ a_base+0(FP), AX
	MOVQ b_base+24(FP), BX
	MOVQ result_base+48(FP), CX
	MOVQ a_len+8(FP), DX
	
	CMPQ DX, $8
	JL scalar_add
	
	// Process 8 elements at a time with AVX2
	SUBQ $8, DX
	
loop_add:
	VMOVUPS (AX), Y0
	VMOVUPS (BX), Y1
	VADDPS Y0, Y1, Y2
	VMOVUPS Y2, (CX)
	
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	SUBQ $8, DX
	JGE loop_add
	
	ADDQ $8, DX
	
scalar_add:
	CMPQ DX, $0
	JLE done_add
	
scalar_loop_add:
	MOVSS (AX), X0
	MOVSS (BX), X1
	ADDSS X0, X1
	MOVSS X1, (CX)
	
	ADDQ $4, AX
	ADDQ $4, BX
	ADDQ $4, CX
	DECQ DX
	JNZ scalar_loop_add
	
done_add:
	VZEROUPPER
	RET

// vectorMulASM performs vectorized multiplication using AVX2
// func vectorMulASM(a, b, result []float32)
TEXT ·vectorMulASM(SB), NOSPLIT, $0-72
	MOVQ a_base+0(FP), AX
	MOVQ b_base+24(FP), BX
	MOVQ result_base+48(FP), CX
	MOVQ a_len+8(FP), DX
	
	CMPQ DX, $8
	JL scalar_mul
	
	SUBQ $8, DX
	
loop_mul:
	VMOVUPS (AX), Y0
	VMOVUPS (BX), Y1
	VMULPS Y0, Y1, Y2
	VMOVUPS Y2, (CX)
	
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	SUBQ $8, DX
	JGE loop_mul
	
	ADDQ $8, DX
	
scalar_mul:
	CMPQ DX, $0
	JLE done_mul
	
scalar_loop_mul:
	MOVSS (AX), X0
	MOVSS (BX), X1
	MULSS X0, X1
	MOVSS X1, (CX)
	
	ADDQ $4, AX
	ADDQ $4, BX
	ADDQ $4, CX
	DECQ DX
	JNZ scalar_loop_mul
	
done_mul:
	VZEROUPPER
	RET

// vectorDotASM computes dot product using AVX2
// func vectorDotASM(a, b []float32) float32
TEXT ·vectorDotASM(SB), NOSPLIT, $0-52
	MOVQ a_base+0(FP), AX
	MOVQ b_base+24(FP), BX
	MOVQ a_len+8(FP), DX
	
	VXORPS Y0, Y0, Y0  // accumulator
	
	CMPQ DX, $8
	JL scalar_dot
	
	SUBQ $8, DX
	
loop_dot:
	VMOVUPS (AX), Y1
	VMOVUPS (BX), Y2
	VFMADD231PS Y1, Y2, Y0
	
	ADDQ $32, AX
	ADDQ $32, BX
	SUBQ $8, DX
	JGE loop_dot
	
	ADDQ $8, DX
	
	// Horizontal sum of Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X0, X1, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	
scalar_dot:
	CMPQ DX, $0
	JLE done_dot
	
scalar_loop_dot:
	MOVSS (AX), X1
	MOVSS (BX), X2
	MULSS X1, X2
	ADDSS X2, X0
	
	ADDQ $4, AX
	ADDQ $4, BX
	DECQ DX
	JNZ scalar_loop_dot
	
done_dot:
	MOVSS X0, ret+48(FP)
	VZEROUPPER
	RET

// Simplified stubs for other functions
TEXT ·matMulASM(SB), NOSPLIT, $0-64
	RET

TEXT ·axpyASM(SB), NOSPLIT, $0-52
	RET

TEXT ·gemvASM(SB), NOSPLIT, $0-76
	RET
