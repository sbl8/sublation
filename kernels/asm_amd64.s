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

// func axpyASM(alpha float32, x, y []float32)
// Y = alpha * X + Y
// Frame size: alpha (4B, pad to 8B) + x_slice (24B) + y_slice (24B) = 56 bytes
TEXT ·axpyASM(SB), NOSPLIT, $0-56
    VMOVSS alpha+0(FP), X0       // Load alpha into X0
    VBROADCASTSS X0, Y0          // Broadcast alpha to Y0 (all 8 floats in YMM0)

    MOVQ x_base+8(FP), SI       // SI = pointer to x.Data
    MOVQ y_base+32(FP), DI      // DI = pointer to y.Data
    MOVQ x_len+16(FP), CX       // CX = n (length of vector x, must be same as y.Len)

    CMPQ CX, $0
    JE   axpy_done              // If n == 0, nothing to do

    // Main AVX loop: process 8 floats (32 bytes) at a time
    MOVQ CX, DX                 // DX = loop counter for AVX part
    ANDQ $~7, DX                // DX = n - (n % 8), number of elements for AVX loop
    JZ   axpy_scalar_prologue   // If n < 8, jump to scalar loop

axpy_loop:
    VMOVUPS (SI), Y1            // Y1 = x[i:i+7]
    VMULPS  Y0, Y1, Y2          // Y2 = alpha * x[i:i+7]
    VMOVUPS (DI), Y3            // Y3 = y[i:i+7]
    VADDPS  Y2, Y3, Y3          // Y3 = (alpha * x[i:i+7]) + y[i:i+7]
    VMOVUPS Y3, (DI)            // y[i:i+7] = Y3 (store result back to y)

    ADDQ $32, SI                // Advance x pointer by 8 floats
    ADDQ $32, DI                // Advance y pointer by 8 floats
    SUBQ $8, DX                 // Decrement AVX loop counter
    JNZ  axpy_loop

axpy_scalar_prologue:
    MOVQ CX, DX                 // DX = n
    ANDQ $7, DX                 // DX = n % 8 (number of remaining elements for scalar loop)
    CMPQ DX, $0
    JE   axpy_done              // If n % 8 == 0, all done

axpy_scalar_loop:
    VMOVSS (SI), X1             // X1 = x[i] (scalar)
    VMULSS X0, X1, X2           // X2 = alpha * x[i]
    VMOVSS (DI), X3             // X3 = y[i]
    VADDSS X2, X3, X3           // X3 = (alpha * x[i]) + y[i]
    VMOVSS X3, (DI)             // y[i] = X3

    ADDQ $4, SI                 // Advance x pointer by 1 float
    ADDQ $4, DI                 // Advance y pointer by 1 float
    DECQ DX                     // Decrement scalar loop counter
    JNZ  axpy_scalar_loop

axpy_done:
    VZEROUPPER                  // Zero upper bits of YMM registers
    RET

// func matMulASM(a []float32, aRows, aCols int, b []float32, bCols int, result []float32)
// result = a * b
// Frame size: a_slice (24B) + aRows (8B) + aCols (8B) + b_slice (24B) + bCols (8B) + result_slice (24B) = 96 bytes
TEXT ·matMulASM(SB), NOSPLIT, $0-96
    MOVQ a_base+0(FP), R8           // R8 = pointer to a.Data
    MOVQ aRows+24(FP), R9           // R9 = M (aRows)
    MOVQ aCols+32(FP), R10          // R10 = K (aCols)
    MOVQ b_base+40(FP), R11          // R11 = pointer to b.Data
    MOVQ bCols+64(FP), R12          // R12 = N (bCols)
    MOVQ result_base+72(FP), R13     // R13 = pointer to result.Data

    // Handle zero dimensions to prevent crashes
    CMPQ R9, $0 // M == 0
    JE matMul_done
    CMPQ R10, $0 // K == 0
    JE matMul_done_zero_result // if K is 0, result is all zeros
    CMPQ R12, $0 // N == 0
    JE matMul_done

    XORQ DI, DI                   // DI = i (row index for A and result), 0 to M-1
matMul_loop_i:
    CMPQ DI, R9                    // if i >= M (aRows)
    JGE  matMul_done

    // DX = pointer to result[i][0] = result.Data + i * N * 4
    MOVQ DI, AX                   // AX = i
    IMULQ R12, AX                  // AX = i * N (element offset)
    SHLQ $2, AX                    // AX = (i * N) * 4 (byte offset)
    LEAQ (R13)(AX*1), DX            // DX = result.Data + byte_offset (Corrected scaling for LEAQ)

    // R14 = pointer to A[i][0] = a.Data + i * K * 4
    MOVQ DI, AX                   // AX = i
    IMULQ R10, AX                  // AX = i * K (element offset)
    SHLQ $2, AX                    // AX = (i * K) * 4 (byte offset)
    LEAQ (R8)(AX*1), R14             // R14 = a.Data + byte_offset (Corrected scaling for LEAQ)

    XORQ SI, SI                   // SI = j (col index for B and result), 0 to N-1
matMul_loop_j_avx:
    MOVQ R12, BP                   // BP = N (bCols)
    SUBQ SI, BP                   // BP = N - j (remaining columns in current row of result)
    CMPQ BP, $8
    JL   matMul_loop_j_scalar_prologue // If < 8 columns left, handle scalar

    VXORPS Y0, Y0, Y0               // Y0 accumulates sums for result[i][j:j+7]

    XORQ BX, BX                   // BX = k (common dimension index), 0 to K-1
matMul_loop_k_avx:
    CMPQ BX, R10                   // if k >= K (aCols)
    JGE  matMul_store_avx_results

    // Load A[i][k] and broadcast
    VMOVSS (R14)(BX*4), X4        // X4 = A[i][k] (R14 is &A[i][0])
    VBROADCASTSS X4, Y4             // Y4 = {A[i][k], ..., A[i][k]}

    // Load B[k][j:j+7]
    // CX = pointer to B[k][0] = b.Data + k * N * 4
    MOVQ BX, AX
    IMULQ R12, AX                  // AX = k * N
    LEAQ (R11)(AX*4), CX          // CX = &B[k][0]
    VMOVUPS (CX)(SI*4), Y5       // Y5 = B[k][j:j+7] (SI is j)

    VFMADD231PS Y4, Y5, Y0          // Y0 += Y4 * Y5 (vectorized multiply-add)

    INCQ BX                        // k++
    JMP  matMul_loop_k_avx

matMul_store_avx_results:
    VMOVUPS Y0, (DX)(SI*4)       // Store result[i][j:j+7] (DX is &result[i][0])
    ADDQ $8, SI                    // j += 8
    JMP  matMul_loop_j_avx

matMul_loop_j_scalar_prologue:
    CMPQ BP, $0                    // BP = remaining columns for scalar part
    JE   matMul_next_i

matMul_loop_j_scalar:
    VXORPS X0, X0, X0               // X0 (scalar part of Y0) for sum of result[i][j]

    XORQ BX, BX                   // k = 0
matMul_loop_k_scalar:
    CMPQ BX, R10                   // if k >= K (aCols)
    JGE  matMul_store_scalar_result

    VMOVSS (R14)(BX*4), X1        // X1 = A[i][k]

    // Pointer to B[k][j] = b.Data + (k * N + j) * 4
    MOVQ BX, AX
    IMULQ R12, AX                  // AX = k * N
    ADDQ SI, AX                   // AX = k * N + j
    VMOVSS (R11)(AX*4), X2        // X2 = B[k][j]

    VFMADD231SS X1, X2, X0          // X0 += X1 * X2 (scalar multiply-add)

    INCQ BX                        // k++
    JMP  matMul_loop_k_scalar

matMul_store_scalar_result:
    VMOVSS X0, (DX)(SI*4)        // Store result[i][j]
    INCQ SI                        // j++
    DECQ BP
    JNZ  matMul_loop_j_scalar

matMul_next_i:
    INCQ DI                        // i++
    JMP  matMul_loop_i

matMul_done_zero_result:
    // If K=0, result matrix should be all zeros.
    // R13 = result.Data, R9 = M (aRows), R12 = N (bCols)
    // Total elements = M * N
    MOVQ R9, AX
    IMULQ R12, AX // AX = M * N (total elements in result)
    CMPQ AX, $0
    JE matMul_done // If M*N is 0, nothing to zero out

    MOVQ R13, DI // DI = pointer to result
    VXORPS X0, X0, X0 // X0 = 0.0
    MOVQ AX, CX // CX = loop counter (number of floats)

    // Check if we can use AVX to zero out (8 floats at a time)
    MOVQ CX, DX
    ANDQ $~7, DX // Number of elements for AVX loop
    JZ matMul_zero_scalar_loop_prologue

    VXORPS Y0, Y0, Y0 // Y0 = {0.0, ...}
matMul_zero_avx_loop:
    VMOVUPS Y0, (DI)
    ADDQ $32, DI
    SUBQ $8, DX
    JNZ matMul_zero_avx_loop

matMul_zero_scalar_loop_prologue:
    ANDQ $7, CX // Remaining elements
    CMPQ CX, $0
    JE matMul_done

matMul_zero_scalar_loop:
    MOVSS   X0, (DI)
    ADDQ $4, DI
    DECQ CX
    JNZ matMul_zero_scalar_loop

matMul_done:
    VZEROUPPER
    RET
// func gemvASM(alpha float32, a []float32, rows, cols int, x []float32, beta float32, y []float32)
// Y = alpha * A * X + beta * Y
// Frame size: alpha (8B) + a_slice (24B) + rows (8B) + cols (8B) + x_slice (24B) + beta (8B) + y_slice (24B) = 104 bytes
TEXT ·gemvASM(SB), NOSPLIT, $0-104
    VMOVSS alpha+0(FP), X0          // X0 = alpha
    VBROADCASTSS X0, Y0             // Y0 = {alpha, ..., alpha}

    MOVQ a_base+8(FP), R8           // R8 = pointer to a.Data
    MOVQ rows+32(FP), R9            // R9 = M (rows)
    MOVQ cols+40(FP), R10           // R10 = N (cols)
    MOVQ x_base+48(FP), R11          // R11 = pointer to x.Data

    VMOVSS beta+72(FP), X1          // X1 = beta
    VBROADCASTSS X1, Y1             // Y1 = {beta, ..., beta}
    MOVQ y_base+80(FP), R12          // R12 = pointer to y.Data

    // Handle M=0 or N=0 cases
    CMPQ R9, $0 // M == 0
    JE gemv_done
    // If N (cols) is 0, dot product is 0. y = beta * y
    CMPQ R10, $0 // N == 0
    JE gemv_N_is_zero


    XORQ DI, DI                   // DI = i (row index for A and y), 0 to M-1
gemv_loop_i:
    CMPQ DI, R9                    // if i >= M (rows)
    JGE  gemv_done

    // SI = pointer to A[i][0] = a.Data + i * N * 4
    MOVQ DI, AX
    IMULQ R10, AX                  // AX = i * N
    LEAQ (R8)(AX*4), SI          // SI = &A[i][0]

    // Dot product: sum_dot = sum(A[i][j] * x[j]) for j = 0 to N-1
    VXORPS Y2, Y2, Y2               // Y2 accumulates dot product vector sum
    MOVQ R11, BX                   // BX = x.Data pointer
    XORQ CX, CX                   // CX = j (column index for dot product)

gemv_loop_j_avx:
    MOVQ R10, DX                   // DX = N (cols)
    SUBQ CX, DX                   // DX = N - j (remaining elements for dot product)
    CMPQ DX, $8
    JL   gemv_j_avx_sum_prologue    // If < 8 elements, finish AVX part and sum

    VMOVUPS (SI)(CX*4), Y3       // Y3 = A[i][j:j+7]
    VMOVUPS (BX)(CX*4), Y4       // Y4 = x[j:j+7]
    VFMADD231PS Y3, Y4, Y2          // Y2 += Y3 * Y4

    ADDQ $8, CX                    // j += 8
    JMP  gemv_loop_j_avx

gemv_j_avx_sum_prologue:
    // Horizontal sum of Y2 to get scalar sum_vec in X5 (low part of Y5)
    VEXTRACTF128 $1, Y2, X3         // X3 = Y2[4-7] (high 128 bits)
                                    // X2 implicitly has Y2[0-3] (low 128 bits)
    VADDPS X2, X3, X2               // X2 = [y0+y4, y1+y5, y2+y6, y3+y7]
    VHADDPS X2, X2, X2              // X2[0] = (y0+y4)+(y1+y5), X2[1]=(y2+y6)+(y3+y7)
    VHADDPS X2, X2, X5              // X5[0] = ((y0+y4)+(y1+y5)) + ((y2+y6)+(y3+y7))
                                    // X5 now holds sum of 8 partial products from Y2

    // Scalar part for remaining elements of dot product (N % 8)
gemv_loop_j_scalar:
    CMPQ CX, R10                   // if j >= N (cols)
    JGE  gemv_j_scalar_done

    VMOVSS (SI)(CX*4), X3        // X3 = A[i][j]
    VMOVSS (BX)(CX*4), X4        // X4 = x[j]
    VFMADD231SS X3, X4, X5          // X5 += X3 * X4 (accumulate into scalar sum)

    INCQ CX                        // j++
    JMP  gemv_loop_j_scalar

gemv_j_scalar_done:
    // X5 contains sum_dot = sum(A[i][j] * x[j])
    // Now calculate y[i] = alpha * sum_dot + beta * y[i]
    // X0 is alpha (scalar), X1 is beta (scalar)
    VMULSS X0, X5, X5               // X5 = alpha * sum_dot

    // Load y[i]
    MOVQ DI, AX                   // AX = i
    VMOVSS (R12)(AX*4), X6        // X6 = y[i] (R12 is y.Data)

    VFMADD231SS X1, X6, X5          // X5 = beta * y[i] + (alpha * sum_dot)
                                    // X5 = X1*X6 + X5 (Fused Multiply Add: X5 = X1*X6 + X5)

    VMOVSS X5, (R12)(AX*4)        // Store y[i] = X5

    INCQ DI                        // i++
    JMP  gemv_loop_i

gemv_N_is_zero: // Handle N=0 case: y = beta * y
    // R9 = M (rows), R12 = y.Data, X1 = beta (scalar)
    // Y1 = broadcasted beta
    XORQ DI, DI // DI = i = 0
gemv_N_is_zero_loop_i:
    CMPQ DI, R9 // if i >= M
    JGE gemv_done

    // Scalar loop for y[i] = beta * y
    // AVX version for y = beta * y
    MOVQ R9, CX // CX = M (number of elements in y)
    SUBQ DI, CX // Remaining elements
    CMPQ CX, $8
    JL gemv_N_is_zero_scalar_prologue

gemv_N_is_zero_avx_loop:
    VMOVUPS (R12)(DI*4), Y2 // Y2 = y[i:i+7]
    VMULPS Y1, Y2, Y2       // Y2 = beta * y[i:i+7]
    VMOVUPS Y2, (R12)(DI*4) // Store updated y
    ADDQ $8, DI
    SUBQ $8, CX
    CMPQ CX, $8
    JGE gemv_N_is_zero_avx_loop

gemv_N_is_zero_scalar_prologue:
    CMPQ CX, $0
    JE gemv_done

gemv_N_is_zero_scalar_loop:
    VMOVSS (R12)(DI*4), X2      // X2 = y[i] (R12 is y.Data, DI is i)
    VMULSS X1, X2, X2           // X2 = beta * y[i] (X1 is beta)
    VMOVSS X2, (R12)(DI*4)      // y[i] = X2
    ADDQ $4, DI                 // Advance y pointer by 1 float (increment i effectively)
    DECQ CX                     // Decrement scalar loop counter (remaining elements)
    JNZ gemv_N_is_zero_scalar_loop

    JMP gemv_done // Corrected jump

gemv_done:
    VZEROUPPER
    RET
