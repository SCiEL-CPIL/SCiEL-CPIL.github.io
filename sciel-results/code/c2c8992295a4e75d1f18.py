import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 8  # A is n x m, B is m x p, C is n x p

    # We use a block decomposition:
    # A (3x4) = [A1 | A2] where A1 is 3x2, A2 is 3x2
    # B (4x8) = [[B11, B12, B13, B14], [B21, B22, B23, B24]] where each B_ij is 2x2
    #
    # C (3x8) = A1*B1 + A2*B2 where B1 = [B11 B12 B13 B14] (2x8) and B2 = [B21 B22 B23 B24] (2x8)
    #
    # C = A1 * B1 + A2 * B2
    # Each term: (3x2) * (2x8)
    #
    # We can further decompose. Let's use Strassen for the 2x2 blocks where possible.
    #
    # Actually, let me think about this differently.
    #
    # M_{3,4,8} = M_{3,2,8} ⊕ M_{3,2,8} (splitting the middle dimension)
    # Each M_{3,2,8} can be further decomposed.
    #
    # M_{3,2,8}: A is 3x2, B is 2x8
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # This is a sum of 2 terms, each a rank-1 outer product.
    # So M_{3,2,8} has rank at most 2*3*8/2 = 24 (schoolbook: 3*2*8 = 48)
    # Wait, schoolbook for M_{3,2,8} is 3*2*8 = 48.
    #
    # But we can do better. M_{3,2,8} = M_{3,2,1} ⊗ M_{1,1,8}
    # M_{3,2,1}: C[i] = sum_j A[i,j]*B[j], rank = 3*2*1 = 6 schoolbook
    # M_{1,1,8}: rank 8
    # Tensor product rank: 6*8 = 48. Same.
    #
    # Alternative: M_{3,2,8} = M_{1,2,8} ⊗ M_{3,1,1}
    # M_{1,2,8}: rank 16 schoolbook
    # M_{3,1,1}: rank 3
    # Total: 48. Same.
    #
    # Let me try a different approach. Use the fact that we can combine terms.
    #
    # For M_{3,4,8}, schoolbook is 96.
    #
    # Let me try to use a rank-7 decomposition for 2x2*2x2 if I can find one.
    #
    # Actually, I recall that the explicit rank-7 tensor decomposition for 2x2*2x2 
    # is known. Let me look it up from memory.
    #
    # The 7 products in Strassen's algorithm are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(g+h)  -- wait, this isn't right either
    #
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a-b)(g+h)
    # M4 = (c-d)(e+f)  -- no
    #
    # Let me be more careful. For A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    # C[0,0] = ae + cg
    # C[0,1] = af + dh
    # C[1,0] = ce + dg
    # C[1,1] = cf + dh
    #
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a-b)(g+h)
    # M4 = c(f-h)
    # M5 = d(g-e)
    # M6 = (a+c)(e-g)
    # M7 = (b+d)(h-f)
    #
    # Then:
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M1 - M2 + M5 + M7
    # C[1,0] = M2 - M1 + M4 + M6
    # C[1,1] = M3 + M1 - M4 - M6
    #
    # Wait, let me verify:
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # M4 = c(f-h) = cf - ch
    # M5 = d(g-e) = dg - de
    # M7 = (b+d)(h-f) = bh - bf + dh - df
    #
    # C[0,0] = M1 + M4 - M5 - M7
    # = ae + ah + de + dh + cf - ch - dg + de - bh + bf - dh + df
    # = ae + ah + 2de + cf - ch - dg - bh + bf + df
    # That's not ae + cg. This is wrong.
    #
    # Let me use the correct Strassen formulas:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)(e + f)
    # M3 = (a - b)(g + h)
    # M4 = c(f - h)
    # M5 = d(g - e)
    # M6 = (a + c)(e - g)
    # M7 = (b + d)(h - f)
    #
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M1 - M2 + M5 + M7
    # C[1,0] = M2 - M1 + M4 + M6
    # C[1,1] = M3 + M1 - M4 - M6
    #
    # Let me verify C[0,0]:
    # M1 = ae + ah + de + dh
    # M4 = cf - ch
    # M5 = dg - de
    # M7 = bh - bf + dh - df
    #
    # M1 + M4 - M5 - M7 = ae + ah + de + dh + cf - ch - dg + de - bh + bf - dh + df
    # = ae + ah + 2de + cf - ch - dg - bh + bf + df
    #
    # This should equal ae + cg. It doesn't. So my formula is wrong.
    #
    # Let me look up the correct Strassen algorithm:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a-b)(g+h)
    # M4 = c(f-h)
    # M5 = d(g-e)
    # M6 = (a+c)(e-g)
    # M7 = (b+d)(h-f)
    #
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M1 - M2 + M5 + M7
    # C[1,0] = M2 - M1 + M4 + M6
    # C[1,1] = M3 + M1 - M4 - M6
    #
    # Hmm, let me just verify with a simple example.
    # A = [[1,0],[0,1]], B = [[1,0],[0,1]]
    # C should be [[1,0],[0,1]]
    #
    # a=1, b=0, c=0, d=1, e=1, f=0, g=0, h=1
    # M1 = (1+1)(1+1) = 4
    # M2 = (0+1)(1+0) = 1
    # M3 = (1-0)(0+1) = 1
    # M4 = 0*(0-1) = 0
    # M5 = 1*(0-1) = -1
    # M6 = (1+0)(1-0) = 1
    # M7 = (0+1)(1-0) = 1
    #
    # C[0,0] = M1 + M4 - M5 - M7 = 4 + 0 - (-1) - 1 = 4 ✓
    # C[0,1] = M1 - M2 + M5 + M7 = 4 - 1 + (-1) + 1 = 3 ✗ (should be 0)
    #
    # So this is definitely wrong. Let me find the correct formulas.
    #
    # Correct Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a-b)(g+h)
    # M4 = c(f-h)
    # M5 = d(g-e)
    # M6 = (a+c)(e-g)
    # M7 = (b+d)(h-f)
    #
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M1 - M2 + M5 + M7
    # C[1,0] = M2 - M1 + M4 + M6
    # C[1,1] = M3 + M1 - M4 - M6
    #
    # Let me try another example. A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    #
    # a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8
    # M1 = (1+4)(5+8) = 5*13 = 65
    # M2 = (3+4)(5+6) = 7*11 = 77
    # M3 = (1-2)(7+8) = -1*15 = -15
    # M4 = 3*(6-8) = 3*(-2) = -6
    # M5 = 4*(7-5) = 4*2 = 8
    # M6 = (1+3)(5-7) = 4*(-2) = -8
    # M7 = (2+4)(8-6) = 6*2 = 12
    #
    # C[0,0] = M1 + M4 - M5 - M7 = 65 + (-6) - 8 - 12 = 39 ✗ (should be 19)
    #
    # This is clearly wrong. I'm mixing up the Strassen formulas.
    #
    # Let me just use schoolbook for the 2x2 blocks to be safe. The rank will be 96.
    #
    # Actually, let me try a completely different approach. Let me just do schoolbook
    # but see if I can find any savings.
    
    rank = n * m * p  # 96
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        raise TimeoutError("Too slow")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
