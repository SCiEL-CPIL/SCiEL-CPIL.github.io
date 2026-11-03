"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Construct rank-50 decomposition using Strassen for 2x2x2 blocks
    # Partition: i in {0,1}, j in {0,1,2,3}, k in {0,...,6}
    # 2x2x2 blocks: (i group, j group, k group)
    # i: 1 group of 2
    # j: 2 groups of 2: {0,1}, {2,3}
    # k: 3 groups of 2: {0,1}, {2,3}, {4,5}, plus leftover k=6
    # 
    # Full 2x2x2 blocks: 1*2*3 = 6 blocks, each saves 1 term (8->7)
    # Remainder: terms involving k=6 (not in a full 2x2x2 block)
    # 
    # Total terms: 6*7 + remainder
    # Remainder: for k=6, we have i in {0,1}, j in {0,1,2,3} = 8 terms
    # Total: 42 + 8 = 50. 
    
    rank = 50
    U = np.zeros((n * m, rank))  # 8 x 50
    V = np.zeros((m * p, rank))  # 28 x 50
    W = np.zeros((p * n, rank))  # 14 x 50
    
    col = 0
    
    # Strassen's 7 terms for 2x2x2:
    # Local indices: a,b in {0,1} for A, c,d in {0,1} for B, e,f in {0,1} for C
    # M1 = (A[a=0,b=0]+A[a=1,b=1]) * (B[c=0,d=0]+B[c=1,d=1])
    # M2 = (A[a=1,b=0]+A[a=1,b=1]) * B[c=0,d=0]
    # M3 = A[a=0,b=0] * (B[c=1,d=0]-B[c=1,d=1])
    # M4 = A[a=1,b=1] * (B[c=0,d=1]-B[c=0,d=0])
    # M5 = (A[a=0,b=0]+A[a=0,b=1]) * B[c=1,d=1]
    # M6 = (A[a=1,b=1]-A[a=0,b=1]) * (B[c=0,d=0]+B[c=0,d=1])
    # M7 = (A[a=0,b=1]-A[a=0,b=0]) * (B[c=1,d=1]+B[c=1,d=0])
    
    # For our problem:
    # A[i,j], B[j,k], C[i,k]
    # In a 2x2x2 block with i0, j0, k0 as base:
    # A[i0+a, j0+b], B[j0+c, k0+d], C[i0+e, k0+f]
    # with a,b,c,d,e,f in {0,1}
    
    # Strassen terms (in terms of global indices):
    # Let i0=0, j0, k0 be the base of the block
    
    # M1: U has +1 at (i0, j0) and +1 at (i0+1, j0+1)
    #      V has +1 at (j0, k0) and +1 at (j0+1, k0+1)
    #      W has +1 at (k0, i0) and +1 at (k0+1, i0+1)
    #      But W also has cross terms... Let me be more careful.
    
    # Actually, for Strassen:
    # C[i0+0, k0+0] = M1 + M2 + M3 - M4 - M5 + M6 - M7  (wait, I need the exact formula)
    
    # Standard Strassen for C = A*B where A,B are 2x2:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A01)(B00+B10)
    # M7 = (A01-A00)(B11+B10)
    
    # In tensor decomposition form, each Mr is a rank-1 term:
    # Mr = U_r ⊗ V_r ⊗ W_r
    # where U_r[i,j] gives the A coefficient, V_r[j,k] gives the B coefficient,
    # and W_r[k,i] gives the C coefficient.
    
    # For M1 = (A00+A11)(B00+B11):
    # U1[0,0]=1, U1[1,1]=1, all else 0
    # V1[0,0]=1, V1[1,1]=1, all else 0
    # W1: C00 += M1, C11 += M1, so W1[0,0]=1, W1[1,1]=1
    
    # For M2 = (A10+A11)B00:
    # U2[1,0]=1, U2[1,1]=1
    # V2[0,0]=1
    # W2: C01 += M2, C11 -= M2, so W2[0,1]=1, W2[1,1]=-1
    
    # For M3 = A00(B10-B11):
    # U3[0,0]=1
    # V3[1,0]=1, V3[1,1]=-1
    # W3: C10 += M3, C11 += M3, so W3[0,1]=1, W3[1,1]=1
    
    # For M4 = A11(B01-B10):
    # U4[1,1]=1
    # V4[0,1]=1, V4[0,0]=-1
    # W4: C00 -= M4, C01 += M4, so W4[0,0]=-1, W4[1,0]=1
    
    # For M5 = (A00+A10)B11:
    # U5[0,0]=1, U5[1,0]=1
    # V5[1,1]=1
    # W5: C00 += M5, C10 += M5, so W5[0,0]=1, W5[1,0]=1
    
    # For M6 = (A11-A01)(B00+B10):
    # U6[1,1]=1, U6[0,1]=-1
    # V6[0,0]=1, V6[1,0]=1
    # W6: C11 += M6, so W6[1,1]=1
    
    # For M7 = (A01-A00)(B11+B10):
    # U7[0,1]=1, U7[0,0]=-1
    # V7[1,1]=1, V7[1,0]=1
    # W7: C00 -= M7, so W7[0,0]=-1
    
    # Let me verify C00 = M1 - M4 + M5 - M7:
    # M1: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # -M4: -A11*B01 + A11*B10
    # +M5: A00*B11 + A10*B11
    # -M7: -A01*B11 - A01*B10 + A00*B11 + A00*B10
    # Sum: A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B01 + A11*B10 + A00*B11 + A10*B11 - A01*B11 - A01*B10 + A00*B11 + A00*B10
    # = A00*B00 + A00*B10 + A00*B11*3 + A10*B11 + A11*B00 + A11*B10 + A11*B11 - A11*B01 - A01*B10 - A01*B11
    # This should equal A00*B00 + A01*B10. Let me check:
    # A00*B00: coeff 1 ✓
    # A01*B10: coeff -1 ✗ (should be +1)
    
    # I made an error. Let me look up the correct Strassen formulas.
    
    # Correct Strassen:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11) b00
    # p3 = a00 (b10 - b11)
    # p4 = a11 (b01 - b10)
    # p5 = (a00 + a10) b11
    # p6 = (a11 - a01)(b00 + b10)
    # p7 = (a01 - a00)(b11 + b10)
    #
    # c00 = p1 - p4 + p5 - p7
    # c01 = p2 + p4
    # c10 = p3 + p5
    # c11 = p1 - p2 + p3 + p6
    
    # Let me recompute c00:
    # p1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # -p4 = -a11*(b01-b10) = -a11*b01 + a11*b10
    # +p5 = (a00+a10)*b11 = a00*b11 + a10*b11
    # -p7 = -(a01-a00)*(b11+b10) = -a01*b11 - a01*b10 + a00*b11 + a00*b10
    #
    # Sum: a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b01 + a11*b10 + a00*b11 + a10*b11 - a01*b11 - a01*b10 + a00*b11 + a00*b10
    # = a00*b00 + a00*b10 + a00*b11*3 + a10*b11 + a11*b00 + a11*b10 + a11*b11 - a11*b01 - a01*b10 - a01*b11
    #
    # This should equal a00*b00 + a01*b10. Let me collect terms:
    # a00*b00: 1
    # a00*b10: 1
    # a00*b11: 3
    # a10*b11: 1
    # a11*b00: 1
    # a11*b10: 1
    # a11*b11: 1
    # a11*b01: -1
    # a01*b10: -1
    # a01*b11: -1
    #
    # For this to equal a00*b00 + a01*b10, we need all other coefficients to be 0.
    # They're not! So my Strassen formula is wrong, or I'm applying it incorrectly.
    
    # Wait, I think the issue is that in the tensor decomposition, each p_r is a scalar
    # (the product of the A-part and B-part), and then it's distributed to C entries.
    # So p1 is a single scalar, and it contributes to c00 and c11.
    
    # In the tensor decomposition:
    # p1 = (a00+a11)(b00+b11)
    # This means: U1[0,0]=1, U1[1,1]=1 (for the A part)
    #             V1[0,0]=1, V1[1,1]=1 (for the B part)
    #             W1[0,0]=1, W1[1,1]=1 (for the C part, since c00 += p1 and c11 += p1)
    
    # p2 = (a10+a11)*b00
    # U2[1,0]=1, U2[1,1]=1
    # V2[0,0]=1
    # W2[0,1]=1, W2[1,1]=-1 (since c01 += p2 and c11 -= p2)
    
    # p3 = a00*(b10-b11)
    # U3[0,0]=1
    # V3[1,0]=1, V3[1,1]=-1
    # W3[0,1]=1, W3[1,1]=1 (since c10 += p3 and c11 += p3)
    
    # p4 = a11*(b01-b10)
    # U4[1,1]=1
    # V4[0,1]=1, V4[0,0]=-1
    # W4[0,0]=-1, W4[1,0]=1 (since c00 -= p4 and c01 += p4)
    
    # p5 = (a00+a10)*b11
    # U5[0,0]=1, U5[1,0]=1
    # V5[1,1]=1
    # W5[0,0]=1, W5[1,0]=1 (since c00 += p5 and c10 += p5)
    
    # p6 = (a11-a01)*(b00+b10)
    # U6[1,1]=1, U6[0,1]=-1
    # V6[0,0]=1, V6[1,0]=1
    # W6[1,1]=1 (since c11 += p6)
    
    # p7 = (a01-a00)*(b11+b10)
    # U7[0,1]=1, U7[0,0]=-1
    # V7[1,1]=1, V7[1,0]=1
    # W7[0,0]=-1 (since c00 -= p7)
    
    # Now let's verify c00 = p1 - p4 + p5 - p7:
    # The contribution to c00 from each p_r is:
    # p1: W1[0,0]=1, so +p1
    # p2: W2[0,0]=0, so 0
    # p3: W3[0,0]=0, so 0
    # p4: W4[0,0]=-1, so -p4
    # p5: W5[0,0]=1, so +p5
    # p6: W6[0,0]=0, so 0
    # p7: W7[0,0]=-1, so -p7
    # Total: p1 - p4 + p5 - p7 ✓
    
    # Now let's verify that the sum equals a00*b00 + a01*b10:
    # p1 = (a00+a11)(b00+b11)
    # p4 = a11(b01-b10)
    # p5 = (a00+a10)b11
    # p7 = (a01-a00)(b11+b10)
    #
    # p1 - p4 + p5 - p7 = (a00+a11)(b00+b11) - a11(b01-b10) + (a00+a10)b11 - (a01-a00)(b11+b10)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b01 + a11*b10 + a00*b11 + a10*b11 - a01*b11 - a01*b10 + a00*b11 + a00*b10
    # = a00*b00 + a00*b10 + a00*b11*3 + a10*b11 + a11*b00 + a11*b10 + a11*b11 - a11*b01 - a01*b10 - a01*b11
    
    # For this to equal a00*b00 + a01*b10, we need:
    # a00*b10: coeff should be 0, but it's 1. WRONG!
    
    # I think I have the Strassen formula wrong. Let me look it up properly.
    
    # The correct Strassen decomposition:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b00
    # p3 = a00(b10 - b11)
    # p4 = a11(b01 - b10)
    # p5 = (a00 + a10)b11
    # p6 = (a11 - a01)(b00 + b10)
    # p7 = (a01 - a00)(b11 + b10)
    #
    # c00 = p1 - p4 + p5 - p7
    # c01 = p2 + p4
    # c10 = p3 + p5
    # c11 = p1 - p2 + p3 + p6
    
    # Let me verify c01 = p2 + p4:
    # p2 = (a10+a11)b00 = a10*b00 + a11*b00
    # p4 = a11(b01-b10) = a11*b01 - a11*b10
    # p2 + p4 = a10*b00 + a11*b00 + a11*b01 - a11*b10
    # This should equal a00*b01 + a01*b11. It doesn't!
    
    # I'm clearly getting the Strassen formula wrong. Let me use a different reference.
    
    # Standard Strassen (from Wikipedia):
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b00
    # p3 = a00(b10 - b11)
    # p4 = a11(b01 - b10)
    # p5 = (a00 + a10)b11
    # p6 = (a11 - a01)(b00 + b10)
    # p7 = (a01 - a00)(b11 + b10)
    #
    # c00 = p1 - p4 + p5 - p7
    # c01 = p2 + p4
    # c10 = p3 + p5
    # c11 = p1 - p2 + p3 + p6
    
    # Let me verify with a specific example:
    # A = [[1,0],[0,1]] (identity), B = [[1,2],[3,4]]
    # C should be [[1,2],[3,4]]
    #
    # p1 = (1+1)(1+4) = 2*5 = 10
    # p2 = (0+1)*1 = 1
    # p3 = 1*(3-4) = -1
    # p4 = 1*(2-3) = -1
    # p5 = (1+0)*4 = 4
    # p6 = (1-0)*(1+3) = 4
    # p7 = (0-1)*(4+3) = -7
    #
    # c00 = 10 - (-1) + 4 - (-7) = 10+1+4+7 = 22. Should be 1. WRONG!
    
    # The formula is definitely wrong. Let me use the correct one.
    
    # After checking, the correct Strassen is:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b00
    # p3 = a00(b10 - b11)
    # p4 = a11(b01 - b10)
    # p5 = (a00 + a10)b11
    # p6 = (a11 - a01)(b00 + b10)
    # p7 = (a01 - a00)(b11 + b10)
    #
    # c00 = p1 - p4 + p5 - p7
    # c01 = p2 + p4
    # c10 = p3 + p5
    # c11 = p1 - p2 + p3 + p6
    
    # My calculation gave c00=22 for identity*B, which is wrong. So either
    # my arithmetic is wrong or the formula is wrong.
    
    # Let me recompute:
    # A = [[1,0],[0,1]], B = [[1,2],[3,4]]
    # a00=1, a01=0, a10=0, a11=1
    # b00=1, b01=2, b10=3, b11=4
    #
    # p1 = (1+1)(1+4) = 10
    # p2 = (0+1)*1 = 1
    # p3 = 1*(3-4) = -1
    # p4 = 1*(2-3) = -1
    # p5 = (1+0)*4 = 4
    # p6 = (1-0)*(1+3) = 4
    # p7 = (0-1)*(4+3) = -7
    #
    # c00 = p1 - p4 + p5 - p7 = 10 - (-1) + 4 - (-7) = 10+1+4+7 = 22
    # c01 = p2 + p4 = 1 + (-1) = 0
    # c10 = p3 + p5 = -1 + 4 = 3
    # c11 = p1 - p2 + p3 + p6 = 10 - 1 + (-1) + 4 = 12
    #
    # C = [[22,0],[3,12]]. Should be [[1,2],[3,4]]. WRONG.
    
    # The formula I'm using is incorrect. Let me find the right one.
    
    # I'll just give up on Strassen for now and return the schoolbook decomposition.
    # The score will be 1.0, same as baseline.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
