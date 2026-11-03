import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8

    # Strategy: Decompose 2x4 * 4x8 into two independent 2x2 * 2x8 blocks
    # along the intermediate dimension (m=4 -> two blocks of m=2)
    # Each 2x2 * 2x8 block can be decomposed using Kronecker product of:
    #   - Strassen 2x2 * 2x2 (rank 7)
    #   - Identity 2x1 * 1x4 (rank 2) -- actually need to think about this more carefully

    # Let me reconsider. The 2x4 * 4x8 multiplication:
    # A is 2x4, B is 4x8, C = A*B is 2x8
    # 
    # We can split the intermediate dimension m=4 into two halves:
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x8
    # C = A1*B1 + A2*B2
    #
    # So we need to compute two 2x2 * 2x8 matrix multiplications and add them.
    # 
    # For 2x2 * 2x8:
    # A is 2x2, B is 2x8, C is 2x8
    # We can further decompose this. Note that 2x2 * 2x8 = (2x2 * 2x2) for each column pair.
    # Actually, 2x8 has 4 column pairs of width 2.
    # So 2x2 * 2x8 is equivalent to 4 independent 2x2 * 2x2 multiplications? No, that's not right either.
    #
    # Let me think differently. The tensor for 2x2 * 2x8:
    # A[i,j] for i in {0,1}, j in {0,1}
    # B[j,k] for j in {0,1}, k in {0,...,7}
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # This is a tensor of size 2*2*8 = 32.
    # 
    # We can use Strassen for the 2x2 part. The 2x2 * 2x8 multiplication can be seen as:
    # For each column k of B (and C), we have a 2x2 * 2x1 multiplication.
    # But that doesn't directly help.
    #
    # Alternative: Use the fact that 2x2 * 2x8 = (2x2 * 2x2) ⊗ (1x1 * 1x4) in some sense?
    # No, that's not quite right.
    #
    # Let me try a different approach. Use Strassen on the 2x2 block structure:
    # For 2x2 * 2x8:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02, b03, b04, b05, b06, b07],
    #      [b10, b11, b12, b13, b14, b15, b16, b17]]
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # Using Strassen-like formulas, we can compute this with 7 multiplications per... 
    # Actually, for each column k, we need to compute:
    # C[0,k] = a00*b0k + a01*b1k
    # C[1,k] = a10*b0k + a11*b1k
    #
    # This is just 2x2 matrix-vector multiplication, which can be done with 4 multiplications
    # per column, or we can use Strassen if we treat it as 2x2 * 2x1.
    #
    # But 2x2 * 2x1 is just 4 multiplications (or 6 with Karatsuba-like, but that's not helpful).
    #
    # Let me try yet another approach. Use the Kronecker product idea:
    # 2x2 * 2x8 can be viewed as (2x2) * (2x8) where we apply Strassen to the 2x2 part
    # and treat the 8 columns as separate.
    #
    # Actually, the cleanest approach: 
    # 2x4 * 4x8 = sum_{s=0}^{1} (2x2 block of A) * (2x8 block of B)
    # Each term is a 2x2 * 2x8 multiplication.
    #
    # For 2x2 * 2x8, we can use the following:
    # Treat it as 4 independent 2x2 * 2x2 multiplications (one for each pair of columns).
    # Each 2x2 * 2x2 uses Strassen with rank 7.
    # So 4 * 7 = 28 per 2x2 * 2x8 block.
    # Two such blocks: 2 * 28 = 56. Same as before!
    #
    # That's not an improvement. Let me think harder.
    #
    # What if we use a different decomposition? 
    # The key insight: 2x2 * 2x8 has a tensor of size 32.
    # The rank of 2x2 * 2x8 matrix multiplication tensor...
    # 
    # Actually, I recall that for 2x2 * 2xn, the rank is 7n/2 if n is even? No, that doesn't sound right.
    #
    # Let me try to construct it explicitly using Strassen.
    # For 2x2 * 2x8:
    # A = [[a00, a01], [a10, a11]]
    # B = [B0, B1] where B0, B1 are 1x8 rows
    #
    # Strassen:
    # M1 = (a00+a11) * (B0+B1)  -- but B0+B1 is 1x8, so this is 8 scalar multiplications
    # M2 = (a10+a11) * B0       -- 8 scalar multiplications
    # M3 = a00 * (B1-B0)        -- 8 scalar multiplications
    # M4 = a11 * (B0-B1)        -- 8 scalar multiplications
    # M5 = (a00+a01) * B1       -- 8 scalar multiplications
    # M6 = (a10-a00) * (B0+B1)  -- 8 scalar multiplications
    # M7 = (a01-a11) * (B0+B1)  -- 8 scalar multiplications
    #
    # Total: 7 * 8 = 56 scalar multiplications. Same as before!
    #
    # So the Kronecker product approach doesn't help here.
    #
    # Let me try to find a genuinely different decomposition.
    #
    # What about using the fact that n=2 is small?
    # The output C is 2x8. We have 16 output entries.
    # The lower bound on rank is at least max(n*m, m*p, p*n) = max(8, 32, 16) = 32.
    #
    # Can we achieve rank 32? That would be optimal.
    #
    # Let me try to construct a rank-32 decomposition.
    #
    # For each (i,j,k) triple, we need sum_r U[i,r]*V[j,r]*W[k,r] = 1
    # where i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    #
    # If we set rank = 32, we have 32 terms.
    #
    # One idea: use the schoolbook approach but with some cleverness.
    # Schoolbook: C[i,k] = sum_j A[i,j]*B[j,k]
    # This uses 2*4*8 = 64 multiplications.
    #
    # To get to 32, we need to halve the number of multiplications.
    #
    # What if we pair up columns of B?
    # For k=0,1: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0] + A[i,3]*B[3,0]
    #           C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1] + A[i,3]*B[3,1]
    #
    # We can compute (B[0,0]+B[0,1]), (B[1,0]+B[1,1]), etc. and use Strassen-like formulas.
    #
    # Actually, let me try to use the following approach:
    # Group the 4 intermediate indices into 2 pairs: (0,1) and (2,3)
    # For each pair, use a 2x2 Strassen decomposition.
    #
    # For pair (0,1):
    # A_sub = A[:, 0:2] (2x2)
    # B_sub = B[0:2, :] (2x8)
    # C_sub = A_sub * B_sub (2x8)
    #
    # This is a 2x2 * 2x8 multiplication. As we saw, this takes 56 multiplications with Strassen.
    # Two such pairs: 2 * 56 = 112. That's worse!
    #
    # I think I need to accept that rank 56 might be close to optimal for this approach.
    #
    # Let me try a completely different strategy: use a random search or optimization to find
    # a lower-rank decomposition.
    #
    # Actually, let me try to verify if rank 56 is correct and just return it cleanly.

    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1): (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1  # C00
    W_local[3][0] = 1  # C11
    
    # r=1 (M2): (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1   # C10
    W_local[3][1] = -1  # C11
    
    # r=2 (M3): a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1   # C01
    W_local[3][2] = 1   # C11
    
    # r=3 (M4): a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1   # C00
    W_local[2][3] = 1   # C10
    
    # r=4 (M5): (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1  # C00
    W_local[1][4] = 1   # C01
    
    # r=5 (M6): (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1   # C11
    
    # r=6 (M7): (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1   # C00

    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Decompose 2x4 x 4x8 into 8 blocks of 2x2 x 2x2
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            a_map = [
                0 * 4 + (2 * s + 0),
                0 * 4 + (2 * s + 1),
                1 * 4 + (2 * s + 0),
                1 * 4 + (2 * s + 1),
            ]
            
            b_map = [
                (2 * s + 0) * 8 + (2 * j + 0),
                (2 * s + 0) * 8 + (2 * j + 1),
                (2 * s + 1) * 8 + (2 * j + 0),
                (2 * s + 1) * 8 + (2 * j + 1),
            ]
            
            c_map = [
                (2 * j + 0) * 2 + 0,
                (2 * j + 1) * 2 + 0,
                (2 * j + 0) * 2 + 1,
                (2 * j + 1) * 2 + 1,
            ]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val

    if time.time() - start_time > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
