"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We decompose the 2x3x8 multiplication into 4 independent 2x3x2 sub-problems.
    # For each 2x3x2 sub-problem, we use a rank-6 decomposition.
    # Total rank = 4 * 6 = 24.
    
    # The 2x3x2 tensor T has indices (i, j, k) with i in {0,1}, j in {0,1,2}, k in {0,1}.
    # We want to find U (6x6), V (6x6), W (4x6) such that T = sum_r U[:,r] V[:,r] W[:,r]^T.
    # Actually, the tensor is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
    # Wait, the convention is W[k*n+i, r]. So W is indexed by (k, i).
    # U is indexed by (i, j).
    # V is indexed by (j, k).
    
    # Let's try to find a rank-6 decomposition for 2x3x2.
    # Is rank 6 possible?
    # M(2,3,2) >= M(2,2,2) = 7? No, that's not a direct bound.
    # M(2,3,2) >= 2*3 = 6 (from the structure of the tensor, each of the 6 entries of A needs to be multiplied).
    # Actually, the tensor rank of 2x3x2 is known to be 12? Or 9?
    # Let's check if rank 6 is possible.
    # If rank 6 is possible, it would be a significant improvement.
    # However, I am not confident that rank 6 is possible for 2x3x2.
    # The schoolbook rank is 12.
    # Strassen-like methods might reduce it to 9 or 10.
    # Let's try to use a known rank-9 decomposition for 2x3x2 if possible, or just use schoolbook.
    
    # Actually, let's look at the geometry.
    # A is 2x3, B is 3x2.
    # C = A*B is 2x2.
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # Let's try to find a rank-9 decomposition.
    # One known rank-9 decomposition for 2x3x2:
    # M1 = (a00+a10)*(b00+b10)
    # M2 = (a01+a11)*(b01+b11)
    # M3 = (a02+a12)*(b20+b21)
    # M4 = (a00-a10)*b00
    # M5 = (a01-a11)*b01
    # M6 = (a02-a12)*b20
    # M7 = a00*b10
    # M8 = a01*b11
    # M9 = a02*b21
    
    # Let's verify this.
    # C[0,0] = M1 - M4 + M7 - M7? No.
    # M1 = a00*b00 + a00*b10 + a10*b00 + a10*b10
    # M4 = a00*b00 - a10*b00
    # M7 = a00*b10
    # M1 - M4 - M7 = a00*b10 + a10*b00 + a10*b10 - a00*b00 + a10*b00 - a00*b10 = a10*b00 + a10*b10 - a00*b00? No.
    
    # Let's use a more systematic approach.
    # I will use the schoolbook decomposition for 2x3x2 (rank 12) to be safe.
    # Total rank = 4 * 12 = 48.
    
    # Wait, the directive says "yielding a total rank of 24".
    # This implies rank 6 per block.
    # Is rank 6 possible for 2x3x2?
    # The tensor rank of 2x3x2 is actually 12. (I think)
    # Let's check: M(2,3,2) >= M(2,3,1) * 2 / ... no.
    # M(2,3,2) >= 2*3 = 6.
    # M(2,3,2) >= 2*2 = 4.
    # M(2,3,2) >= 3*2 = 6.
    # So rank 6 is a lower bound.
    # Is it achievable?
    # If we can find 6 multiplications, it would be great.
    # Let's try to construct it.
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # Let's assume rank 6 is possible and try to find the coefficients.
    # I will use a linear algebra solver to find U, V, W.
    # But I can't use a solver in the final code if it's not guaranteed to work fast.
    # I will hardcode a known decomposition.
    
    # Actually, I recall that M(2,3,2) = 12.
    # So rank 6 is likely not possible.
    # The directive might be wrong about rank 6.
    # However, I must follow the directive's approach: "decomposing ... into four independent 2x3x2 sub-problems".
    # If I use rank 12 per block, total rank is 48.
    # If I can find a rank 9 decomposition, total rank is 36.
    # Let's try to find a rank 9 decomposition.
    
    # I will use the following rank-9 decomposition for 2x3x2:
    # M1 = (a00+a10)*(b00+b10)
    # M2 = (a01+a11)*(b01+b11)
    # M3 = (a02+a12)*(b20+b21)
    # M4 = (a00-a10)*b00
    # M5 = (a01-a11)*b01
    # M6 = (a02-a12)*b20
    # M7 = a00*b10
    # M8 = a01*b11
    # M9 = a02*b21
    
    # Let's check if this works.
    # C[0,0] = M1 - M4 - M7 + M7? No.
    # Let's compute C[0,0] using these.
    # M1 = a00*b00 + a00*b10 + a10*b00 + a10*b10
    # M4 = a00*b00 - a10*b00
    # M7 = a00*b10
    # M1 - M4 - M7 = a00*b00 + a00*b10 + a10*b00 + a10*b10 - a00*b00 + a10*b00 - a00*b10 = a10*b00 + a10*b10 - a00*b00? No.
    
    # Let's try a different set.
    # I will use the schoolbook decomposition to be safe.
    # Total rank = 48.
    
    # Wait, I can try to use a rank-10 decomposition?
    # I don't have a verified rank-10 or rank-9 decomposition.
    # I will stick to schoolbook.
    
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 48
    V = [[0] * rank for _ in range(m * p)]  # 24 x 48
    W = [[0] * rank for _ in range(p * n)]  # 16 x 48
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
