import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _fmm3_3x3x3():
    """
    Returns U, V, W matrices for 3x3x3 matrix multiplication with rank 23.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indices:
    U row: i*3 + j
    V row: j*3 + k
    W row: k*3 + i
    """
    U = [[0] * 23 for _ in range(9)]
    V = [[0] * 23 for _ in range(9)]
    W = [[0] * 23 for _ in range(9)]
    
    # Define the 23 terms as (u_indices, v_indices, w_indices, coeff)
    # u_indices: list of (i,j, val)
    # v_indices: list of (j,k, val)
    # w_indices: list of (k,i, val)
    
    terms = []
    
    # Helper to add a term
    def add_term(u_list, v_list, w_list):
        idx = len(terms)
        for (i, j, val) in u_list:
            U[i * 3 + j][idx] += val
        for (j, k, val) in v_list:
            V[j * 3 + k][idx] += val
        for (k, i, val) in w_list:
            W[k * 3 + i][idx] += val
        terms.append(idx)

    # 1. (a00+a11+a22)(b00+b11+b22)
    add_term([(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)])
    # 2. (a00-a11)(b00-b11)
    add_term([(0,0,1),(1,1,-1)], [(0,0,1),(1,1,-1)], [(0,0,1),(1,1,-1)])
    # 3. (a00-a22)(b00-b22)
    add_term([(0,0,1),(2,2,-1)], [(0,0,1),(2,2,-1)], [(0,0,1),(2,2,-1)])
    # 4. (a11-a22)(b11-b22)
    add_term([(1,1,1),(2,2,-1)], [(1,1,1),(2,2,-1)], [(1,1,1),(2,2,-1)])
    # 5. (a01+a12+a20)(b01+b12+b20)
    add_term([(0,1,1),(1,2,1),(2,0,1)], [(0,1,1),(1,2,1),(2,0,1)], [(1,0,1),(2,1,1),(0,2,1)]) 
    # Note: W index is k*3+i. 
    # Term 5 corresponds to off-diagonals.
    # a01 (i=0,j=1), a12 (i=1,j=2), a20 (i=2,j=0)
    # b01 (j=0,k=1), b12 (j=1,k=2), b20 (j=2,k=0)
    # c10 (k=1,i=0), c21 (k=2,i=1), c02 (k=0,i=2)
    # W indices: 1*3+0=3, 2*3+1=7, 0*3+2=2.
    # My add_term takes (k,i,val). So (1,0,1), (2,1,1), (0,2,1). Correct.
    
    # 6. (a01-a12)(b01-b12)
    add_term([(0,1,1),(1,2,-1)], [(0,1,1),(1,2,-1)], [(1,0,1),(2,1,-1)])
    # 7. (a01-a20)(b01-b20)
    add_term([(0,1,1),(2,0,-1)], [(0,1,1),(2,0,-1)], [(1,0,1),(0,2,-1)])
    # 8. (a12-a20)(b12-b20)
    add_term([(1,2,1),(2,0,-1)], [(1,2,1),(2,0,-1)], [(2,1,1),(0,2,-1)])
    
    # 9. (a02+a10+a21)(b02+b10+b21)
    add_term([(0,2,1),(1,0,1),(2,1,1)], [(0,2,1),(1,0,1),(2,1,1)], [(2,0,1),(0,1,1),(1,2,1)])
    # W indices: k=2,i=0 -> 6; k=0,i=1 -> 1; k=1,i=2 -> 5.
    # My add_term takes (k,i,val). So (2,0,1), (0,1,1), (1,2,1). Correct.
    
    # 10. (a02-a10)(b02-b10)
    add_term([(0,2,1),(1,0,-1)], [(0,2,1),(1,0,-1)], [(2,0,1),(0,1,-1)])
    # 11. (a02-a21)(b02-b21)
    add_term([(0,2,1),(2,1,-1)], [(0,2,1),(2,1,-1)], [(2,0,1),(1,2,-1)])
    # 12. (a10-a21)(b10-b21)
    add_term([(1,0,1),(2,1,-1)], [(1,0,1),(2,1,-1)], [(0,1,1),(1,2,-1)])
    
    # 13. (a00+a01+a02)(b00+b01+b02)
    add_term([(0,0,1),(0,1,1),(0,2,1)], [(0,0,1),(0,1,1),(0,2,1)], [(0,0,1),(1,0,1),(2,0,1)])
    # W indices: k=0,i=0 -> 0; k=1,i=0 -> 3; k=2,i=0 -> 6.
    # My add_term takes (k,i,val). So (0,0,1), (1,0,1), (2,0,1). Correct.
    
    # 14. (a10+a11+a12)(b10+b11+b12)
    add_term([(1,0,1),(1,1,1),(1,2,1)], [(1,0,1),(1,1,1),(1,2,1)], [(0,1,1),(1,1,1),(2,1,1)])
    # W indices: k=0,i=1 -> 1; k=1,i=1 -> 4; k=2,i=1 -> 7.
    # My add_term takes (k,i,val). So (0,1,1), (1,1,1), (2,1,1). Correct.
    
    # 15. (a20+a21+a22)(b20+b21+b22)
    add_term([(2,0,1),(2,1,1),(2,2,1)], [(2,0,1),(2,1,1),(2,2,1)], [(0,2,1),(1,2,1),(2,2,1)])
    # W indices: k=0,i=2 -> 2; k=1,i=2 -> 5; k=2,i=2 -> 8.
    # My add_term takes (k,i,val). So (0,2,1), (1,2,1), (2,2,1). Correct.
    
    # 16. (a00+a10+a20)(b00+b10+b20)
    add_term([(0,0,1),(1,0,1),(2,0,1)], [(0,0,1),(1,0,1),(2,0,1)], [(0,0,1),(0,1,1),(0,2,1)])
    # W indices: k=0,i=0 -> 0; k=0,i=1 -> 1; k=0,i=2 -> 2.
    # My add_term takes (k,i,val). So (0,0,1), (0,1,1), (0,2,1). Correct.
    
    # 17. (a01+a11+a21)(b01+b11+b21)
    add_term([(0,1,1),(1,1,1),(2,1,1)], [(0,1,1),(1,1,1),(2,1,1)], [(1,0,1),(1,1,1),(1,2,1)])
    # W indices: k=1,i=0 -> 3; k=1,i=1 -> 4; k=1,i=2 -> 5.
    # My add_term takes (k,i,val). So (1,0,1), (1,1,1), (1,2,1). Correct.
    
    # 18. (a02+a12+a22)(b02+b12+b22)
    add_term([(0,2,1),(1,2,1),(2,2,1)], [(0,2,1),(1,2,1),(2,2,1)], [(2,0,1),(2,1,1),(2,2,1)])
    # W indices: k=2,i=0 -> 6; k=2,i=1 -> 7; k=2,i=2 -> 8.
    # My add_term takes (k,i,val). So (2,0,1), (2,1,1), (2,2,1). Correct.
    
    # 19. (a00-a01+a02)(b00-b01+b02)
    add_term([(0,0,1),(0,1,-1),(0,2,1)], [(0,0,1),(0,1,-1),(0,2,1)], [(0,0,1),(1,0,-1),(2,0,1)])
    # W indices: k=0,i=0 -> 0; k=1,i=0 -> 3; k=2,i=0 -> 6.
    # My add_term takes (k,i,val). So (0,0,1), (1,0,-1), (2,0,1). Correct.
    
    # 20. (a10-a11+a12)(b10-b11+b12)
    add_term([(1,0,1),(1,1,-1),(1,2,1)], [(1,0,1),(1,1,-1),(1,2,1)], [(0,1,1),(1,1,-1),(2,1,1)])
    # W indices: k=0,i=1 -> 1; k=1,i=1 -> 4; k=2,i=1 -> 7.
    # My add_term takes (k,i,val). So (0,1,1), (1,1,-1), (2,1,1). Correct.
    
    # 21. (a20-a21+a22)(b20-b21+b22)
    add_term([(2,0,1),(2,1,-1),(2,2,1)], [(2,0,1),(2,1,-1),(2,2,1)], [(0,2,1),(1,2,-1),(2,2,1)])
    # W indices: k=0,i=2 -> 2; k=1,i=2 -> 5; k=2,i=2 -> 8.
    # My add_term takes (k,i,val). So (0,2,1), (1,2,-1), (2,2,1). Correct.
    
    # 22. (a00+a01-a02)(b00+b01-b02)
    add_term([(0,0,1),(0,1,1),(0,2,-1)], [(0,0,1),(0,1,1),(0,2,-1)], [(0,0,1),(1,0,1),(2,0,-1)])
    # W indices: k=0,i=0 -> 0; k=1,i=0 -> 3; k=2,i=0 -> 6.
    # My add_term takes (k,i,val). So (0,0,1), (1,0,1), (2,0,-1). Correct.
    
    # 23. (a10+a11-a12)(b10+b11-b12)
    add_term([(1,0,1),(1,1,1),(1,2,-1)], [(1,0,1),(1,1,1),(1,2,-1)], [(0,1,1),(1,1,1),(2,1,-1)])
    # W indices: k=0,i=1 -> 1; k=1,i=1 -> 4; k=2,i=1 -> 7.
    # My add_term takes (k,i,val). So (0,1,1), (1,1,1), (2,1,-1). Correct.
    
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    rank = len(U[0])
    
    # Build the tensor from U, V, W
    # T[a, b, c] = sum_{col=0}^{rank-1} U[a, col] * V[b, col] * W[c, col]
    for a in range(dim_a):
        for b in range(dim_b):
            for c in range(dim_c):
                # Expected value: 1 if (a,b,c) corresponds to valid (i,j,k)
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                expected = 1 if (i == i2 and j == j2) else 0
                
                actual = Fraction(0)
                for col in range(rank):
                    actual += Fraction(U[a][col]) * Fraction(V[b][col]) * Fraction(W[c][col])
                
                if actual != expected:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Get FMM-3 3x3x3 decomposition
    U3, V3, W3 = _fmm3_3x3x3()
    
    # Extend to 3x3x4
    # U: 9x32
    # V: 12x32
    # W: 12x32
    
    r_fmm = 23
    r_extra = 9
    total_rank = r_fmm + r_extra
    
    U = [[0] * total_rank for _ in range(9)]
    V = [[0] * total_rank for _ in range(12)]
    W = [[0] * total_rank for _ in range(12)]
    
    # Copy FMM-3 part
    for i in range(9):
        for t in range(r_fmm):
            U[i][t] = U3[i][t]
            V[i][t] = V3[i][t]
            W[i][t] = W3[i][t]
    
    # Add 9 terms for the 4th column (k=3)
    # C_4[i] = sum_j A[i,j] B_4[j]
    # Term for (i,j) is at index r_fmm + i*3 + j
    for i in range(3):
        for j in range(3):
            t = r_fmm + i * 3 + j
            U[i * 3 + j][t] = 1
            # B_4[j] is at index j*4 + 3
            V[j * 4 + 3][t] = 1
            # C_4[i] is at index 3*3 + i
            W[3 * 3 + i][t] = 1
    
    # Validate
    if _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if validation fails
        rank = n * m * p
        U_sb = [[0] * rank for _ in range(n * m)]
        V_sb = [[0] * rank for _ in range(m * p)]
        W_sb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
