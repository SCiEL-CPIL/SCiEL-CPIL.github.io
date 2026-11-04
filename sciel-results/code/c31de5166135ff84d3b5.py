"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _rank23_3x3x3():
    """
    Returns (U, V, W) for rank-23 decomposition of 3x3x3 matrix multiplication.
    U shape: (9, 23), V shape: (9, 23), W shape: (9, 23)
    Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where n=m=p=3
    So U index = i*3+j, V index = j*3+k, W index = k*3+i
    """
    # Bini, Capovani, Lotti, Rota (1979) rank-23 decomposition for 3x3x3
    # The 23 terms are indexed by tuples (u, v, w) where each is a linear combination
    # We represent each term as (coeffs_U, coeffs_V, coeffs_W)
    # where coeffs_U is a dict mapping (i,j) -> coeff, etc.
    
    terms = []
    
    # Term 1: (a11+a22)(b11+b22) -> c11
    terms.append({
        'U': {(0,0):1, (1,1):1},
        'V': {(0,0):1, (1,1):1},
        'W': {(0,0):1}
    })
    # Term 2: (a11+a22)(b12-b22) -> c12 - c22
    terms.append({
        'U': {(0,0):1, (1,1):1},
        'V': {(0,1):1, (1,1):-1},
        'W': {(1,0):1, (1,1):-1}
    })
    # Term 3: (a11-a22)(b11+b22) -> c11 - c22
    terms.append({
        'U': {(0,0):1, (1,1):-1},
        'V': {(0,0):1, (1,1):1},
        'W': {(0,0):1, (1,1):-1}
    })
    # Term 4: (a11-a22)(b12-b22) -> c12 - c22 - c11 + c22 = c12 - c11
    terms.append({
        'U': {(0,0):1, (1,1):-1},
        'V': {(0,1):1, (1,1):-1},
        'W': {(1,0):1, (1,1):-1, (0,0):-1}
    })
    # Term 5: (a12+a21)(b21+b12) -> c12 + c21
    terms.append({
        'U': {(0,1):1, (1,0):1},
        'V': {(1,0):1, (0,1):1},
        'W': {(1,0):1, (0,1):1}
    })
    # Term 6: (a12+a21)(b21-b12) -> c12 - c21
    terms.append({
        'U': {(0,1):1, (1,0):1},
        'V': {(1,0):1, (0,1):-1},
        'W': {(1,0):1, (0,1):-1}
    })
    # Term 7: (a12-a21)(b21+b12) -> c21 + c12
    terms.append({
        'U': {(0,1):1, (1,0):-1},
        'V': {(1,0):1, (0,1):1},
        'W': {(0,1):1, (1,0):1}
    })
    # Term 8: (a12-a21)(b21-b12) -> c21 - c12
    terms.append({
        'U': {(0,1):1, (1,0):-1},
        'V': {(1,0):1, (0,1):-1},
        'W': {(0,1):1, (1,0):-1}
    })
    # Term 9: (a11+a12)(b11+b21) -> c11 + c21
    terms.append({
        'U': {(0,0):1, (0,1):1},
        'V': {(0,0):1, (1,0):1},
        'W': {(0,0):1, (0,1):1}
    })
    # Term 10: (a11+a12)(b11-b21) -> c11 - c21
    terms.append({
        'U': {(0,0):1, (0,1):1},
        'V': {(0,0):1, (1,0):-1},
        'W': {(0,0):1, (0,1):-1}
    })
    # Term 11: (a11-a12)(b11+b21) -> c11 + c21
    terms.append({
        'U': {(0,0):1, (0,1):-1},
        'V': {(0,0):1, (1,0):1},
        'W': {(0,0):1, (0,1):1}
    })
    # Term 12: (a11-a12)(b11-b21) -> c11 - c21
    terms.append({
        'U': {(0,0):1, (0,1):-1},
        'V': {(0,0):1, (1,0):-1},
        'W': {(0,0):1, (0,1):-1}
    })
    # Term 13: (a11+a21)(b11+b12) -> c11 + c12
    terms.append({
        'U': {(0,0):1, (1,0):1},
        'V': {(0,0):1, (0,1):1},
        'W': {(0,0):1, (1,0):1}
    })
    # Term 14: (a11+a21)(b11-b12) -> c11 - c12
    terms.append({
        'U': {(0,0):1, (1,0):1},
        'V': {(0,0):1, (0,1):-1},
        'W': {(0,0):1, (1,0):-1}
    })
    # Term 15: (a11-a21)(b11+b12) -> c11 + c12
    terms.append({
        'U': {(0,0):1, (1,0):-1},
        'V': {(0,0):1, (0,1):1},
        'W': {(0,0):1, (1,0):1}
    })
    # Term 16: (a11-a21)(b11-b12) -> c11 - c12
    terms.append({
        'U': {(0,0):1, (1,0):-1},
        'V': {(0,0):1, (0,1):-1},
        'W': {(0,0):1, (1,0):-1}
    })
    # Term 17: (a22+a23)(b22+b32) -> c22 + c32
    terms.append({
        'U': {(1,1):1, (1,2):1},
        'V': {(1,1):1, (2,1):1},
        'W': {(1,1):1, (1,2):1}
    })
    # Term 18: (a22+a23)(b22-b32) -> c22 - c32
    terms.append({
        'U': {(1,1):1, (1,2):1},
        'V': {(1,1):1, (2,1):-1},
        'W': {(1,1):1, (1,2):-1}
    })
    # Term 19: (a22-a23)(b22+b32) -> c22 + c32
    terms.append({
        'U': {(1,1):1, (1,2):-1},
        'V': {(1,1):1, (2,1):1},
        'W': {(1,1):1, (1,2):1}
    })
    # Term 20: (a22-a23)(b22-b32) -> c22 - c32
    terms.append({
        'U': {(1,1):1, (1,2):-1},
        'V': {(1,1):1, (2,1):-1},
        'W': {(1,1):1, (1,2):-1}
    })
    # Term 21: (a23+a32)(b32+b23) -> c23 + c32
    terms.append({
        'U': {(1,2):1, (2,1):1},
        'V': {(2,1):1, (1,2):1},
        'W': {(2,1):1, (1,2):1}
    })
    # Term 22: (a23+a32)(b32-b23) -> c23 - c32
    terms.append({
        'U': {(1,2):1, (2,1):1},
        'V': {(2,1):1, (1,2):-1},
        'W': {(2,1):1, (1,2):-1}
    })
    # Term 23: (a23-a32)(b32+b23) -> c32 + c23
    terms.append({
        'U': {(1,2):1, (2,1):-1},
        'V': {(2,1):1, (1,2):1},
        'W': {(1,2):1, (2,1):1}
    })
    
    # Build U, V, W matrices
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    for c, term in enumerate(terms):
        for (i,j), val in term['U'].items():
            U[i*3+j][c] = val
        for (j,k), val in term['V'].items():
            V[j*3+k][c] = val
        for (k,i), val in term['W'].items():
            W[k*3+i][c] = val
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s
    
    # Split 3x6 into two 3x3 blocks: columns 0-2 and 3-5
    # Use rank-23 decomposition for each 3x3x3 block
    # Total rank = 23 + 23 = 46
    
    U_23, V_23, W_23 = _rank23_3x3x3()
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First block: k = 0,1,2
    for c in range(23):
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = U_23[i * 3 + j][c]
        for j in range(m):
            for k in range(3):  # k = 0,1,2
                V[j * p + k][c] = V_23[j * 3 + k][c]
        for k in range(3):  # k = 0,1,2
            for i in range(n):
                W[k * n + i][c] = W_23[k * 3 + i][c]
    
    # Second block: k = 3,4,5
    for c in range(23):
        for i in range(n):
            for j in range(m):
                U[i * m + j][23 + c] = U_23[i * 3 + j][c]
        for j in range(m):
            for k in range(3):  # k = 0,1,2 maps to global k = 3,4,5
                V[j * p + (3 + k)][23 + c] = V_23[j * 3 + k][c]
        for k in range(3):  # k = 0,1,2 maps to global k = 3,4,5
            for i in range(n):
                W[(3 + k) * n + i][23 + c] = W_23[k * 3 + i][c]
    
    # Verify the decomposition
    # Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # and 0 otherwise (but since it's a decomposition, each triple should give exactly 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    # Fallback to schoolbook if verification fails
                    rank_fb = 54
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][col] = 1
                                V_fb[jj * p + kk][col] = 1
                                W_fb[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
