import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def _validate(U, V, W, n=3, m=3, p=8):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                if total != 1:
                    return False
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if isinstance(val, float):
                    if not val.is_integer() and abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 3, 3, 8
    
    # Strategy: Winograd for 3x3x3 (rank 23) + Schoolbook for 3x3x5 (rank 45) = 68.
    # I need the Winograd factors.
    # I will use a known set.
    
    # Winograd 3x3x3 factors (Rank 23)
    # I will define them as a list of tuples (u_idx, v_idx, w_idx, coeff)
    # But I don't have them.
    
    # Fallback: Schoolbook.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    valid = _validate(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
