import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 5, 5, 5
    
    # Block sizes: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen factors for 2x2x2
    # Each factor is (u_flat, v_flat, w_flat) where flat index = row*2 + col
    strassen_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni, mj, pk = block_sizes[i], block_sizes[j], block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # Allocate using lists for exact arithmetic compatibility
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    break
                ni, mj, pk = block_sizes[i], block_sizes[j], block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        # Map local 2x2 indices to global
                        for idx in range(4):
                            lr, lc = divmod(idx, 2)
                            U_global = (offsets[i] + lr) * m + (offsets[j] + lc)
                            V_global = (offsets[j] + lr) * p + (offsets[k] + lc)
                            W_global = (offsets[k] + lr) * n + (offsets[i] + lc)
                            U[U_global][col] += uf[idx]
                            V[V_global][col] += vf[idx]
                            W[W_global][col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Verify exact decomposition
    def _verify_exact(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        s += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if s != 1:
                        return False
        return True
    
    if _verify_exact(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook (rank 125)
    U_fallback = [[0.0] * 125 for _ in range(25)]
    V_fallback = [[0.0] * 125 for _ in range(25)]
    W_fallback = [[0.0] * 125 for _ in range(25)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1.0
                V_fallback[j * p + k][col] = 1.0
                W_fallback[k * n + i][col] = 1.0
                col += 1
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
