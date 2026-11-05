import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 5, 5, 5
    
    # Block sizes: 5 = 3 + 2
    block_sizes = [3, 2]
    
    # Block offsets
    offsets = [0, 3, 5]
    
    # Rank for each block size combination
    # For a×b×c block: if a=b=c=2, use Strassen (rank 7)
    # For a×b×c block: use best known or schoolbook
    # 3×3×3: use rank 23 (from known results) or schoolbook 27
    # 3×3×2, 3×2×3, 2×3×3, etc: use schoolbook or known results
    # 3×2×2: 12, 2×3×2: 12, 2×2×3: 12
    # 2×2×2: 7 (Strassen)
    # 3×3×3: 27 (schoolbook, no known better simple decomposition)
    # 3×3×2: 18, 3×2×3: 18, 2×3×3: 18
    # 3×2×2: 12, 2×3×2: 12, 2×2×3: 12
    # 2×2×2: 7
    
    def block_rank(a, b, c):
        if a == 2 and b == 2 and c == 2:
            return 7
        return a * b * c  # Schoolbook fallback
    
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total_rank += block_rank(block_sizes[i], block_sizes[j], block_sizes[k])
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if time.time() > deadline:
                    break
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen decomposition
                    strassen_factors = [
                        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
                        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
                        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
                        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
                        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
                        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
                    ]
                    for (uf, vf, wf) in strassen_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1),
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1),
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 1),
                        ]
                        
                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx]][col] += wf[idx]
                        col += 1
                else:
                    # Use schoolbook decomposition
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
    def _verify_exact(n, m, p, U, V, W):
        rank = len(U[0]) if U else 0
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    u_row = U_frac[i * m + j]
                    v_row = V_frac[j * p + k]
                    w_row = W_frac[k * n + i]
                    for c in range(rank):
                        s += u_row[c] * v_row[c] * w_row[c]
                    if s != 1:
                        return False
        return True
    
    if _verify_exact(n, m, p, U, V, W):
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
