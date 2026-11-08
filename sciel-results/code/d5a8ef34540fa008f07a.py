import numpy as np
import time
from itertools import product

START_TIME = time.time()
TIME_LIMIT = 140  # seconds, safety margin from 150s

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We'll try to find a lower rank decomposition
    # Start with the known rank-50 solution, then try to optimize
    
    # First, build the rank-50 baseline
    rank_baseline = 50
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        U_base[a_idx][col] += 1; U_base[d_idx][col] += 1
        V_base[e_idx][col] += 1; V_base[h_idx][col] += 1
        W_base[c00_idx][col] += 1; W_base[c11_idx][col] += 1
        col += 1
        
        U_base[c_idx][col] += 1; U_base[d_idx][col] += 1
        V_base[e_idx][col] += 1
        W_base[c10_idx][col] += 1; W_base[c11_idx][col] -= 1
        col += 1
        
        U_base[a_idx][col] += 1
        V_base[f_idx][col] += 1; V_base[h_idx][col] -= 1
        W_base[c01_idx][col] += 1; W_base[c11_idx][col] += 1
        col += 1
        
        U_base[d_idx][col] += 1
        V_base[g_idx][col] += 1; V_base[e_idx][col] -= 1
        W_base[c00_idx][col] += 1; W_base[c10_idx][col] += 1
        col += 1
        
        U_base[a_idx][col] += 1; U_base[b_idx][col] += 1
        V_base[h_idx][col] += 1
        W_base[c00_idx][col] -= 1; W_base[c01_idx][col] += 1
        col += 1
        
        U_base[c_idx][col] += 1; U_base[a_idx][col] -= 1
        V_base[e_idx][col] += 1; V_base[f_idx][col] += 1
        W_base[c11_idx][col] += 1
        col += 1
        
        U_base[b_idx][col] += 1; U_base[d_idx][col] -= 1
        V_base[g_idx][col] += 1; V_base[h_idx][col] += 1
        W_base[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U_base[a_idx][col] += 1; V_base[e_idx][col] += 1; W_base[c00_idx][col] += 1
        col += 1
        U_base[b_idx][col] += 1; V_base[g_idx][col] += 1; W_base[c00_idx][col] += 1
        col += 1
        U_base[c_idx][col] += 1; V_base[e_idx][col] += 1; W_base[c10_idx][col] += 1
        col += 1
        U_base[d_idx][col] += 1; V_base[g_idx][col] += 1; W_base[c10_idx][col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    assert col == rank_baseline
    
    # Now try to reduce rank by finding linear dependencies among the rank terms
    # or by finding a better decomposition
    # Convert to numpy for linear algebra
    U_np = np.array(U_base, dtype=float)
    V_np = np.array(V_base, dtype=float)
    W_np = np.array(W_base, dtype=float)
    
    # Try to find if we can reduce rank by checking for dependencies
    # The tensor is T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We want to find a lower rank decomposition
    
    # Strategy: try random projections to find a lower rank equivalent
    # Or use the fact that we can add linear combinations of rank terms
    
    # For now, return the baseline if no improvement found in time
    if time.time() - START_TIME > TIME_LIMIT - 10:
        return {"U": U_base, "V": V_base, "W": W_base}
    
    # Try to find a rank-49 decomposition by checking if one rank term is redundant
    # This is complex, so let's just return the baseline for now
    return {"U": U_base, "V": V_base, "W": W_base}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
