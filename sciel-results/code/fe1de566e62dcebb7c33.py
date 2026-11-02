"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 110s as safety margin
    safe_deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen's algorithm for 4 independent 2x2 blocks.
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # U (A): 7 linear combinations
    # 0: A00 + A11
    # 1: A10 + A11
    # 2: A00
    # 3: A11
    # 4: A00 + A01
    # 5: A10 - A00
    # 6: A01 - A11
    
    U_defs = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    
    # V (B block): 7 linear combinations for each block
    # 0: B00 + B11
    # 1: B00
    # 2: B01 - B11
    # 3: B10 - B00
    # 4: B11
    # 5: B00 + B01
    # 6: B10 + B11
    
    V_defs = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    
    # W (C block): 7 linear combinations
    # 0: C00 + C11
    # 1: C10 - C11
    # 2: C01 + C11
    # 3: C00 + C10
    # 4: -C00 + C01
    # 5: C11
    # 6: C00
    
    W_defs = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            
            # Fill U
            for idx, val in U_defs[s]:
                U[idx][r] = val
                
            # Fill V
            # Local B index mapping:
            # 0: j=0, k=2q
            # 1: j=0, k=2q+1
            # 2: j=1, k=2q
            # 3: j=1, k=2q+1
            for local_idx, val in V_defs[s]:
                j = local_idx // 2
                k_local = local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
                
            # Fill W
            # Local C index mapping (trace convention W[k*n+i, r]):
            # 0: k=2q, i=0
            # 1: k=2q, i=1
            # 2: k=2q+1, i=0
            # 3: k=2q+1, i=1
            for local_idx, val in W_defs[s]:
                k_local = local_idx // 2
                i = local_idx % 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    # Verify the decomposition is correct
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        
        verified = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over r: U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    total = 0.0
                    for r in range(rank):
                        total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                    if abs(total - 1.0) > 1e-10:
                        verified = False
                        break
                if not verified:
                    break
            if not verified:
                break
        
        if not verified:
            # Fallback: return a valid but potentially higher-rank decomposition
            # Use schoolbook: rank = n*m*p = 32
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U_fb[u_idx][idx] = 1
                        V_fb[v_idx][idx] = 1
                        W_fb[w_idx][idx] = 1
                        idx += 1
            
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
