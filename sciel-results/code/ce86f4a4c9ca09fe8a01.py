import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 130.0

    # Rank-56 Strassen decomposition built from 8 independent 2x2x2 blocks
    rank_56 = 56
    U = [[0] * rank_56 for _ in range(n * m)]
    V = [[0] * rank_56 for _ in range(m * p)]
    W = [[0] * rank_56 for _ in range(p * n)]

    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]

    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * m + gj][cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk][cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * n + gi][cr] += val

    # 8 blocks: A is 2x4, B is 4x8
    # A: i_base=0, j_base in {0,2}
    # B: j_base in {0,2}, k_base in {0,2,4,6}
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def verify(U_arr, V_arr, W_arr, rank_val):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vals = U_arr[i*m+j, :rank_val]
                    v_vals = V_arr[j*p+k, :rank_val]
                    w_vals = W_arr[k*n+i, :rank_val]
                    s = np.dot(u_vals, v_vals * w_vals)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U_np, V_np, W_np, rank_56):
        return {"U": U, "V": V, "W": W}

    # Try to find a rank-55 decomposition by checking for linear dependencies
    # We test if removing each column r still yields a valid decomposition
    # This is a greedy approach: check each column independently
    # If any single column can be removed, we have rank 55
    # For a more thorough search, we could try pairs, but that's O(56^2) = 3136 checks
    # Each check is O(n*m*p) = 64 operations, so total ~200k ops, very fast
    
    best_rank = rank_56
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    # Try removing each single column
    for r in range(rank_56):
        if time.time() > deadline - 10:
            break
        
        # Create copies with column r removed
        mask = np.ones(rank_56, dtype=bool)
        mask[r] = False
        U_test = U_np[:, mask]
        V_test = V_np[:, mask]
        W_test = W_np[:, mask]
        
        if verify(U_test, V_test, W_test, rank_56 - 1):
            best_rank = rank_56 - 1
            best_U = U_test
            best_V = V_test
            best_W = W_test
            break
    
    if best_rank < rank_56:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Try pairs of columns to remove (more expensive but still feasible)
    # Only do this if we have time
    if time.time() < deadline - 20:
        for r1 in range(rank_56):
            for r2 in range(r1 + 1, rank_56):
                if time.time() > deadline - 10:
                    break
                mask = np.ones(rank_56, dtype=bool)
                mask[r1] = False
                mask[r2] = False
                U_test = U_np[:, mask]
                V_test = V_np[:, mask]
                W_test = W_np[:, mask]
                
                if verify(U_test, V_test, W_test, rank_56 - 2):
                    best_rank = rank_56 - 2
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    break
            if best_rank < rank_56:
                break
    
    if best_rank < rank_56:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Return the rank-56 solution
    return {
        "U": U_np.tolist(),
        "V": V_np.tolist(),
        "W": W_np.tolist()
    }
