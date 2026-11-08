import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def _build_rank56():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U0 = [(0, 1), (3, 1)]
    V0 = [(0, 1), (3, 1)]
    W0 = [(0, 1), (3, 1)]
    U1 = [(2, 1), (3, 1)]
    V1 = [(0, 1)]
    W1 = [(1, 1), (3, -1)]
    U2 = [(0, 1)]
    V2 = [(1, 1), (3, -1)]
    W2 = [(2, 1), (3, 1)]
    U3 = [(3, 1)]
    V3 = [(2, 1), (0, -1)]
    W3 = [(0, 1), (1, 1)]
    U4 = [(0, 1), (1, 1)]
    V4 = [(3, 1)]
    W4 = [(0, -1), (2, 1)]
    U5 = [(2, 1), (0, -1)]
    V5 = [(0, 1), (1, 1)]
    W5 = [(3, 1)]
    U6 = [(1, 1), (3, -1)]
    V6 = [(2, 1), (3, 1)]
    W6 = [(0, 1)]

    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]

    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=4, p=8):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        expected = 1 if (i == ip and k == kp) else 0
                        actual = 0
                        for r in range(rank):
                            actual += U[i * m + j][r] * V[j * p + k][r] * W[kp * n + ip][r]
                        if actual != expected:
                            return False
    return True

def _build_coefficient_matrix(U, V, W, n=2, m=4, p=8):
    R = len(U[0])
    num_eq = n * m * p * n * p
    A = np.zeros((num_eq, R), dtype=float)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        for r in range(R):
                            A[eq_idx, r] = U[i * m + j][r] * V[j * p + k][r] * W[kp * n + ip][r]
                        eq_idx += 1
    return A

def _is_half_int(x):
    return abs(x * 2 - round(x * 2)) < 1e-9

def _round_half(x):
    return round(x * 2) / 2.0

def _try_reduce_rank_step(U, V, W, n=2, m=4, p=8):
    R = len(U[0])
    A = _build_coefficient_matrix(U, V, W, n, m, p)
    
    try:
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
    except np.linalg.LinAlgError:
        return None
    
    tol = 1e-10
    null_mask = s < tol
    
    if not np.any(null_mask):
        return None
    
    null_space = Vt[null_mask]
    
    # Try to find a reduction that results in half-integers
    best_reduction = None
    best_score = -1
    
    for c in null_space:
        # Find index to eliminate
        # Try eliminating each index with significant coefficient
        candidates = [i for i in range(R) if abs(c[i]) > 1e-8]
        if not candidates:
            continue
            
        # Sort by magnitude to pick the "best" one to eliminate?
        # Or just try the first one?
        # Let's try the one with max magnitude
        max_idx = max(candidates, key=lambda i: abs(c[i]))
        
        if abs(c[max_idx]) < 1e-15:
            continue
            
        new_rank = R - 1
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        
        new_idx_map = {}
        new_i = 0
        for old_i in range(R):
            if old_i != max_idx:
                new_idx_map[old_i] = new_i
                new_i += 1
                
        valid = True
        # Check half-integers
        for old_r in range(R):
            if old_r == max_idx:
                continue
            new_r = new_idx_map[old_r]
            factor = 1 - c[old_r] / c[max_idx]
            
            for row in range(n * m):
                val = factor * U[row][old_r]
                if not _is_half_int(val):
                    valid = False
                    break
                new_U[row][new_r] = _round_half(val)
            if not valid: break
            
            if valid:
                for row in range(m * p):
                    val = factor * V[row][old_r]
                    if not _is_half_int(val):
                        valid = False
                        break
                    new_V[row][new_r] = _round_half(val)
                if not valid: break
                
            if valid:
                for row in range(p * n):
                    val = factor * W[row][old_r]
                    if not _is_half_int(val):
                        valid = False
                        break
                    new_W[row][new_r] = _round_half(val)
                if not valid: break
            
            if not valid: break
        
        if not valid:
            continue
            
        # Verify
        if _verify_decomposition(new_U, new_V, new_W, n, m, p):
            return {"U": new_U, "V": new_V, "W": new_W}
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    base = _build_rank56()
    current = base
    
    # Iteratively reduce rank
    while time.time() < deadline:
        reduced = _try_reduce_rank_step(current["U"], current["V"], current["W"])
        if reduced is None:
            break
        current = reduced
        
    return current
