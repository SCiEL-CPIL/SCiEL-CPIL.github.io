"""
Strategy: Randomized Algebraic Search over Z2.
Instead of constructing a fixed Strassen decomposition, we randomly generate
candidate rank-r decompositions over Z2 (where Strassen's 7-term algorithm
is known to be minimal for 2x2), lift to Q, and verify exactness against
the 4x4x8 matrix multiplication tensor. This explores a different solution
space than recursive block-Strassen and may find lower-rank decompositions
that exploit the non-square structure.
"""

import time
import numpy as np
import random

DIMENSIONS = [4, 4, 8]

def _get_target_tensor():
    """Construct the target matrix multiplication tensor T[a,b,c]."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition, rank = n*m*p = 128."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_k U[:,k] V[:,k] W[:,k] == T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _random_z2_decomposition(n, m, p, rank, rng):
    """
    Generate a random rank-r decomposition over Z2.
    U, V, W are binary matrices of shapes (n*m, r), (m*p, r), (p*n, r).
    """
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)
    
    # Random binary entries
    for r in range(rank):
        U[:, r] = rng.getrandbits(n*m * rank) if False else np.array([rng.randint(0,1) for _ in range(n*m)], dtype=np.int8)
        V[:, r] = np.array([rng.randint(0,1) for _ in range(m*p)], dtype=np.int8)
        W[:, r] = np.array([rng.randint(0,1) for _ in range(p*n)], dtype=np.int8)
    
    return U, V, W

def _verify_z2_decomposition(U, V, W, n, m, p):
    """
    Verify decomposition over Z2: for all (i,j,k),
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 (mod 2)
    """
    rank = U.shape[1]
    # Use integer arithmetic mod 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0
                for r in range(rank):
                    s ^= (U[a, r] & V[b, r] & W[c, r])
                if s != 1:
                    return False
    return True

def _lift_z2_to_q(U, V, W):
    """Lift Z2 decomposition to Q (entries become 0 or 1)."""
    U_q = U.astype(np.float64)
    V_q = V.astype(np.float64)
    W_q = W.astype(np.float64)
    return U_q, V_q, W_q

def _try_strassen_based_4x4x8():
    """
    Try to build a 4x4x8 decomposition using Strassen at multiple levels.
    Returns (U, V, W, rank) or None if failed.
    """
    n, m, p = 4, 4, 8
    
    # Build 2x2 Strassen terms
    terms_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Top-level Strassen for 4x4 (2x2 blocks of 2x2)
    U_top = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    V_top = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    W_top = np.zeros((7, 4))
    W_top[0, 0] += 1; W_top[3, 0] += -1; W_top[5, 0] += 1
    W_top[4, 1] += 1; W_top[6, 1] += 1
    W_top[1, 2] += 1; W_top[3, 2] += 1
    W_top[0, 3] += 1; W_top[2, 3] += -1; W_top[6, 3] += 1
    
    # Build 4x4x4 decomposition (rank 49)
    rank4 = 49
    U4 = np.zeros((16, rank4))
    V4 = np.zeros((16, rank4))
    W4 = np.zeros((16, rank4))
    
    col = 0
    for k in range(7):
        u_top_vec = U_top[k]
        v_top_vec = V_top[k]
        w_top_vec = W_top[k]
        
        for t in range(7):
            u_bot, v_bot, w_bot = terms_2x2[t]
            
            u_full = np.zeros(16)
            v_full = np.zeros(16)
            w_full = np.zeros(16)
            
            for b_idx in range(4):
                if u_top_vec[b_idx] != 0:
                    for l in range(4):
                        if u_bot[l] != 0:
                            br = b_idx // 2
                            bc = b_idx % 2
                            lr = l // 2
                            lc = l % 2
                            gr = br * 2 + lr
                            gc = bc * 2 + lc
                            gidx = gr * 4 + gc
                            u_full[gidx] += u_top_vec[b_idx] * u_bot[l]
                
                if v_top_vec[b_idx] != 0:
                    for l in range(4):
                        if v_bot[l] != 0:
                            br = b_idx // 2
                            bc = b_idx % 2
                            lr = l // 2
                            lc = l % 2
                            gr = br * 2 + lr
                            gc = bc * 2 + lc
                            gidx = gr * 4 + gc
                            v_full[gidx] += v_top_vec[b_idx] * v_bot[l]
                
                if w_top_vec[b_idx] != 0:
                    for l in range(4):
                        if w_bot[l] != 0:
                            br = b_idx // 2
                            bc = b_idx % 2
                            lr = l // 2
                            lc = l % 2
                            gr = br * 2 + lr
                            gc = bc * 2 + lc
                            gidx = gr * 4 + gc
                            w_full[gidx] += w_top_vec[b_idx] * w_bot[l]
            
            U4[:, col] = u_full
            V4[:, col] = v_full
            W4[:, col] = w_full
            col += 1
    
    # Extend to 4x4x8: two independent 4x4x4 multiplications
    total_rank = 2 * rank4
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # Left half
    U_full[:, 0:rank4] = U4
    for r in range(rank4):
        for idx in range(16):
            val = V4[idx, r]
            if val != 0:
                j = idx // 4
                k_local = idx % 4
                v_idx = j * 8 + k_local
                V_full[v_idx, r] += val
            val = W4[idx, r]
            if val != 0:
                k_local = idx // 4
                i = idx % 4
                w_idx = k_local * 4 + i
                W_full[w_idx, r] += val
    
    # Right half
    U_full[:, rank4:2*rank4] = U4
    for r in range(rank4):
        for idx in range(16):
            val = V4[idx, r]
            if val != 0:
                j = idx // 4
                k_local = idx % 4
                v_idx = j * 8 + (k_local + 4)
                V_full[v_idx, r + rank4] += val
            val = W4[idx, r]
            if val != 0:
                k_local = idx // 4
                i = idx % 4
                w_idx = (k_local + 4) * 4 + i
                W_full[w_idx, r + rank4] += val
    
    return U_full, V_full, W_full, total_rank

def _try_random_search(n, m, p, max_rank, start_rank, rng, deadline):
    """
    Try random Z2 decompositions for ranks from start_rank to max_rank.
    Returns (U, V, W, rank) if found, else None.
    """
    T = _get_target_tensor()
    
    for rank in range(start_rank, max_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random samples per rank
        max_tries = 5000 if rank <= 50 else 1000
        for attempt in range(max_tries):
            if time.time() > deadline - 5:
                break
            
            U, V, W = _random_z2_decomposition(n, m, p, rank, rng)
            
            if _verify_z2_decomposition(U, V, W, n, m, p):
                # Found a valid Z2 decomposition, lift to Q
                U_q, V_q, W_q = _lift_z2_to_q(U, V, W)
                
                # Verify over Q
                if _verify_decomposition(U_q, V_q, W_q, T):
                    return U_q, V_q, W_q, rank
        
        # Progress indicator (silent)
        print(f"  Tried rank {rank}, no solution found in {max_tries} attempts")
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    T = _get_target_tensor()
    
    best_result = None
    best_rank = n * m * p  # 128
    
    # 1. First, get the Strassen-based decomposition (rank 98)
    strassen_result = _try_strassen_based_4x4x8()
    if strassen_result is not None:
        U_s, V_s, W_s, rank_s = strassen_result
        if _verify_decomposition(U_s, V_s, W_s, T):
            best_result = (U_s, V_s, W_s)
            best_rank = rank_s
            print(f"Strassen-based decomposition: rank {rank_s}")
    
    # 2. Try random search for lower ranks
    # Start from ranks below the Strassen rank
    search_start = max(40, best_rank - 10)
    search_end = min(128, best_rank - 1)
    
    if time.time() < deadline - 10:
        rng = random.Random(42)
        random_result = _try_random_search(n, m, p, search_end, search_start, rng, deadline)
        
        if random_result is not None:
            U_r, V_r, W_r, rank_r = random_result
            if rank_r < best_rank:
                best_result = (U_r, V_r, W_r)
                best_rank = rank_r
                print(f"Random search found decomposition: rank {rank_r}")
    
    # 3. Fallback to schoolbook if nothing else worked
    if best_result is None:
        U_fb, V_fb, W_fb = _schoolbook_decomposition()
        best_result = (U_fb, V_fb, W_fb)
        best_rank = n * m * p
        print(f"Using schoolbook fallback: rank {best_rank}")
    
    U_out, V_out, W_out = best_result
    
    # Convert to half-integer representation
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        return mat.tolist()
    
    U_final = to_half_integer(U_out)
    V_final = to_half_integer(V_out)
    W_final = to_half_integer(W_out)
    
    return {"U": U_final, "V": V_final, "W": W_final}
