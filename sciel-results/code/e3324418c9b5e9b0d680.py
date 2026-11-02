import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 115

    def build_schoolbook(n, m, p):
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    col = i * m * p + j * p + k
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
        return U, V, W

    def verify_fast(U_mat, V_mat, W_mat, n, m, p, tol=1e-8):
        """
        Vectorized verification using matrix multiplication.
        The tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        We need T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
        and T[a,b,c] = 0 for all invalid (a,b,c).
        
        We construct the full tensor via einsum and check.
        """
        U_mat = np.asarray(U_mat, dtype=float)
        V_mat = np.asarray(V_mat, dtype=float)
        W_mat = np.asarray(W_mat, dtype=float)
        r = U_mat.shape[1]
        if U_mat.shape[0] != n * m or V_mat.shape[0] != m * p or W_mat.shape[0] != p * n:
            return False
        if U_mat.shape[1] != r or V_mat.shape[1] != r or W_mat.shape[1] != r:
            return False
        
        # Compute T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Use einsum: 'ar,br,cr->abc'
        T = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
        
        # Check required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1) > tol:
                        return False
        
        # Check all other entries are 0
        # Create a mask of valid entries
        valid_mask = np.zeros((n * m, m * p, p * n), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid_mask[a, b, c] = True
        
        # All entries not in valid_mask must be ~0
        invalid_entries = T[~valid_mask]
        if invalid_entries.size > 0 and np.max(np.abs(invalid_entries)) > tol:
            return False
        
        return True

    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-8):
        """Wrapper that uses fast verification."""
        return verify_fast(U_mat, V_mat, W_mat, n, m, p, tol)

    # Build schoolbook for the full 3x3x7 tensor
    U, V, W = build_schoolbook(n, m, p)
    
    # Verify schoolbook
    if not verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook

    def find_decomposition(n, m, p, target_rank, max_time=10, seed=42):
        """
        Try to find a rank-target_rank decomposition for n x m * p multiplication.
        Uses fast vectorized verification and batch column removal.
        Returns (U, V, W, rank) or None.
        """
        start = time.time()
        rng = np.random.RandomState(seed)
        
        best_rank = n * m * p
        best_U, best_V, best_W = None, None, None
        
        max_stuck = 200  # Restart if we fail this many random removal attempts in a row
        batch_size = 3   # Try removing this many columns at once
        
        while time.time() - start < max_time:
            # Restart with schoolbook
            U_cur, V_cur, W_cur = build_schoolbook(n, m, p)
            cur_rank = n * m * p
            
            stuck_counter = 0
            
            while cur_rank > target_rank and time.time() - start < max_time:
                if stuck_counter > max_stuck:
                    break  # Restart outer loop
                
                # Try to remove a batch of random columns
                n_to_remove = min(batch_size, cur_rank - target_rank)
                cols_to_try = rng.choice(cur_rank, size=n_to_remove, replace=False)
                
                # Create mask: keep all except cols_to_try
                mask = np.ones(cur_rank, dtype=bool)
                mask[cols_to_try] = False
                
                U_cand = U_cur[:, mask]
                V_cand = V_cur[:, mask]
                W_cand = W_cur[:, mask]
                
                if verify_fast(U_cand, V_cand, W_cand, n, m, p):
                    U_cur = U_cand
                    V_cur = V_cand
                    W_cur = W_cand
                    cur_rank -= n_to_remove
                    stuck_counter = 0
                    # Update best if this is better
                    if cur_rank < best_rank:
                        best_rank = cur_rank
                        best_U = U_cur.copy()
                        best_V = V_cur.copy()
                        best_W = W_cur.copy()
                else:
                    # Try removing just one column at a time
                    removed_any = False
                    for col in cols_to_try:
                        mask_single = np.ones(cur_rank, dtype=bool)
                        mask_single[col] = False
                        U_cand = U_cur[:, mask_single]
                        V_cand = V_cur[:, mask_single]
                        W_cand = W_cur[:, mask_single]
                        if verify_fast(U_cand, V_cand, W_cand, n, m, p):
                            U_cur = U_cand
                            V_cur = V_cand
                            W_cur = W_cand
                            cur_rank -= 1
                            stuck_counter = 0
                            removed_any = True
                            if cur_rank < best_rank:
                                best_rank = cur_rank
                                best_U = U_cur.copy()
                                best_V = V_cur.copy()
                                best_W = W_cur.copy()
                            break
                    if not removed_any:
                        stuck_counter += 1
                    
            if cur_rank <= target_rank:
                break
        
        if best_U is not None:
            return best_U, best_V, best_W, best_rank
        return None

    # Let's try to find a decomposition for 3x3x4
    # We will allocate 70 seconds for this search.
    t_start = time.time()
    result_334 = find_decomposition(3, 3, 4, 30, max_time=70, seed=12345)
    t_end = time.time()
    
    if result_334:
        U4, V4, W4, rank_334 = result_334
        print(f"Found 3x3x4 decomposition with rank {rank_334}")
    else:
        # Fall back to schoolbook
        U4, V4, W4 = build_schoolbook(3, 3, 4)
        rank_334 = 36
        print("Failed to find better 3x3x4 decomposition, using schoolbook")

    # Now we need a 3x3x3 decomposition.
    # We know rank 23 is possible.
    # Let's try to find it or use a known one.
    # If we can't find it quickly, use schoolbook (rank 27).
    
    # Let's try to find a 3x3x3 decomposition with rank 23.
    # Allocate 35 seconds.
    t_start = time.time()
    result_333 = find_decomposition(3, 3, 3, 23, max_time=35, seed=67890)
    t_end = time.time()
    
    if result_333:
        U3, V3, W3, rank_333 = result_333
        print(f"Found 3x3x3 decomposition with rank {rank_333}")
    else:
        # Fall back to schoolbook
        U3, V3, W3 = build_schoolbook(3, 3, 3)
        rank_333 = 27
        print("Failed to find better 3x3x3 decomposition, using schoolbook")

    # Combine the decompositions.
    # The 3x3x7 tensor is the direct sum of 3x3x3 (k=0,1,2) and 3x3x4 (k=3,4,5,6).
    # We need to construct the full U, V, W matrices.
    
    total_rank = rank_333 + rank_334
    
    U_full = np.zeros((3 * 3, total_rank), dtype=float)
    V_full = np.zeros((3 * 7, total_rank), dtype=float)
    W_full = np.zeros((7 * 3, total_rank), dtype=float)
    
    # Copy 3x3x3 part (k=0,1,2)
    U_full[:, :rank_333] = U3
    # V3 corresponds to j in 0..2, k in 0..2.
    # In V_full, the rows are indexed by j*p + k.
    # For the 3x3x3 part, k is 0,1,2.
    # So rows 0..8 of V_full (j=0, k=0..2; j=1, k=0..2; j=2, k=0..2) correspond to V3.
    V_full[0:9, :rank_333] = V3
    # W3 corresponds to k in 0..2, i in 0..2.
    # In W_full, the rows are indexed by k*n + i.
    # For k=0,1,2, rows are 0..8.
    W_full[0:9, :rank_333] = W3
    
    # Copy 3x3x4 part (k=3,4,5,6)
    U_full[:, rank_333:] = U4
    
    # V4 corresponds to j in 0..2, k in 0..3 (local index).
    # In the full tensor, k is 3,4,5,6.
    # So rows in V_full are j*7 + k.
    for j in range(3):
        for l in range(4):
            k_full = 3 + l
            row_v4 = j * 4 + l
            row_v_full = j * 7 + k_full
            V_full[row_v_full, rank_333:] = V4[row_v4, :]
            
    # W4 corresponds to k_local in 0..3, i in 0..2.
    # In the full tensor, k is 3,4,5,6.
    # So rows in W_full are k*n + i.
    for l in range(4):
        k_full = 3 + l
        for i in range(3):
            row_w4 = l * 3 + i
            row_w_full = k_full * 3 + i
            W_full[row_w_full, rank_333:] = W4[row_w4, :]

    # Verify the full decomposition
    if verify(U_full, V_full, W_full, 3, 3, 7):
        best_U, best_V, best_W = U_full, V_full, W_full
        best_rank = total_rank
    else:
        # Fallback to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = rank_schoolbook

    # Convert to half-integers if possible
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-9):
            return (rounded / 2).tolist()
        return mat.tolist()

    return {"U": to_half_integer(best_U), "V": to_half_integer(best_V), "W": to_half_integer(best_W)}
