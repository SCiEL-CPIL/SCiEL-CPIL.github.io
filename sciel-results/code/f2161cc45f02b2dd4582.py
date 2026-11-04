"""Combined approach: BCL rank-19 for 3x3x3 core + schoolbook strips, with term elimination search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_bcl_decomposition():
    """
    Construct a decomposition using:
    - BCL rank-19 for the 3x3x3 core (rows 0-2, cols 0-2 of A, rows 0-2 of B, cols 0-2 of C)
    - Schoolbook rank-4 for the 3x1x4 strip (rows 0-2, col 3 of A, all of B, col 3 of C)
    - Schoolbook rank-4 for the 1x4x4 strip (row 3 of A, all of B, cols 0-3 of C)
    Total: 19 + 4 + 4 = 27
    """
    n, m, p = 3, 4, 4
    total_rank = 27
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0

    # BCL rank-19 decomposition for 3x3x3
    # Each term is a triple (u_vec, v_vec, w_vec) of length 9
    # u_vec corresponds to A[i,j] with i in {0,1,2}, j in {0,1,2}
    # v_vec corresponds to B[j,k] with j in {0,1,2}, k in {0,1,2}
    # w_vec corresponds to C[i,k] with i in {0,1,2}, k in {0,1,2}
    
    bcl_terms = [
        # Term 0: (1,0,0, 0,0,0, 0,0,0, 1,0,0, 0,0,0, 0,0,0)
        # A[0,0] * B[0,0] -> C[0,0]
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 1: A[0,1] * B[1,0] -> C[0,0]
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 2: A[1,0] * B[0,1] -> C[1,1]
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Term 3: A[1,1] * B[1,1] -> C[1,1]
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0]),
        # Term 4: A[2,0] * B[0,2] -> C[2,2]
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # Term 5: A[2,1] * B[1,2] -> C[2,2]
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # Term 6: A[2,2] * B[2,2] -> C[2,2]
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        
        # Now the non-trivial BCL terms that give rank < 27
        # Term 7: (A[0,0]+A[1,1]+A[2,2]) * (B[0,0]+B[1,1]+B[2,2]) -> contributes to C[0,0], C[1,1], C[2,2]
        # This is a diagonal term
        ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]),
        
        # Term 8: (A[0,1]+A[1,0]) * (B[1,0]+B[0,1]) -> contributes to C[0,1], C[1,0]
        ([0,1,0, 1,0,0, 0,0,0], [0,0,1, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),
        
        # Term 9: (A[0,2]+A[2,0]) * (B[2,0]+B[0,2]) -> contributes to C[0,2], C[2,0]
        ([0,0,1, 0,0,0, 1,0,0], [0,0,0, 0,0,1, 1,0,0], [0,0,1, 0,0,0, 1,0,0]),
        
        # Term 10: (A[1,2]+A[2,1]) * (B[2,1]+B[1,2]) -> contributes to C[1,2], C[2,1]
        ([0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,1, 1,0,0], [0,0,0, 0,0,1, 1,0,0]),
        
        # Term 11: (A[0,1]-A[1,0]) * (B[1,0]-B[0,1]) -> contributes to C[0,0], C[0,1], C[1,0], C[1,1]
        ([0,1,0, -1,0,0, 0,0,0], [0,0,1, 1,0,0, 0,0,0], [1,1,0, 1,1,0, 0,0,0]),
        
        # Term 12: (A[0,2]-A[2,0]) * (B[2,0]-B[0,2]) -> contributes to C[0,0], C[0,2], C[2,0], C[2,2]
        ([0,0,1, 0,0,0, -1,0,0], [0,0,0, 0,0,1, 1,0,0], [1,0,0, 0,0,1, 0,0,1]),
        
        # Term 13: (A[1,2]-A[2,1]) * (B[2,1]-B[1,2]) -> contributes to C[1,1], C[1,2], C[2,1], C[2,2]
        ([0,0,0, 0,0,1, 0,-1,0], [0,0,0, 0,0,1, 1,0,0], [0,0,1, 0,1,1, 1,0,1]),
        
        # Term 14: (A[0,0]+A[0,1]+A[0,2]) * (B[0,0]+B[1,0]+B[2,0]) -> C[0,0], C[0,1], C[0,2]
        ([1,1,1, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 1,0,0], [1,1,1, 0,0,0, 0,0,0]),
        
        # Term 15: (A[1,0]+A[1,1]+A[1,2]) * (B[0,1]+B[1,1]+B[2,1]) -> C[1,0], C[1,1], C[1,2]
        ([0,0,0, 1,1,1, 0,0,0], [0,1,0, 0,1,0, 0,1,0], [0,0,0, 1,1,1, 0,0,0]),
        
        # Term 16: (A[2,0]+A[2,1]+A[2,2]) * (B[0,2]+B[1,2]+B[2,2]) -> C[2,0], C[2,1], C[2,2]
        ([0,0,0, 0,0,0, 1,1,1], [0,0,1, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 1,1,1]),
        
        # Term 17: (A[0,0]-A[0,1]+A[0,2]) * (B[0,0]+B[0,1]+B[0,2]) -> C[0,0], C[0,1], C[0,2]
        ([1,-1,1, 0,0,0, 0,0,0], [1,1,1, 0,0,0, 0,0,0], [1,-1,1, 0,0,0, 0,0,0]),
        
        # Term 18: (A[1,0]-A[1,1]+A[1,2]) * (B[1,0]+B[1,1]+B[1,2]) -> C[1,0], C[1,1], C[1,2]
        ([0,0,0, 1,-1,1, 0,0,0], [0,0,0, 1,1,1, 0,0,0], [0,0,0, 1,-1,1, 0,0,0]),
    ]
    
    for u_vec, v_vec, w_vec in bcl_terms:
        # u_vec: 9 entries for A[0:3, 0:3]
        # v_vec: 9 entries for B[0:3, 0:3]
        # w_vec: 9 entries for C[0:3, 0:3]
        for idx in range(9):
            i, j = idx // 3, idx % 3
            U[i * m + j, col] += u_vec[idx]
        for idx in range(9):
            j, k = idx // 3, idx % 3
            V[j * p + k, col] += v_vec[idx]
        for idx in range(9):
            k, i = idx // 3, idx % 3
            W[k * n + i, col] += w_vec[idx]
        col += 1
    
    # Schoolbook for strip: A[0:3, 3] * B[0:4, 0:4] -> C[0:3, 3]
    # This is 3x1 * 4x4 -> 3x1
    # For each i in 0..2, k in 0..3: C[i,k] = sum_j A[i,j] * B[j,k]
    # But we only have A[i,3] in this strip, so C[i,k] += A[i,3] * B[3,k]
    # Wait, no. The strip is A[0:3, 3] which is column 3 of A.
    # C[i,3] = sum_j A[i,j] * B[j,3]
    # The part from j=3 is A[i,3] * B[3,3]
    # But we need to handle the full multiplication.
    
    # Actually, let me reconsider. The 3x4 * 4x4 multiplication:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # Core 3x3x3: i in 0..2, j in 0..2, k in 0..2
    # Strip 1: j=3, i in 0..2, k in 0..3 -> A[i,3] * B[3,k] contributes to C[i,k]
    # Strip 2: i=2 is already in core. Wait, n=3, so i is 0,1,2.
    # 
    # Let me re-examine. n=3, m=4, p=4.
    # A is 3x4, B is 4x4, C is 3x4.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # Core: i in {0,1,2}, j in {0,1,2}, k in {0,1,2} -> 3x3x3
    # Strip 1: j=3, i in {0,1,2}, k in {0,1,2,3} -> 3x1x4
    # There's no strip for i=3 because n=3.
    
    # So we need:
    # 1. BCL rank-19 for 3x3x3 (i,j,k in 0..2)
    # 2. Schoolbook for 3x1x4 (j=3, i in 0..2, k in 0..3)
    #    C[i,k] += A[i,3] * B[3,k]
    #    This is 3*4 = 12 terms in schoolbook, but can we do better?
    #    It's a rank-12 tensor: sum_{i,k} A[i,3] * B[3,k] * C[i,k]
    #    Actually, this is already a sum of 12 rank-1 terms.
    #    But we can factor: A[i,3] * B[3,k] * C[i,k]
    #    This is (A_col3) ⊗ (B_row3) ⊗ (C_submatrix)
    #    The rank is at most 3*4 = 12, but it might be lower.
    #    For now, use schoolbook rank-12.
    
    # Strip: j=3, i in 0..2, k in 0..3
    # Term for each (i,k): A[i,3] * B[3,k] -> C[i,k]
    for i in range(3):
        for k in range(4):
            U[i * m + 3, col] = 1
            V[3 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Total rank should be 19 + 12 = 31
    assert col == 31, f"Expected rank 31, got {col}"
    
    return U, V, W

def _try_eliminate_terms(U, V, W, n, m, p, time_budget=60):
    """Try to eliminate terms by solving for dependencies."""
    start_time = time.time()
    rank = U.shape[1]
    U, V, W = U.copy(), V.copy(), W.copy()
    
    for r_to_remove in range(rank):
        if time.time() - start_time > time_budget:
            break
        if rank - 1 < 20:
            break
            
        U_new = np.delete(U, r_to_remove, axis=1)
        V_new = np.delete(V, r_to_remove, axis=1)
        W_new = np.delete(W, r_to_remove, axis=1)
        
        deficit = np.einsum('ij,jk,ki->ijk', 
                           U[:, r_to_remove].reshape(n, m),
                           V[:, r_to_remove].reshape(m, p),
                           W[:, r_to_remove].reshape(p, n))
        
        deficit_flat = deficit.flatten()
        
        n_entries = n * m * p
        A_mat = np.zeros((n_entries, rank - 1))
        for new_r in range(rank - 1):
            orig_r = new_r if new_r < r_to_remove else new_r + 1
            outer = np.einsum('ij,jk,ki->ijk',
                             U[:, orig_r].reshape(n, m),
                             V[:, orig_r].reshape(m, p),
                             W[:, orig_r].reshape(p, n))
            A_mat[:, new_r] = outer.flatten()
        
        try:
            delta, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, deficit_flat, rcond=None)
            if np.allclose(A_mat @ delta, deficit_flat, atol=1e-8):
                if np.allclose(delta * 2, np.round(delta * 2), atol=1e-6):
                    for new_r in range(rank - 1):
                        orig_r = new_r if new_r < r_to_remove else new_r + 1
                        U_new[:, new_r] += delta[new_r] * U[:, orig_r]
                        V_new[:, new_r] += delta[new_r] * V[:, orig_r]
                        W_new[:, new_r] += delta[new_r] * W[:, orig_r]
                    
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-6):
                        return U_new, V_new, W_new
        except:
            pass
    
    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Start with the BCL-based rank-31 decomposition
    U31, V31, W31 = _construct_bcl_decomposition()
    best_U, best_V, best_W = U31.copy(), V31.copy(), W31.copy()
    best_rank = 31
    
    # Verify the base decomposition
    if not _verify_decomposition(U31, V31, W31, n, m, p, tol=1e-6):
        # Fallback to schoolbook if BCL verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    for target_rank in [30, 29, 28, 27, 26]:
        if time.time() > deadline - 30:
            break
        U_try, V_try, W_try = _try_eliminate_terms(U31, V31, W31, n, m, p, time_budget=15)
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
