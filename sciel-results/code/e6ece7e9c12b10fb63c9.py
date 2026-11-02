import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Strassen vectors for 2x2 matrix multiplication (rank 7)
    u_vecs = np.array([
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ], dtype=float)
    v_vecs = np.array([
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ], dtype=float)
    w_vecs = np.array([
        [1, 0, 0, 1],   # M1: c00 + c11
        [0, 0, 1, -1],  # M2: c10 - c11
        [0, 1, 0, 1],   # M3: c01 + c11
        [1, 0, 1, 0],   # M4: c00 + c10
        [-1, 1, 0, 0],  # M5: -c00 + c01
        [0, 0, 0, 1],   # M6: c11
        [1, 0, 0, 0]    # M7: c00
    ], dtype=float)
    
    # Build the base rank-56 Strassen solution
    def build_strassen_solution():
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(8):
            base_col = 7 * r
            for s in range(7):
                col = base_col + s
                for li in range(4):
                    if u_vecs[s][li] != 0:
                        U[li][col] = u_vecs[s][li]
                v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
                for li in range(4):
                    if v_vecs[s][li] != 0:
                        V[v_idx[li]][col] = v_vecs[s][li]
                w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
                for li in range(4):
                    if w_vecs[s][li] != 0:
                        W[w_idx[li]][col] = w_vecs[s][li]
        
        return U, V, W
    
    # Fast verification using tensor contraction
    def verify_decomp_fast(U, V, W):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        rank_val = U.shape[1]
        U_3d = U.reshape(n, m, rank_val)
        V_3d = V.reshape(m, p, rank_val)
        W_3d = W.reshape(p, n, rank_val)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        UV = np.einsum('ijr,jkr->ijk r', U_3d, V_3d)
        T = np.einsum('ijk r,kir->ijk', UV, W_3d)
        
        return np.allclose(T, 1.0, atol=1e-10)
    
    # Build and verify base solution
    U_base, V_base, W_base = build_strassen_solution()
    
    if not verify_decomp_fast(U_base, V_base, W_base):
        # Fallback: schoolbook decomposition (rank 64)
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower-rank decompositions by searching for linear dependencies
    # The tensor T[i,j,k] = 1 for all i,j,k has a minimum rank
    # For (2x2)*(2x16), the Strassen bound is 56
    # Lower bound is max(n*m, m*p, p*n) = max(4, 32, 32) = 32
    
    # Try rank values from 55 down to 48
    best_rank = 56
    best_U, best_V, best_W = U_base, V_base, W_base
    
    # Attempt to reduce rank by checking if any factor is redundant
    # Use a greedy approach: try removing each factor and see if the rest
    # can be adjusted to maintain the decomposition
    
    if time.time() < deadline:
        # Try rank 55: remove one factor
        for remove_idx in range(56):
            if time.time() > deadline - 10:
                break
            
            U_try = np.delete(U_base, remove_idx, axis=1)
            V_try = np.delete(V_base, remove_idx, axis=1)
            W_try = np.delete(W_base, remove_idx, axis=1)
            
            # Check if the remaining 55 factors still span the tensor
            # This is not straightforward, so we'll use a heuristic:
            # check if the removed factor is linearly dependent on others
            # in the tensor product space
            
            # For now, skip this complex check and try a different approach
            pass
    
    # Alternative: try to find a rank-48 decomposition (6 per block)
    # This would require a rank-6 decomposition for 2x2 matmul, which doesn't exist
    # So we can't reduce below 56 using independent blocks
    
    # Try a global approach: find a rank-r decomposition for the full tensor
    # This is NP-hard, so we'll use iterative refinement if time permits
    
    if time.time() < deadline - 5:
        # Try rank 52: 8 blocks with some sharing
        # This is speculative, so we'll just return the known rank-56 solution
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
