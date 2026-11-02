"""Improved rank-minimal decomposition for 2x5 @ 5x6 using block decomposition + dependency reduction."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np
from collections import defaultdict
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Step 1: Build the valid 54-rank decomposition
    rank = 54
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    # Part A: j=0 (2x1x6 tensor), rank 12
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Strassen 2x2x2 terms
    strassen_2x2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Part B: j=1,2 and j=3,4 (two 2x2x6 tensors)
    for block_j in [1, 3]:
        for k_block in range(3):
            k_base = k_block * 2
            for term in strassen_2x2x2:
                u_list, v_list, w_list = term
                for (li, lj, val) in u_list:
                    U[li * m + (block_j + lj), col] += val
                for (lj, lk, val) in v_list:
                    V[(block_j + lj) * p + (k_base + lk), col] += val
                for (li, lk, val) in w_list:
                    W[(k_base + lk) * n + li, col] += val
                col += 1
    
    # Verify the decomposition is valid
    def verify(U_mat, V_mat, W_mat, trials=5):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for _ in range(trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C = np.random.randn(n, p)
            # Left side: sum_{i,j,k} A[i,j] B[j,k] C[i,k]
            left = np.sum(A * B * C)  # element-wise product, then sum
            # Wait, this is sum_{i,j,k} A[i,j]*B[j,k]*C[i,k] but dimensions don't match directly
            # A is n×m, B is m×p, C is n×p
            # sum_{i,j,k} A[i,j] B[j,k] C[i,k]
            # = sum_i sum_j A[i,j] (sum_k B[j,k] C[i,k])
            # = sum_i sum_j A[i,j] (B @ C.T)[j,i]
            # = sum_{i,j} A[i,j] * (B @ C.T)[j,i]
            # = sum_{i,j} A[i,j] * (C @ B.T)[i,j]
            left = np.sum(A * (C @ B.T))
            
            # Right side: sum_r (A·u_r) * (B·v_r) * (C·w_r)
            # where A·u_r = sum_{i,j} A[i,j] U[i*m+j, r]
            # B·v_r = sum_{j,k} B[j,k] V[j*p+k, r]
            # C·w_r = sum_{i,k} C[i,k] W[k*n+i, r]
            right = 0
            for r in range(U_mat.shape[1]):
                a_term = np.dot(A.flatten(), U_mat[:, r])
                b_term = np.dot(B.flatten(), V_mat[:, r])
                c_term = np.dot(C.flatten(), W_mat[:, r])
                right += a_term * b_term * c_term
            
            if abs(left - right) > 1e-9:
                return False
        return True
    
    is_valid = verify(U, V, W, trials=3)
    if not is_valid:
        # Should not happen, but fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Step 2: Try to reduce rank by finding linear dependencies in the tensor space
    # Each rank-1 component T_r = u_r ⊗ v_r ⊗ w_r corresponds to a vector in R^{nmp}
    # vec_r[i*m*p + j*p + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    tensor_dim = n * m * p  # 60
    
    def tensor_to_vec(U_col, V_col, W_col):
        vec = np.zeros(tensor_dim)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    vec[idx] = U_col[i * m + j] * V_col[j * p + k] * W_col[k * n + i]
        return vec
    
    # Build all rank-1 vectors
    rank1_vecs = np.zeros((rank, tensor_dim))
    for r in range(rank):
        rank1_vecs[r, :] = tensor_to_vec(U[:, r], V[:, r], W[:, r])
    
    # Total tensor
    total_tensor = np.sum(rank1_vecs, axis=0)
    
    # Greedy removal: try to remove columns that are in the span of others
    active = list(range(rank))
    
    for iteration in range(30):
        if time.time() > deadline:
            break
        removed = False
        for idx in range(len(active)):
            r = active[idx]
            remaining = [s for s in active if s != r]
            if len(remaining) < 1:
                continue
            
            # Check if (total_tensor - rank1_vecs[r, :]) is in span of remaining
            A_mat = rank1_vecs[remaining, :].T  # (60, len(remaining))
            b_vec = total_tensor - rank1_vecs[r, :]
            
            try:
                # Use least squares to check if b_vec is in the column space
                coeffs, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                if len(residuals) > 0 and residuals[0] < 1e-8 * (np.linalg.norm(b_vec) + 1e-10):
                    active.remove(r)
                    removed = True
                    break
            except np.linalg.LinAlgError:
                continue
        
        if not removed:
            break
    
    final_rank = len(active)
    
    if final_rank < rank:
        U_final = U[:, active]
        V_final = V[:, active]
        W_final = W[:, active]
        
        if verify(U_final, V_final, W_final, trials=3):
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    # Return the original 54-rank solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

