import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Construct Schoolbook Decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # 2. Attempt Rank Reduction via BCLR
    # We look for linear dependencies in U, V, or W.
    # If U * c = 0, we can potentially reduce rank.
    
    def try_reduce(U, V, W, max_iters=1000):
        n, m, p = DIMENSIONS
        R = U.shape[1]
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find null space of U_curr
            # Use SVD to find a null vector
            try:
                U_svd = np.linalg.svd(U_curr, full_matrices=False)
            except:
                break
            
            # Find the smallest singular value
            s_min = U_svd[1][-1]
            if s_min > 1e-9:
                # No dependency
                break
            
            # Get the right singular vector corresponding to s_min
            c = U_svd[2][:, -1]
            
            # Verify U_curr * c is close to 0
            if np.linalg.norm(U_curr @ c) > 1e-9:
                break
                
            # We have a dependency U_curr * c = 0.
            # We want to eliminate one term, say the last one (index R-1).
            # We need c[R-1] != 0.
            if abs(c[R-1]) < 1e-9:
                # Swap with a non-zero element
                idx = np.where(np.abs(c) > 1e-9)[0]
                if len(idx) == 0:
                    break
                swap_idx = idx[0]
                c[swap_idx], c[R-1] = c[R-1], c[swap_idx]
                U_curr[:, swap_idx], U_curr[:, R-1] = U_curr[:, R-1], U_curr[:, swap_idx]
                V_curr[:, swap_idx], V_curr[:, R-1] = V_curr[:, R-1], V_curr[:, swap_idx]
                W_curr[:, swap_idx], W_curr[:, R-1] = W_curr[:, R-1], W_curr[:, swap_idx]
                
            if abs(c[R-1]) < 1e-9:
                break
                
            # Perform the transformation
            # U_new = U_curr (without last col)
            # V_new = V_curr + V_curr * c * e_last^T ?
            # The standard BCLR update:
            # Let U' = U_curr[:, :-1]
            # Let V' = V_curr[:, :-1] + (V_curr @ c)[:, :-1] * np.ones((1, R-1)) ? No.
            
            # Correct update:
            # We know sum_r c_r U_r = 0.
            # So U_{R-1} = -1/c_{R-1} sum_{r < R-1} c_r U_r.
            # Substitute into tensor:
            # T = sum_{r < R-1} U_r V_r W_r + U_{R-1} V_{R-1} W_{R-1}
            # T = sum_{r < R-1} U_r V_r W_r - 1/c_{R-1} (sum_{r < R-1} c_r U_r) V_{R-1} W_{R-1}
            # T = sum_{r < R-1} U_r (V_r - 1/c_{R-1} c_r V_{R-1}) W_{R-1} + U_{R-1} V_{R-1} W_{R-1} ? No.
            # The term U_{R-1} is replaced.
            # T = sum_{r < R-1} U_r V_r W_r + (-1/c_{R-1} sum_{r < R-1} c_r U_r) V_{R-1} W_{R-1}
            # T = sum_{r < R-1} U_r [ V_r W_r - 1/c_{R-1} c_r V_{R-1} W_{R-1} ]
            # This is not a sum of rank-1 terms in the standard form U' V' W'.
            
            # Alternative BCLR step:
            # Update V and W to absorb the dependency.
            # V_new = V_curr[:, :-1] + (V_curr @ c)[:, :-1] * np.ones((1, R-1)) / c[R-1] ?
            
            # Let's use the explicit formula:
            # U_new = U_curr[:, :-1]
            # V_new = V_curr[:, :-1] + np.outer(V_curr @ c, np.ones(R-1)) / c[R-1]
            # W_new = W_curr[:, :-1] - np.outer(W_curr @ c, np.ones(R-1)) / c[R-1]
            
            # Check if this preserves the tensor.
            # This specific transformation is valid if U_curr * c = 0.
            
            V_col = V_curr @ c
            W_col = W_curr @ c
            
            V_new = V_curr[:, :-1] + np.outer(V_col, np.ones(R-1)) / c[R-1]
            W_new = W_curr[:, :-1] - np.outer(W_col, np.ones(R-1)) / c[R-1]
            U_new = U_curr[:, :-1]
            
            # Verify exactness
            # Tensor T = sum U V W
            # We can check a few random entries or the whole tensor if small.
            # For 3x5x7, tensor size is 15x35x21? No, it's a 3D tensor.
            # Flattened: (15*35) x 21?
            # Let's check the matrix multiplication property.
            # A (3x5), B (5x7). C (3x7).
            # T should satisfy C = A B.
            
            # Reconstruct C for a random A, B
            A_rand = np.random.randint(0, 10, (n, m))
            B_rand = np.random.randint(0, 10, (m, p))
            C_true = A_rand @ B_rand
            
            # Reconstruct C using factors
            # C[i, k] = sum_j A[i, j] B[j, k] ?
            # The decomposition is U (nm x R), V (mp x R), W (pn x R).
            # Index mapping:
            # U index: i*m + j
            # V index: j*p + k
            # W index: k*n + i
            # C[i, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] ??
            # No, the sum is over r. The indices i, j, k are fixed.
            # This implies the tensor T_{i,j,k} = sum_r U_{ij, r} V_{jk, r} W_{ki, r}.
            # And C_{ik} = sum_j A_{ij} B_{jk} T_{i,j,k} ?
            # If T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'}?
            # In schoolbook, T_{i,j,k} = 1 if it's the "correct" term?
            # Actually, the standard matmul tensor has T_{i,j,k} = 1 for all i,j,k? No.
            # T_{i,j,k} is 1 if the indices match the multiplication path.
            # For C_ik = sum_j A_ij B_jk, the tensor entry is 1 for the term (i,j,k).
            # So T_{i,j,k} = 1.
            # Then C_ik = sum_{i,j,k} A_{ij} B_{jk} T_{i,j,k} ?
            # No, C_ik = sum_j A_{ij} B_{jk}.
            # So the tensor must be T_{i,j,k} = 1.
            # Wait, if T_{i,j,k} = 1, then sum_{i,j,k} A_{ij} B_{jk} T_{i,j,k} = sum_{i,j,k} A_{ij} B_{jk} = sum_i sum_k (sum_j A_{ij} B_{jk}) = sum_i sum_k C_{ik}.
            # This sums all elements.
            
            # The correct tensor definition for matmul is:
            # T_{i, j, k} such that C_{ik} = sum_j A_{ij} B_{jk}.
            # This is a bilinear form.
            # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k ? No.
            # T = sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ki} ?
            
            # Let's check the seed's W index: k*n + i.
            # This suggests the output is indexed by (k, i).
            # So C^T_{ki} = C_{ik}.
            # The contraction is sum_r U_{ij, r} V_{jk, r} W_{ki, r}.
            # If this equals 1 for the "active" term and 0 otherwise, then
            # C_{ik} = sum_j A_{ij} B_{jk} * (sum_r U_{ij, r} V_{jk, r} W_{ki, r}).
            # If the sum is 1, then C_{ik} = sum_j A_{ij} B_{jk}.
            
            # So we just need to verify that the reconstructed tensor T_{i,j,k} is 1.
            
            # Check T_{i,j,k} for all i,j,k
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                U_curr, V_curr, W_curr = U_new, V_new, W_new
                R -= 1
                best_rank = R
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
            else:
                break
                
        return best_rank, best_U, best_V, best_W

    # Run reduction
    best_rank, best_U, best_V, best_W = try_reduce(U, V, W)
    
    # Convert to lists of floats (integers or half-integers)
    # The problem asks for integer/half-integer entries.
    # Our operations involve division by c[R-1].
    # If c[R-1] is 1 or -1, we get integers.
    # If not, we might get fractions.
    # We should check if the entries are close to half-integers.
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
