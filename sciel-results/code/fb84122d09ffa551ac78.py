import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    def build_strassen_56():
        """Build the standard 56-rank Strassen decomposition."""
        U_local = np.zeros((4, 7), dtype=int)
        V_local = np.zeros((4, 7), dtype=int)
        W_local = np.zeros((4, 7), dtype=int)
        
        # r=0 (M1)
        U_local[0,0] = 1; U_local[3,0] = 1
        V_local[0,0] = 1; V_local[3,0] = 1
        W_local[0,0] = 1; W_local[3,0] = 1
        
        # r=1 (M2)
        U_local[2,1] = 1; U_local[3,1] = 1
        V_local[0,1] = 1
        W_local[2,1] = 1; W_local[3,1] = -1
        
        # r=2 (M3)
        U_local[0,2] = 1
        V_local[1,2] = 1; V_local[3,2] = -1
        W_local[1,2] = 1; W_local[3,2] = 1
        
        # r=3 (M4)
        U_local[3,3] = 1
        V_local[2,3] = 1; V_local[0,3] = -1
        W_local[0,3] = 1; W_local[2,3] = 1
        
        # r=4 (M5)
        U_local[0,4] = 1; U_local[1,4] = 1
        V_local[3,4] = 1
        W_local[0,4] = -1; W_local[1,4] = 1
        
        # r=5 (M6)
        U_local[2,5] = 1; U_local[0,5] = -1
        V_local[0,5] = 1; V_local[1,5] = 1
        W_local[3,5] = 1
        
        # r=6 (M7)
        U_local[1,6] = 1; U_local[3,6] = -1
        V_local[2,6] = 1; V_local[3,6] = 1
        W_local[0,6] = 1
        
        total_rank = 56
        U = np.zeros((8, total_rank), dtype=int)
        V = np.zeros((32, total_rank), dtype=int)
        W = np.zeros((16, total_rank), dtype=int)
        
        for j in range(4):
            for s in range(2):
                base_r = (j * 2 + s) * 7
                
                a_map = [0, 1, 4, 5] if s == 0 else [2, 3, 6, 7]
                b_map = [0*8+2*j, 0*8+2*j+1, 1*8+2*j, 1*8+2*j+1] if s == 0 else \
                        [2*8+2*j, 2*8+2*j+1, 3*8+2*j, 3*8+2*j+1]
                c_map = [2*j*2+0, (2*j+1)*2+0, 2*j*2+1, (2*j+1)*2+1]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la, l]
                        if val != 0:
                            U[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb, l]
                        if val != 0:
                            V[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc, l]
                        if val != 0:
                            W[c_map[lc], r] += val
        
        return U, V, W
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        R = U.shape[1]
        n, m, p = 2, 4, 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False, (i, j, k, s)
        return True, None

    U56, V56, W56 = build_strassen_56()
    valid, err = verify_decomposition(U56, V56, W56)
    
    if not valid:
        # Fallback to naive 64-rank if Strassen is broken
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to reduce rank by checking for linear dependencies among the 56 rank-1 terms.
    # Each term r defines a tensor T_r[i,j,k] = U[i,r]*V[j,r]*W[k,r]
    # We flatten these tensors into vectors of length 8*32*16 = 4096.
    # If the 56 vectors are linearly dependent, we can eliminate one term.
    
    def try_reduce_rank(U, V, W):
        if time.time() > deadline:
            return None
        
        R = U.shape[1]
        n, m, p = 2, 4, 8
        dim = n*m * m*p * p*n
        # Create a matrix of shape (R, dim) where each row is a flattened tensor
        # T_r[i*m+j, j*p+k, k*n+i] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # To save memory and time, we only need to check dependencies.
        # We can use QR decomposition or SVD on the (R, dim) matrix.
        # But dim is 4096, R is 56. Matrix is 56x4096.
        
        # Construct the matrix efficiently
        # T_r is sparse? No, U, V, W are sparse, but the product tensor is dense-ish.
        # Actually, U, V, W have small support.
        
        # Let's just construct the full matrix. 56 * 4096 * 8 bytes = ~1.8 MB. Fine.
        T_matrix = np.zeros((R, dim), dtype=np.float64)
        
        # Mapping from (i,j,k) to flat index
        # flat = (i*m + j) * (m*p) * (p*n) + (j*p + k) * (p*n) + (k*n + i)
        # This is huge. Let's iterate.
        
        # Optimization: Only non-zero entries of U, V, W contribute.
        # But it's easier to just loop over r, and for each r, loop over non-zero entries of U[:,r], V[:,r], W[:,r].
        
        for r in range(R):
            u_vals = U[:, r]
            v_vals = V[:, r]
            w_vals = W[:, r]
            
            u_idx = np.nonzero(u_vals)[0]
            v_idx = np.nonzero(v_vals)[0]
            w_idx = np.nonzero(w_vals)[0]
            
            for ui in u_idx:
                u_val = u_vals[ui]
                # ui = i*m + j
                i = ui // m
                j = ui % m
                
                for vi in v_idx:
                    v_val = v_vals[vi]
                    # vi = j*p + k
                    # j must match the j from u? No, the tensor is T[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
                    # So j in U and V must be the same.
                    v_j = vi // p
                    v_k = vi % p
                    if v_j != j:
                        continue
                    
                    for wi in w_idx:
                        w_val = w_vals[wi]
                        # wi = k*n + i
                        # k must match v_k, i must match u_i
                        w_k = wi // n
                        w_i = wi % n
                        if w_k != v_k or w_i != i:
                            continue
                        
                        # Compute flat index
                        # Index in T_matrix[r, :]
                        # The tensor is indexed by (i, j, k)
                        # Let's use a consistent flat index for the tensor T
                        # T_flat = ((i * m + j) * (m * p) + (j * p + k)) * (p * n) + (k * n + i)
                        # This is the index in the flattened tensor.
                        flat_idx = ((i * m + j) * (m * p) + (j * p + k)) * (p * n) + (k * n + i)
                        T_matrix[r, flat_idx] = u_val * v_val * w_val
        
        # Now find the rank of T_matrix
        # Use SVD
        try:
            # SVD of 56x4096 matrix
            U_svd, s, Vt = np.linalg.svd(T_matrix, full_matrices=False)
        except:
            return None
        
        # Determine numerical rank
        # Tolerance
        tol = 1e-10
        rank = np.sum(s > tol)
        
        if rank < R:
            # We found a dependency.
            # The last (R - rank) singular vectors are in the null space.
            # We can eliminate one of the terms corresponding to a null vector.
            # However, the null space vectors are linear combinations of the original terms.
            # To get a valid decomposition, we need to eliminate a term and adjust the others?
            # Actually, if T_r = sum c_s T_s, then we can just drop term r and add c_s * term_s to the other terms?
            # No, the decomposition is sum_r T_r. If T_r is dependent, the sum is still the same.
            # We can just drop the term r? No, that changes the sum.
            # We need to find a subset of R-1 terms that sums to the same tensor.
            # This is not trivial from SVD alone.
            
            # A simpler approach: If rank < R, the problem is under-determined.
            # We can try to solve for a rank R-1 decomposition directly using the null space.
            # But for now, let's just return the 56-rank solution if we can't easily construct a lower one.
            # The probability of finding a simple integer dependency is low.
            
            pass
        
        return None

    result_reduced = try_reduce_rank(U56, V56, W56)
    if result_reduced is not None:
        return result_reduced

    # Return the verified 56-rank solution
    return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
# EVOLVE-BLOCK-END
