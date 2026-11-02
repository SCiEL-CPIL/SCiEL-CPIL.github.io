import numpy as np
from fractions import Fraction
from itertools import combinations, product
import time

def construct():
    n, m, p = 2, 4, 6
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # The matrix multiplication tensor T for 2x4 * 4x6 -> 2x6
    # T[i, j, k] = 1 if the (i,k) entry of C depends on A[i,j] * B[j,k]
    # In our indexing:
    # A is n x m = 2 x 4, indexed by (i, j) where i in [0,2), j in [0,4)
    # B is m x p = 4 x 6, indexed by (j, k) where j in [0,4), k in [0,6)
    # C is n x p = 2 x 6, indexed by (i, k) where i in [0,2), k in [0,6)
    # W index: W[k*n + i, r] for C[i, k]
    
    # The tensor has entries T[i, j, k] = 1 if C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i, j, k] = 1 for all valid i, j, k
    
    # Rank of this tensor: we need to find the minimum r such that
    # T = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m} = R^8, v_r in R^{m*p} = R^{24}, w_r in R^{p*n} = R^{12}
    
    # Current best: rank 42 via Strassen
    # Let's try to find a better decomposition using tensor slicing / ALS
    
    # First, let's verify the Strassen-based rank 42 solution
    # Then try to improve via search
    
    # Build the tensor in flat form
    # T_flat[i*m + j, k] ... no, let's think in terms of the factorization
    
    # The tensor T can be viewed as a 3D array T[i, j, k] where:
    # i in [0, n*m) = [0, 8)  -> i = row_A * m + col_A = iA * 4 + j
    # j in [0, m*p) = [0, 24) -> j = row_B * p + col_B = jB * 6 + k  
    # k in [0, p*n) = [0, 12) -> k = col_C * n + row_C = kC * 2 + iC
    
    # T[i, j, k] = 1 if and only if:
    # iA = i // 4, jA = i % 4
    # jB = j // 6, kC = j % 6
    # iC = k % 2, kC_out = k // 2
    # AND jA == jB AND iA == iC AND kC == kC_out
    
    # So T[i, j, k] = 1 iff (i%4 == j//6) and (i//4 == k%2) and (j%6 == k//2)
    
    # Let's build the tensor as a 3D numpy array for verification
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for iA in range(n):
        for j in range(m):
            for k in range(p):
                i_idx = iA * m + j
                j_idx = j * p + k
                k_idx = k * n + iA
                T[i_idx, j_idx, k_idx] = 1
    
    # Verify Strassen rank 42
    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        # Reconstruct tensor
        T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(U.shape[1]):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        # Check against original
        if not np.allclose(T_recon, T, atol=1e-10):
            return False
        # Also check that entries are integers or half-integers
        for mat in [U, V, W]:
            for val in mat:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True
    
    # Strassen-based rank 42
    def strassen_rank42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen(a, b, c, d, e, f, g, h, c11, c12, c21, c22, col):
            U[a][col] += 1; U[d][col] += 1; V[e][col] += 1; V[h][col] += 1
            W[c11][col] += 1; W[c22][col] += 1; col += 1
            U[c][col] += 1; U[d][col] += 1; V[e][col] += 1
            W[c21][col] += 1; W[c22][col] -= 1; col += 1
            U[a][col] += 1; V[f][col] += 1; V[h][col] -= 1
            W[c12][col] += 1; W[c22][col] += 1; col += 1
            U[d][col] += 1; V[g][col] += 1; V[e][col] -= 1
            W[c11][col] += 1; W[c21][col] += 1; col += 1
            U[a][col] += 1; U[b][col] += 1; V[h][col] += 1
            W[c11][col] -= 1; W[c12][col] += 1; col += 1
            U[c][col] += 1; U[a][col] -= 1; V[e][col] += 1; V[f][col] += 1
            W[c22][col] += 1; col += 1
            U[b][col] += 1; U[d][col] -= 1; V[g][col] += 1; V[h][col] += 1
            W[c11][col] += 1; col += 7
            return col
        
        col = 0
        for k in range(3):
            col = add_strassen(0, 1, 4, 5, 2*k, 2*k+1, 6+2*k, 6+2*k+1,
                              2*k*2, (2*k+1)*2, 2*k*2+1, (2*k+1)*2+1, col)
            col = add_strassen(2, 3, 6, 7, 12+2*k, 12+2*k+1, 18+2*k, 18+2*k+1,
                              2*k*2, (2*k+1)*2, 2*k*2+1, (2*k+1)*2+1, col)
        
        return {"U": U, "V": V, "W": W}
    
    best_result = strassen_rank42()
    
    # Try to find lower rank via random initialization + ALS
    # The tensor rank problem is NP-hard, but we can try heuristic approaches
    
    # Approach: Use the fact that the tensor has a specific structure
    # C = A * B where A is 2x4, B is 4x6
    # We can try different block decompositions
    
    # Let's try a more systematic search over possible rank values
    # Starting from rank 41 down to some minimum
    
    def try_rank(r, max_iters=50, time_limit=10):
        """Try to find a rank-r decomposition using ALS"""
        start_time = time.time()
        
        # Random initialization
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
            
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.array(T.reshape(n*m, m*p*p*n))  # This won't work directly
            
            # Better: use the tensor structure
            # T[i, j, k] for each r, we have U[i,r] * V[j,r] * W[k,r]
            
            # Let's use a different approach: fix one factor and solve for others
            # This is the tensor completion / ALS approach
            
            # Initialize all three factors
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize
            for col in range(r):
                norm = np.linalg.norm(U[:, col])
                if norm > 0:
                    U[:, col] /= norm
                    V[:, col] *= norm
                norm = np.linalg.norm(V[:, col])
                if norm > 0:
                    V[:, col] /= norm
                    W[:, col] *= norm
            
            for iteration in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U: for each i, U[i, :] should satisfy
                # sum_j,k T[i,j,k] * V[j,:] * W[k,:] / (V[j,:] * W[k,:]) ... 
                # Actually, ALS for tensor decomposition:
                # Fix V, W, solve for U
                # U[i, r] = sum_{j,k} T[i,j,k] * V[j,r] * W[k,r] / sum_{j,k} (V[j,r] * W[k,r])^2
                
                # More precisely, for each column r:
                # The contribution to T[i,j,k] from factor r is U[i,r]*V[j,r]*W[k,r]
                # To update U, we fix V and W, and for each i:
                # U[i,r] = <T_i, V_r ⊗ W_r> / ||V_r ⊗ W_r||^2
                # where T_i is the slice T[i, :, :]
                
                V_W = np.einsum('jr,kr->jk r', V, W)  # (m*p, p*n, r)
                # This is expensive. Let's do it column by column.
                
                U_new = np.zeros_like(U)
                for i in range(n*m):
                    for r in range(r):
                        # T[i, :, :] dot (V[:, r] * W[:, r])
                        # V[:, r] is length m*p, W[:, r] is length p*n
                        # T[i, j, k] * V[j, r] * W[k, r] summed over j, k
                        num = 0.0
                        den = 0.0
                        for j in range(m*p):
                            for k in range(p*n):
                                num += T[i, j, k] * V[j, r] * W[k, r]
                                den += (V[j, r] * W[k, r]) ** 2
                        if den > 1e-12:
                            U_new[i, r] = num / den
                        else:
                            U_new[i, r] = 0
                U = U_new
                
                # Similarly update V and W
                V_new = np.zeros_like(V)
                for j in range(m*p):
                    for r in range(r):
                        num = 0.0
                        den = 0.0
                        for i in range(n*m):
                            for k in range(p*n):
                                num += T[i, j, k] * U[i, r] * W[k, r]
                                den += (U[i, r] * W[k, r]) ** 2
                        if den > 1e-12:
                            V_new[j, r] = num / den
                        else:
                            V_new[j, r] = 0
                V = V_new
                
                W_new = np.zeros_like(W)
                for k in range(p*n):
                    for r in range(r):
                        num = 0.0
                        den = 0.0
                        for i in range(n*m):
                            for j in range(m*p):
                                num += T[i, j, k] * U[i, r] * V[j, r]
                                den += (U[i, r] * V[j, r]) ** 2
                        if den > 1e-12:
                            W_new[k, r] = num / den
                        else:
                            W_new[k, r] = 0
                W = W_new
                
                # Check convergence
                T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                for r in range(r):
                    T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                err = np.linalg.norm(T_recon - T)
                if err < 1e-6:
                    # Check if entries are close to integers/half-integers
                    all_good = True
                    for mat in [U, V, W]:
                        for val in mat:
                            if abs(val * 2 - round(val * 2)) > 1e-4:
                                all_good = False
                                break
                        if not all_good:
                            break
                    if all_good:
                        # Round to nearest half-integer
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        # Verify
                        T_recon2 = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                        for r in range(r):
                            T_recon2 += np.einsum('i,j,k->ijk', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                        if np.allclose(T_recon2, T, atol=1e-9):
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
        
        return None
    
    # Try ranks from 41 down
    for r in range(41, 35, -1):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=30, time_limit=15)
        if result is not None:
            best_result = result
            break
    
    return best_result

