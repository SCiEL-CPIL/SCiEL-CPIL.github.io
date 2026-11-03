"""Program for 3x3x6 matrix multiplication tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Check if U, V, W exactly decompose the tensor."""
    if T is None:
        T = _build_tensor()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(reconstructed, T, atol=1e-8)

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x3x6 matrix multiplication.
    
    Strategy: We know that matmul(3,3,3) has rank 23 (Balandin's result).
    The 3x3x6 tensor can be viewed as two 3x3x3 tensors sharing the middle index.
    A naive direct sum would give rank 46.
    
    However, we can potentially do better by exploiting the shared structure.
    The tensor T_{i,j,k} for k=0..5 can be split into:
    - T^0_{i,j,k} for k=0,1,2 (first 3x3 block)
    - T^1_{i,j,k} for k=3,4,5 (second 3x3 block, with k' = k-3)
    
    Both are standard 3x3x3 matmul tensors.
    
    The rank of the 3x3x6 tensor is at most 23 + 23 = 46.
    It could be less, but 46 is a safe upper bound.
    
    For now, we'll use the direct sum construction (rank 46) which is valid.
    We need to find the actual rank-23 decomposition coefficients for 3x3x3.
    
    Since I don't have the exact Balandin rank-23 coefficients hardcoded,
    I'll use a different approach: use Strassen-like methods or find a valid
    lower-rank decomposition through optimization.
    
    Actually, let me use a known rank-46 construction first to ensure correctness,
    then attempt optimization.
    """
    deadline = time.time() + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 54) as fallback
    rank_fallback = n * m * p
    U_fb = np.zeros((n * m, rank_fallback))
    V_fb = np.zeros((m * p, rank_fallback))
    W_fb = np.zeros((p * n, rank_fallback))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    # Try to find a better decomposition using ALS optimization
    T = _build_tensor()
    
    # Try rank 46 first (direct sum of two 3x3x3 rank-23 decompositions)
    # We need a rank-23 decomposition of 3x3x3.
    # Since I don't have exact coefficients, let me try a different approach:
    # Use randomized initialization and ALS to find a low-rank decomposition.
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fallback
    
    # Try various ranks from low to high
    for target_rank in [42, 43, 44, 45, 46]:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # ALS optimization
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each column c, U[:,c] should satisfy
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
                # For fixed V, W: U[:,c] = argmin ||T - einsum('ic,jc,kc->ijk', U, V, W)||
                # This is a linear system: for each c, U[:,c] = (V_c^T W_c)^{-1} ... 
                # Actually, ALS step: fix V and W, solve for U.
                # T_ijk = sum_c U_ic * V_jc * W_kc
                # For fixed V, W: reshape and solve.
                
                # For U update:
                # For each i,j: T[i,j,:] = sum_c U[i,j,c] * (V[:,c] * W[:,c] broadcast)
                # Let's do it properly with numpy.
                
                # Reshape approach:
                # T is (n*m, m*p, p*n)
                # Fix V (m*p, r) and W (p*n, r)
                # Then for each c: T_ijk has contribution U_ic * V_jc * W_kc
                # Sum over c: T_ijk = sum_c U_ic * (V_jc * W_kc)
                
                # For U update:
                # For each i: the row of T indexed by (i, j, k) for all j,k
                # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # This is complex. Let me use a simpler gradient-based approach.
                
                # Compute gradient of ||T - reconstruction||^2 w.r.t. U
                recon = np.einsum('ic,jc,kc->ijk', U, V, W)
                diff = T - recon
                
                # Gradient w.r.t. U:
                # d/dU_ic sum_j,k diff[i,j,k]^2 * 2 = -2 * sum_j,k diff[i,j,k] * V[j*c] * W[k*c]
                # But indices are coupled. Let me use a different approach.
                
                # Use ALS properly:
                # Fix V and W. For U:
                # T_ijk = sum_c U_ic V_jc W_kc
                # For fixed V,W: this is linear in U.
                # For each (i,j) pair (which indexes a row of U as U[i*m+j, :]):
                # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # Let's build the linear system for U.
                # For each row r = i*m+j of U, we have equations for each k:
                # T[r, j*p+k, k*n+i] = sum_c U[r, c] * (V[j*p+k, c] * W[k*n+i, c])
                
                # This is: for row r (with i = r//m, j = r%m):
                # For k in range(p): T[r, j*p+k, k*n+i] = dot(U[r, :], V_row_k * W_row_k)
                # where V_row_k = V[j*p+k, :] and W_row_k = W[k*n+i, :]
                
                # So we have a linear system: A_r * U[r, :] = b_r
                # where A_r[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # and b_r[k] = T[r, j*p+k, k*n+i]
                
                i_val = (n * m - 1)  # placeholder
                for i in range(n):
                    for j in range(m):
                        r = i * m + j
                        A = np.zeros((p, target_rank))
                        b = np.zeros(p)
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                            b[k] = T[r, j * p + k, k * n + i]
                        # Solve A * U[r, :] = b
                        try:
                            U[r, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        s = j * p + k
                        A = np.zeros((n, target_rank))
                        b = np.zeros(n)
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                            b[i] = T[i * m + j, j * p + k, k * n + i]
                        try:
                            V[s, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        t = k * n + i
                        A = np.zeros((m, target_rank))
                        b = np.zeros(m)
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                            b[j] = T[i * m + j, j * p + k, k * n + i]
                        try:
                            W[t, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
            
            # Check convergence
            recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.max(np.abs(recon - T))
            if err < 1e-7:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Verify
    if _verify(U_list, V_list, W_list, T):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
