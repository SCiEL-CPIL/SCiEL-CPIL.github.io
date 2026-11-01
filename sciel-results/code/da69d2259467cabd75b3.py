"""Matrix multiplication 2x3x9 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W


def _verify(U, V, W, n, m, p, atol=1e-8):
    """Check T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] equals target."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1
    return np.allclose(T_recon, T_target, atol=atol)


def _structured_search(n, m, p, deadline, max_rank):
    """
    Search for a rank-<54 decomposition by exploiting n=2 structure.
    
    Key insight: For M_{2,m,p}, the tensor can be decomposed as:
      C[0,k] = sum_j A[0,j] * B[j,k]   (layer 0)
      C[1,k] = sum_j A[1,j] * B[j,k]   (layer 1)
    
    Each layer is an M_{1,m,p} tensor of rank m*p = 27.
    The full tensor is the "sum" of two M_{1,m,p} tensors sharing the B-mode.
    
    We try to find a decomposition where the B-mode vectors are shared
    between the two layers, potentially reducing total rank.
    
    Strategy: Parameterize as
      U[:,r] = (u0_r, u1_r)  where u0_r in R^m, u1_r in R^m
      V[:,r] = v_r in R^p
      W[:,r] = (w0_r, w1_r)  where w0_r in R^p, w1_r in R^p
      
    Constraints:
      For all j,k: sum_r u0_r[j] * v_r[k] * w0_r[k*n+0] = delta_{j,k}  (for layer 0)
      For all j,k: sum_r u1_r[j] * v_r[k] * w1_r[k*n+1] = delta_{j,k}  (for layer 1)
    
    Actually, let me reformulate more carefully.
    
    The tensor T has:
      T[i*m+j, j*p+k, k*n+i] = 1
    
    For a rank-r decomposition:
      T = sum_{r=0}^{r-1} u_r ⊗ v_r ⊗ w_r
    
    where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    
    For n=2:
      u_r = (a_r, b_r) where a_r in R^m (for i=0), b_r in R^m (for i=1)
      v_r = c_r in R^{m*p}
      w_r = (d_r, e_r) where d_r in R^p (for i=0, indices k*2+0), e_r in R^p (for i=1, indices k*2+1)
    
    The constraint T[i*m+j, j*p+k, k*n+i] = 1 becomes:
      sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1  for all valid i,j,k
    
    For i=0: sum_r a_r[j] * c_r[j*p+k] * d_r[k] = 1  for all j,k
    For i=1: sum_r b_r[j] * c_r[j*p+k] * e_r[k] = 1  for all j,k
    
    This is equivalent to:
      A_0 = sum_r a_r ⊗ d_r ⊗ (c_r restricted) ... this is getting complex.
    
    Let me try a different approach: use the fact that for each j,
    the "slice" of the tensor at fixed j involves A[i,j], B[j,k], C[k,i].
    
    Actually, let me try a direct numerical approach: initialize random
    rank-r decomposition and use gradient descent to minimize the error.
    """
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Build target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1
    
    rng = np.random.RandomState(12345)
    
    # Try a few ranks below schoolbook
    for target_rank in range(min(max_rank, n * m * p - 1), min(max_rank, n * m * p - 1) - 8, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
            
            # Simple iterative refinement: alternating least squares
            # Minimize ||T_target - sum_r u_r ⊗ v_r ⊗ w_r||_F^2
            # 
            # For fixed V, W: U[a,r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / (sum_{b,c} V[b,r]^2 W[c,r]^2)
            # This is exact for each mode.
            
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Update U
                for a in range(n * m):
                    for r in range(target_rank):
                        num = np.sum(T_target[a, :, :] * V[:, r][:, None] * W[:, r][None, :])
                        den = np.sum(V[:, r] ** 2) * np.sum(W[:, r] ** 2)
                        if abs(den) > 1e-12:
                            U[a, r] = num / den
                
                # Update V
                for b in range(m * p):
                    for r in range(target_rank):
                        num = np.sum(T_target[:, b, :] * U[:, r][:, None] * W[:, r][None, :])
                        den = np.sum(U[:, r] ** 2) * np.sum(W[:, r] ** 2)
                        if abs(den) > 1e-12:
                            V[b, r] = num / den
                
                # Update W
                for c in range(p * n):
                    for r in range(target_rank):
                        num = np.sum(T_target[:, :, c] * U[:, r][:, None] * V[:, r][None, :])
                        den = np.sum(U[:, r] ** 2) * np.sum(V[:, r] ** 2)
                        if abs(den) > 1e-12:
                            W[c, r] = num / den
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_recon - T_target)
                if err < 1e-6:
                    break
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T_target)
            
            if err < 1e-6 and target_rank < best_rank:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                break  # Found a valid decomposition at this rank
    
    if best_U is not None:
        return best_U, best_V, best_W, best_rank
    
    return None, None, None, n * m * p


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Schoolbook decomposition of rank n*m*p = 54
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Verify schoolbook decomposition is correct
    assert _verify(U_sb, V_sb, W_sb, n, m, p), "Schoolbook verification failed"
    
    # Try to find a lower-rank decomposition using structured search
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    if time.time() < deadline - 30:
        U_new, V_new, W_new, new_rank = _structured_search(n, m, p, deadline, best_rank)
        if U_new is not None and new_rank < best_rank:
            # Final verification with tighter tolerance
            if _verify(U_new, V_new, W_new, n, m, p, atol=1e-6):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_rank
    
    # Convert to list-of-lists format as required
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
