"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_result = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120
    
    # Build the matrix multiplication tensor
    # T[i, j, k] = 1 if this corresponds to a valid multiplication term
    # For (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor has shape (n, m, p) where T[i,j,k] = 1 for all i,j,k
    # Actually, the multiplication tensor is: for each output element (i,k),
    # it's a sum over j of A[i,j]*B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # But wait - that's just the all-ones tensor, which isn't right.
    
    # Let me reconsider. The matrix multiplication tensor M_{n,m,p} is defined as:
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (e_i^T A e_j)(e_j^T B e_k)
    # Actually, the standard definition: the tensor T has T[i,j,k] = 1 if the 
    # (i,k) entry of the product depends on A[i,j] and B[j,k]
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
    # So T is the all-ones tensor of shape (n, m, p)
    
    # Wait, that's not right either. Let me think more carefully.
    # The matrix multiplication map: (A, B) -> AB
    # (AB)[i,k] = sum_{j=1}^{m} A[i,j] * B[j,k]
    # So the bilinear form for output (i,k) is: sum_j A[i,j] B[j,k]
    # The tensor T is defined by T[i,j,k] = 1 if A[i,j] and B[j,k] contribute to (AB)[i,k]
    # This is true for all i, j, k, so T is indeed the all-ones tensor.
    
    # But that can't be right because then the rank would just be the rank of the 
    # all-ones matrix, which is 1. That's clearly wrong.
    
    # I think I'm confusing myself. Let me use the standard definition:
    # The matrix multiplication tensor M_{n,m,p} is an element of R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p}
    # where M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^{(n)} ⊗ e_j^{(m)} ⊗ e_k^{(p)} ⊗ ... 
    # Actually, it's: M = sum_{i,j,k} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_i ⊗ e_k)
    # where the first factor is indexed by (i,j) for A, second by (j,k) for B, third by (i,k) for AB.
    
    # So the tensor T has shape (n*m, m*p, n*p) and T[(i,j), (j',k'), (i'',k'')] = 1 
    # if i=i'', j=j', and k=k'' (i.e., the indices match for a valid multiplication term)
    
    # Let me build this tensor properly
    U_dim = n * m  # rows of A
    V_dim = m * p  # columns of B  
    W_dim = n * p  # entries of AB
    
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = i * p + k  # Note: output is (i,k), so index is i*p + k
                T[u_idx, v_idx, w_idx] = 1
    
    # Now try to find low-rank decomposition
    # Try various ranks using ALS
    
    def snap_to_half_integers(arr, tol=1e-6):
        """Snap values to nearest half-integer if close enough."""
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None
    
    def verify_decomposition(U_mat, V_mat, W_mat, T_tensor, tol=1e-6):
        """Verify that U, V, W exactly decompose T."""
        rank = U_mat.shape[1]
        # Reconstruct tensor
        T_recon = np.zeros_like(T_tensor)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.allclose(T_recon, T_tensor, atol=tol)
    
    def try_als(T_tensor, rank, max_iters=200, tol=1e-8, seed=42):
        """Try ALS to find rank decomposition."""
        np.random.seed(seed)
        U_dim, V_dim, W_dim = T_tensor.shape
        
        # Initialize randomly
        U_mat = np.random.randn(U_dim, rank) * 0.1
        V_mat = np.random.randn(V_dim, rank) * 0.1
        W_mat = np.random.randn(W_dim, rank) * 0.1
        
        for it in range(max_iters):
            # Update U
            # For each column r, U[:, r] should minimize ||T - sum_r U[:,r] V[:,r]^T W[:,r]||
            # This is a linear system: for each (i,j), T[i,j,:] = sum_r U[i,r] V[j,r] W[:,r]
            # Reshape T to (U_dim * V_dim, W_dim)
            T_flat = T_tensor.reshape(U_dim * V_dim, W_dim)
            
            # For U update: fix V, W
            # T_flat[i*V_dim+j, :] = sum_r U[i,r] * V[j,r] * W[:,r]
            # Let's define for each i: T_i = T_tensor[i, :, :] reshaped to (V_dim, W_dim)
            # T_i[j, k] = sum_r U[i,r] V[j,r] W[k,r]
            # This is: T_i = sum_r U[i,r] (V[:,r] ⊗ W[:,r])
            # So T_i = U[i,:] @ M where M[r, (j,k)] = V[j,r] * W[k,r]
            
            M = np.zeros((rank, V_dim * W_dim))
            for r in range(rank):
                M[r, :] = np.outer(V_mat[:, r], W_mat[:, r]).flatten()
            
            # Solve for U: T_flat = U @ M
            # U = T_flat @ pinv(M)
            U_new, _, _, _ = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
            U_mat = U_new.T
            
            # Update V
            # T_tensor[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
            # For fixed U, W: T_tensor[:, j, k] = sum_r U[:,r] V[j,r] W[k,r]
            # Let's reshape: for each (j,k), T[:, j, k] = sum_r U[:,r] * V[j,r] * W[k,r]
            # Define N[r, (i,k)] = U[i,r] * W[k,r]
            # Then T_reshaped[(j), (i,k)] = sum_r V[j,r] * N[r, (i,k)]
            # T_reshaped = V @ N.T
            
            N = np.zeros((rank, U_dim * W_dim))
            for r in range(rank):
                N[r, :] = np.outer(U_mat[:, r], W_mat[:, r]).flatten()
            
            T_reshaped = T_tensor.transpose(1, 0, 2).reshape(V_dim, U_dim * W_dim)
            V_new, _, _, _ = np.linalg.lstsq(N.T, T_reshaped.T, rcond=None)
            V_mat = V_new.T
            
            # Update W
            # T_tensor[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
            # For fixed U, V: T_tensor[i, j, k] = sum_r W[k,r] * (U[i,r] * V[j,r])
            # Define P[r, (i,j)] = U[i,r] * V[j,r]
            # T_reshaped[(k), (i,j)] = sum_r W[k,r] * P[r, (i,j)]
            # T_reshaped = W @ P.T
            
            P = np.zeros((rank, U_dim * V_dim))
            for r in range(rank):
                P[r, :] = np.outer(U_mat[:, r], V_mat[:, r]).flatten()
            
            T_reshaped2 = T_tensor.transpose(2, 0, 1).reshape(W_dim, U_dim * V_dim)
            W_new, _, _, _ = np.linalg.lstsq(P.T, T_reshaped2.T, rcond=None)
            W_mat = W_new.T
            
            # Check convergence
            if it % 10 == 0:
                T_recon = np.zeros_like(T_tensor)
                for r in range(rank):
                    T_recon += np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
                err = np.linalg.norm(T_recon - T_tensor) / np.linalg.norm(T_tensor)
                if err < tol:
                    break
        
        # Check if we got an exact solution
        T_recon = np.zeros_like(T_tensor)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        err = np.linalg.norm(T_recon - T_tensor) / (np.linalg.norm(T_tensor) + 1e-12)
        
        if err < 1e-6:
            # Try to snap to half-integers
            U_snapped = snap_to_half_integers(U_mat)
            V_snapped = snap_to_half_integers(V_mat)
            W_snapped = snap_to_half_integers(W_mat)
            
            if U_snapped is not None and V_snapped is not None and W_snapped is not None:
                if verify_decomposition(U_snapped, V_snapped, W_snapped, T_tensor, tol=1e-6):
                    return U_snapped, V_snapped, W_snapped
            
            # Try with smaller tolerance for snapping
            U_snapped2 = snap_to_half_integers(U_mat, tol=1e-4)
            V_snapped2 = snap_to_half_integers(V_mat, tol=1e-4)
            W_snapped2 = snap_to_half_integers(W_mat, tol=1e-4)
            
            if U_snapped2 is not None and V_snapped2 is not None and W_snapped2 is not None:
                if verify_decomposition(U_snapped2, V_snapped2, W_snapped2, T_tensor, tol=1e-4):
                    return U_snapped2, V_snapped2, W_snapped2
        
        return None
    
    # Try ranks from low to high
    # Known lower bounds: for 2x3x8, the rank is at least max(n*m, m*p, n*p) = max(6, 24, 16) = 24
    # But actually the lower bound is more complex. Let's try from 24 upward.
    
    # Actually, let's try a range of ranks
    for rank in range(24, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        result = try_als(T, rank, max_iters=100, tol=1e-8, seed=42)
        if result is not None:
            U_mat, V_mat, W_mat = result
            best_rank = rank
            best_result = {
                "U": U_mat.tolist(),
                "V": V_mat.tolist(),
                "W": W_mat.tolist()
            }
            break
    
    # If no better result found, return schoolbook
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
