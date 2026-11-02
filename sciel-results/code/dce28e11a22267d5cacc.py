"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition search."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    return tensor

def _verify(U, V, W, tensor, tol=1e-6):
    """Check if U @ V^T @ W^T approximates the tensor"""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reconstructed: sum over rank of U[:,r] * V[:,r] * W[:,r]
    # Check a few entries
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Use einsum for efficient check
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    # Compare with tensor
    diff = np.abs(recon - tensor)
    return np.max(diff) < tol

def _als_step(U, V, W, tensor, n, m, p, lr=0.1, max_iter=200):
    """Alternating Least Squares to refine the decomposition"""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Update U: for each row i, U[i,:] = argmin sum_j,k tensor[i,j,k] * V[j,:] * W[k,:]
        # This is a least squares problem
        for i in range(n*m):
            # Build the system: U[i,:] * (V[j,:] * W[k,:])^T = tensor[i,j,k]
            # Reshape V and W
            VW = np.einsum('jr,kr->jkr', V, W)  # (m*p, p*n, rank)
            VW_flat = VW.reshape(-1, rank)  # (m*p*p*n, rank)
            target = tensor[i].reshape(-1)  # (m*p*p*n,)
            # Solve: U[i,:] @ VW_flat^T = target
            # U[i,:] = (VW_flat @ VW_flat^T)^-1 @ VW_flat @ target
            A = VW_flat @ VW_flat.T  # (rank, rank)
            b = VW_flat @ target  # (rank,)
            try:
                U[i,:] = np.linalg.solve(A + 1e-8*np.eye(rank), b)
            except:
                U[i,:] = np.linalg.lstsq(VW_flat, target, rcond=None)[0]
        
        # Update V
        for j in range(m*p):
            UW = np.einsum('ir,kr->ikr', U, W)  # (n*m, p*n, rank)
            UW_flat = UW.reshape(-1, rank)
            target = tensor[:,j,:].reshape(-1)
            A = UW_flat @ UW_flat.T
            b = UW_flat @ target
            try:
                V[j,:] = np.linalg.solve(A + 1e-8*np.eye(rank), b)
            except:
                V[j,:] = np.linalg.lstsq(UW_flat, target, rcond=None)[0]
        
        # Update W
        for k in range(p*n):
            UV = np.einsum('ir,jr->ijr', U, V)  # (n*m, m*p, rank)
            UV_flat = UV.reshape(-1, rank)
            target = tensor[:,:,k].reshape(-1)
            A = UV_flat @ UV_flat.T
            b = UV_flat @ target
            try:
                W[k,:] = np.linalg.solve(A + 1e-8*np.eye(rank), b)
            except:
                W[k,:] = np.linalg.lstsq(UV_flat, target, rcond=None)[0]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep safety margin
    
    tensor = _build_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank via random initialization + ALS
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But for matrix multiplication, better bounds exist
    
    # Try various ranks from low to high
    min_possible = max(n*m, m*p, p*n)
    
    for target_rank in range(min_possible, rank_schoolbook + 1):
        if time.time() > deadline - 30:  # Safety margin
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(random.randint(0, 1000000))
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Run ALS
            U, V, W = _als_step(U, V, W, tensor, n, m, p, max_iter=100)
            
            # Check if it's a valid decomposition
            if _verify(U, V, W, tensor, tol=1e-5):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break  # Found valid decomposition at this rank, try lower next
        
        if best_rank == target_rank:
            # Found it, can stop trying higher ranks (but we want lower)
            # Actually we're going from low to high, so if we found at target_rank,
            # we should continue to see if we can do better (lower) - but we already tried lower
            # So we can stop
            break
    
    # Convert to list format as required
    U_list = [[float(U[i,j]) for j in range(best_rank)] for i in range(n*m)]
    V_list = [[float(V[i,j]) for j in range(best_rank)] for i in range(m*p)]
    W_list = [[float(W[i,j]) for j in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
