import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T[i,j,k] = 1 if i,j,k are "aligned"
    # For matrix mult C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i, j, k] = 1 (it's a standard matrix mult tensor)
    # Actually the tensor is: T[i,j,k] = 1 for all valid i,j,k
    # But we need to find rank-1 decompositions
    
    # The multiplication tensor for (n,m,p) has entries:
    # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T is a 3D array of all 1s with shape (n, m, p)
    
    # We want to find minimum rank r such that:
    # T[i,j,k] = sum_{s=1}^{r} U[i,j,s] * V[j,k,s] * W[k,i,s]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    
    # Using slice-and-fiber: fix k, look at the "fiber" T[:,:,k] which is an n×m matrix
    # For each k, T[:,:,k] = sum_s U[:,:,s] * V[:,k,s] * W[k,:,s]
    # This is a sum of rank-1 matrices in the (i,j) space
    
    # Let's use a systematic search approach
    # First, establish schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return mat.tolist()
    
    # Try to find lower rank using iterative refinement
    # Strategy: start with a candidate rank, try to find a valid decomposition
    
    best_result = {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    best_rank = rank_sb
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) / min(n,m,p) roughly, but let's be more careful
    # For (3,3,7): n*m=9, m*p=21, p*n=21
    # Known bounds suggest rank around 20-30 might be achievable
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U,V,W correctly decompose the multiplication tensor."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        # Reconstruct tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] should equal sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    val = 0.0
                    for s in range(r):
                        val += U[i*m + j, s] * V[j*p + k, s] * W[k*n + i, s]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Use a randomized approach to find lower rank decompositions
    def try_rank(r, max_iter=10000, seed=42):
        """Try to find a rank-r decomposition using iterative methods."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize
        for s in range(r):
            norm = np.linalg.norm(U[:, s])
            if norm > 1e-10:
                U[:, s] /= norm
                V[:, s] *= norm
                W[:, s] *= norm
        
        # Iterative refinement: alternate between solving for each factor
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each (i,j), U[i*m+j, s] should satisfy
            # sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1 for all k
            # This is: U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:] elementwise) = 1 for all k
            # Let's form a system: for fixed i,j, we have p equations
            # sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1, k=0..p-1
            
            # This is a linear system in U[i*m+j,:]
            # A_ij @ U[i*m+j,:] = 1 (vector of ones, length p)
            # where A_ij[k, s] = V[j*p+k, s] * W[k*n+i, s]
            
            # Similarly for V and W
            
            # Let's do coordinate descent: update one factor at a time
            for factor in range(3):
                if factor == 0:  # Update U
                    for i in range(n):
                        for j in range(m):
                            # Build system for U[i*m+j, :]
                            A_mat = np.zeros((p, r))
                            for k in range(p):
                                for s in range(r):
                                    A_mat[k, s] = V[j*p + k, s] * W[k*n + i, s]
                            b = np.ones(p)
                            # Solve least squares
                            if r <= p:
                                try:
                                    U[i*m + j, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, b, rcond=None)
                                except:
                                    pass
                            else:
                                # Underdetermined, use minimum norm solution
                                try:
                                    U[i*m + j, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                                except:
                                    pass
                elif factor == 1:  # Update V
                    for j in range(m):
                        for k in range(p):
                            # Build system for V[j*p+k, :]
                            A_mat = np.zeros((n, r))
                            for i in range(n):
                                for s in range(r):
                                    A_mat[i, s] = U[i*m + j, s] * W[k*n + i, s]
                            b = np.ones(n)
                            try:
                                V[j*p + k, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                            except:
                                pass
                else:  # Update W
                    for k in range(p):
                        for i in range(n):
                            # Build system for W[k*n+i, :]
                            A_mat = np.zeros((m, r))
                            for j in range(m):
                                for s in range(r):
                                    A_mat[j, s] = U[i*m + j, s] * V[j*p + k, s]
                            b = np.ones(m)
                            try:
                                W[k*n + i, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                            except:
                                pass
            
            # Check convergence
            if it % 10 == 0 or it == max_iter - 1:
                if verify_decomposition(U, V, W, tol=1e-6):
                    return U, V, W
        
        # Final check
        if verify_decomposition(U, V, W, tol=1e-8):
            return U, V, W
        return None
    
    # Try various ranks
    # Based on known results for similar problems, try ranks from ~15 to schoolbook
    for r in range(15, min(rank_sb, 40)):
        if time.time() > deadline - 10:
            break
        
        # Try a few random seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            result = try_rank(r, max_iter=2000, seed=seed)
            if result is not None:
                U, V, W = result
                if verify_decomposition(U, V, W, tol=1e-8):
                    if r < best_rank:
                        best_rank = r
                        best_result = {
                            "U": to_list(U),
                            "V": to_list(V),
                            "W": to_list(W)
                        }
                    break  # Found valid decomposition for this rank, move to next rank
        
        # If we found a valid decomposition for this rank, we can stop
        if best_rank == r:
            break
    
    return best_result
# EVOLVE-BLOCK-END
