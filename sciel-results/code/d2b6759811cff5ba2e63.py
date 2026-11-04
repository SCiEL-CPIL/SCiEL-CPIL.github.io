"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U:(n*m, r), V:(m*p, r), W:(p*n, r)
    # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    deadline = time.time() + 140  # safety margin
    
    # Start with schoolbook rank 50
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank using random search + verification
    # The constraint: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    def verify(U, V, W, tol=1e-6):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for rr in range(r):
                        val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def try_rank(r):
        """Try to find a rank-r decomposition using random initialization and verification."""
        if r >= best_rank:
            return None, None, None
        
        # Random search with multiple restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Scale to help with magnitude
            U *= 2.0
            V *= 2.0
            W *= 0.5
            
            # Simple iterative refinement: adjust to satisfy constraints
            # Use a simple approach: for each (i,j,k), the constraint is
            # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is nonlinear. Let's try a different approach.
            
            # Alternative: use the fact that we can fix W to be simple
            # Try W[k*n+i, r] = 1 for all (structured)
            # Then need: sum_r U[i*m+j,r]*V[j*p+k,r] = 1
            # This is like a matrix factorization problem
            
            # Let's try: fix W to identity-like structure
            # Actually, let's try a more systematic approach
            
            # Try rank r with structured W
            W_struct = np.zeros((p*n, r))
            for k in range(p):
                for i in range(n):
                    W_struct[k*n+i, (k*n+i) % r] = 1.0 if (k*n+i) < r else 0.0
            # This doesn't work well for r < p*n
            
            # Better: try to use known low-rank constructions
            # For 2 x m x p, try rank = m*p + 2*min(m,p) - something
            
            break  # Random search is too slow, try structured approach
        
        return None, None, None
    
    # Try structured constructions
    # For 2 x m x p matrix multiplication, one known bound is rank <= m*p + 2*min(m,p) - 1
    # But let's try to find explicit low-rank decompositions
    
    # Try rank = 25 (m*p) + 2 = 27? Or other values
    # Actually, let's try a different approach: use the fact that
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as:
    # C[i,k] = sum_r f_r(A) * g_r(B)
    # where f_r and g_r are linear forms
    
    # For rank r, we need r pairs of linear forms (f_r, g_r) such that
    # sum_r f_r(A) * g_r(B) = sum_j A[i,j] B[j,k] for each (i,k)
    
    # This means: for each (i,k), the bilinear form (A,B) -> sum_j A[i,j] B[j,k]
    # must equal sum_r f_r(A) g_r(B)
    
    # Let f_r(A) = sum_{i,j} alpha_{r,i,j} A[i,j]
    # g_r(B) = sum_{j,k} beta_{r,j,k} B[j,k]
    
    # Then: sum_r alpha_{r,i',j'} beta_{r,j',k'} = delta[i',i] delta[j',j] delta[k',k]... 
    # No, we need: sum_r alpha_{r,i,j} beta_{r,j,k} = 1 for the specific (i,j,k) in the sum
    
    # Actually, the condition is:
    # sum_r alpha_{r,i,j} beta_{r,j,k} = 1 for all i,j,k (since each term A[i,j]B[j,k] has coeff 1)
    
    # So we need: for all i in [0,2), j in [0,5), k in [0,5):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try to find this with rank as low as possible
    
    # Try various ranks from low to high
    for r in range(10, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize to help convergence
            for rr in range(r):
                if np.abs(U[:, rr]).max() > 0:
                    U[:, rr] /= np.abs(U[:, rr]).max()
                if np.abs(V[:, rr]).max() > 0:
                    V[:, rr] /= np.abs(V[:, rr]).max()
                if np.abs(W[:, rr]).max() > 0:
                    W[:, rr] /= np.abs(W[:, rr]).max()
            
            # Check if this random init happens to work (unlikely)
            if verify(U, V, W):
                found = True
                break
        
        if found:
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
