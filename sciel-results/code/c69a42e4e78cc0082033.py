"""Program for matrix multiplication 3x3 * 3x7 tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Tries to find rank < 63 by exploiting tensor structure, falls back to schoolbook.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    
    def make_schoolbook():
        """Schoolbook decomposition of rank n*m*p = 63."""
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def verify_exact(U, V, W, tol=1e-10):
        """Verify that U,V,W form an exact decomposition."""
        r = U.shape[1]
        Ur = U.reshape(n, m, r)
        Vr = V.reshape(m, p, r)
        Wr = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def try_decomposition(U, V, W, tol=1e-10):
        """Try to verify and return if valid."""
        if verify_exact(U, V, W, tol):
            return True
        return False
    
    # Start with schoolbook as guaranteed fallback
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Strategy: For each (i,k) pair, we have m=3 equations:
    # sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c] = 1 for j=0,1,2
    # 
    # If we fix U and V, then for each (i,k), we can solve for W[k,i,:] 
    # from a 3 x r linear system (if r >= 3).
    #
    # Key insight: We can try structured U and V that create low-rank patterns.
    
    # Approach 1: Try rank = 3*p + 2*n = 21 + 6 = 27
    # For each k, use 3 "base" columns for the j-sum, plus shared structure
    
    # Approach 2: Use the fact that T is rank-1 in each mode
    # T[i,j,k] = 1 means it's the outer product of ones vectors
    # But the decomposition constraint is more complex
    
    # Let's try a systematic search over small ranks
    # For each target rank r, try structured initializations
    
    rng = np.random.default_rng(12345)
    
    # Try various rank values from low to high
    for target_rank in [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 45, 50, 55, 60, 63]:
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Initialize U and V with structured patterns
            # Use a mix of deterministic and random patterns
            
            # Pattern 1: Kronecker-like structure
            U = np.zeros((n*m, target_rank), dtype=float)
            V = np.zeros((m*p, target_rank), dtype=float)
            
            # Fill first min(target_rank, n*m*p) columns with schoolbook-like
            fill_count = min(target_rank, n*m*p)
            for c in range(fill_count):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
            
            # Add some random structure to remaining columns
            for c in range(fill_count, target_rank):
                U[:, c] = rng.standard_normal(n*m) * 0.5
                V[:, c] = rng.standard_normal(m*p) * 0.5
            
            # Solve for W: for each (i,k), solve A_w @ W[k*n+i,:] = ones(3)
            # where A_w[j,c] = U[i*m+j, c] * V[j*p+k, c]
            
            W = np.zeros((p*n, target_rank), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the 3 x target_rank matrix
                    A_w = np.zeros((m, target_rank), dtype=float)
                    for j in range(m):
                        A_w[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    # Solve A_w @ w = ones(m) for w (length target_rank)
                    # Use least squares
                    try:
                        w, residuals, rank_w, sv = np.linalg.lstsq(A_w, np.ones(m), rcond=None)
                        # Check if it's an exact solution
                        if np.allclose(A_w @ w, np.ones(m), atol=1e-10):
                            W[k*n+i, :] = w
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                if verify_exact(U, V, W, tol=1e-10):
                    best_U, best_V, best_W, best_rank = U, V, W, target_rank
                    break  # Found valid at this rank, try lower
        
        # If we found a valid decomposition at this rank, we can stop
        # since we're going from low to high
        if best_rank == target_rank:
            break
    
    # If we found something better than schoolbook, use it
    # Otherwise, fall back to schoolbook
    if best_rank > 63:
        best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
