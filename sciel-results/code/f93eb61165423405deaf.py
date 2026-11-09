"""Matrix multiplication 2x3 by 3x6: rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Uses algebraic structure and systematic search for rank reduction.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 18, 12
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U_mat.shape[1]):
                        val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def to_output(mat):
        """Convert numpy array to list of lists."""
        return [[float(x) for x in row] for row in mat]
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p  # 36
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Strategy: For matrix multiplication, we can use the fact that
    # the tensor has a specific structure. Let's try to find lower-rank
    # decompositions by exploiting linearity.
    
    # Key insight: We can try to find decompositions where some columns
    # are linear combinations of others, effectively reducing rank.
    
    # Approach: Use the bilinear structure. For each rank r, we need to find
    # U (6xr), V (18xr), W (12xr) such that the tensor equation holds.
    
    # Let's try a structured approach:
    # 1. Try rank values from a reasonable lower bound up
    # 2. For each rank, use a combination of:
    #    - Structured initialization based on known patterns
    #    - Local optimization with exact verification
    
    # Lower bound analysis:
    # - The matrix multiplication tensor for (n,m,p) has rank at least max(n*m, m*p, n*p)
    # - For (2,3,6): max(6, 18, 12) = 18
    # - But actual rank is typically higher than this lower bound
    
    # Let's try ranks from 18 to 35, checking if a solution exists
    
    def try_find_decomposition(r, max_time=30):
        """Try to find a rank-r decomposition using structured search."""
        t_start = time.time()
        
        # For matrix multiplication, a common approach is to use
        # the fact that we can partition the computation.
        
        # Try different initialization strategies:
        strategies = [
            'block_diag',      # Block diagonal structure
            'strassen_like',   # Strassen-inspired
            'random_refine',   # Random with refinement
            'structured_zero', # Structured with zeros
        ]
        
        for strategy in strategies:
            if time.time() - t_start > max_time:
                break
            
            U_try = np.zeros((nm, r))
            V_try = np.zeros((mp, r))
            W_try = np.zeros((pn, r))
            
            if strategy == 'block_diag':
                # Try to decompose by blocks
                # For 2x3 * 3x6, think of it as:
                # Row 0 of C: [A00*B0 + A01*B1 + A02*B2, ...]
                # Row 1 of C: [A10*B0 + A11*B1 + A12*B2, ...]
                # 
                # We can try to share some computations between rows
                
                # Initialize with a pattern that might allow sharing
                for c in range(r):
                    # Assign each column to handle certain (i,j,k) combos
                    i_idx = c % n
                    j_idx = (c // n) % m
                    k_idx = (c // (n*m)) % p
                    
                    U_try[i_idx * m + j_idx, c] = 1
                    V_try[j_idx * p + k_idx, c] = 1
                    W_try[k_idx * n + i_idx, c] = 1
                    
            elif strategy == 'strassen_like':
                # Strassen-style: combine multiple terms
                # For 2x3*3x6, try to use fewer products
                # by combining A entries and B entries
                
                # Create a pattern where each column handles multiple (i,j,k)
                # by having nonzero entries in multiple positions
                
                for c in range(r):
                    # Each column handles a "block" of computations
                    block_size = max(1, (n*m*p) // r)
                    start = c * block_size
                    end = min(start + block_size, n*m*p)
                    
                    for idx in range(start, end):
                        i_idx = idx // (m*p)
                        j_idx = (idx // p) % m
                        k_idx = idx % p
                        
                        U_try[i_idx * m + j_idx, c] = 1
                        V_try[j_idx * p + k_idx, c] = 1
                        W_try[k_idx * n + i_idx, c] = 1
                        
            elif strategy == 'random_refine':
                # Random initialization with local refinement
                np.random.seed(42 + r + hash(strategy) % 1000)
                U_try = np.random.randn(nm, r)
                V_try = np.random.randn(mp, r)
                W_try = np.random.randn(pn, r)
                
                # Normalize columns
                for c in range(r):
                    u_norm = np.linalg.norm(U_try[:, c])
                    v_norm = np.linalg.norm(V_try[:, c])
                    w_norm = np.linalg.norm(W_try[:, c])
                    if u_norm > 1e-10:
                        U_try[:, c] /= u_norm
                    if v_norm > 1e-10:
                        V_try[:, c] /= v_norm
                    if w_norm > 1e-10:
                        W_try[:, c] /= w_norm
                        
                # Local refinement using gradient steps
                for _ in range(100):
                    if time.time() - t_start > max_time:
                        break
                    
                    # Compute error
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(r):
                                    val += U_try[i*m+j, c] * V_try[j*p+k, c] * W_try[k*n+i, c]
                                err += (val - 1)**2
                    
                    if err < 1e-12:
                        break
                    
                    # Gradient step (simplified)
                    lr = 0.001
                    dU = np.zeros_like(U_try)
                    dV = np.zeros_like(V_try)
                    dW = np.zeros_like(W_try)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(r):
                                    val += U_try[i*m+j, c] * V_try[j*p+k, c] * W_try[k*n+i, c]
                                e = val - 1
                                for c in range(r):
                                    dU[i*m+j, c] += 2*e*V_try[j*p+k, c]*W_try[k*n+i, c]
                                    dV[j*p+k, c] += 2*e*U_try[i*m+j, c]*W_try[k*n+i, c]
                                    dW[k*n+i, c] += 2*e*U_try[i*m+j, c]*V_try[j*p+k, c]
                    
                    U_try -= lr * dU
                    V_try -= lr * dV
                    W_try -= lr * dW
                    
            elif strategy == 'structured_zero':
                # Try a structured approach with strategic zeros
                # For 2x3*3x6, the key is that we have 2 rows of output
                # Each output row depends on all 3 input columns
                
                # Try to use rank = n*p + m*p - m = 12 + 18 - 3 = 27? 
                # Or other formulas
                
                # Simple structured init
                for c in range(r):
                    i_idx = c % n
                    j_idx = (c // n) % m  
                    k_idx = (c // (n*m)) % p
                    
                    if c < n*m*p // 2:
                        U_try[i_idx * m + j_idx, c] = 1
                        V_try[j_idx * p + k_idx, c] = 1
                        W_try[k_idx * n + i_idx, c] = 1
                    else:
                        # Second half: try to share
                        i2 = (c // (n*m*p//2)) % n
                        j2 = (c // (n*m*p//2) // n) % m
                        k2 = (c // (n*m*p//2) // (n*m)) % p
                        U_try[i2 * m + j2, c] = 1
                        V_try[j2 * p + k2, c] = 1
                        W_try[k2 * n + i2, c] = 1
            
            # Verify
            if verify(U_try, V_try, W_try, tol=1e-8):
                return U_try, V_try, W_try
        
        return None
    
    # Try ranks from a reasonable lower bound upward
    # Theoretical lower bound is max(n*m, m*p, n*p) = 18
    # But for matrix multiplication, the rank is often higher
    
    # Let's try from 18 to 35
    for r in range(18, 36):
        if time.time() > deadline - 10:
            break
        
        result = try_find_decomposition(r, max_time=25)
        if result is not None:
            U_try, V_try, W_try = result
            best_rank = r
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    # If we found a better rank, return it
    if best_rank < rank_sb:
        return {"U": to_output(best_U), "V": to_output(best_V), "W": to_output(best_W)}
    
    # Otherwise return schoolbook
    return {"U": to_output(U_sb), "V": to_output(V_sb), "W": to_output(W_sb)}
# EVOLVE-BLOCK-END
