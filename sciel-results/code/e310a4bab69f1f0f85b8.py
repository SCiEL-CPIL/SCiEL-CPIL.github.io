"""Matrix multiplication 3x3 x 3x7 - rank-minimal tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # In the W convention: W index is k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    def schoolbook():
        """Schoolbook decomposition, rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    best_rank = rank_sb
    
    def verify(U, V, W):
        """Verify that T = sum_r u_r ⊗ v_r ⊗ w_r."""
        # Reconstruct: T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Use einsum for efficiency
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_recon, T, atol=1e-9, rtol=1e-9)
    
    def round_half_integer(U, V, W, tol=1e-6):
        """Round entries to nearest half-integer and verify."""
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None
    
    # Try to find lower rank decompositions using optimization
    # Strategy: try ranks from lower bounds upward
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Also rank >= number of non-zero entries in any slice
    # For matrix multiplication, known lower bounds:
    # rank(MM_{n,m,p}) >= max(n*m, m*p, p*n, n*p)
    # Also: rank >= n*m*p / max(n,m,p) (from slicing arguments)
    lb = max(n*m, m*p, p*n, n*p)
    # For 3x3x7: lb = max(9, 21, 21, 21) = 21
    
    # Try optimization-based approach for ranks between lb and schoolbook
    # Use a simple iterative approach: random init + gradient-free local search
    
    # First, try a structured approach based on known results:
    # For MM_{3,3,7}, try rank 21, 22, ... up to some limit
    
    # Use a deterministic approach: try to construct rank-r decompositions
    # by solving the system of bilinear equations
    
    def try_rank_via_optimization(r, max_iter=5000, seed=42):
        """Try to find a rank-r decomposition via optimization."""
        rng = np.random.RandomState(seed)
        
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = rng.randn(n*m, r)
            V = rng.randn(m*p, r)
            W = rng.randn(p*n, r)
            
            # Use a simple iterative method: alternate least squares style
            # For each r, fix two of U,V,W and solve for the third
            # This is non-convex but can work for small problems
            
            # Actually, let's use a simpler approach:
            # The tensor T can be flattened to a matrix in 3 ways:
            # T_a (shape n*m x m*p*p*n) - not quite right
            # Let's use the standard tensor unfolding approach
            
            # For a rank-r decomposition T = sum u_r ⊗ v_r ⊗ w_r:
            # The 1-mode unfolding: T_(1) = U * (V^T ⊗ W^T)
            # T_(2) = V * (U^T ⊗ W^T)  
            # T_(3) = W * (U^T ⊗ V^T)
            
            # Flatten T to 1-mode: shape (n*m, m*p * p*n)
            T1 = T.reshape(n*m, m*p * p*n)
            T2 = T.transpose(1,0,2).reshape(m*p, n*m * p*n)
            T3 = T.transpose(2,0,1).reshape(p*n, n*m * m*p)
            
            # Alternating least squares:
            # Fix V,W: U = T1 @ (V⊗W)^+ 
            # Fix U,W: V = T2 @ (U⊗W)^+
            # Fix U,V: W = T3 @ (U⊗V)^+
            
            success = False
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Update U: T1 = U @ kron(V, W)
                # T1[a, b*p*n + c] = sum_r U[a,r] V[b,r] W[c,r]
                # So T1 = U @ (V ⊗ W) where V⊗W has shape (m*p*p*n, r)
                # Actually: (V ⊗ W)[b*p*n + c, r] = V[b,r] * W[c,r]
                VW = np.outer(V.ravel(), W.ravel()).reshape(m*p, p*n, r).transpose(1,0,2).reshape(p*n * m*p, r)
                # Wait, let me be more careful
                # T1[a, (b,c)] = sum_r U[a,r] V[b,r] W[c,r]
                # So T1 = U @ M where M[(b,c), r] = V[b,r] W[c,r]
                M = np.zeros((m*p * p*n, r))
                for b in range(m*p):
                    for c in range(p*n):
                        M[b * p*n + c, :] = V[b, :] * W[c, :]
                # U = T1 @ M^+
                try:
                    U_new = T1 @ np.linalg.lstsq(M, T1.T, rcond=None)[0].T
                except:
                    U_new = U
                
                # Update V: T2[b, (a,c)] = sum_r U[a,r] V[b,r] W[c,r]
                # T2 = V @ M2 where M2[(a,c), r] = U[a,r] W[c,r]
                M2 = np.zeros((n*m * p*n, r))
                for a in range(n*m):
                    for c in range(p*n):
                        M2[a * p*n + c, :] = U[a, :] * W[c, :]
                try:
                    V_new = T2 @ np.linalg.lstsq(M2, T2.T, rcond=None)[0].T
                except:
                    V_new = V
                
                # Update W: T3[c, (a,b)] = sum_r U[a,r] V[b,r] W[c,r]
                # T3 = W @ M3 where M3[(a,b), r] = U[a,r] V[b,r]
                M3 = np.zeros((n*m * m*p, r))
                for a in range(n*m):
                    for b in range(m*p):
                        M3[a * m*p + b, :] = U[a, :] * V[b, :]
                try:
                    W_new = T3 @ np.linalg.lstsq(M3, T3.T, rcond=None)[0].T
                except:
                    W_new = W
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if verify(U, V, W):
                    success = True
                    break
                    
            if success:
                # Try rounding
                rounded = round_half_integer(U, V, W)
                if rounded:
                    U_r, V_r, W_r = rounded
                    return U_r, V_r, W_r, r
                # Return without rounding if exact
                return U, V, W, r
        
        return None
    
    # Try ranks from lower bound upward
    # But this optimization is expensive, so let's be strategic
    # Known: for MM_{3,3,7}, the rank is at most 21 (from slicing: each column of C 
    # depends on at most 3*7=21 products, but we can do better)
    # Actually the rank could be as low as max(n*m, m*p, p*n) = 21
    
    # Let's try a few specific ranks with limited computation
    for r in [21, 22, 23, 24, 25, 27, 30, 35, 40, 45, 50, 55, 60]:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            break
        try:
            result = try_rank_via_optimization(r, max_iter=200, seed=42 + r)
            if result:
                U_r, V_r, W_r, rank_r = result
                if verify(U_r, V_r, W_r):
                    best_result = (U_r, V_r, W_r, rank_r)
                    best_rank = rank_r
        except Exception:
            continue
    
    # Convert best result to list-of-lists format
    U_best, V_best, W_best, rank_best = best_result
    U_out = U_best.tolist()
    V_out = V_best.tolist()
    W_out = W_best.tolist()
    
    # Ensure entries are exactly 0, 0.5, 1, 1.5, 2, etc. (half-integers)
    # Clean up floating point
    def clean_half_int(matrix):
        result = []
        for row in matrix:
            new_row = []
            for val in row:
                # Round to nearest half integer
                val2 = round(val * 2) / 2.0
                # Represent as exact float
                if val2 == int(val2):
                    new_row.append(int(val2))
                else:
                    new_row.append(val2)
            result.append(new_row)
        return result
    
    U_out = clean_half_int(U_out)
    V_out = clean_half_int(V_out)
    W_out = clean_half_int(W_out)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
