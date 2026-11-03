"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook as fallback
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try various rank targets with random initialization + optimization
    # Use a simple approach: try to find low-rank decomposition via
    # iterative refinement of random factor matrices
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        # Reconstruct: sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(recon, T, atol=tol, rtol=0)
    
    def try_rank(r, n_iter=2000, seed=None):
        """Try to find rank-r decomposition."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Simple iterative optimization: minimize ||T - sum u_c ⊗ v_c ⊗ w_c||^2
        # Use alternating least squares / gradient descent
        
        for it in range(n_iter):
            if time.time() > deadline:
                break
            
            # Update U: for each column c, U[:,c] should satisfy
            # sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
            # => U[:,c] = (V[:,c] ⊗ W[:,c])^T * T_flattened / ||V[:,c] ⊗ W[:,c]||^2
            # This is a linear system
            
            # Flatten T to (n*m * m*p * p*n,)
            # The equation is: sum_c vec(U[:,c]) * vec(V[:,c])^T * ... 
            # Actually, let's use: T = sum_c U_c ⊗ V_c ⊗ W_c
            # Residual R = T - sum_c U_c ⊗ V_c ⊗ W_c
            # Update U_c: minimize ||R + U_c ⊗ V_c ⊗ W_c - sum_{c'≠c} U_c' ⊗ V_c' ⊗ W_c'||^2
            # => U_c = argmin ||(sum_{c'≠c} U_c'⊗V_c'⊗W_c') - T + U_c⊗V_c⊗W_c||^2
            # This is a linear system in U_c
            
            # For efficiency, compute the "cross term" tensor
            # R_others = T - sum_c U_c ⊗ V_c ⊗ W_c + U_c ⊗ V_c ⊗ W_c = T - sum_{c'≠c} ...
            # Actually simpler: residual when we change U_c by delta:
            # ||(T - sum_all) + delta_U ⊗ V_c ⊗ W_c||^2
            # = ||residual||^2 + 2*delta_U^T*(V_c⊗W_c)^T*residual + delta_U^T*(V_c⊗W_c)^T*(V_c⊗W_c)*delta_U
            # Optimal: delta_U = -(V_c⊗W_c)^T*residual / ||V_c⊗W_c||^2
            # So U_c_new = U_c + delta_U = U_c - (V_c⊗W_c)^T*residual / ||V_c⊗W_c||^2
            
            # But computing residual each time is O(n*m*m*p*p*n*r) which is expensive
            # Let's use a simpler approach: batch update
            
            # Compute current reconstruction
            recon = np.einsum('ac,bc,dc->abd', U, V, W)
            residual = T - recon
            
            for c in range(r):
                # V_c ⊗ W_c has shape (m*p * p*n,)
                vw = np.outer(V[:, c], W[:, c])
                # (V_c ⊗ W_c)^T * residual_flattened
                # residual_flattened has shape (n*m, m*p, p*n)
                # We need: sum over b,d of residual[a,b,d] * V_c[b] * W_c[d] for each a
                rhs = np.einsum('abd,b,d->a', residual, V[:, c], W[:, c])
                denom = np.dot(V[:, c], V[:, c]) * np.dot(W[:, c], W[:, c])
                if denom > 1e-12:
                    U[:, c] += rhs / denom
            
            # Similarly update V
            recon = np.einsum('ac,bc,dc->abd', U, V, W)
            residual = T - recon
            for c in range(r):
                uw = np.outer(U[:, c], W[:, c])
                rhs = np.einsum('abd,a,d->b', residual, U[:, c], W[:, c])
                denom = np.dot(U[:, c], U[:, c]) * np.dot(W[:, c], W[:, c])
                if denom > 1e-12:
                    V[:, c] += rhs / denom
            
            # Similarly update W
            recon = np.einsum('ac,bc,dc->abd', U, V, W)
            residual = T - recon
            for c in range(r):
                uv = np.outer(U[:, c], V[:, c])
                rhs = np.einsum('abd,a,b->d', residual, U[:, c], V[:, c])
                denom = np.dot(U[:, c], U[:, c]) * np.dot(V[:, c], V[:, c])
                if denom > 1e-12:
                    W[:, c] += rhs / denom
        
        # Check if we found an exact solution
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.max(np.abs(recon - T))
        if err < 1e-8:
            return U, V, W, True
        return U, V, W, False
    
    # Try ranks from low to high
    for r in range(20, best_rank + 1, 2):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            U_try, V_try, W_try, success = try_rank(r, n_iter=1000, seed=42 + attempt * 1000 + r)
            if success:
                # Verify more strictly
                if verify(U_try, V_try, W_try, tol=1e-10):
                    # Try to round to rational values
                    U_round = np.round(U_try, 6)
                    V_round = np.round(V_try, 6)
                    W_round = np.round(W_try, 6)
                    if verify(U_round, V_round, W_round, tol=1e-9):
                        best_U, best_V, best_W = U_round, V_round, W_round
                        best_rank = r
                        found = True
                        break
                    else:
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = r
                        found = True
                        break
        
        if found:
            # Try to reduce further
            continue
        else:
            # If we can't find a solution at this rank, try slightly higher
            continue
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list(mat):
        """Convert numpy array to list, trying to use half-integers."""
        result = []
        for row in mat:
            r = []
            for val in row:
                # Try to represent as half-integer
                v = float(val)
                # Check if close to a half-integer
                half = round(v * 2) / 2.0
                if abs(v - half) < 1e-6:
                    r.append(half)
                else:
                    r.append(v)
            result.append(r)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
