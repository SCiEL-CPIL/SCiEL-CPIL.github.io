"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Try to find a low-rank decomposition
    best_result = None
    best_rank = n * m * p  # schoolbook fallback
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    best_rank = rank_sb
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def try_rank(r, max_iters=200, seed=42):
        """Try to find a rank-r decomposition using alternating minimization."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Solve for W given U, V
            # For each (i,j,k): sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1
            # This is linear in W entries.
            # W[k*n+i, s] appears in constraint (i,j,k) with coefficient U[i*m+j,s]*V[j*p+k,s]
            
            # Build the linear system for W
            # Unknowns: W[0..p*n-1, 0..r-1] → p*n*r unknowns
            # Equations: n*m*p = 45
            
            # Actually, let's vectorize. For each (i,j,k), the equation is:
            # sum_s (U[i*m+j,s] * V[j*p+k,s]) * W[k*n+i,s] = 1
            # Let a_{(i,j,k),s} = U[i*m+j,s] * V[j*p+k,s]
            # Then sum_s a_{(i,j,k),s} * W[k*n+i,s] = 1
            
            # But W[k*n+i,s] is the same W entry used across different (i,j,k) for same k,i.
            # For fixed (k,i), W[k*n+i,s] appears in constraints for all j.
            
            # Let's build the full linear system.
            # Number of unknowns: p*n*r
            # Number of equations: n*m*p
            
            num_unk = p * n * r
            num_eq = n * m * p
            
            A_mat = np.zeros((num_eq, num_unk))
            b_vec = np.ones(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for s in range(r):
                            # W[k*n+i, s] is at index (k*n+i)*r + s
                            A_mat[eq_idx, (k*n+i)*r + s] = U[i*m+j, s] * V[j*p+k, s]
                        eq_idx += 1
            
            # Solve least squares
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W = W_flat.reshape(p*n, r)
            
            # Solve for U given V, W
            A_mat = np.zeros((num_eq, num_unk))  # reuse, but now unknowns are U
            # U[i*m+j, s] appears in constraint (i,j,k) with coefficient V[j*p+k,s]*W[k*n+i,s]
            A_mat = np.zeros((num_eq, n*m*r))
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for s in range(r):
                            A_mat[eq_idx, (i*m+j)*r + s] = V[j*p+k, s] * W[k*n+i, s]
                        eq_idx += 1
            
            U_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            U = U_flat.reshape(n*m, r)
            
            # Solve for V given U, W
            A_mat = np.zeros((num_eq, m*p*r))
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for s in range(r):
                            A_mat[eq_idx, (j*p+k)*r + s] = U[i*m+j, s] * W[k*n+i, s]
                        eq_idx += 1
            
            V_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            V = V_flat.reshape(m*p, r)
            
            # Check convergence
            if it % 20 == 0:
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1)**2
                if err < 1e-12:
                    break
        
        # Try rounding to integers and half-integers
        # First try rounding to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify(U_int, V_int, W_int):
            return U_int, V_int, W_int, r
        
        # Try rounding to half-integers
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        if verify(U_half, V_half, W_half):
            return U_half, V_half, W_half, r
        
        # Try mixed: some integer, some half
        # Check if close to integers
        U_close = np.abs(U - np.round(U)) < 0.01
        V_close = np.abs(V - np.round(V)) < 0.01
        W_close = np.abs(W - np.round(W)) < 0.01
        if U_close.all() and V_close.all() and W_close.all():
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r, r
        
        return None
    
    # Try various ranks
    for r in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        if time.time() > deadline - 15:
            break
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            result = try_rank(r, max_iters=100, seed=seed + r*100)
            if result is not None:
                U, V, W, rank = result
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U, V, W, rank)
                break
    
    U, V, W, rank = best_result
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
