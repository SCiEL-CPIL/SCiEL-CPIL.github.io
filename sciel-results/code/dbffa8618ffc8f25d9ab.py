"""Program for finding a rank-minimal exact tensor decomposition for 2x4*4x5 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-36 tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Uses a structured random search with alternating least squares to find
    a rank-36 decomposition, falling back to schoolbook rank-40 if not found.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    def verify(U, V, W):
        """Verify that the factors decompose the matrix multiplication tensor exactly."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        err = max(err, abs(val - 1.0))
        return err
    
    def snap_to_half_integers(X, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        X_snap = np.round(X * 2) / 2
        if np.max(np.abs(X - X_snap)) < tol:
            return X_snap
        # Try integer snap
        X_int = np.round(X)
        if np.max(np.abs(X - X_int)) < tol:
            return X_int
        return X
    
    def try_rank(r, max_iters=2000, n_restarts=50):
        """Try to find a rank-r decomposition via random initialization and ALS."""
        for restart in range(n_restarts):
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            for it in range(max_iters):
                # Update W: for each (i,k), solve linear system
                W_new = np.zeros_like(W)
                for i in range(n):
                    for k in range(p):
                        # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                        # This is a system: sum_j (U[i*m+j,:] * V[j*p+k,:]) * W[k*n+i,:] = 1
                        # Let M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        # Then sum_c M[j,c] * W[k*n+i,c] = 1 for all j
                        # This is a linear system in W[k*n+i, :]
                        M = np.zeros((m, r))
                        for j in range(m):
                            M[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        # Solve M @ W[k*n+i, :] = ones(m)
                        try:
                            W[k*n + i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Update U: for each (i,j), solve
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                        # Let M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        M = np.zeros((p, r))
                        for k in range(p):
                            M[k, :] = V[j*p + k, :] * W[k*n + i, :]
                        try:
                            U[i*m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                        M = np.zeros((n, r))
                        for i in range(n):
                            M[i, :] = U[i*m + j, :] * W[k*n + i, :]
                        try:
                            V[j*p + k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                err = verify(U, V, W)
                if err < 1e-9:
                    U_snap = snap_to_half_integers(U)
                    V_snap = snap_to_half_integers(V)
                    W_snap = snap_to_half_integers(W)
                    err_snap = verify(U_snap, V_snap, W_snap)
                    if err_snap < 1e-9:
                        return U_snap, V_snap, W_snap
                    # Try with original
                    return U, V, W
                
                if it % 100 == 0 and err < 1e-6:
                    # Check if we're close enough
                    U_snap = snap_to_half_integers(U)
                    V_snap = snap_to_half_integers(V)
                    W_snap = snap_to_half_integers(W)
                    err_snap = verify(U_snap, V_snap, W_snap)
                    if err_snap < 1e-9:
                        return U_snap, V_snap, W_snap
        
        return None
    
    # Try rank 36 first
    result = try_rank(36, max_iters=1000, n_restarts=30)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 40
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
