"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook fallback (rank 24)
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    def verify(U, V, W, n, m, p):
        """Check if the decomposition is valid with exact integer/half-integer arithmetic."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    # Check if s is exactly 1.0
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def round_to_half_int(arr):
        """Round a numpy array to the nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def try_rank(r, max_als_iters=500, max_restarts=20):
        """Try to find a valid rank-r decomposition using ALS and rounding."""
        if time.time() > deadline:
            return None
            
        best_err = float('inf')
        best_solution = None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # ALS iterations
            for iteration in range(max_als_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        except:
                            pass
                
                # Calculate error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (s - 1.0)**2
                            
                if err < best_err:
                    best_err = err
                    if err < 1e-4:
                        # Try to round and verify
                        U_r = round_to_half_int(U)
                        V_r = round_to_half_int(V)
                        W_r = round_to_half_int(W)
                        if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
                        # If rounding failed, try a local search around the real solution
                        # This is a heuristic: perturb slightly and re-round
                        for _ in range(10):
                            U_p = U_r + np.random.choice([-0.5, 0, 0.5], U.shape)
                            V_p = V_r + np.random.choice([-0.5, 0, 0.5], V.shape)
                            W_p = W_r + np.random.choice([-0.5, 0, 0.5], W.shape)
                            if verify(U_p.tolist(), V_p.tolist(), W_p.tolist(), n, m, p):
                                return {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}
                            
                if err < 1e-6:
                    # Converged to a real solution, try harder to find discrete
                    U_r = round_to_half_int(U)
                    V_r = round_to_half_int(V)
                    W_r = round_to_half_int(W)
                    if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
            
        return None

    # Try rank 11 first
    result = try_rank(11)
    if result is not None:
        best_result = result
        best_rank = 11
        return best_result
    
    # Try rank 12
    result = try_rank(12)
    if result is not None:
        best_result = result
        best_rank = 12
        return best_result

    # If random search fails, return the best we have (schoolbook)
    return best_result
# EVOLVE-BLOCK-END
