"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Returns a rank-minimal (or best found) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x7 matrix.
    
    The decomposition is defined by three matrices U, V, W of shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Such that for all i, j, k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    num_eqs = n * m * p  # 84
    
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank), dtype=int)
        V = np.zeros((dim_V, rank), dtype=int)
        W = np.zeros((dim_W, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, r):
        """Verify exact decomposition with half-integer entries."""
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :r] * V[j * p + k, :r], W[k * n + i, :r])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Precompute index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    U_rows = i_idx * m + j_idx
    V_rows = j_idx * p + k_idx
    W_rows = k_idx * n + i_idx
    
    def try_solve_rank(r, U_init=None, V_init=None, num_attempts=10):
        """Try to find a valid W for given rank r and optional initial U, V."""
        best_result = None
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 3:
                break
            
            # Initialize U and V
            if U_init is not None and V_init is not None and attempt == 0:
                U = U_init.copy()
                V = V_init.copy()
            else:
                # Dense random initialization with values in {-2,-1,0,1,2}
                density_U = min(0.7, 0.3 + 0.02 * (84 - r) / 10)
                density_V = min(0.7, 0.3 + 0.02 * (84 - r) / 10)
                
                U = np.zeros((dim_U, r), dtype=float)
                V = np.zeros((dim_V, r), dtype=float)
                
                for c in range(r):
                    # Random pattern for U column
                    mask_U = np.random.rand(dim_U) < density_U
                    if not np.any(mask_U):
                        mask_U[np.random.randint(dim_U)] = True
                    U[mask_U, c] = np.random.choice([-2, -1, 1, 2], size=np.sum(mask_U))
                    
                    # Random pattern for V column
                    mask_V = np.random.rand(dim_V) < density_V
                    if not np.any(mask_V):
                        mask_V[np.random.randint(dim_V)] = True
                    V[mask_V, c] = np.random.choice([-2, -1, 1, 2], size=np.sum(mask_V))
                
                # Round to integers for cleaner structure
                U = np.round(U).astype(int)
                V = np.round(V).astype(int)
            
            # Build the linear system A W = b
            A = np.zeros((num_eqs, dim_W * r))
            b = np.ones(num_eqs)
            
            for eq in range(num_eqs):
                u_row = U_rows[eq]
                v_row = V_rows[eq]
                w_row = W_rows[eq]
                for c in range(r):
                    coeff = U[u_row, c] * V[v_row, c]
                    if coeff != 0:
                        A[eq, w_row * r + c] = coeff
            
            # Check if system is well-posed
            if np.all(A == 0):
                continue
            
            # Solve using least squares
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=1e-12)
            except Exception:
                continue
            
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            # Reshape W_flat to W
            W_sol = W_flat.reshape((dim_W, r))
            
            # Check if W_sol is close to half-integers
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            max_err = np.max(np.abs(W_scaled - W_rounded))
            
            if max_err < 1e-5:
                W_int = W_rounded / 2.0
                
                if verify(U, V, W_int, r):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist(), "r": r}
            
            # Try rounding U and V to nearest integers and re-solve
            U_rounded = np.round(U).astype(int)
            V_rounded = np.round(V).astype(int)
            
            if not np.array_equal(U, U_rounded) or not np.array_equal(V, V_rounded):
                A2 = np.zeros((num_eqs, dim_W * r))
                for eq in range(num_eqs):
                    u_row = U_rows[eq]
                    v_row = V_rows[eq]
                    w_row = W_rows[eq]
                    for c in range(r):
                        coeff = U_rounded[u_row, c] * V_rounded[v_row, c]
                        if coeff != 0:
                            A2[eq, w_row * r + c] = coeff
                
                try:
                    W_flat2, residuals2, _, _ = np.linalg.lstsq(A2, b, rcond=1e-12)
                    if len(residuals2) > 0 and residuals2[0] < 1e-8:
                        W_sol2 = W_flat2.reshape((dim_W, r))
                        W_scaled2 = W_sol2 * 2
                        W_rounded2 = np.round(W_scaled2)
                        if np.max(np.abs(W_scaled2 - W_rounded2)) < 1e-5:
                            W_int2 = W_rounded2 / 2.0
                            if verify(U_rounded, V_rounded, W_int2, r):
                                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_int2.tolist(), "r": r}
                except Exception:
                    pass
        
        return best_result
    
    # Strategy 1: Try structured low-rank decompositions
    # Use the fact that matrix multiplication has algebraic structure
    
    # Strategy 2: Systematic search from rank 28 upward
    best_solution = None
    best_rank = 84
    
    # Try a few structured approaches first
    # Approach: Use Kronecker product structure
    for r in [28, 30, 32, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 83]:
        if time.time() > deadline - 10:
            break
        
        # Allocate more time for lower ranks
        if r <= 40:
            num_attempts = 20
        elif r <= 60:
            num_attempts = 15
        else:
            num_attempts = 8
        
        result = try_solve_rank(r, num_attempts=num_attempts)
        if result is not None:
            best_solution = result
            best_rank = r
            break
    
    # If we found a solution, try to improve it
    if best_solution is not None:
        for r in range(best_rank - 1, 27, -1):
            if time.time() > deadline - 5:
                break
            # Use the previous solution as initialization
            U_prev = np.array(best_solution["U"])[:, :r] if best_solution["r"] > r else np.zeros((dim_U, r))
            V_prev = np.array(best_solution["V"])[:, :r] if best_solution["r"] > r else np.zeros((dim_V, r))
            
            result = try_solve_rank(r, U_init=U_prev, V_init=V_prev, num_attempts=10)
            if result is not None:
                best_solution = result
                best_rank = r
    
    if best_solution is not None:
        return {"U": best_solution["U"], "V": best_solution["V"], "W": best_solution["W"]}
    
    # If no better rank found, return schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
