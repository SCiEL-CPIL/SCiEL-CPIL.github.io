"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    Strategy:
    1. Construct the schoolbook decomposition (Rank 24) as a fallback.
    2. Attempt to search for a lower rank decomposition using local optimization.
    3. If a valid lower-rank solution with half-integer entries is found within the time limit, return it.
    4. Otherwise, return the schoolbook solution.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_sb = n * m * p
    
    # 1. Schoolbook Decomposition (Fallback)
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    max_time = 110.0  # Safety margin for 120s limit
    
    def verify_solution(U, V, W, r):
        """
        Check if the solution is valid:
        1. Entries are half-integers.
        2. The tensor decomposition is exact.
        """
        # Check half-integer constraint
        for c in range(r):
            for val in U[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
            for val in V[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
            for val in W[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
        
        # Check exact decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[c][u_idx] * V[c][v_idx] * W[c][w_idx]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(U, V, W, r):
        """
        Attempt to snap the floating point solution to the nearest half-integers.
        Returns True if the snapped solution is exact.
        """
        U_snap = [[round(val * 2) / 2.0 for val in row] for row in U]
        V_snap = [[round(val * 2) / 2.0 for val in row] for row in V]
        W_snap = [[round(val * 2) / 2.0 for val in row] for row in W]
        
        return verify_solution(U_snap, V_snap, W_snap, r)

    # 2. Search for lower rank
    # We try ranks from rank_sb - 1 down to a lower bound.
    # Given the difficulty, we limit the search to a few ranks and short times.
    for r in range(rank_sb - 1, max(10, rank_sb - 5), -1):
        if time.time() - start_time > max_time:
            break
            
        # Define the error function
        def error_func(params):
            U_flat = params[:r * nm]
            V_flat = params[r * nm : r * nm + r * mp]
            W_flat = params[r * nm + r * mp :]
            
            U = U_flat.reshape(r, nm)
            V = V_flat.reshape(r, mp)
            W = W_flat.reshape(r, pn)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = 0.0
                        for c in range(r):
                            s += U[c, u_idx] * V[c, v_idx] * W[c, w_idx]
                        err += (s - 1.0) ** 2
            return err

        # Try a few random initializations
        found = False
        for attempt in range(3):
            if time.time() - start_time > max_time:
                break
                
            # Random initialization
            x0 = np.random.uniform(-1, 1, size=r * (nm + mp + pn))
            
            try:
                # Use Nelder-Mead for small dimensions
                res = minimize(error_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Extract U, V, W
                U_flat = res.x[:r * nm]
                V_flat = res.x[r * nm : r * nm + r * mp]
                W_flat = res.x[r * nm + r * mp :]
                
                U = U_flat.reshape(r, nm).tolist()
                V = V_flat.reshape(r, mp).tolist()
                W = W_flat.reshape(r, pn).tolist()
                
                if snap_to_half_integers(U, V, W, r):
                    return {"U": U, "V": V, "W": W}
        
    # 3. Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
