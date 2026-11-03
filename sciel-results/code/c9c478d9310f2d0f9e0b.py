import time
import numpy as np
from scipy.optimize import minimize
import random

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS

def _verify_scaled(U2, V2, W2):
    """Verify with scaled integer entries where product sum should be 8."""
    n, m, p = N, M, P
    r = len(U2[0])
    if len(U2) != n * m: return False
    if len(V2) != m * p: return False
    if len(W2) != p * n: return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U2[i * m + j][c] * V2[j * p + k][c] * W2[k * n + i][c]
                if s != 8:
                    return False
    return True

def _als_step_u(V2, W2, r):
    """Update U2 given V2 and W2 using least squares."""
    n, m, p = N, M, P
    U2 = [[0] * r for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            # Build A_ij: p x r
            # A_ij[k, c] = V2[j*p+k][c] * W2[k*n+i][c]
            A = np.zeros((p, r))
            for k in range(p):
                for c in range(r):
                    A[k, c] = V2[j * p + k][c] * W2[k * n + i][c]
            b = np.ones(p) * 8.0
            # Solve A * u = b in least squares sense
            # u = argmin ||A*u - b||^2
            try:
                u, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                u = np.zeros(r)
            # Round to nearest integer
            u_int = np.round(u).astype(int)
            for c in range(r):
                U2[i * m + j][c] = u_int[c]
    return U2

def _als_step_v(U2, W2, r):
    """Update V2 given U2 and W2 using least squares."""
    n, m, p = N, M, P
    V2 = [[0] * r for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            # Build B_jk: n x r
            # B_jk[i, c] = U2[i*m+j][c] * W2[k*n+i][c]
            B = np.zeros((n, r))
            for i in range(n):
                for c in range(r):
                    B[i, c] = U2[i * m + j][c] * W2[k * n + i][c]
            b = np.ones(n) * 8.0
            try:
                v, residuals, rank, sv = np.linalg.lstsq(B, b, rcond=None)
            except np.linalg.LinAlgError:
                v = np.zeros(r)
            v_int = np.round(v).astype(int)
            for c in range(r):
                V2[j * p + k][c] = v_int[c]
    return V2

def _als_step_w(U2, V2, r):
    """Update W2 given U2 and V2 using least squares."""
    n, m, p = N, M, P
    W2 = [[0] * r for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            # Build C_ki: m x r
            # C_ki[j, c] = U2[i*m+j][c] * V2[j*p+k][c]
            C = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    C[j, c] = U2[i * m + j][c] * V2[j * p + k][c]
            b = np.ones(m) * 8.0
            try:
                w, residuals, rank, sv = np.linalg.lstsq(C, b, rcond=None)
            except np.linalg.LinAlgError:
                w = np.zeros(r)
            w_int = np.round(w).astype(int)
            for c in range(r):
                W2[k * n + i][c] = w_int[c]
    return W2

def _local_search(U2, V2, W2, r, max_iter=100):
    """Try to fix errors by flipping individual entries."""
    n, m, p = N, M, P
    
    def calc_error():
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U2[i * m + j][c] * V2[j * p + k][c] * W2[k * n + i][c]
                    err += (s - 8) ** 2
        return err

    current_err = calc_error()
    if current_err == 0:
        return True

    for _ in range(max_iter):
        improved = False
        # Try flipping a random entry in U, V, or W
        mat_idx = random.choice([0, 1, 2])
        if mat_idx == 0:
            row = random.randint(0, n * m - 1)
            col = random.randint(0, r - 1)
            old_val = U2[row][col]
            # Try adding -1, 1
            for delta in [-1, 1]:
                U2[row][col] = old_val + delta
                new_err = calc_error()
                if new_err < current_err:
                    current_err = new_err
                    improved = True
                    break
                else:
                    U2[row][col] = old_val
        elif mat_idx == 1:
            row = random.randint(0, m * p - 1)
            col = random.randint(0, r - 1)
            old_val = V2[row][col]
            for delta in [-1, 1]:
                V2[row][col] = old_val + delta
                new_err = calc_error()
                if new_err < current_err:
                    current_err = new_err
                    improved = True
                    break
                else:
                    V2[row][col] = old_val
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, r - 1)
            old_val = W2[row][col]
            for delta in [-1, 1]:
                W2[row][col] = old_val + delta
                new_err = calc_error()
                if new_err < current_err:
                    current_err = new_err
                    improved = True
                    break
                else:
                    W2[row][col] = old_val
        
        if not improved:
            break
            
    return current_err == 0

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = N, M, P
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U0 = [[0] * rank_sb for _ in range(dim_u)]
    V0 = [[0] * rank_sb for _ in range(dim_v)]
    W0 = [[0] * rank_sb for _ in range(dim_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    best_result = {"U": [[x/2 for x in row] for row in U0], 
                   "V": [[x/2 for x in row] for row in V0], 
                   "W": [[x/2 for x in row] for row in W0]}
    best_rank = rank_sb

    # Search for lower ranks
    # Lower bound is max(dim_u, dim_v, dim_w) = 20
    # We try ranks from 25 to 59
    for r in range(25, rank_sb):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize V2, W2 with small random integers
            V2 = [[random.randint(-2, 2) for _ in range(r)] for _ in range(dim_v)]
            W2 = [[random.randint(-2, 2) for _ in range(r)] for _ in range(dim_w)]
            
            # Run ALS for a few iterations
            for _ in range(10):
                if time.time() > deadline:
                    break
                U2 = _als_step_u(V2, W2, r)
                V2 = _als_step_v(U2, W2, r)
                W2 = _als_step_w(U2, V2, r)
                
                # Check if we have a solution
                if _verify_scaled(U2, V2, W2):
                    # Convert to half-integers
                    U_res = [[x / 2 for x in row] for row in U2]
                    V_res = [[x / 2 for x in row] for row in V2]
                    W_res = [[x / 2 for x in row] for row in W2]
                    best_result = {"U": U_res, "V": V_res, "W": W_res}
                    best_rank = r
                    found = True
                    break
            
            if found:
                break
                
        if found:
            break

    return best_result
