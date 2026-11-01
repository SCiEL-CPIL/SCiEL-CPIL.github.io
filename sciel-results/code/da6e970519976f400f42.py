import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    U_sb, V_sb, W_sb, R_sb = schoolbook()
    best_U, best_V, best_W, best_R = U_sb, V_sb, W_sb, R_sb

    # Hybrid Strassen/Schoolbook construction
    # Split 2x3 as [2x2 | 2x1], 3x7 as [2x7; 1x7]
    # A1 (2x2) * B1 (2x7) via Strassen (rank 7)
    # A2 (2x1) * B2 (1x7) via schoolbook (rank 7)
    # Total: 7 + 7 = 14? No, we need to be more careful.
    
    # Actually, let's use a block structure:
    # C = A * B where A is 2x3, B is 3x7
    # Split B into B_top (2x7) and B_bottom (1x7)
    # C = A[:, :2] * B_top + A[:, 2:3] * B_bottom
    
    # For A[:, :2] (2x2) * B_top (2x7): use Strassen
    # For A[:, 2:3] (2x1) * B_bottom (1x7): schoolbook rank 7
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # But B is 2x7, not 2x2. We can apply Strassen to each 2x2 block of B.
    # B_top is 2x7. Split into 3 blocks: B11(2x2), B12(2x2), B13(2x2), B14(2x1)
    # Actually, let's just use schoolbook for simplicity but try to find better.
    
    # Let me try a direct approach: use the known optimal or near-optimal decompositions.
    # For 2x3x7, the rank is known to be at most 24 (by various bounds).
    # Let's try to find a decomposition with rank < 42.
    
    # Strategy: Use random continuous initialization with least_squares, but with
    # better initialization and more attempts.
    
    def verify(U, V, W, R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integer(mat, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.round(mat * 2) / 2
        if np.max(np.abs(mat - snapped)) < tol:
            return snapped
        return mat

    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: rank >= max(n*p, m*p, n*m) is not strict, but rank >= n*p = 14
    # Actually, for matrix multiplication, rank >= n*m*p / (something), but let's try from 20
    
    min_rank_to_try = 20
    max_attempts_per_rank = 8
    
    for R in range(min_rank_to_try, min(R_sb + 1, 42)):
        if time.time() > deadline - 15:
            break
            
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline - 15:
                break
                
            # Better initialization: use structured random values
            rng = np.random.default_rng(attempt * 7919 + R * 104729)
            
            # Initialize with small random values from a set that includes half-integers
            vals = [-1, -0.5, 0, 0.5, 1]
            U = rng.choice(vals, size=(n * m, R)).astype(float)
            V = rng.choice(vals, size=(m * p, R)).astype(float)
            W = rng.choice(vals, size=(p * n, R)).astype(float)
            
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten(x):
                u_size = n * m * R
                v_size = m * p * R
                U = x[:u_size].reshape(n * m, R)
                V = x[u_size:u_size+v_size].reshape(m * p, R)
                W = x[u_size+v_size:].reshape(p * n, R)
                return U, V, W
            
            def residuals(x):
                U, V, W = unflatten(x)
                res = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            res[idx] = s - 1.0
                            idx += 1
                return res
            
            x0 = flatten(U, V, W)
            
            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=2000, ftol=1e-12, xtol=1e-12)
            except Exception:
                continue
            
            if sol.cost < 1e-10:
                U_sol, V_sol, W_sol = unflatten(sol.x)
                
                # Try to snap to half-integers
                U_snapped = snap_to_half_integer(U_sol)
                V_snapped = snap_to_half_integer(V_sol)
                W_snapped = snap_to_half_integer(W_sol)
                
                if verify(U_snapped, V_snapped, W_snapped, R):
                    return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                
                # If snapping doesn't work, try the raw solution
                if verify(U_sol, V_sol, W_sol, R):
                    # Check if entries are close to half-integers
                    all_close = True
                    for mat in [U_sol, V_sol, W_sol]:
                        snapped = snap_to_half_integer(mat, tol=1e-4)
                        if np.max(np.abs(mat - snapped)) > 1e-4:
                            all_close = False
                            break
                    if all_close:
                        return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                    
                    # Otherwise, use the solution as-is (but it needs to be half-integer)
                    # If not half-integer, skip
                    pass

    # If no better solution found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
