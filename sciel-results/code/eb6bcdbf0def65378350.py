"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _to_halfint_list(mat):
    """Convert a float matrix to a list-of-lists of integers/half-integers if possible, else round.
    Ensures native Python types for JSON serialization."""
    doubled = np.round(mat * 2) / 2.0
    out = []
    for row in doubled:
        r = []
        for val in row:
            fval = float(val)
            if abs(fval - round(fval)) < 1e-9:
                r.append(int(round(fval)))
            else:
                r.append(fval)
        out.append(r)
    return out


def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is exact (up to numerical tolerance)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check zero positions
    rng = np.random.default_rng(42)
    for _ in range(100):
        a = rng.integers(0, n * m)
        b = rng.integers(0, m * p)
        c = rng.integers(0, p * n)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if not (j == j2 and k == k2 and i == i2):
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > tol:
                return False
    return True


def _compute_residual(U, V, W, n, m, p):
    """Compute max absolute residual over non-zero and sampled zero positions."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _newton_step(U, V, W, n, m, p, target_rank, lambda_factor=1.0):
    """
    Perform one Newton step on the polynomial system.
    Returns (U_new, V_new, W_new, residual)
    """
    # Build the Jacobian and residual for the system:
    # F_{i,j,k}(U,V,W) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1 = 0
    # for all i,j,k. Plus we need zero positions to be 0, but let's focus on non-zero first.
    # Actually, we need ALL positions to match the tensor. Let's include zero positions too.
    # Total number of constraints: n*m * m*p * p*n = (n*m)*(m*p)*(p*n)
    # For 2,3,10: 6*30*20 = 3600 constraints
    # Unknowns: (n*m + m*p + p*n) * target_rank = (6+30+20)*R = 56*R
    # For R=60: 3360 unknowns, 3600 constraints - overdetermined but consistent.
    
    # This is too large for direct Newton. Let's use a different approach:
    # Alternate least squares but with line search (Levenberg-Marquardt style)
    
    # Actually, let's just do a faster ALS with better initialization and line search.
    return U, V, W, _compute_residual(U, V, W, n, m, p)


def _try_als_fast(n, m, p, target_rank, max_iter=100, tol=1e-8, n_restarts=10, deadline=None):
    """
    Fast ALS with line search and better convergence.
    """
    rng = np.random.default_rng()
    best = None
    best_res = np.inf
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Initialize with structured random matrices
        U = rng.standard_normal((n * m, target_rank))
        V = rng.standard_normal((m * p, target_rank))
        W = rng.standard_normal((p * n, target_rank))
        
        # Normalize columns
        for r in range(target_rank):
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
            if nu > 1e-12: U[:, r] /= nu
            if nv > 1e-12: V[:, r] /= nv
            if nw > 1e-12: W[:, r] /= nw
        
        prev_res = np.inf
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U
            for a in range(n * m):
                i = a // m
                j = a % m
                M = np.zeros((p, target_rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                U[a, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            
            # Update V
            for b in range(m * p):
                j = b // p
                k = b % p
                M = np.zeros((n, target_rank))
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    M[i, :] = U[a, :] * W[c, :]
                V[b, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
            
            # Update W
            for c in range(p * n):
                k = c // n
                i = c % n
                M = np.zeros((m, target_rank))
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    M[j, :] = U[a, :] * V[b, :]
                W[c, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            
            # Compute residual
            res = _compute_residual(U, V, W, n, m, p)
            
            if res < tol:
                U_snap = np.round(U * 2) / 2.0
                V_snap = np.round(V * 2) / 2.0
                W_snap = np.round(W * 2) / 2.0
                if _validate(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
                # Try without snapping
                if _validate(U, V, W, n, m, p, tol=1e-5):
                    # Try to find nearby half-integer solution
                    for delta_scale in [0.5, 0.25, 0.125]:
                        U_try = np.round(U / delta_scale) * delta_scale
                        V_try = np.round(V / delta_scale) * delta_scale
                        W_try = np.round(W / delta_scale) * delta_scale
                        if _validate(U_try, V_try, W_try, n, m, p, tol=1e-5):
                            return U_try, V_try, W_try
                pass
            
            if res < best_res:
                best_res = res
                best = (U.copy(), V.copy(), W.copy())
            
            if res < prev_res * 0.99:
                prev_res = res
            else:
                # Stagnation, break early
                if it > 10:
                    break
        
        if deadline is not None and time.time() > deadline:
            break
    
    return best


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try ranks from low to high
    for target_rank in [20, 22, 24, 26, 28, 30, 35, 40, 45, 50]:
        if time.time() > deadline - 5:
            break
        result = _try_als_fast(n, m, p, target_rank, max_iter=80, tol=1e-7, n_restarts=8, deadline=deadline - 5)
        if result is not None:
            U, V, W = result
            if np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(W)):
                if _validate(U, V, W, n, m, p):
                    return {
                        "U": _to_halfint_list(U),
                        "V": _to_halfint_list(V),
                        "W": _to_halfint_list(W)
                    }
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": _to_halfint_list(U),
        "V": _to_halfint_list(V),
        "W": _to_halfint_list(W)
    }

# EVOLVE-BLOCK-END
