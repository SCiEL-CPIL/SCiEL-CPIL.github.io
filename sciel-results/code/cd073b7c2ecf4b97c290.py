import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matmul."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vals = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                s = np.sum(vals)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(arr, tol=1e-8):
    """Check if all entries are integers or half-integers."""
    for val in arr:
        doubled = 2.0 * val
        if abs(doubled - round(doubled)) > tol:
            return False
    return True

def _snap_to_half_int(arr, tol=1e-8):
    """Snap values to nearest half-integer if close."""
    result = []
    for val in arr:
        doubled = 2.0 * val
        nearest = round(doubled)
        if abs(doubled - nearest) < tol:
            result.append(nearest / 2.0)
        else:
            result.append(val)
    return result

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors."""
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vals = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                s = np.sum(vals)
                error += (s - 1.0) ** 2
    return error

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _try_rank_r(r, n, m, p, max_iters=5000, seed=42):
    """Try to find a decomposition of rank r."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    np.random.seed(seed)
    
    # Candidate values: half-integers in a reasonable range
    candidates = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    best_error = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    # Multiple restarts
    for restart in range(5):
        U = np.random.choice(candidates, size=(nm, r))
        V = np.random.choice(candidates, size=(mp, r))
        W = np.random.choice(candidates, size=(pn, r))
        
        current_error = _compute_error(U, V, W, n, m, p)
        
        for it in range(max_iters):
            if current_error < best_error:
                best_error = current_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if current_error < 1e-12:
                break
            
            # Pick a random matrix and entry to perturb
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                idx = np.random.randint(nm)
                col = np.random.randint(r)
                old_val = U[idx, col]
                new_val = np.random.choice(candidates)
                U[idx, col] = new_val
                new_error = _compute_error(U, V, W, n, m, p)
                if new_error > current_error * 1.01:
                    U[idx, col] = old_val
                else:
                    current_error = new_error
            elif mat_choice == 1:
                idx = np.random.randint(mp)
                col = np.random.randint(r)
                old_val = V[idx, col]
                new_val = np.random.choice(candidates)
                V[idx, col] = new_val
                new_error = _compute_error(U, V, W, n, m, p)
                if new_error > current_error * 1.01:
                    V[idx, col] = old_val
                else:
                    current_error = new_error
            else:
                idx = np.random.randint(pn)
                col = np.random.randint(r)
                old_val = W[idx, col]
                new_val = np.random.choice(candidates)
                W[idx, col] = new_val
                new_error = _compute_error(U, V, W, n, m, p)
                if new_error > current_error * 1.01:
                    W[idx, col] = old_val
                else:
                    current_error = new_error
        
        if best_error < 1e-12:
            break
    
    if best_error < 1e-10:
        # Snap to half-integers
        best_U = np.array([_snap_to_half_int(row) for row in best_U])
        best_V = np.array([_snap_to_half_int(row) for row in best_V])
        best_W = np.array([_snap_to_half_int(row) for row in best_W])
        
        if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            if _is_half_integer(best_U) and _is_half_integer(best_V) and _is_half_integer(best_W):
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    U_fb, V_fb, W_fb, rank_schoolbook = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    for r in range(30, 36):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_r(r, n, m, p, max_iters=3000, seed=42 + r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Return schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
