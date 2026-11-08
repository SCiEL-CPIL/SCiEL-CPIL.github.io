"""Tensor decomposition search for 2x3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]


def _verify(U, V, W):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r_idx in range(r):
                    s += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True


def _is_half_integer(arr, tol=1e-6):
    """Check if all entries are close to half-integers."""
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    return np.allclose(doubled, rounded, atol=tol)


def _round_to_half_integer(arr):
    """Round entries to nearest half-integer."""
    return np.round(2.0 * arr) / 2.0


def _solve_for_W(U, V, n, m, p, rank):
    """
    Given U (n*m x rank) and V (m*p x rank), solve for W (p*n x rank)
    such that for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1.
    
    This is a linear system in W.
    Unknowns: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,rank)
    Total unknowns: p * n * rank
    Constraints: n * m * p
    """
    num_unknowns = p * n * rank
    num_constraints = n * m * p
    
    # Build linear system A @ w_flat = b
    A = np.zeros((num_constraints, num_unknowns))
    b = np.zeros(num_constraints)
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                for r in range(rank):
                    # W[k*n+i, r] is at position (k*n+i) * rank + r in w_flat
                    w_pos = (k * n + i) * rank + r
                    A[idx, w_pos] = U[i * m + j, r] * V[j * p + k, r]
                b[idx] = 1.0
                idx += 1
    
    # Solve least squares
    try:
        W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-12)
    except np.linalg.LinAlgError:
        return None
    
    W = W_flat.reshape(p * n, rank)
    
    # Check if the solution satisfies the constraints
    if not np.allclose(A @ W_flat, b, atol=1e-8):
        return None
    
    return W


def _try_rank(n, m, p, rank, deadline, max_trials=500):
    """Try to find a valid decomposition of given rank."""
    nm, mp, pn = n * m, m * p, p * n
    
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        
        # Use deterministic seeds for reproducibility
        seed = rank * 10000 + trial
        rng = np.random.RandomState(seed)
        
        # Generate random half-integer U and V
        # Keep values small to improve chance of W being half-integer
        U = 0.5 * rng.randint(-3, 4, size=(nm, rank))
        V = 0.5 * rng.randint(-3, 4, size=(mp, rank))
        
        W = _solve_for_W(U, V, n, m, p, rank)
        if W is None:
            continue
        
        # Check if W is close to half-integers
        if _is_half_integer(W, tol=1e-4):
            # Round all to half-integers and verify
            U_rounded = _round_to_half_integer(U)
            V_rounded = _round_to_half_integer(V)
            W_rounded = _round_to_half_integer(W)
            
            if _verify(U_rounded, V_rounded, W_rounded):
                return (U_rounded, V_rounded, W_rounded)
    
    return None


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Try to find lower-rank decompositions
    # Lower bound: for matrix multiplication, rank >= max(n*m, m*p, p*n) is a simple bound
    # But better bounds exist. Let's try from 21 upward (heuristic lower bound)
    lower_bound = 21
    
    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank(n, m, p, target_rank, deadline)
        if result is not None:
            U_found, V_found, W_found = result
            best_U, best_V, best_W = U_found, V_found, W_found
            best_rank = target_rank
            break  # Found the lowest rank we tried, stop
    
    # Verify the best solution
    if not _verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = schoolbook_rank
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
