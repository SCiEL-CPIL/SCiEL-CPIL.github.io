import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def _validate(U, V, W, n=3, m=3, p=8):
    """Check if U,V,W exactly decompose the matrix multiplication tensor.
    The condition: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] == W[k*n+i,c]
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U[i * m + j, :], V[j * p + k, :])
                rhs = W[k * n + i, :]
                if np.max(np.abs(lhs - rhs)) > 1e-9:
                    return False
    # Check half-integer constraint
    for mat in [U, V, W]:
        doubled = 2.0 * mat
        rounded = np.round(doubled)
        if not np.all(np.abs(doubled - rounded) < 1e-9):
            return False
    return True

def _build_schoolbook(n=3, m=3, p=8):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _round_half(x):
    return np.round(2.0 * x) / 2.0

def _check_half_integer(arr, tol=1e-9):
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    return np.all(np.abs(doubled - rounded) < tol)

def _try_rank_structured(r, n, m, p, deadline, num_attempts=300):
    """Try to find rank-r decomposition using structured random search."""
    rng = np.random.default_rng()
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            return None
        
        # Strategy: random U, V, then compute W as outer product sum
        U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, r))
        V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, r))
        
        # W[k*n+i, c] should equal sum_j U[i*m+j, c] * V[j*p+k, c]
        W = np.zeros((p * n, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    W[k * n + i, :] += U[i * m + j, :] * V[j * p + k, :]
        
        # Round W to half-integers
        W_r = _round_half(W)
        
        # Verify with rounded values
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U[i * m + j, :], V[j * p + k, :])
                    rhs = W_r[k * n + i, :]
                    if np.max(np.abs(lhs - rhs)) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid and _check_half_integer(W_r):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W_r.tolist()
            }
        
        # Alternative: fix W, solve for U and V
        W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, r))
        
        # For each c, we need U[i*m+j,c] * V[j*p+k,c] = W[k*n+i,c] for all i,j,k
        # This is a rank-1 matrix factorization problem per column c
        success = True
        U_sol = np.zeros((n * m, r))
        V_sol = np.zeros((m * p, r))
        
        for c in range(r):
            # Build matrix M[i,k] = W[k*n+i, c]
            M = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    M[i, k] = W[k * n + i, c]
            
            # Check if M has rank 1 (or is zero)
            if np.all(M == 0):
                U_sol[:, c] = 0
                V_sol[:, c] = 0
                continue
            
            # Try to factor M = u * v^T where u is n-vector, v is p-vector
            # Use SVD
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                if len(s) > 0 and s[0] > 1e-10:
                    if len(s) > 1 and s[1] > 1e-10 * s[0]:
                        success = False
                        break
                    u = U_svd[:, 0] * np.sqrt(s[0])
                    v = Vt[0, :] * np.sqrt(s[0])
                    
                    # Round to half-integers
                    u_r = _round_half(u)
                    v_r = _round_half(v)
                    
                    # Check if rounded factorization is exact
                    M_check = np.outer(u_r, v_r)
                    if not np.all(np.abs(M_check - M) < 1e-9):
                        # Try without rounding if possible
                        if not _check_half_integer(u_r) or not _check_half_integer(v_r):
                            success = False
                            break
                    else:
                        U_sol[:, c] = u_r
                        V_sol[:, c] = v_r
                else:
                    U_sol[:, c] = 0
                    V_sol[:, c] = 0
            except:
                success = False
                break
        
        if success:
            # Verify full decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        lhs = np.dot(U_sol[i * m + j, :], V_sol[j * p + k, :])
                        rhs = W[k * n + i, :]
                        if np.max(np.abs(lhs - rhs)) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {
                    "U": U_sol.tolist(),
                    "V": V_sol.tolist(),
                    "W": W.tolist()
                }
    
    return None

def _try_rank_solve(r, n, m, p, deadline, num_attempts=200):
    """Try solving linear systems for rank-r decomposition."""
    rng = np.random.default_rng()
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            return None
        
        # Random U, solve for V and W
        U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, r))
        
        # Build system: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] = W[k*n+i,c]
        # This is bilinear in V and W, so we need a different approach
        
        # Try: fix U and V, solve for W
        V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, r))
        
        # W[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        W = np.zeros((p * n, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    W[k * n + i, :] += U[i * m + j, :] * V[j * p + k, :]
        
        W_r = _round_half(W)
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U[i * m + j, :], V[j * p + k, :])
                    rhs = W_r[k * n + i, :]
                    if np.max(np.abs(lhs - rhs)) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid and _check_half_integer(W_r):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W_r.tolist()
            }
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 8
    
    # Start with schoolbook
    result = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Lower bound: max(n*p, m*p, n*m) = max(24, 24, 9) = 24
    # But for matrix multiplication, the known lower bound is higher
    # For 3x3 by 3x8, let's try from a reasonable lower bound
    lower_bound = 24
    
    # Try ranks from lower_bound up to best_rank-1
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Budget per rank
        remaining = deadline - time.time()
        ranks_remaining = best_rank - r
        budget_per_rank = max(5, remaining / max(1, ranks_remaining))
        
        # Try structured search
        num_attempts = int(min(200, max(30, budget_per_rank * 5)))
        res = _try_rank_structured(r, n, m, p, deadline, num_attempts)
        if res is not None:
            if _validate(res["U"], res["V"], res["W"], n, m, p):
                best_rank = r
                result = res
                break
        
        # Try solve-based search
        num_attempts = int(min(100, max(20, budget_per_rank * 3)))
        res = _try_rank_solve(r, n, m, p, deadline, num_attempts)
        if res is not None:
            if _validate(res["U"], res["V"], res["W"], n, m, p):
                best_rank = r
                result = res
                break
    
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    valid = _validate(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
