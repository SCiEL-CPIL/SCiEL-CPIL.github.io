"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 7]

def _build_tensor_constraints(n, m, p):
    """Build the list of (U_idx, V_idx, W_idx, target) tuples."""
    constraints = []
    # Valid entries: T[(i,j), (j,k), (i,k)] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i  # W[k*n+i, r] per problem statement
                constraints.append((u_idx, v_idx, w_idx, 1.0))
    # Invalid entries: all other combinations
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                # Check if this is a valid entry
                i1 = u_idx // m
                j1 = u_idx % m
                j2 = v_idx // p
                k1 = v_idx % p
                k2 = w_idx // n
                i2 = w_idx % n
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue  # Already handled
                constraints.append((u_idx, v_idx, w_idx, 0.0))
    return constraints

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W give a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = sum(U[u_idx, c] * V[v_idx, c] * W[w_idx, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    
    # Check invalid entries (sample for speed, but check all for correctness)
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                i1 = u_idx // m
                j1 = u_idx % m
                j2 = v_idx // p
                k1 = v_idx % p
                k2 = w_idx // n
                i2 = w_idx % n
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue
                val = sum(U[u_idx, c] * V[v_idx, c] * W[w_idx, c] for c in range(rank))
                if abs(val) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, max_iters=1000, tol=1e-8, time_limit=30):
    """Try to find a rank-r decomposition using optimization."""
    start_time = time.time()
    
    # Variables: U (n*m x r), V (m*p x r), W (p*n x r)
    # Total variables: (n*m + m*p + p*n) * r
    total_vars = (n * m + m * p + p * n) * r
    
    def unpack(params):
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:(n*m+m*p)*r].reshape(m*p, r)
        W = params[(n*m+m*p)*r:].reshape(p*n, r)
        return U, V, W
    
    def residuals(params):
        if time.time() - start_time > time_limit:
            return None
        U, V, W = unpack(params)
        res = []
        # Valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = sum(U[u_idx, c] * V[v_idx, c] * W[w_idx, c] for c in range(r))
                    res.append(val - 1.0)
        # Invalid entries
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    i1 = u_idx // m
                    j1 = u_idx % m
                    j2 = v_idx // p
                    k1 = v_idx % p
                    k2 = w_idx // n
                    i2 = w_idx % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    val = sum(U[u_idx, c] * V[v_idx, c] * W[w_idx, c] for c in range(r))
                    res.append(val)
        return np.array(res)
    
    # Try multiple random initializations
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        params = np.random.randn(total_vars) * 0.5
        
        best_params = params
        best_cost = float('inf')
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            res = residuals(params)
            if res is None:
                break
            cost = np.sum(res**2)
            
            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()
            
            if cost < tol:
                U, V, W = unpack(best_params)
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-5):
                    return U, V, W
                break
            
            # Simple gradient step (this is slow, but let's try)
            # Actually, let's use scipy
            break  # Use scipy instead
        
        # Use scipy least_squares
        try:
            result = least_squares(residuals, params, method='lm', max_nfev=500, xtol=1e-10)
            if result.cost < 1e-10:
                U, V, W = unpack(result.x)
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-5):
                    return U, V, W
        except Exception:
            pass
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s margin for 150s limit
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = 21, but let's start a bit higher
    # Actually, let's try a range of ranks
    
    # First, try to find a low-rank decomposition
    # Known: schoolbook is 42. Let's try ranks 30-42
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 25 to 42 (or lower if we find one)
    for r in range(25, best_rank + 1):
        if time.time() > deadline:
            break
        
        result = _try_rank(r, n, m, p, max_iters=200, tol=1e-10, time_limit=5)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = r
            break  # Found the lowest rank, stop
    
    if best_result is None:
        # Fall back to schoolbook
        best_result = _schoolbook_decomposition(n, m, p)
    
    return best_result
