"""Program for 2x5x5 matrix multiplication tensor decomposition."""
import numpy as np
from scipy.optimize import linprog
import itertools
import time

DIMENSIONS = [2, 5, 5]

def _is_integer_array(arr, tol=1e-9):
    """Check if array entries are (half-)integers."""
    arr = np.asarray(arr, dtype=float)
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    return np.allclose(doubled, rounded, atol=tol)

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify that U, V, W exactly decompose the tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Expected: T[i,j,k] = 1 for all i,j,k (since it's the identity of multiplication)
    rank = U.shape[1]
    for c in range(rank):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        # Uc is (n*m,), Vc is (m*p,), Wc is (p*n,)
        # Reshape Uc to (n, m), Vc to (m, p), Wc to (p, n)
        Uc_m = Uc.reshape(n, m)
        Vc_m = Vc.reshape(m, p)
        Wc_m = Wc.reshape(p, n)
        # For this rank-1 slice, the contribution to T[i,j,k] is Uc_m[i,j] * Vc_m[j,k] * Wc_m[k,i]
        # We need sum over c of these to equal 1 for all (i,j,k)
    # Compute the full tensor and check
    # Efficient way: for each c, form outer products
    # T = sum_c np.einsum('ij,jk,ki->ijk', Uc_m, Vc_m, Wc_m)
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        Uc_m = U[:, c].reshape(n, m)
        Vc_m = V[:, c].reshape(m, p)
        Wc_m = W[:, c].reshape(p, n)
        # T[i,j,k] += Uc_m[i,j] * Vc_m[j,k] * Wc_m[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc_m[i, j] * Vc_m[j, k] * Wc_m[k, i]
    return np.allclose(T, 1.0, atol=tol)

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _try_lp_decomposition(n, m, p, max_rank=25, time_budget=100):
    """Try to find a low-rank decomposition using linear programming.
    
    The idea: we want T[i,j,k] = 1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    This is a degree-3 equation. We can fix U and W to be sparse/structured and solve for V,
    or use a different approach.
    
    Alternative: Use the fact that if we fix the rank, we can set up a system where
    we choose U and V to be 0-1 or -1,0,1 matrices and solve for W, then check.
    
    Actually, let's try a different approach: 
    For rank-r decomposition, we need to find r triples of vectors (u_c, v_c, w_c)
    such that sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1.
    
    Let's try: fix U to have a specific structure. For 2x5x5, maybe we can use
    the structure of the 2 rows.
    
    Another idea: use the "rank 10 for 1x5x5" approach if it exists, but we don't know it.
    
    Let's try a heuristic search with random initialization and optimization.
    """
    pass

def construct():
    """Construct the decomposition. Returns dict with 'U', 'V', 'W' keys."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, return the schoolbook decomposition as a safe fallback
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Try to find a better decomposition
    # Strategy: try rank 10, 12, 15, etc. using random search with verification
    
    # For a tensor of size 50 (2*5*5), the tensor has 50 entries all equal to 1.
    # Rank r means we need r terms.
    
    # Let's try a structured approach:
    # Assume U and V have entries in {-1, 0, 1} and W is solved linearly.
    # For fixed U, V, the equation is linear in W.
    
    # Let's try rank 10 first (if 1x5x5 has rank 10, then 2x5x5 might have rank 10 or 20)
    
    best_rank = rank
    best_result = None
    
    for target_rank in [10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 45, 49]:
        if time.time() - start_time > 120:
            break
            
        # Try random search for this rank
        found = _try_random_search(n, m, p, target_rank, max_iters=50, time_budget=120 - (time.time() - start_time))
        if found is not None:
            U_f, V_f, W_f = found
            if _verify(U_f, V_f, W_f, n, m, p):
                best_rank = target_rank
                best_result = _to_lists(U_f, V_f, W_f)
                break
    
    if best_result is not None:
        return best_result
    else:
        return _to_lists(U, V, W)

def _try_random_search(n, m, p, target_rank, max_iters=50, time_budget=10):
    """Try random initialization and check if the system is solvable.
    
    We set U and V randomly with entries in {-1, 0, 1} and solve for W.
    The equation for each (i,j,k) is:
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    This is linear in W for fixed U, V.
    """
    start_time = time.time()
    np.random.seed(42)
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Generate random U and V with entries in {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n * m, target_rank))
        V = np.random.randint(-1, 2, size=(m * p, target_rank))
        
        # Build the linear system for W
        # Unknowns: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,rank)
        # Total unknowns: p * n * target_rank
        num_unknowns = p * n * target_rank
        
        # Equations: for each (i, j, k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Number of equations: n * m * p
        
        # Build A matrix (n*m*p) x (p*n*target_rank)
        # A[row, col] where row = i*m*p + j*p + k, col = (k*n + i)*target_rank + c
        # A[row, col] = U[i*m+j,c] * V[j*p+k,c] if col corresponds to (k,n+i,c)
        
        # This is sparse, so let's build it efficiently
        rows = []
        cols = []
        vals = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i * m * p + j * p + k
                    for c in range(target_rank):
                        u_val = U[i * m + j, c]
                        v_val = V[j * p + k, c]
                        if u_val != 0 and v_val != 0:
                            col = (k * n + i) * target_rank + c
                            rows.append(row)
                            cols.append(col)
                            vals.append(u_val * v_val)
        
        if len(rows) == 0:
            continue
        
        # Build sparse A matrix
        A = np.zeros((len(rows), len(cols)), dtype=float)
        # Actually, let's use a different approach. The system might be overdetermined.
        # Let's just check if a solution exists by solving the least squares problem
        # and checking the residual.
        
        # For efficiency, let's sample a subset of equations if there are too many
        # n*m*p = 50 equations, p*n*rank unknowns
        # For rank 10: 50 * 10 = 500 unknowns, 50 equations. Underdetermined.
        # We can find a minimum-norm solution and check if it works.
        
        # Build the full A matrix
        A_full = np.zeros((n * m * p, num_unknowns))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i * m * p + j * p + k
                    for c in range(target_rank):
                        u_val = U[i * m + j, c]
                        v_val = V[j * p + k, c]
                        if u_val != 0 and v_val != 0:
                            col = (k * n + i) * target_rank + c
                            A_full[row, col] = u_val * v_val
        
        b = np.ones(n * m * p)
        
        # Solve least squares
        try:
            W_flat, residuals, rank_solve, sv = np.linalg.lstsq(A_full, b, rcond=None)
        except:
            continue
        
        # Check residual
        if residuals is not None and len(residuals) > 0:
            residual = residuals[0]
        else:
            residual = np.linalg.norm(A_full @ W_flat - b)
        
        if residual < 1e-7:
            # Check if W is (half-)integer
            W = W_flat.reshape(p * n, target_rank)
            if _is_integer_array(W):
                return (U, V, W)
        
        # Even if not integer, we might be able to round
        W = W_flat.reshape(p * n, target_rank)
        W_rounded = np.round(W * 2) / 2
        if _verify(U, V, W_rounded, n, m, p):
            return (U, V, W_rounded)
    
    return None

if __name__ == "__main__":
    result = construct()
    print("Rank:", result["U"][0].__len__() if result["U"] else 0)
