"""Program for rank-minimal tensor decomposition of 3x3 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the rank-27 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-6):
    """
    Check that the decomposition is valid.
    
    The condition is:
    sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
    
    for all i, j, j', k.
    """
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += Ua[i*m+j, c] * Va[jp*p+k, c] * Wa[k*n+i, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def _check_half_ints(U, V, W, tol=1e-6):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for v in row:
                if abs(v * 2 - round(v * 2)) > tol:
                    return False
    return True

def _round_to_half_ints(U, V, W):
    """Round all entries to nearest half-integer."""
    def r(x):
        return round(x * 2) / 2.0
    return (
        [[r(v) for v in row] for row in U],
        [[r(v) for v in row] for row in V],
        [[r(v) for v in row] for row in W],
    )

def _objective(x, n=3, m=3, p=3, rank=23):
    """
    Objective function: sum of squared errors over all (i, j, j', k) tuples.
    
    x is a flat vector of length 3 * 3 * 3 * rank = 69 * 3 = 207.
    x[0:9*rank] = U flattened (9 x rank)
    x[9*rank:18*rank] = V flattened (9 x rank)
    x[18*rank:27*rank] = W flattened (9 x rank)
    """
    Ua = x[0:9*rank].reshape(9, rank)
    Va = x[9*rank:18*rank].reshape(9, rank)
    Wa = x[18*rank:27*rank].reshape(9, rank)
    
    err = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += Ua[i*m+j, c] * Va[jp*p+k, c] * Wa[k*n+i, c]
                    expected = 1.0 if j == jp else 0.0
                    err += (val - expected) ** 2
    return err

def _objective_grad(x, n=3, m=3, p=3, rank=23):
    """Gradient of the objective function."""
    Ua = x[0:9*rank].reshape(9, rank)
    Va = x[9*rank:18*rank].reshape(9, rank)
    Wa = x[18*rank:27*rank].reshape(9, rank)
    
    gU = np.zeros_like(Ua)
    gV = np.zeros_like(Va)
    gW = np.zeros_like(Wa)
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    expected = 1.0 if j == jp else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += Ua[i*m+j, c] * Va[jp*p+k, c] * Wa[k*n+i, c]
                    diff = val - expected
                    for c in range(rank):
                        gU[i*m+j, c] += 2 * diff * Va[jp*p+k, c] * Wa[k*n+i, c]
                        gV[jp*p+k, c] += 2 * diff * Ua[i*m+j, c] * Wa[k*n+i, c]
                        gW[k*n+i, c] += 2 * diff * Ua[i*m+j, c] * Va[jp*p+k, c]
    
    gx = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    return gx

def _try_rank_23_numerical(deadline):
    """
    Try to find a rank-23 decomposition using numerical optimization.
    Returns None if not found within time limit.
    """
    n, m, p = DIMENSIONS
    rank = 23
    dim = 27 * rank  # 9*3 for each of U, V, W
    
    best_obj = float('inf')
    best_x = None
    
    # Try multiple random initializations
    np.random.seed(42)
    n_attempts = 0
    
    while time.time() < deadline - 10:
        n_attempts += 1
        
        # Random initialization with small values
        x0 = np.random.randn(dim) * 0.1
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(
                _objective,
                x0,
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
                
                # Check if we've found a good solution
                if best_obj < 1e-10:
                    break
        except Exception as e:
            continue
        
        # Check time
        if time.time() > deadline - 15:
            break
    
    # If we found a good solution, extract U, V, W
    if best_x is not None and best_obj < 1e-8:
        Ua = best_x[0:9*rank].reshape(9, rank)
        Va = best_x[9*rank:18*rank].reshape(9, rank)
        Wa = best_x[18*rank:27*rank].reshape(9, rank)
        
        # Convert to lists and round to half-integers
        U = [[Ua[i, c] for c in range(rank)] for i in range(9)]
        V = [[Va[i, c] for c in range(rank)] for i in range(9)]
        W = [[Wa[i, c] for c in range(rank)] for i in range(9)]
        
        # Round to half-integers
        U, V, W = _round_to_half_ints(U, V, W)
        
        # Verify
        if _verify(U, V, W) and _check_half_ints(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_known_constructions(deadline):
    """
    Try known explicit constructions for rank-23 decomposition.
    
    Based on the Winograd construction, we can try to build the decomposition
    using the structure of 3x3 matrix multiplication.
    
    One approach: Use the fact that 3x3 matmul can be embedded in a larger
    structure and use known lower-rank decompositions.
    """
    n, m, p = DIMENSIONS
    rank = 23
    
    # Try a structured initialization based on the schoolbook decomposition
    # with some entries combined
    
    # Start with schoolbook and try to merge some columns
    U_school = np.zeros((9, 27))
    V_school = np.zeros((9, 27))
    W_school = np.zeros((9, 27))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try to find a rank-23 subset or linear combination
    # This is a hard problem, so we'll use a heuristic:
    # Randomly select 23 columns and try to adjust them
    
    best_obj = float('inf')
    best_selection = None
    
    np.random.seed(123)
    n_attempts = 0
    
    while time.time() < deadline - 20:
        n_attempts += 1
        
        # Select 23 random columns
        selected = np.random.choice(27, size=23, replace=False)
        
        # Initialize with these columns
        x0 = np.concatenate([
            U_school[:, selected].flatten(),
            V_school[:, selected].flatten(),
            W_school[:, selected].flatten()
        ])
        
        # Optimize
        try:
            result = minimize(
                _objective,
                x0,
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_selection = result.x
        except Exception:
            continue
    
    if best_selection is not None and best_obj < 1e-8:
        Ua = best_selection[0:9*rank].reshape(9, rank)
        Va = best_selection[9*rank:18*rank].reshape(9, rank)
        Wa = best_selection[18*rank:27*rank].reshape(9, rank)
        
        U = [[Ua[i, c] for c in range(rank)] for i in range(9)]
        V = [[Va[i, c] for c in range(rank)] for i in range(9)]
        W = [[Wa[i, c] for c in range(rank)] for i in range(9)]
        
        U, V, W = _round_to_half_ints(U, V, W)
        
        if _verify(U, V, W) and _check_half_ints(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Try rank-23 constructions
    result = _try_known_constructions(deadline)
    if result is not None:
        return result
    
    result = _try_rank_23_numerical(deadline)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
