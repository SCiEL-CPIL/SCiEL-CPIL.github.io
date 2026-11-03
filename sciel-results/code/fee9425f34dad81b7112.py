import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Check shapes
    if U_arr.shape[0] != n * m:
        return False
    if V_arr.shape[0] != m * p:
        return False
    if W_arr.shape[0] != p * n:
        return False
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False
    
    # Build the target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1
    
    # Compute reconstructed tensor: T[α,β,γ] = sum_r U[α,r] * V[β,r] * W[γ,r]
    # Use einsum for efficiency
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check if they match
    return np.allclose(T_recon, T_target, atol=tol, rtol=0)

def _to_fraction_list(matrix):
    """Convert a numpy array to a list of lists of Fractions."""
    result = []
    for row in matrix:
        frac_row = []
        for val in row:
            if abs(val - round(val)) < 1e-10:
                frac_row.append(Fraction(int(round(val))))
            else:
                # Try to find a simple fraction
                frac = Fraction(val).limit_denominator(1000)
                frac_row.append(frac)
        result.append(frac_row)
    return result

def _try_rank_reduction(rank, n, m, p, max_iterations=500, deadline=None):
    """Try to find a rank-r decomposition using gradient descent."""
    if deadline is not None and time.time() > deadline:
        return None
    
    # Random initialization with small values
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Build target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute reconstruction
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((T_recon - T_target) ** 2)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-12:
            break
        
        # Compute gradients
        # dE/dU[α,r] = 2 * sum_{β,γ} (T_recon[α,β,γ] - T_target[α,β,γ]) * V[β,r] * W[γ,r]
        diff = T_recon - T_target
        
        grad_U = np.einsum('abc,br,cr->ar', diff, V, W) * 2
        grad_V = np.einsum('abc,ar,cr->br', diff, U, W) * 2
        grad_W = np.einsum('abc,ar,br->cr', diff, U, V) * 2
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if error > best_error * 1.1:
            lr *= 0.5
    
    # Check if we found a valid decomposition
    if best_error < 1e-10:
        # Try to round to integers/half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
            return {
                'U': _to_fraction_list(U_rounded),
                'V': _to_fraction_list(V_rounded),
                'W': _to_fraction_list(W_rounded),
                'rank': rank
            }
    
    return None

def _try_structured_decomposition(n, m, p, deadline=None):
    """Try structured decompositions based on algebraic properties."""
    # For 3x5x5, try to exploit the structure
    
    # Strategy: Try to find a decomposition where some entries are shared
    # This is a heuristic approach - try various patterns
    
    results = []
    
    # Try rank from 60 to 74 (schoolbook is 75)
    for rank in range(60, 75):
        if deadline is not None and time.time() > deadline:
            break
        
        result = _try_rank_reduction(rank, n, m, p, max_iterations=200, deadline=deadline)
        if result is not None:
            results.append(result)
            if rank <= 65:  # Found a good rank
                break
    
    if results:
        # Return the best (lowest rank)
        return min(results, key=lambda x: x['rank'])
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][column] = 1
                V_schoolbook[j * p + k][column] = 1
                W_schoolbook[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower-rank decomposition
    best_result = None
    
    if time.time() < deadline - 5:
        best_result = _try_structured_decomposition(n, m, p, deadline=deadline)
    
    if best_result is not None:
        return {
            "U": best_result['U'],
            "V": best_result['V'],
            "W": best_result['W']
        }
    
    # Return the schoolbook decomposition
    return {"U": U_schoolbook, "V": V_schoolbook, "W": W_schoolbook}
