"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition as numpy arrays (float)."""
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
    return U, V, W

def _tensor_error(U, V, W, n, m, p):
    """Compute Frobenius norm of error between decomposition and true tensor."""
    # True tensor: T[i,j,k] = 1
    # Decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape to (n, m, p, rank) and contract
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
                error += (val - 1.0) ** 2
    return np.sqrt(error)

def _try_reduce_rank(U, V, W, n, m, p, max_time=100.0):
    """Try to find a lower-rank decomposition via local search."""
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    best_err = _tensor_error(best_U, best_V, best_W, n, m, p)
    
    current_rank = best_rank
    U_curr, V_curr, W_curr = best_U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by 1 at a time
    for target_rank in range(current_rank - 1, max(20, current_rank - 20), -1):
        if time.time() - start_time > max_time - 30:
            break
        
        # Try random column removal and re-solve
        for attempt in range(50):
            if time.time() - start_time > max_time - 30:
                break
            
            # Select a random subset of columns
            if target_rank >= current_rank:
                continue
                
            col_indices = list(range(current_rank))
            np.random.shuffle(col_indices)
            keep = col_indices[:target_rank]
            keep.sort()
            
            U_try = U_curr[:, keep].copy()
            V_try = V_curr[:, keep].copy()
            W_try = W_curr[:, keep].copy()
            
            err = _tensor_error(U_try, V_try, W_try, n, m, p)
            if err < 1e-8:
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                best_err = err
                current_rank = target_rank
                U_curr, V_curr, W_curr = U_try.copy(), V_try.copy(), W_try.copy()
                break
    
    # If we found a valid lower-rank decomposition, try to polish it
    if best_err < 1e-8 and best_rank < U.shape[1]:
        # Try to find exact rational values
        pass
    
    return best_U, best_V, best_W, best_rank

def _to_fraction_list(matrix):
    """Convert numpy array to list of lists of Fractions for exactness."""
    result = []
    for row in matrix:
        frac_row = []
        for val in row:
            if abs(val) < 1e-10:
                frac_row.append(Fraction(0))
            elif abs(val - round(val)) < 1e-10:
                frac_row.append(Fraction(int(round(val))))
            else:
                # Try to find a simple fraction
                frac = Fraction(val).limit_denominator(1000)
                frac_row.append(frac)
        result.append(frac_row)
    return result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Get schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    schoolbook_rank = U.shape[1]
    
    # Try to reduce rank
    U_best, V_best, W_best, best_rank = _try_reduce_rank(U, V, W, n, m, p, max_time=120.0)
    
    # If we found a better decomposition, use it; otherwise use schoolbook
    if best_rank < schoolbook_rank:
        U_out, V_out, W_out = U_best, V_best, W_best
    else:
        U_out, V_out, W_out = U, V, W
    
    # Convert to list of lists of integers (or floats if needed, but schoolbook is integer)
    # The problem asks for integer/half-integer entries. Schoolbook has 0s and 1s.
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
