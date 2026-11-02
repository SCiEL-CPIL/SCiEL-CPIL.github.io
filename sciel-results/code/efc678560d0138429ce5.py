"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, deadline, seed=0):
    """Try to find a rank-r decomposition using random initialization + local search."""
    if time.time() > deadline:
        return None
    
    np.random.seed(seed)
    
    # Initialize with random sparse matrices
    density = 0.3
    U = np.random.choice([-1, 0, 1], size=(n*m, r))
    V = np.random.choice([-1, 0, 1], size=(m*p, r))
    W = np.random.choice([-1, 0, 1], size=(p*n, r))
    
    # Try to solve for W given U, V using least squares
    # The constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a linear system in W entries
    
    # Build the linear system: A * w_flat = b
    # where w_flat is the flattened W matrix
    # Each equation (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    equations = []
    b = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = np.zeros(p*n)
                for c in range(r):
                    coeff = U[i*m+j, c] * V[j*p+k, c]
                    if coeff != 0:
                        row[k*n+i] += coeff
                if np.any(row != 0):
                    equations.append(row)
                    b.append(1.0)
    
    if not equations:
        return None
    
    A = np.array(equations)
    b = np.array(b)
    
    # Check if system is consistent and solvable
    try:
        # Use least squares
        w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        W_sol = w_flat.reshape(p*n, r)
        
        # Check if solution is exact
        if np.allclose(A @ w_flat, b, atol=1e-8):
            # Round to nearest integers/half-integers if possible
            W_rounded = np.round(W_sol * 2) / 2
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            
            if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
            else:
                # Try with original values
                if _verify(U.tolist(), V.tolist(), W_sol.tolist(), n, m, p):
                    return (U.tolist(), V.tolist(), W_sol.tolist())
    except:
        pass
    
    return None

def _structured_search(n, m, p, deadline):
    """Try structured decompositions using block methods."""
    # For 4x4 * 4x8, try various structured approaches
    
    # Approach 1: Try rank = 32 (n*m*p/4)
    # Approach 2: Try rank = 48 (n*m*p/3)  
    # Approach 3: Try rank = 64 (n*m*p/2)
    
    for target_rank in [32, 40, 48, 56, 64, 80, 96, 112, 127]:
        if time.time() > deadline:
            break
        
        for seed in range(10):
            if time.time() > deadline:
                break
            result = _try_rank(target_rank, n, m, p, deadline, seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # Try to find a lower-rank decomposition
    result = _structured_search(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
