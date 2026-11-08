"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _check_decomposition(U, V, W, n, m, p):
    """Check if decomposition is valid."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(val, tol=1e-6):
    """Check if val is close to a half-integer."""
    closest = np.round(val * 2) / 2
    return abs(val - closest) < tol

def _to_half_integer(val):
    """Round to nearest half-integer."""
    return np.round(val * 2) / 2

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    # Search for lower rank decompositions
    # Try ranks from rank_trivial - 1 down to a reasonable lower bound
    # Lower bound: at least max(n*m, m*p, p*n) is not valid
    # A better lower bound: the rank must be at least the max of the "slice" dimensions
    # For now, search down to 16
    
    half_ints = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for rank in range(rank_trivial - 1, 15, -1):
        if time.time() > deadline:
            break
        
        # Number of random attempts for this rank
        max_attempts = 300 if rank > 25 else 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 7919 + rank * 104729)
            
            # Generate random U and V with small integer/half-integer values
            # Use a mix: some entries 0, some ±0.5, some ±1
            probs = [0.3, 0.2, 0.2, 0.15, 0.15]  # P(-1), P(-0.5), P(0), P(0.5), P(1)
            
            U_r = np.random.choice(half_ints, size=(n * m, rank), p=probs)
            V_r = np.random.choice(half_ints, size=(m * p, rank), p=probs)
            
            U_np = U_r
            V_np = V_r
            
            W_np = np.zeros((p * n, rank))
            valid = True
            
            # For each (i, k), solve for W_{k*n+i, :}
            for i in range(n):
                for k in range(p):
                    # Construct the system A W = 1
                    # A has shape (m, rank)
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U_np[i * m + j, r] * V_np[j * p + k, r]
                    
                    # Solve A W_col = 1
                    # W_col has shape (rank,)
                    try:
                        if rank == m:
                            # Square system
                            det = np.linalg.det(A)
                            if abs(det) < 1e-10:
                                valid = False
                                break
                            W_col = np.linalg.solve(A, np.ones(m))
                        elif rank > m:
                            # Underdetermined: use least squares, but need exact solution
                            # Check if system is consistent first
                            W_col, residuals, rank_l, singular_values = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            if np.linalg.norm(A @ W_col - np.ones(m)) > 1e-6:
                                valid = False
                                break
                        else:
                            # rank < m: overdetermined
                            W_col, residuals, rank_l, singular_values = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            if np.linalg.norm(A @ W_col - np.ones(m)) > 1e-6:
                                valid = False
                                break
                        
                        # Check if W_col entries are close to half-integers
                        for r in range(rank):
                            val = W_col[r]
                            if not _is_half_integer(val, tol=1e-5):
                                valid = False
                                break
                            W_np[k * n + i, r] = _to_half_integer(val)
                        if not valid:
                            break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    if not valid:
                        break
            
            if valid:
                # Verify the full decomposition
                if _check_decomposition(U_r.tolist(), V_r.tolist(), W_np.tolist(), n, m, p):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_np.tolist()}
                    best_rank = rank
                    break  # Found a valid decomposition for this rank, move to next lower rank
        
        if best_rank < rank_trivial:
            # Found improvement, continue searching for even lower rank
            pass
    
    return best_result
# EVOLVE-BLOCK-END
