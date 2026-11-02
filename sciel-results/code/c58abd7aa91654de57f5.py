"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import time
import itertools
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _check_exact(U, V, W, n, m, p):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = len(U[0])
    # Convert to numpy for faster summation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # We need to check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is not a simple matrix product. It's a batched tensor contraction.
    # We can iterate over c and accumulate.
    # For small r, this is fast enough.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vj[c] * wk[c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _error_metric(U, V, W, n, m, p):
    """Calculate the squared error of the tensor decomposition."""
    r = len(U[0])
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vj[c] * wk[c]
                total_error += (s - 1.0) ** 2
    return total_error


def _local_search(U, V, W, n, m, p, deadline, max_steps=2000):
    """
    Perform local search to minimize the error.
    Moves: change a value in U, V, or W to another integer in {-2, -1, 0, 1, 2}.
    """
    r = len(U[0])
    current_error = _error_metric(U, V, W, n, m, p)
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_error = current_error
    
    candidates = [-2, -1, 0, 1, 2]
    
    while time.time() < deadline and max_steps > 0:
        max_steps -= 1
        # Pick a random matrix, row, and column
        mat_choice = random.randint(0, 2)
        if mat_choice == 0:
            rows = n * m
            mat = U
            mat_name = 'U'
        elif mat_choice == 1:
            rows = m * p
            mat = V
            mat_name = 'V'
        else:
            rows = p * n
            mat = W
            mat_name = 'W'
            
        row_idx = random.randint(0, rows - 1)
        col_idx = random.randint(0, r - 1)
        
        old_val = mat[row_idx][col_idx]
        # Pick a new value
        new_val = random.choice([v for v in candidates if v != old_val])
        
        # Apply change
        mat[row_idx][col_idx] = new_val
        
        # Calculate new error
        new_error = _error_metric(U, V, W, n, m, p)
        
        if new_error < best_error:
            best_error = new_error
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
            if best_error < 1e-9:
                return best_U, best_V, best_W
        else:
            # Revert change (simple hill climbing, no simulated annealing for speed)
            mat[row_idx][col_idx] = old_val
            # If stuck, maybe try a random restart or accept a worse move with small prob?
            # For now, simple hill climbing.
            
        if new_error == current_error and random.random() < 0.01:
             # Random walk to escape local minima
             pass

    return best_U, best_V, best_W


def _try_sparse_decomposition(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of target_rank using local search.
    Initializes with sparse random integers.
    """
    r = target_rank
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Initialize with sparse random values
    # Ensure each column has at least one non-zero in each matrix to avoid trivial zeros
    for c in range(r):
        # Pick random non-zero entries
        # For U
        idx_u = random.randint(0, n * m - 1)
        U[idx_u][c] = random.choice([-1, 1])
        
        # For V
        idx_v = random.randint(0, m * p - 1)
        V[idx_v][c] = random.choice([-1, 1])
        
        # For W
        idx_w = random.randint(0, p * n - 1)
        W[idx_w][c] = random.choice([-1, 1])
        
        # Add a few more non-zeros to allow for cancellation/sums
        for _ in range(2):
            idx_u2 = random.randint(0, n * m - 1)
            U[idx_u2][c] = random.choice([-1, 1])
            idx_v2 = random.randint(0, m * p - 1)
            V[idx_v2][c] = random.choice([-1, 1])
            idx_w2 = random.randint(0, p * n - 1)
            W[idx_w2][c] = random.choice([-1, 1])

    U, V, W = _local_search(U, V, W, n, m, p, deadline)
    
    if _check_exact(U, V, W, n, m, p):
        return U, V, W
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Known lower bound is 51 for 3x5x6.
    # We will try ranks from 51 up to 60.
    # If we find a lower one, great. If not, we return the best found.
    
    random.seed(42)
    
    # Try ranks in increasing order
    for target_rank in range(51, 65):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        found = False
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            res = _try_sparse_decomposition(n, m, p, target_rank, deadline)
            if res:
                U, V, W = res
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            # If we found a solution at rank r, can we find a lower one?
            # The loop continues to r+1, but we want the MINIMAL rank.
            # So we should break if we found a solution?
            # No, we are iterating upwards. If we find rank 51, that's likely optimal.
            # If we find rank 52, we might still find 51 in a later iteration?
            # No, we iterate 51, 52, 53...
            # So if we find 51, we should return it immediately as it's the lower bound.
            # If we don't find 51, we try 52, etc.
            # So if we find a solution, we should return it?
            # Wait, if I find rank 52, I should continue to see if 51 is possible?
            # No, I am iterating 51 first. If 51 fails, I try 52.
            # If 52 succeeds, I have a rank 52 solution.
            # I should stop if I hit the deadline.
            # Should I continue searching for lower ranks? I already tried 51.
            # So if I find 52, it's the best so far.
            # But maybe 53 has a better chance of being found quickly?
            # Usually, lower rank is harder.
            # So if I find 52, I should probably return it, unless I have lots of time left.
            # Let's just return the first valid one found in the increasing sequence.
            return best_result

    return best_result
# EVOLVE-BLOCK-END
