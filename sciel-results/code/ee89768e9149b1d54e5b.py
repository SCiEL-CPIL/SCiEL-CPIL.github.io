import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _get_schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def _calc_error(U, V, W, n, m, p, rank):
    """Calculate the sum of squared errors."""
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.sum((T - 1.0)**2)

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2

def _polish(U, V, W, n, m, p, rank, time_limit=5):
    """Polish a near-solution by trying small perturbations and rounding."""
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    best_error = _calc_error(U, V, W, n, m, p, rank)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try local perturbations
    while time.time() - start_time < time_limit:
        # Pick a random matrix and entry
        mat_idx = np.random.randint(3)
        if mat_idx == 0:
            row, col = np.random.randint(n * m), np.random.randint(rank)
            old_val = U[row, col]
            # Try small perturbation
            delta = np.random.uniform(-0.5, 0.5)
            U[row, col] = old_val + delta
            new_error = _calc_error(U, V, W, n, m, p, rank)
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                U[row, col] = old_val
        elif mat_idx == 1:
            row, col = np.random.randint(m * p), np.random.randint(rank)
            old_val = V[row, col]
            delta = np.random.uniform(-0.5, 0.5)
            V[row, col] = old_val + delta
            new_error = _calc_error(U, V, W, n, m, p, rank)
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[row, col] = old_val
        else:
            row, col = np.random.randint(p * n), np.random.randint(rank)
            old_val = W[row, col]
            delta = np.random.uniform(-0.5, 0.5)
            W[row, col] = old_val + delta
            new_error = _calc_error(U, V, W, n, m, p, rank)
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[row, col] = old_val
    
    # Try rounding the best solution
    U_rounded = _round_to_half_integer(best_U)
    V_rounded = _round_to_half_integer(best_V)
    W_rounded = _round_to_half_integer(best_W)
    
    error_rounded = _calc_error(U_rounded, V_rounded, W_rounded, n, m, p, rank)
    if error_rounded < 1e-9:
        return U_rounded, V_rounded, W_rounded
    
    # Also try the unrounded best if it's very close
    if best_error < 1e-9:
        return best_U, best_V, best_W
    
    return None

def _structured_search(n, m, p, rank, time_limit=10):
    """Search for a decomposition using structured initialization and optimization."""
    start_time = time.time()
    
    # Try multiple structured initializations
    n_restarts = 5
    best_result = None
    best_error = float('inf')
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Strategy 1: Random continuous
            U = np.random.uniform(-1, 1, size=(n * m, rank))
            V = np.random.uniform(-1, 1, size=(m * p, rank))
            W = np.random.uniform(-1, 1, size=(p * n, rank))
        elif restart == 1:
            # Strategy 2: Sparse random - many zeros
            U = np.random.choice([0, 0, 0, 0.5, 1, -0.5, -1], size=(n * m, rank))
            V = np.random.choice([0, 0, 0, 0.5, 1, -0.5, -1], size=(m * p, rank))
            W = np.random.choice([0, 0, 0, 0.5, 1, -0.5, -1], size=(p * n, rank))
        elif restart == 2:
            # Strategy 3: Block structure - exploit the fact that tensor is all-ones
            # Try to make each "rank component" contribute equally
            U = np.ones((n * m, rank)) / np.sqrt(rank)
            V = np.ones((m * p, rank)) / np.sqrt(rank)
            W = np.ones((p * n, rank)) / np.sqrt(rank)
        elif restart == 3:
            # Strategy 4: Alternating signs
            U = np.random.choice([-1, 1], size=(n * m, rank)) / np.sqrt(rank)
            V = np.random.choice([-1, 1], size=(m * p, rank)) / np.sqrt(rank)
            W = np.random.choice([-1, 1], size=(p * n, rank)) / np.sqrt(rank)
        else:
            # Strategy 5: Mixed
            U = np.random.uniform(-1, 1, size=(n * m, rank))
            V = np.random.choice([0, 1, -1], size=(m * p, rank))
            W = np.random.choice([0, 1, -1], size=(p * n, rank))
        
        # Polish this initialization
        result = _polish(U, V, W, n, m, p, rank, time_limit=2)
        
        if result is not None:
            U_r, V_r, W_r = result
            error = _calc_error(U_r, V_r, W_r, n, m, p, rank)
            if error < best_error:
                best_error = error
                best_result = (U_r, V_r, W_r)
                
                if error < 1e-9:
                    break
    
    return best_result

def _discrete_search(n, m, p, rank, time_limit=5):
    """Search using discrete half-integer values with simulated annealing."""
    start_time = time.time()
    
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Initialize with random discrete values
    U = np.random.choice(values, size=(n * m, rank))
    V = np.random.choice(values, size=(m * p, rank))
    W = np.random.choice(values, size=(p * n, rank))
    
    current_error = _calc_error(U, V, W, n, m, p, rank)
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    T_current = 1.0
    T_min = 1e-6
    alpha = 0.995
    
    while time.time() - start_time < time_limit:
        mat_idx = np.random.randint(3)
        if mat_idx == 0:
            row, col = np.random.randint(n * m), np.random.randint(rank)
            old_val = U[row, col]
            new_val = np.random.choice(values)
            if new_val == old_val:
                continue
            U[row, col] = new_val
            new_error = _calc_error(U, V, W, n, m, p, rank)
            delta = new_error - current_error
            if delta < 0 or np.random.random() < np.exp(-delta / T_current):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                U[row, col] = old_val
        elif mat_idx == 1:
            row, col = np.random.randint(m * p), np.random.randint(rank)
            old_val = V[row, col]
            new_val = np.random.choice(values)
            if new_val == old_val:
                continue
            V[row, col] = new_val
            new_error = _calc_error(U, V, W, n, m, p, rank)
            delta = new_error - current_error
            if delta < 0 or np.random.random() < np.exp(-delta / T_current):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[row, col] = old_val
        else:
            row, col = np.random.randint(p * n), np.random.randint(rank)
            old_val = W[row, col]
            new_val = np.random.choice(values)
            if new_val == old_val:
                continue
            W[row, col] = new_val
            new_error = _calc_error(U, V, W, n, m, p, rank)
            delta = new_error - current_error
            if delta < 0 or np.random.random() < np.exp(-delta / T_current):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[row, col] = old_val
        
        T_current *= alpha
        if T_current < T_min:
            T_current = T_min
    
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # Leave 10s safety margin for the 150s kill
    
    # Get schoolbook as fallback
    schoolbook = _get_schoolbook(n, m, p)
    
    # Try ranks from low to high
    # Lower bound is 12, but realistic minimum might be higher
    ranks_to_try = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    for r in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
        
        # Allocate time based on rank - lower ranks get more time
        if r <= 15:
            time_per_rank = 12
        elif r <= 18:
            time_per_rank = 8
        else:
            time_per_rank = 5
        
        # First try structured search
        result = _structured_search(n, m, p, r, time_limit=time_per_rank * 0.6)
        
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Then try discrete search if time remains
        remaining_time = time_per_rank - (time.time() - start_time - (time_limit - time_per_rank))
        if time.time() - start_time < time_limit and remaining_time > 2:
            result = _discrete_search(n, m, p, r, time_limit=min(remaining_time, 5))
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Check if we're running out of time
        if time.time() - start_time > time_limit - 5:
            break
    
    return schoolbook
# EVOLVE-BLOCK-END
