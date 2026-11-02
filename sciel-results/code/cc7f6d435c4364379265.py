"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_list(arr):
    return np.array(arr).tolist()

def _structured_search(target_rank, time_budget, n=2, m=3, p=3):
    """Structured search for rank decomposition using algebraic patterns."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Try various structured patterns for U and V
    # Pattern 1: Binary patterns with different densities
    # Pattern 2: Cyclic patterns
    # Pattern 3: Kronecker-like patterns
    
    np.random.seed(12345)
    
    # Generate candidate patterns
    candidates = []
    
    # Add random patterns with various densities
    for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for _ in range(50):
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            for c in range(target_rank):
                u_mask = np.random.random(n*m) < density
                v_mask = np.random.random(m*p) < density
                U[u_mask, c] = 1
                V[v_mask, c] = 1
                # Ensure at least one 1 in each column
                if U[:, c].sum() == 0:
                    U[np.random.randint(n*m), c] = 1
                if V[:, c].sum() == 0:
                    V[np.random.randint(m*p), c] = 1
            candidates.append((U, V))
    
    # Add structured cyclic patterns
    for offset in range(target_rank):
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        for c in range(target_rank):
            # Cyclic pattern for U
            for idx in range(n*m):
                if (idx + c + offset) % 3 == 0:
                    U[idx, c] = 1
            # Cyclic pattern for V
            for idx in range(m*p):
                if (idx + c + offset) % 3 == 0:
                    V[idx, c] = 1
        candidates.append((U, V))
    
    # Add Kronecker-like patterns
    for base in range(3):
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        for c in range(target_rank):
            pattern_u = [(c + base) % 3]
            pattern_v = [(c + base) % 3]
            for idx in range(n*m):
                if idx % 3 in pattern_u:
                    U[idx, c] = 1
            for idx in range(m*p):
                if idx % 3 in pattern_v:
                    V[idx, c] = 1
        candidates.append((U, V))
    
    # Try each candidate
    for U, V in candidates:
        if time.time() > deadline:
            break
        
        # Build linear system for W
        num_eqs = n * m * p
        num_unk = p * n * target_rank
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        u_val = U[i*m+j, c]
                        v_val = V[j*p+k, c]
                        if u_val != 0 and v_val != 0:
                            w_row = k * n + i
                            w_col = c
                            unk_idx = w_row * target_rank + w_col
                            A[eq_idx, unk_idx] = u_val * v_val
                    eq_idx += 1
        
        # Check if system is consistent
        rank_A = np.linalg.matrix_rank(A)
        if rank_A == 0:
            continue
        
        # Solve A @ W_vec = b
        try:
            W_vec, residuals, rank_solved, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if solution is exact
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        W_mat = W_vec.reshape(p*n, target_rank)
        
        # Check if W is close to half-integers
        W_check = 2 * W_mat
        max_err = np.max(np.abs(W_check - np.round(W_check)))
        if max_err < 1e-6:
            W_rounded = np.round(W_check) / 2.0
            if _verify(U.tolist(), V.tolist(), W_rounded.tolist()):
                return {
                    "U": _to_list(U),
                    "V": _to_list(V),
                    "W": _to_list(W_rounded)
                }
    
    return None

def _random_search(target_rank, time_budget, n=2, m=3, p=3):
    """Random search for rank decomposition."""
    start_time = time.time()
    deadline = start_time + time_budget
    np.random.seed(42)
    
    for trial in range(50000):
        if time.time() > deadline:
            break
        
        # Random U and V with entries from {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n*m, target_rank))
        V = np.random.randint(-1, 2, size=(m*p, target_rank))
        
        # Build linear system for W
        num_eqs = n * m * p
        num_unk = p * n * target_rank
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        u_val = U[i*m+j, c]
                        v_val = V[j*p+k, c]
                        if u_val != 0 and v_val != 0:
                            w_row = k * n + i
                            w_col = c
                            unk_idx = w_row * target_rank + w_col
                            A[eq_idx, unk_idx] = u_val * v_val
                    eq_idx += 1
        
        # Solve A @ W_vec = b
        try:
            W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        W_mat = W_vec.reshape(p*n, target_rank)
        
        # Check if W is close to half-integers
        W_check = 2 * W_mat
        max_err = np.max(np.abs(W_check - np.round(W_check)))
        if max_err < 1e-6:
            W_rounded = np.round(W_check) / 2.0
            if _verify(U.tolist(), V.tolist(), W_rounded.tolist()):
                return {
                    "U": _to_list(U),
                    "V": _to_list(V),
                    "W": _to_list(W_rounded)
                }
    
    return None

def construct():
    start_time = time.time()
    total_time = 140  # Safety margin before 150s kill
    
    # Try structured search first for lower ranks
    for target_rank in [12, 13, 14, 15, 16]:
        elapsed = time.time() - start_time
        remaining = total_time - elapsed
        if remaining < 10:
            break
        
        # Allocate time: structured gets 60%, random gets 40%
        struct_time = min(remaining * 0.6, 50)
        random_time = min(remaining * 0.4, 40)
        
        result = _structured_search(target_rank, struct_time)
        if result is not None:
            return result
        
        elapsed = time.time() - start_time
        remaining = total_time - elapsed
        if remaining < 5:
            break
        
        result = _random_search(target_rank, random_time)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
