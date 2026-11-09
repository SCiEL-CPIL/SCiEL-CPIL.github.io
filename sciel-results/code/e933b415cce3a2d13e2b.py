"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition with rank = n*m*p."""
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual for all (i,j,k) triples."""
    rank = U.shape[1]
    residual = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                residual[i, j, k] = s - 1.0
    return residual

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    residual = _compute_residual(U, V, W, n, m, p)
    return np.max(np.abs(residual)) <= tol

def _count_nonzero_columns(U, V, W):
    """Count columns that have any non-zero entry."""
    rank = U.shape[1]
    count = 0
    for c in range(rank):
        if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
            count += 1
    return count

def _local_search(n, m, p, deadline):
    """
    Local search to minimize rank. Start from schoolbook, then:
    1. Try removing columns
    2. Try setting individual entries to zero (sparsification)
    3. Repeat until no improvement or deadline
    """
    schoolbook = _build_schoolbook(n, m, p)
    U = np.array(schoolbook["U"], dtype=float)
    V = np.array(schoolbook["V"], dtype=float)
    W = np.array(schoolbook["W"], dtype=float)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Precompute indices for speed
    indices_U = [(i, j) for i in range(n) for j in range(m)]
    indices_V = [(j, k) for j in range(m) for k in range(p)]
    indices_W = [(k, i) for k in range(p) for i in range(n)]
    
    # Map (i,j) to row in U
    u_row = {}
    for i in range(n):
        for j in range(m):
            u_row[(i, j)] = i * m + j
    v_row = {}
    for j in range(m):
        for k in range(p):
            v_row[(j, k)] = j * p + k
    w_row = {}
    for k in range(p):
        for i in range(n):
            w_row[(k, i)] = k * n + i
    
    def try_remove_column(U, V, W, c):
        """Try removing column c. Return True if valid after removal."""
        mask = np.ones(U.shape[1], dtype=bool)
        mask[c] = False
        U_r = U[:, mask]
        V_r = V[:, mask]
        W_r = W[:, mask]
        if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-9):
            return True, U_r, V_r, W_r
        return False, U, V, W
    
    def try_zero_entry(U, V, W, matrix, row, col):
        """Try zeroing an entry. Return (success, U, V, W)."""
        if matrix == 'U' and U[row, col] == 0:
            return False, U, V, W
        if matrix == 'V' and V[row, col] == 0:
            return False, U, V, W
        if matrix == 'W' and W[row, col] == 0:
            return False, U, V, W
        
        old_val = 0.0
        if matrix == 'U':
            old_val = U[row, col]
            U[row, col] = 0.0
        elif matrix == 'V':
            old_val = V[row, col]
            V[row, col] = 0.0
        else:
            old_val = W[row, col]
            W[row, col] = 0.0
        
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return True, U, V, W
        else:
            # Restore
            if matrix == 'U':
                U[row, col] = old_val
            elif matrix == 'V':
                V[row, col] = old_val
            else:
                W[row, col] = old_val
            return False, U, V, W
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try removing columns
        for c in range(U.shape[1]):
            if time.time() >= deadline:
                break
            success, U_new, V_new, W_new = try_remove_column(U, V, W, c)
            if success:
                U, V, W = U_new, V_new, W_new
                improved = True
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = U.shape[1]
                break  # Restart column scan after removal
        
        # Phase 2: Try zeroing entries (sparsification)
        if time.time() < deadline:
            # Try zeroing entries in U
            for i in range(n):
                if time.time() >= deadline:
                    break
                for j in range(m):
                    if time.time() >= deadline:
                        break
                    row = u_row[(i, j)]
                    for c in range(U.shape[1]):
                        if time.time() >= deadline:
                            break
                        if U[row, c] != 0:
                            success, U_new, V_new, W_new = try_zero_entry(U, V, W, 'U', row, c)
                            if success:
                                U, V, W = U_new, V_new, W_new
                                improved = True
                                nz = _count_nonzero_columns(U, V, W)
                                if nz < best_rank:
                                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                    best_rank = nz
                                break  # Restart after change
            
            # Try zeroing entries in V
            if time.time() < deadline:
                for j in range(m):
                    if time.time() >= deadline:
                        break
                    for k in range(p):
                        if time.time() >= deadline:
                            break
                        row = v_row[(j, k)]
                        for c in range(V.shape[1]):
                            if time.time() >= deadline:
                                break
                            if V[row, c] != 0:
                                success, U_new, V_new, W_new = try_zero_entry(U, V, W, 'V', row, c)
                                if success:
                                    U, V, W = U_new, V_new, W_new
                                    improved = True
                                    nz = _count_nonzero_columns(U, V, W)
                                    if nz < best_rank:
                                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                        best_rank = nz
                                    break
            
            # Try zeroing entries in W
            if time.time() < deadline:
                for k in range(p):
                    if time.time() >= deadline:
                        break
                    for i in range(n):
                        if time.time() >= deadline:
                            break
                        row = w_row[(k, i)]
                        for c in range(W.shape[1]):
                            if time.time() >= deadline:
                                break
                            if W[row, c] != 0:
                                success, U_new, V_new, W_new = try_zero_entry(U, V, W, 'W', row, c)
                                if success:
                                    U, V, W = U_new, V_new, W_new
                                    improved = True
                                    nz = _count_nonzero_columns(U, V, W)
                                    if nz < best_rank:
                                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                        best_rank = nz
                                    break
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4×5 × 5×6 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition
    result = _build_schoolbook(n, m, p)
    
    # Verify it's correct
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        pass
    
    # Try to reduce rank via local search
    U_red, V_red, W_red, reduced_rank = _local_search(n, m, p, deadline)
    
    if reduced_rank < result["rank"]:
        result = {"U": U_red, "V": V_red, "W": W_red}
    
    return result
# EVOLVE-BLOCK-END
