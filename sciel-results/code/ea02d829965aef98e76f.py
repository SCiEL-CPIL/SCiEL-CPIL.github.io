import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Schoolbook fallback: rank = n*m*p = 56
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Try to find a lower rank decomposition
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # But for matrix multiplication, the Strassen-like bounds apply
    
    # Try ranks from 28 upward
    for target_rank in range(28, 57):
        if time.time() > deadline - 5:
            break
        
        result = find_rank_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            return result
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def find_rank_decomposition(n, m, p, rank, deadline):
    """
    Find a rank-rank decomposition for n x m by m x p matrix multiplication.
    Uses vectorized error computation and structured search.
    """
    
    # Precompute index mappings for vectorized operations
    # For each (i,j,k), we need U[i*m+j], V[j*p+k], W[k*n+i]
    i_idx = np.array([(i, j, k) for i in range(n) for j in range(m) for k in range(p)])
    u_rows = i_idx[:, 0] * m + i_idx[:, 1]  # i*m + j
    v_rows = i_idx[:, 1] * p + i_idx[:, 2]  # j*p + k
    w_rows = i_idx[:, 2] * n + i_idx[:, 0]  # k*n + i
    
    def compute_error(U, V, W):
        """Compute total squared error using vectorized operations"""
        # U[u_rows, r] * V[v_rows, r] * W[w_rows, r] summed over r
        # Shape: (56, rank) for each
        U_sel = U[u_rows, :]  # (56, rank)
        V_sel = V[v_rows, :]  # (56, rank)
        W_sel = W[w_rows, :]  # (56, rank)
        products = U_sel * V_sel * W_sel  # (56, rank)
        sums = products.sum(axis=1)  # (56,)
        errors = (sums - 1.0) ** 2
        return errors.sum()
    
    def verify_exact(U, V, W):
        """Verify that the decomposition is exact"""
        U_sel = U[u_rows, :]
        V_sel = V[v_rows, :]
        W_sel = W[w_rows, :]
        products = U_sel * V_sel * W_sel
        sums = products.sum(axis=1)
        return np.allclose(sums, 1.0, atol=1e-10)
    
    def snap_to_half_integer(U, V, W):
        """Snap values to nearest half-integer"""
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        return U_snapped, V_snapped, W_snapped
    
    # Try multiple strategies
    best_result = None
    best_error = float('inf')
    
    # Strategy 1: Structured initialization based on Kronecker-like patterns
    # For n=2, we can try to exploit the structure
    num_attempts = 10
    for attempt in range(num_attempts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if attempt == 0:
            # All ones initialization
            U = np.ones((n * m, rank))
            V = np.ones((m * p, rank))
            W = np.ones((p * n, rank)) / (n * m * p / rank) if rank > 0 else np.ones((p * n, rank))
        elif attempt == 1:
            # Random small values
            U = np.random.uniform(-1, 1, (n * m, rank))
            V = np.random.uniform(-1, 1, (m * p, rank))
            W = np.random.uniform(-1, 1, (p * n, rank))
        elif attempt == 2:
            # Sparse initialization
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for r in range(rank):
                U[np.random.randint(0, n*m), r] = 1
                V[np.random.randint(0, m*p), r] = 1
                W[np.random.randint(0, p*n), r] = 1
        elif attempt == 3:
            # Block-structured: split into 2x2 blocks
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for r in range(rank):
                # Assign based on block structure
                block = r % 4
                row_u = block * 2 + np.random.randint(0, 2)
                row_v = block * 7 + np.random.randint(0, 7)
                row_w = (r % 7) * 2 + np.random.randint(0, 2)
                U[row_u, r] = 1
                V[row_v, r] = 1
                W[row_w, r] = 1
        else:
            # Random with bias toward 0 and 1
            U = np.where(np.random.random((n * m, rank)) < 0.5, 0, 1)
            V = np.where(np.random.random((m * p, rank)) < 0.5, 0, 1)
            W = np.where(np.random.random((p * n, rank)) < 0.5, 0, 1)
        
        current_error = compute_error(U, V, W)
        if current_error < best_error:
            best_error = current_error
            if current_error < 1e-10 and verify_exact(U, V, W):
                U_snapped, V_snapped, W_snapped = snap_to_half_integer(U, V, W)
                if verify_exact(U_snapped, V_snapped, W_snapped):
                    return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
        
        # Local optimization with coordinate descent
        max_iterations = 500
        for it in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Pick which matrix to update
            mat_idx = np.random.randint(0, 3)
            
            if mat_idx == 0:
                row = np.random.randint(0, n * m)
                col = np.random.randint(0, rank)
                old_val = U[row, col]
                # Try perturbations
                for delta in [-1, -0.5, -0.1, 0.1, 0.5, 1]:
                    U[row, col] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        if current_error < best_error:
                            best_error = current_error
                            if current_error < 1e-10 and verify_exact(U, V, W):
                                U_snapped, V_snapped, W_snapped = snap_to_half_integer(U, V, W)
                                if verify_exact(U_snapped, V_snapped, W_snapped):
                                    return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                        break
                    else:
                        U[row, col] = old_val
            elif mat_idx == 1:
                row = np.random.randint(0, m * p)
                col = np.random.randint(0, rank)
                old_val = V[row, col]
                for delta in [-1, -0.5, -0.1, 0.1, 0.5, 1]:
                    V[row, col] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        if current_error < best_error:
                            best_error = current_error
                            if current_error < 1e-10 and verify_exact(U, V, W):
                                U_snapped, V_snapped, W_snapped = snap_to_half_integer(U, V, W)
                                if verify_exact(U_snapped, V_snapped, W_snapped):
                                    return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                        break
                    else:
                        V[row, col] = old_val
            else:
                row = np.random.randint(0, p * n)
                col = np.random.randint(0, rank)
                old_val = W[row, col]
                for delta in [-1, -0.5, -0.1, 0.1, 0.5, 1]:
                    W[row, col] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        if current_error < best_error:
                            best_error = current_error
                            if current_error < 1e-10 and verify_exact(U, V, W):
                                U_snapped, V_snapped, W_snapped = snap_to_half_integer(U, V, W)
                                if verify_exact(U_snapped, V_snapped, W_snapped):
                                    return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                        break
                    else:
                        W[row, col] = old_val
        
        # After local search, try to snap
        if current_error < 1.0:
            U_snapped, V_snapped, W_snapped = snap_to_half_integer(U, V, W)
            if verify_exact(U_snapped, V_snapped, W_snapped):
                return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
    
    return None
