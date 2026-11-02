import time
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 7]

def _build_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _support_indices(n, m, p):
    """Return list of (idx_u, idx_v, idx_w) for all support entries."""
    sup = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                sup.append((i * m + j, j * p + k, k * n + i))
    return sup

def _check_exact(U, V, W, n, m, p, support):
    """
    Check that the decomposition is exact using Fraction.
    U, V, W are lists of lists of floats (representing half-integers).
    """
    rank = len(U[0]) if U else 0
    # Convert to Fraction for exactness
    U_frac = [[Fraction(U[i][j]).limit_denominator(100) for j in range(rank)] for i in range(n*m)]
    V_frac = [[Fraction(V[i][j]).limit_denominator(100) for j in range(rank)] for i in range(m*p)]
    W_frac = [[Fraction(W[i][j]).limit_denominator(100) for j in range(rank)] for i in range(p*n)]
    
    support_set = set(support)
    
    # Check support entries sum to 1
    for (a, b, c) in support:
        s = sum(U_frac[a][r] * V_frac[b][r] * W_frac[c][r] for r in range(rank))
        if s != 1:
            return False
            
    # Check non-support entries sum to 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) in support_set:
                    continue
                s = sum(U_frac[a][r] * V_frac[b][r] * W_frac[c][r] for r in range(rank))
                if s != 0:
                    return False
    return True

def _calculate_error(U, V, W, n, m, p, target_tensor):
    """
    Calculate the squared error between the reconstructed tensor and the target.
    U, V, W are numpy arrays.
    """
    rank = U.shape[1]
    # Reshape U, V, W for efficient computation
    # R[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # Step 1: Compute UV[a, b, r] = U[a, r] * V[b, r]
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # We want a tensor of shape (n*m, m*p, rank)
    UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
    
    # Step 2: Compute R[a, b, c] = sum_r UV[a, b, r] * W[c, r]
    # W shape: (p*n, rank)
    # We want to multiply UV (n*m, m*p, rank) with W (rank, p*n)
    # This is a batched matrix multiplication or einsum
    # R shape: (n*m, m*p, p*n)
    R = np.einsum('abr,cr->abc', UV, W)
    
    # target_tensor is a numpy array of shape (n*m, m*p, p*n) with 1s at support
    diff = R - target_tensor
    return np.sum(diff**2)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    n, m, p = DIMENSIONS
    support = _support_indices(n, m, p)
    support_set = set(support)
    
    # Build target tensor
    target_tensor = np.zeros((n*m, m*p, p*n))
    for (a, b, c) in support:
        target_tensor[a, b, c] = 1.0

    # Schoolbook fallback
    best_result = _build_schoolbook(n, m, p)
    best_rank = n * m * p

    val_set = np.array([-1, -0.5, 0, 0.5, 1], dtype=float)
    
    # We will try ranks from 41 down to 21
    # To save time, we prioritize lower ranks but start with a few attempts at each
    ranks_to_try = list(range(41, 20, -1))
    
    np.random.seed(42)
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        # Number of attempts per rank
        # Higher ranks are easier, so fewer attempts? No, we want to find minimal.
        # So we try lower ranks more? Or just iterate.
        max_attempts = 10 if rank < 35 else 5
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
                
            # Initialize U, V, W
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Random initialization with some density
            # Density 0.5 seems reasonable
            mask_u = np.random.rand(n*m, rank) < 0.5
            mask_v = np.random.rand(m*p, rank) < 0.5
            mask_w = np.random.rand(p*n, rank) < 0.5
            
            U[mask_u] = val_set[np.random.randint(0, len(val_set), size=np.sum(mask_u))]
            V[mask_v] = val_set[np.random.randint(0, len(val_set), size=np.sum(mask_v))]
            W[mask_w] = val_set[np.random.randint(0, len(val_set), size=np.sum(mask_w))]
            
            current_error = _calculate_error(U, V, W, n, m, p, target_tensor)
            
            # Simulated Annealing
            T_temp = 1.0
            cooling = 0.99
            min_temp = 0.01
            
            # If initial error is already very low, check immediately
            if current_error < 1e-9:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _check_exact(U_list, V_list, W_list, n, m, p, support):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result

            for step in range(2000):
                if time.time() > deadline - 2:
                    break
                
                # Propose a move
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    mat = U
                    rows = n * m
                elif mat_choice == 1:
                    mat = V
                    rows = m * p
                else:
                    mat = W
                    rows = p * n
                
                idx = np.random.randint(0, rows)
                col = np.random.randint(0, rank)
                old_val = mat[idx, col]
                new_val = val_set[np.random.randint(0, len(val_set))]
                
                if old_val == new_val:
                    continue
                    
                mat[idx, col] = new_val
                new_error = _calculate_error(U, V, W, n, m, p, target_tensor)
                
                delta = new_error - current_error
                if delta < 0 or np.random.rand() < np.exp(-delta / T_temp):
                    current_error = new_error
                    if current_error < 1e-9:
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        if _check_exact(U_list, V_list, W_list, n, m, p, support):
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = rank
                            return best_result
                else:
                    mat[idx, col] = old_val
                    
                T_temp *= cooling
                if T_temp < min_temp:
                    T_temp = min_temp

    return best_result
