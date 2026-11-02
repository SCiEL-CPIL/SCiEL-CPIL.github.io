"""
Matrix multiplication 2x3, 3x9 tensor decomposition.
Strategy: Structured random search with sparsity constraints.
We assume the rank-53 decomposition is "close" to the schoolbook one, meaning the U and V matrices are sparse.
We restrict the search space by allowing at most 2 non-zero entries per column in U and V, with values from {-1, -0.5, 0, 0.5, 1}.
This is a different design family from the previous "random dense/sparse with pinv" approach, focusing on explicit sparsity patterns and integer/half-integer arithmetic.
"""

import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for evaluation overhead and safety
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    rank_schoolbook = n * m * p  # 54
    target_rank = rank_schoolbook - 1  # 53
    
    # Build the target tensor flattened: T_flat[u*dim_v+v, w] = 1 if (u,v,w) is a valid multiplication term
    # u = i*m + j, v = j*p + k, w = k*n + i
    T_flat = np.zeros((dim_u * dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T_flat[u_idx * dim_v + v_idx, w_idx] = 1.0
    
    # Fallback: Schoolbook
    def get_schoolbook():
        rank = rank_schoolbook
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
    
    # Values allowed for entries
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # We will perform a structured random search.
    # Each column of U and V will have exactly 2 non-zero entries (or 1, but 2 is more likely to allow merging).
    # We will iterate through many random patterns.
    
    num_trials = 5000
    for t in range(num_trials):
        if time.time() > deadline:
            break
            
        U_rand = np.zeros((dim_u, target_rank))
        V_rand = np.zeros((dim_v, target_rank))
        
        # Generate U and V with sparsity pattern: 2 non-zeros per column
        for r in range(target_rank):
            # Choose 2 indices for U
            u_indices = random.sample(range(dim_u), 2)
            u_vals = random.choices(values, k=2)
            # Ensure not both zero
            if u_vals[0] == 0 and u_vals[1] == 0:
                u_vals[0] = 1.0
            U_rand[u_indices[0], r] = u_vals[0]
            U_rand[u_indices[1], r] = u_vals[1]
            
            # Choose 2 indices for V
            v_indices = random.sample(range(dim_v), 2)
            v_vals = random.choices(values, k=2)
            if v_vals[0] == 0 and v_vals[1] == 0:
                v_vals[0] = 1.0
            V_rand[v_indices[0], r] = v_vals[0]
            V_rand[v_indices[1], r] = v_vals[1]
            
        # Construct B
        # B[u*dim_v+v, r] = U_rand[u, r] * V_rand[v, r]
        B = np.einsum('ur,vr->uvr', U_rand, V_rand).reshape(dim_u * dim_v, target_rank)
        
        # Solve B @ W^T = T_flat
        # We use least squares, but check for exactness
        try:
            W_t, residuals, rank_B, s = np.linalg.lstsq(B, T_flat, rcond=None)
            W_candidate = W_t.T
            
            # Check if W_candidate is half-integer
            W_scaled = W_candidate * 2
            W_rounded = np.round(W_scaled)
            error = np.max(np.abs(W_scaled - W_rounded))
            
            if error < 1e-6:
                # Verify exactness
                B_check = np.einsum('ur,vr->uvr', U_rand, V_rand).reshape(dim_u * dim_v, target_rank)
                T_reconstructed = B_check @ W_candidate.T
                if np.allclose(T_reconstructed, T_flat, atol=1e-6):
                    # Convert to lists
                    # Ensure entries are clean integers or half-integers
                    U_out = [[(x / 2).round() * 2 for x in row] for row in U_rand] # This is just rounding to 0.5
                    # Actually, U and V are already from the set, so they are clean.
                    # W is rounded.
                    W_out = W_rounded / 2.0
                    
                    return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_out.tolist()}
        except Exception:
            pass
    
    # If structured search fails, try a less constrained random search with higher density
    num_trials_2 = 2000
    for t in range(num_trials_2):
        if time.time() > deadline:
            break
        
        U_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_u, target_rank), p=[0.1, 0.1, 0.6, 0.1, 0.1])
        V_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_v, target_rank), p=[0.05, 0.05, 0.8, 0.05, 0.05])
        
        B = np.einsum('ur,vr->uvr', U_rand, V_rand).reshape(dim_u * dim_v, target_rank)
        
        try:
            W_t, residuals, rank_B, s = np.linalg.lstsq(B, T_flat, rcond=None)
            W_candidate = W_t.T
            
            W_scaled = W_candidate * 2
            W_rounded = np.round(W_scaled)
            error = np.max(np.abs(W_scaled - W_rounded))
            
            if error < 1e-6:
                B_check = np.einsum('ur,vr->uvr', U_rand, V_rand).reshape(dim_u * dim_v, target_rank)
                T_reconstructed = B_check @ W_candidate.T
                if np.allclose(T_reconstructed, T_flat, atol=1e-6):
                    W_out = W_rounded / 2.0
                    return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_out.tolist()}
        except Exception:
            pass
            
    return get_schoolbook()
