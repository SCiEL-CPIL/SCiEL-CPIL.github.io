"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Uses vectorized validation and structured random search.
    """
    n, m, p = 2, 3, 4
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Target tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # T[a, b, c] = 1 if indices match, 0 otherwise.
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def validate(U, V, W):
        """Check if U, V, W exactly decompose the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute T_calc[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # Using einsum for efficiency
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Check exact match
        if not np.allclose(T_calc, T_target, atol=1e-9):
            return False
        return True

    # Schoolbook solution of rank 24
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_solution = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook

    random.seed(42)
    
    # Candidate values for entries
    candidates = [0, 0.5, -0.5, 1, -1]
    
    # Try to find lower rank decompositions
    # Theoretical lower bound for 2x3x4 is likely 12 or higher.
    # We search from 12 upwards.
    for target_rank in range(12, 24):
        if time.time() > deadline - 5:
            break
        
        num_attempts = 50 if target_rank <= 15 else 20
        for attempt in range(num_attempts):
            if time.time() > deadline - 3:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with sparse random values
            for r in range(target_rank):
                # Ensure each rank component is "active"
                # U: pick 1-2 random (i,j)
                num_u = random.choice([1, 2])
                u_indices = random.sample(range(n * m), num_u)
                for idx in u_indices:
                    U[idx, r] = random.choice(candidates[1:])
                
                # V: pick 1-2 random (j,k)
                num_v = random.choice([1, 2])
                v_indices = random.sample(range(m * p), num_v)
                for idx in v_indices:
                    V[idx, r] = random.choice(candidates[1:])
                    
                # W: pick 1-2 random (i,k)
                num_w = random.choice([1, 2])
                w_indices = random.sample(range(p * n), num_w)
                for idx in w_indices:
                    W[idx, r] = random.choice(candidates[1:])
            
            if validate(U, V, W):
                best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
            
    return best_solution
# EVOLVE-BLOCK-END
