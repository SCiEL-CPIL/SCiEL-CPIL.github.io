import numpy as np
import time
import random
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    nm, mp, pn = n * m, m * p, p * n

    # Target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def error_metric(U, V, W):
        T_prime = compute_tensor(U, V, W)
        return np.sum((T_prime - T_target) ** 2)

    def is_valid(U, V, W, tol=1e-6):
        return error_metric(U, V, W) < tol

    # Schoolbook decomposition (rank 12)
    rank_school = n * m * p
    U_sb = np.zeros((nm, rank_school))
    V_sb = np.zeros((mp, rank_school))
    W_sb = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school

    # Strategy: Discrete Local Search on Half-Integer Lattice
    # We attempt to find a solution of rank 10, then 11.
    # We use a "jump and climb" approach:
    # 1. Initialize with random half-integers.
    # 2. Use a local search that moves to neighboring half-integers.
    # 3. If stuck, perform a larger random jump.
    
    target_ranks = [10, 11]
    
    def discrete_local_search(rank, time_limit):
        """
        Attempts to find a half-integer decomposition of given rank.
        Uses a combination of random restarts and local hill climbing.
        """
        if time.time() > time_limit:
            return None

        # Parameters for the search
        max_iterations_per_run = 5000
        jump_prob = 0.05
        step_size = 0.5 # We are on a half-integer lattice
        
        # Pre-compute indices for faster error calculation?
        # np.einsum is fast enough for small matrices.

        for attempt in range(50): # 50 restarts
            if time.time() > time_limit:
                break
                
            # Initialize U, V, W with random half-integers in [-2, 2]
            # Scaling is important. Entries are usually small.
            scale = 2.0
            U = np.round(np.random.uniform(-scale, scale, (nm, rank)) / step_size) * step_size
            V = np.round(np.random.uniform(-scale, scale, (mp, rank)) / step_size) * step_size
            W = np.round(np.random.uniform(-scale, scale, (pn, rank)) / step_size) * step_size
            
            current_error = error_metric(U, V, W)
            
            # If we start with a valid solution (unlikely), return
            if current_error < 1e-6:
                return U, V, W

            # Local Search Loop
            for it in range(max_iterations_per_run):
                if time.time() > time_limit:
                    return None
                
                # Decide whether to jump or step
                if random.random() < jump_prob:
                    # Large jump: re-randomize a subset of entries
                    # Randomly select 10% of entries to re-randomize
                    num_jumps = max(1, int(0.1 * rank))
                    
                    # Jump in U
                    idx_u = np.random.choice(nm * rank, num_jumps, replace=False)
                    U_flat = U.flatten()
                    for idx in idx_u:
                        U_flat[idx] = np.round(random.uniform(-scale, scale) / step_size) * step_size
                    U = U_flat.reshape(nm, rank)
                    
                    # Jump in V
                    idx_v = np.random.choice(mp * rank, num_jumps, replace=False)
                    V_flat = V.flatten()
                    for idx in idx_v:
                        V_flat[idx] = np.round(random.uniform(-scale, scale) / step_size) * step_size
                    V = V_flat.reshape(mp, rank)
                    
                    # Jump in W
                    idx_w = np.random.choice(pn * rank, num_jumps, replace=False)
                    W_flat = W.flatten()
                    for idx in idx_w:
                        W_flat[idx] = np.round(random.uniform(-scale, scale) / step_size) * step_size
                    W = W_flat.reshape(pn, rank)
                    
                else:
                    # Small step: change one random entry by +/- step_size
                    # Choose which matrix to update
                    r = random.random()
                    if r < 0.33:
                        i, j = np.random.randint(0, nm), np.random.randint(0, rank)
                        delta = random.choice([-step_size, step_size])
                        U[i, j] += delta
                        new_error = error_metric(U, V, W)
                        if new_error < current_error:
                            current_error = new_error
                        else:
                            U[i, j] -= delta # Revert
                    elif r < 0.66:
                        i, j = np.random.randint(0, mp), np.random.randint(0, rank)
                        delta = random.choice([-step_size, step_size])
                        V[i, j] += delta
                        new_error = error_metric(U, V, W)
                        if new_error < current_error:
                            current_error = new_error
                        else:
                            V[i, j] -= delta # Revert
                    else:
                        i, j = np.random.randint(0, pn), np.random.randint(0, rank)
                        delta = random.choice([-step_size, step_size])
                        W[i, j] += delta
                        new_error = error_metric(U, V, W)
                        if new_error < current_error:
                            current_error = new_error
                        else:
                            W[i, j] -= delta # Revert
                
                if current_error < 1e-6:
                    return U, V, W
                    
        return None

    # Try rank 10 first
    time_for_rank_10 = min(60.0, deadline - time.time() - 10.0)
    if time_for_rank_10 > 5.0:
        res = discrete_local_search(10, time.time() + time_for_rank_10)
        if res:
            U, V, W = res
            best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
            best_rank = 10
            return {"U": best_U, "V": best_V, "W": best_W}

    # Try rank 11
    time_for_rank_11 = min(50.0, deadline - time.time() - 10.0)
    if time_for_rank_11 > 5.0:
        res = discrete_local_search(11, time.time() + time_for_rank_11)
        if res:
            U, V, W = res
            best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
            best_rank = 11
            return {"U": best_U, "V": best_V, "W": best_W}

    # Fallback to schoolbook if discrete search fails
    return {"U": best_U, "V": best_V, "W": best_W}
