import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    deadline = time.time() + 145  # Safety margin
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the list of (u_idx, v_idx, w_idx) for each tensor entry
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(entries)  # 24

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    np.random.seed(42)
    
    # Candidate values for half-integers
    # We restrict to small values to keep the linear system well-conditioned and likely to have half-integer solutions
    candidate_vals = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])

    def verify_solution(U, V, W, target_rank):
        """Check if the snapped solution is exact"""
        for (u_idx, v_idx, w_idx) in entries:
            s = 0.0
            for r in range(target_rank):
                s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            if abs(s - 1.0) > 1e-9:
                return False
        return True

    def snap_to_half_int(x):
        """Snap to nearest half-integer"""
        return np.round(2.0 * x) / 2.0

    # Try ranks from 19 upward (known lower bound for 2x3x4 is 19)
    # We start at 19 because 18 is theoretically impossible for 2x3x4 (rank >= 19)
    for target_rank in range(19, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Number of random restarts. We can afford many because the check is fast.
        num_restarts = 5000
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Randomly initialize U and V with small half-integers
            # Using sparse initialization might help? Let's try dense first.
            U_init = np.random.choice(candidate_vals, size=(nm, target_rank))
            V_init = np.random.choice(candidate_vals, size=(mp, target_rank))
            
            # Construct the linear system for W
            # Equation: sum_r U[u,r] V[v,r] W[w,r] = 1
            # For each entry e, we have a linear combination of W variables.
            # The variables are W[w_idx, r].
            # There are pn * target_rank variables.
            # There are num_entries equations.
            
            # We can represent this as M W_flat = 1
            # M has shape (num_entries, pn * target_rank)
            
            # To make it faster, we can just iterate.
            # But solving a linear system is fast.
            
            # Let's build the matrix M
            M = np.zeros((num_entries, pn * target_rank))
            for e_idx, (u_idx, v_idx, w_idx) in enumerate(entries):
                for r in range(target_rank):
                    val = U_init[u_idx, r] * V_init[v_idx, r]
                    if val != 0:
                        M[e_idx, w_idx * target_rank + r] = val
            
            # Solve M W_flat = 1
            # Since the system is underdetermined, we use least squares (minimum norm)
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(M, np.ones(num_entries), rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Reshape W_flat
            W_opt = W_flat.reshape(pn, target_rank)
            
            # Snap to half-integers
            W_snapped = snap_to_half_int(W_opt)
            
            # Verify
            if verify_solution(U_init, V_init, W_snapped, target_rank):
                # Found a valid solution!
                # Convert to lists of floats (representing half-integers)
                U_list = [[float(x) for x in row] for row in U_init]
                V_list = [[float(x) for x in row] for row in V_init]
                W_list = [[float(x) for x in row] for row in W_snapped]
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
