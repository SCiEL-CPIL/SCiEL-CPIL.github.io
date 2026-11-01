"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # 24
    
    # Schoolbook fallback (rank 24)
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def verify_solution(U, V, W, rank):
        """Check if the decomposition is exact."""
        # Check tensor entries
        for (u, v, w) in tensor_entries:
            val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
            if abs(val - 1.0) > 1e-10:
                return False
        # Check non-entries
        entry_set = set(tensor_entries)
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) not in entry_set:
                        val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                        if abs(val) > 1e-10:
                            return False
        return True

    def verify_solution_fast(U, V, W, rank):
        """Faster verification using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check tensor entries
        for (u, v, w) in tensor_entries:
            val = np.dot(U_arr[u], V_arr[v] * W_arr[w])
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check a sample of non-entries
        entry_set = set(tensor_entries)
        np.random.seed(12345)
        # Check all non-entries for correctness
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) not in entry_set:
                        val = np.dot(U_arr[u], V_arr[v] * W_arr[w])
                        if abs(val) > 1e-9:
                            return False
        return True

    # Search for minimal rank using randomized linear solving
    # Theoretical lower bound is 18 for 2x3x4. We start from 18 to save time.
    # If 18 fails, we go up to 24.
    
    # We use a strategy where we fix U and V randomly (sparse, small values)
    # and solve for W. Then we check if W is "simple" (half-integer).
    
    # To make this work, we need the system A_W * W_flat = b_W to be consistent.
    # A_W is built from U and V.
    
    # We will try ranks from 18 to 23.
    for target_rank in range(18, best_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        np.random.seed(42 + target_rank)
        
        # Number of random attempts
        num_attempts = 5000
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 2:
                break
            
            # Generate random U and V with entries in {-1, 0, 1}
            # Using sparse matrices increases the chance of solvability and simple W
            U_init = np.random.choice([-1, 0, 1], size=(nm, target_rank))
            V_init = np.random.choice([-1, 0, 1], size=(mp, target_rank))
            
            # Ensure U and V are not zero vectors
            if np.all(U_init == 0) or np.all(V_init == 0):
                continue
            
            # Build A_W and b_W
            # Equation: sum_r U[u,r] * V[v,r] * W[w,r] = T[u,v,w]
            # For each (u,v,w) in tensor_entries:
            #   sum_r (U[u,r] * V[v,r]) * W[w,r] = 1
            # This is a linear system in W[w,r].
            
            # We have num_entries equations.
            # Unknowns are W[w,r] for w in 0..pn-1, r in 0..target_rank-1.
            # Total unknowns: pn * target_rank.
            
            A_W = np.zeros((num_entries, pn * target_rank))
            b_W = np.ones(num_entries)
            
            for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                for r in range(target_rank):
                    col = w_idx * target_rank + r
                    A_W[e_idx, col] = U_init[u_idx, r] * V_init[v_idx, r]
            
            # Solve least squares
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A_W, b_W, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            W_curr = W_flat.reshape(pn, target_rank)
            
            # Check if W_curr is close to half-integers
            # We want W_curr * 2 to be close to integers
            W_check = np.round(W_curr * 2) / 2.0
            
            # Calculate error if we snap to half-integers
            # We need to verify the full tensor equation with the snapped W
            # and the original U, V
            
            # Fast check: only check tensor entries first
            error = 0.0
            for (u, v, w) in tensor_entries:
                val = sum(U_init[u, r] * V_init[v, r] * W_check[w, r] for r in range(target_rank))
                error += abs(val - 1.0)
                
            if error > 1e-6:
                continue
                
            # Full verification
            if verify_solution_fast(U_init.tolist(), V_init.tolist(), W_check.tolist(), target_rank):
                best_rank = target_rank
                best_result = {
                    "U": U_init.tolist(),
                    "V": V_init.tolist(),
                    "W": W_check.tolist()
                }
                found = True
                break
        
        if found:
            break
    
    # If we found a solution, return it
    if best_rank < n * m * p:
        return best_result
    
    # Return schoolbook if nothing better found
    return best_result
# EVOLVE-BLOCK-END
