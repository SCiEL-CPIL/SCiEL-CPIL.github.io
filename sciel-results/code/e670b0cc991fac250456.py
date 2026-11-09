import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 120
    
    # Precompute index mappings
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i, j, k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook solution as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find lower rank solutions using simulated annealing
    # We work in scaled space: entries are integers, divide by 2 at the end
    # This allows half-integer entries
    
    # For a given rank R, we want to find U, V, W with integer entries (scaled)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 2 for all (i,j,k)
    # (because we scale by 2, so 1.0 becomes 2)
    
    # Actually, let's think about this differently.
    # The constraint is: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # We want integer or half-integer entries.
    
    # Strategy: Use structured random search with pruning
    # For small ranks, try to find solutions where most entries are 0 or 1
    
    # First, let's try a more systematic approach:
    # For rank R, we can try to find a solution where each column r
    # corresponds to a "term" in the decomposition.
    
    # Key insight: If we can find a set of R triples (i_r, j_r, k_r) such that
    # each (i,j,k) is covered exactly once, then rank R = n*m*p works (schoolbook).
    # To get lower rank, we need some columns to cover multiple (i,j,k) tuples.
    
    # Let's try a different approach: use the fact that for matrix multiplication,
    # there are known lower bounds. For 4x5x6, the minimum rank is at least max(n*m, m*p, p*n) = max(20, 30, 24) = 30.
    # But in practice, for small matrices, the rank is often close to n*m*p.
    
    # Let's try random search with a smarter initialization:
    # Start with rank R and try to assign entries such that the constraint is satisfied.
    
    # Use a greedy/iterative approach:
    # For each (i,j,k), we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # This is a system of n*m*p = 120 equations in R* (n*m + m*p + p*n) = R*74 unknowns.
    # For R = 30, we have 2220 unknowns and 120 equations - very underdetermined.
    
    # Let's try to find solutions with small integer entries.
    # Use simulated annealing with the objective:
    # E = sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
    
    # To make this efficient, we can precompute contributions.
    
    def compute_energy(U, V, W, R):
        """Compute the energy (sum of squared errors)."""
        energy = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    energy += (s - 1.0) ** 2
        return energy
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 30
    # But let's start from a reasonable estimate
    
    # For efficiency, let's try a few specific ranks
    # and use random restarts with simulated annealing
    
    # Define the search space: entries in {-2, -1, 0, 1, 2} (scaled by 0.5)
    # So actual entries are in {-1, -0.5, 0, 0.5, 1}
    
    # Actually, let's allow a wider range: {-3, -2, -1, 0, 1, 2, 3}
    # Scaled: actual entries in {-1.5, -1, -0.5, 0, 0.5, 1, 1.5}
    
    # Let's try a more targeted approach:
    # For each candidate rank, initialize with random small integers
    # and use simulated annealing to minimize the energy.
    
    # To speed up energy computation, we can precompute for each (i,j,k)
    # the current sum and update incrementally.
    
    # Let's implement a faster version with incremental updates.
    
    def fast_simulated_annealing(R, max_iters=5000, init_temp=1.0, final_temp=0.01):
        """Run simulated annealing for rank R. Returns (U, V, W, energy) or None."""
        # Initialize with random small integers
        U = np.random.randint(-2, 3, size=(n*m, R))
        V = np.random.randint(-2, 3, size=(m*p, R))
        W = np.random.randint(-2, 3, size=(p*n, R))
        
        # Compute initial energy
        # Precompute S[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        S = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    S[i,j,k] = s
        
        energy = np.sum((S - 1.0) ** 2)
        
        best_energy = energy
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        # Simulated annealing
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Temperature schedule
            temp = init_temp * (final_temp / init_temp) ** (it / max_iters)
            
            # Choose a random move: pick a random matrix, random entry, change by +/-1
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                mat = U
                rows = n*m
            elif mat_choice == 1:
                mat = V
                rows = m*p
            else:
                mat = W
                rows = p*n
            
            row = np.random.randint(0, rows)
            col = np.random.randint(0, R)
            delta = np.random.choice([-1, 1])
            
            old_val = mat[row, col]
            new_val = old_val + delta
            
            # Check bounds
            if new_val < -3 or new_val > 3:
                continue
            
            # Compute change in energy
            # If we change U[row, col] by delta, then for each (i,j,k) where i*m+j == row,
            # S[i,j,k] changes by delta * V[j*p+k, col] * W[k*n+i, col]
            # Similarly for V and W.
            
            old_energy = energy
            
            # Apply the change
            mat[row, col] = new_val
            
            # Update S and energy
            if mat_choice == 0:
                # U changed
                i = row // m
                j = row % m
                for k in range(p):
                    old_s = S[i,j,k]
                    new_s = old_s + delta * V[j*p+k, col] * W[k*n+i, col]
                    energy -= old_s ** 2 - 2*old_s + 1
                    energy += new_s ** 2 - 2*new_s + 1
                    S[i,j,k] = new_s
            elif mat_choice == 1:
                # V changed
                j = row // p
                k = row % p
                for i in range(n):
                    old_s = S[i,j,k]
                    new_s = old_s + delta * U[i*m+j, col] * W[k*n+i, col]
                    energy -= old_s ** 2 - 2*old_s + 1
                    energy += new_s ** 2 - 2*new_s + 1
                    S[i,j,k] = new_s
            else:
                # W changed
                k = row // n
                i = row % n
                for j in range(m):
                    old_s = S[i,j,k]
                    new_s = old_s + delta * U[i*m+j, col] * V[j*p+k, col]
                    energy -= old_s ** 2 - 2*old_s + 1
                    energy += new_s ** 2 - 2*new_s + 1
                    S[i,j,k] = new_s
            
            new_energy = energy
            
            # Accept or reject
            if new_energy < old_energy or np.random.random() < np.exp(-(new_energy - old_energy) / temp):
                # Accept
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
            else:
                # Reject: revert
                mat[row, col] = old_val
                if mat_choice == 0:
                    i = row // m
                    j = row % m
                    for k in range(p):
                        old_s = S[i,j,k]
                        new_s = old_s - delta * V[j*p+k, col] * W[k*n+i, col]
                        energy -= old_s ** 2 - 2*old_s + 1
                        energy += new_s ** 2 - 2*new_s + 1
                        S[i,j,k] = new_s
                elif mat_choice == 1:
                    j = row // p
                    k = row % p
                    for i in range(n):
                        old_s = S[i,j,k]
                        new_s = old_s - delta * U[i*m+j, col] * W[k*n+i, col]
                        energy -= old_s ** 2 - 2*old_s + 1
                        energy += new_s ** 2 - 2*new_s + 1
                        S[i,j,k] = new_s
                else:
                    k = row // n
                    i = row % n
                    for j in range(m):
                        old_s = S[i,j,k]
                        new_s = old_s - delta * U[i*m+j, col] * V[j*p+k, col]
                        energy -= old_s ** 2 - 2*old_s + 1
                        energy += new_s ** 2 - 2*new_s + 1
                        S[i,j,k] = new_s
        
        # Check if we found a valid solution
        if best_energy < 1e-6:
            return best_U, best_V, best_W, best_energy
        return None
    
    # Try different ranks
    # Start from a reasonable lower bound
    # For 4x5x6, known results suggest rank around 60-80 might be achievable
    # Let's try ranks from 40 to 119
    
    ranks_to_try = list(range(40, 120))
    # Shuffle to try different ranks
    random.shuffle(ranks_to_try)
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(restart * 1000 + R)
            result = fast_simulated_annealing(R, max_iters=3000)
            if result is not None:
                U_sol, V_sol, W_sol, energy = result
                # Convert from scaled integers to actual values (divide by 2)
                U_sol = U_sol / 2.0
                V_sol = V_sol / 2.0
                W_sol = W_sol / 2.0
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U_sol[i*m+j, r] * V_sol[j*p+k, r] * W_sol[k*n+i, r]
                            if abs(s - 1.0) > 1e-4:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = R
                    best_U = U_sol
                    best_V = V_sol
                    best_W = W_sol
                    break
        
        if best_rank < schoolbook_rank:
            break
    
    # Return as lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
