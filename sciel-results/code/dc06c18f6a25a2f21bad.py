import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # 1. Precompute APs and Incidence Matrix
    # APs are defined by start s, step d.
    # We include all APs of length >= 2.
    # Note: Length 1 APs have sum +/- 1, so max is at least 1.
    
    ap_terms = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            # AP: s, s+d, s+2d, ... < N
            terms = list(range(s, N, d))
            if len(terms) > 1:
                ap_terms.append(terms)
                
    num_aps = len(ap_terms)
    
    # Build M (num_aps x N)
    M = np.zeros((num_aps, N), dtype=np.float32)
    # Also store the list of AP indices for each position for fast updates
    pos_to_aps = [[] for _ in range(N)]
    
    for j, terms in enumerate(ap_terms):
        for i in terms:
            M[j, i] = 1.0
            pos_to_aps[i].append(j)
            
    # Convert pos_to_aps to numpy arrays for indexing
    pos_to_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]
    
    # Compute C = M.T @ M
    # C[i, k] = number of APs containing both i and k
    # C[i, i] = number of APs containing i (K_i)
    C = M.T @ M
    K = np.diag(C).astype(np.float32)
    
    # Helper to get hard max
    def get_hard_max(sums):
        return int(np.max(np.abs(sums)))
    
    # Soft-max surrogate: smooth approximation to max|sums|
    # Used as auxiliary objective to guide search toward reducing the hard max
    def soft_max_obj(sums, beta):
        abs_sums = np.abs(sums)
        m = np.max(abs_sums)
        if m == 0:
            return 0.0
        # log-sum-exp trick for stability
        # F = (1/beta) * log(sum(exp(beta * |S|)))
        # To avoid overflow, we shift:
        # beta * |S| - beta * m
        shifted = beta * (abs_sums - m)
        sum_exp = np.sum(np.exp(shifted))
        return (m + np.log(sum_exp) / beta)

    best_signs = None
    best_hard_max = N # Worst case
    
    # Strategy:
    # 1. Greedy Initialization
    # 2. Simulated Annealing with Soft-Max Objective
    # 3. Exact Local Search Polish
    # 4. Perturb and Repeat Polish if time permits
    
    # --- Phase 1: Greedy Initialization ---
    signs = np.ones(N, dtype=np.float32)
    sums = M @ signs
    current_hard_max = get_hard_max(sums)
    
    # Iterate through positions, flip if it reduces the current max absolute sum
    # This is a simple greedy heuristic to get a decent starting point
    improved = True
    while improved:
        improved = False
        for k in range(N):
            delta = -2.0 * signs[k]
            idxs = pos_to_aps_arr[k]
            old_max = current_hard_max
            sums[idxs] += delta
            new_max = get_hard_max(sums)
            if new_max < old_max:
                signs[k] = -signs[k]
                current_hard_max = new_max
                improved = True
            else:
                sums[idxs] -= delta # Revert
                # Optional: accept if equal and random? No, keep deterministic for init.
                
    if current_hard_max < best_hard_max:
        best_hard_max = current_hard_max
        best_signs = signs.copy()
        
    # --- Phase 2: Simulated Annealing with Soft-Max ---
    # We will run SA starting from the greedy result.
    # Objective: Minimize Soft-Max of |Sums|
    
    rng = random.Random(42)
    
    # Initialize SA state
    sa_signs = signs.copy()
    sa_sums = M @ sa_signs
    sa_G = C @ sa_signs # Not strictly needed for soft-max delta, but good to have? 
    # Actually, for soft-max, we need to update the objective value.
    # Delta in soft-max is expensive to compute exactly for all APs.
    # Approximation: Only the APs containing k change.
    
    # To make SA efficient, we use a "Delta" approximation for the soft-max objective.
    # However, exact delta for log-sum-exp is tricky.
    # Alternative: Use the Sum of Squares (J) for SA dynamics (fast, exact delta)
    # but track the Hard Max and Soft Max for acceptance if possible?
    # The current program's hybrid approach was decent.
    # Let's stick to the Sum of Squares (J) for the fast delta calculation in SA,
    # as it is a standard proxy for variance/discrepancy, but we will rely heavily
    # on the Polish phase to fix the Max.
    # Wait, the prompt suggests combining with Inspiration 1 (Soft-Max).
    # Let's try to implement a fast Soft-Max delta.
    
    def calc_soft_delta(sums, idxs, delta_val, beta):
        """
        Calculate the change in soft-max objective when flipping sign at k.
        idxs: APs containing k.
        delta_val: change in sum for those APs (-2 * sign[k]).
        """
        old_abs = np.abs(sums[idxs])
        new_sums_vals = sums[idxs] + delta_val
        new_abs = np.abs(new_sums_vals)
        
        # We need to update the global sum_exp.
        # This is hard without storing the global state of exps.
        # For simplicity and speed, we will use the Sum of Squares (J) for SA
        # but use a "Hard Max" check frequently to guide the Polish phase.
        # This is a compromise: SA explores using J (fast), Polish refines using Hard Max (exact).
        
        # Let's stick to the J-based SA from the current program but optimize it.
        # The current program used: delta_J + 5.0 * delta_soft (approx).
        # The approximation of delta_soft was expensive.
        
        # New Plan:
        # 1. SA with J (Sum of Squares) as the primary objective for speed.
        # 2. Frequent Hard Max checks.
        # 3. Aggressive Polish.
        
        pass # Not used in this implementation, relying on J.

    # Re-initialize SA with Greedy Start
    sa_signs = best_signs.copy()
    sa_sums = M @ sa_signs
    sa_G = C @ sa_signs
    
    T = 10.0
    T_min = 0.1
    
    steps = 0
    best_local_hard_max = get_hard_max(sa_sums)
    best_local_signs = sa_signs.copy()
    
    # SA Loop
    while time.time() < deadline - 10: # Leave 10s for Polish
        # Pick random index to flip
        k = rng.randint(0, N - 1)
        
        # Delta for signs
        delta = -2.0 * sa_signs[k]
        
        # Change in J = sum S_P^2
        delta_J = 2.0 * delta * sa_G[k] + (delta * delta) * K[k]
        
        # Metropolis Criterion on J
        accept = False
        if delta_J < 0:
            accept = True
        else:
            if delta_J / T > 100:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta_J / T)
                
        if accept:
            # Apply move
            sa_signs[k] = -sa_signs[k]
            
            # Update sums for all APs containing k
            idxs = pos_to_aps_arr[k]
            sa_sums[idxs] += delta
            
            # Update G
            sa_G += delta * C[:, k]
            
            steps += 1
            
            # Check hard max periodically
            if steps % 20 == 0:
                current_hard_max = get_hard_max(sa_sums)
                if current_hard_max < best_local_hard_max:
                    best_local_hard_max = current_hard_max
                    best_local_signs = sa_signs.copy()
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = sa_signs.copy()
                        if best_hard_max <= 1:
                            break
                # If we are stuck at a high hard max, we might want to force exploration
                # but Metropolis on J should handle that.
                
        # Annealing
        if steps % 500 == 0:
            T *= 0.97
            if T < T_min:
                T = T_min
                
        # Early stop if we hit 1
        if best_hard_max <= 1:
            break
    
    # --- Phase 3: Polish Phase (Exact Local Search) ---
    # Start from the best found so far
    if time.time() < deadline - 1:
        polish_signs = best_local_signs.copy()
        polish_sums = M @ polish_signs
        polish_hard_max = get_hard_max(polish_sums)
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for k in range(N):
                delta = -2.0 * polish_signs[k]
                idxs = pos_to_aps_arr[k]
                
                # Try flip
                # Save old sums for affected APs to revert if needed
                old_vals = polish_sums[idxs].copy()
                polish_sums[idxs] += delta
                new_hard_max = get_hard_max(polish_sums)
                
                if new_hard_max < polish_hard_max:
                    polish_signs[k] = -polish_signs[k]
                    polish_hard_max = new_hard_max
                    improved = True
                    if polish_hard_max < best_hard_max:
                        best_hard_max = polish_hard_max
                        best_signs = polish_signs.copy()
                        if best_hard_max <= 1:
                            break
                else:
                    # Revert
                    polish_sums[idxs] = old_vals
                    
            if best_hard_max <= 1:
                break

    # --- Phase 4: Perturb and Polish (if time remains) ---
    # If we haven't reached 1, try to escape local minima
    if best_hard_max > 1 and time.time() < deadline - 2:
        # Create a perturbed version of the best solution
        perturb_signs = best_signs.copy()
        num_perturb = 5
        for _ in range(num_perturb):
            k = rng.randint(0, N - 1)
            perturb_signs[k] = -perturb_signs[k]
        
        perturb_sums = M @ perturb_signs
        perturb_hard_max = get_hard_max(perturb_sums)
        
        # Run a quick polish on the perturbed solution
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for k in range(N):
                delta = -2.0 * perturb_signs[k]
                idxs = pos_to_aps_arr[k]
                old_vals = perturb_sums[idxs].copy()
                perturb_sums[idxs] += delta
                new_hard_max = get_hard_max(perturb_sums)
                
                if new_hard_max < perturb_hard_max:
                    perturb_signs[k] = -perturb_signs[k]
                    perturb_hard_max = new_hard_max
                    improved = True
                    if perturb_hard_max < best_hard_max:
                        best_hard_max = perturb_hard_max
                        best_signs = perturb_signs.copy()
                        if best_hard_max <= 1:
                            break
                else:
                    perturb_sums[idxs] = old_vals
                if best_hard_max <= 1:
                    break
            if best_hard_max <= 1:
                break

    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
