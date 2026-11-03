import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Precompute conflict matrix: conflict[i,j] = True if M[i,j] > 2
    conflict = (M > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    best_set = None
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Vectorized greedy select
    def greedy_select(order, conflict_mat, N):
        selected_orig = np.zeros(N, dtype=bool)
        count = 0
        for idx in order:
            if count == 0:
                selected_orig[idx] = True
                count += 1
            else:
                if not np.any(conflict_mat[idx] & selected_orig):
                    selected_orig[idx] = True
                    count += 1
        return selected_orig, count
    
    # Strategy A: Many random orderings (quick initial solutions)
    n_trials = 500
    for trial in range(n_trials):
        if time.time() > deadline - 60:
            break
        
        if trial == 0:
            order = np.arange(N)
        else:
            order = rng.permutation(N)
        
        selected_mask, count = greedy_select(order, conflict, N)
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy B: Least conflicts first
    if time.time() < deadline - 50:
        conflict_counts = np.sum(conflict, axis=1)
        order_b = np.argsort(conflict_counts, kind='stable')
        selected_mask, count = greedy_select(order_b, conflict, N)
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy C: Simulated Annealing / QUBO-inspired optimization
    # QUBO objective: maximize sum(x_i) - penalty * sum(conflict_ij * x_i * x_j)
    # We use a penalty large enough that any valid solution beats any invalid one
    penalty = N + 1  # ensures validity dominates count
    
    def compute_energy(x):
        """Compute QUBO energy: -|S| + penalty * (number of conflicts in S)"""
        # Number of selected
        n_selected = np.sum(x)
        # Number of conflict pairs in selected set
        # conflict is symmetric, so each pair counted twice; divide by 2
        n_conflicts = np.sum(conflict[x] @ x) / 2
        return -n_selected + penalty * n_conflicts
    
    def sa_optimize(initial_set, n_steps, T_start, T_end, rng):
        """Simulated annealing on the QUBO formulation"""
        x = initial_set.copy()
        current_energy = compute_energy(x)
        best_x = x.copy()
        best_energy = current_energy
        
        # Precompute for speed: conflict matrix as int8
        # For each step, we flip one or a few bits
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            
            # Temperature schedule (geometric cooling)
            if T_end > 0:
                T = T_start * (T_end / T_start) ** (step / n_steps)
            else:
                T = T_start * (1 - step / n_steps)
            
            # Choose move: flip 1-3 random bits
            n_flips = rng.integers(1, 4)
            flip_indices = rng.integers(0, N, size=n_flips)
            
            # Compute delta energy efficiently
            # For each flipped bit, check if adding/removing changes energy
            # Delta for adding bit i: -1 + penalty * (number of selected neighbors of i)
            # Delta for removing bit i: +1 - penalty * (number of selected neighbors of i)
            
            delta = 0.0
            for i in flip_indices:
                if x[i]:
                    # Removing i
                    n_neighbors = np.sum(conflict[i] & x)
                    delta += 1 - penalty * n_neighbors
                else:
                    # Adding i
                    n_neighbors = np.sum(conflict[i] & x)
                    delta += -1 + penalty * n_neighbors
            
            # Acceptance criterion
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / max(T, 1e-10))):
                for i in flip_indices:
                    x[i] = not x[i]
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
        
        return best_x, best_energy
    
    # Run SA from multiple starting points
    if best_set is not None and time.time() < deadline - 10:
        # SA from best greedy solution
        n_sa_steps = 50000
        T_start = 10.0
        T_end = 0.01
        
        # Multiple SA runs with different parameters
        sa_configs = [
            (n_sa_steps, T_start, T_end),
            (n_sa_steps, 5.0, 0.005),
            (n_sa_steps, 20.0, 0.05),
            (n_sa_steps // 2, 8.0, 0.01),
            (n_sa_steps // 2, 15.0, 0.03),
        ]
        
        for config_idx, (steps, T_s, T_e) in enumerate(sa_configs):
            if time.time() > deadline - 5:
                break
            
            # Start from best_set or a random greedy solution
            if config_idx == 0:
                start_set = best_set.copy()
            else:
                # Random greedy restart
                order = rng.permutation(N)
                start_set, _ = greedy_select(order, conflict, N)
            
            sa_set, sa_energy = sa_optimize(start_set, steps, T_s, T_e, rng)
            
            # Extract count from SA result
            sa_count = np.sum(sa_set)
            
            if sa_count > best_count:
                best_count = sa_count
                best_set = sa_set.copy()
        
        # Additional SA runs with more aggressive cooling
        if time.time() < deadline - 5:
            for extra_run in range(3):
                if time.time() > deadline - 3:
                    break
                start_set = best_set.copy()
                # Perturb the best solution
                n_perturb = rng.integers(5, 15)
                perturb_idx = rng.integers(0, N, size=n_perturb)
                for p in perturb_idx:
                    start_set[p] = not start_set[p]
                
                sa_set, sa_energy = sa_optimize(start_set, n_sa_steps, 10.0, 0.01, rng)
                sa_count = np.sum(sa_set)
                
                if sa_count > best_count:
                    best_count = sa_count
                    best_set = sa_set.copy()
    
    # Final phase: Add any compatible unselected vectors
    if best_set is not None and time.time() < deadline - 2:
        unselected = np.where(~best_set)[0]
        for idx in unselected:
            if time.time() > deadline - 1:
                break
            if not np.any(conflict[idx] & best_set):
                best_set[idx] = True
                best_count += 1
    
    if best_set is None:
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_set].astype(np.int64)
