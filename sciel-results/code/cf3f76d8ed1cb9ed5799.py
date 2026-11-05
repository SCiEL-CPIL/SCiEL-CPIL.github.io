import time
import random
import math
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 295  # Safety margin for 330s kill
    
    # Precompute squared distances between all pairs of points in [0, N)^2
    # Points are indexed by p = x * N + y
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # dist_matrix[i, j] = squared distance between point i and point j
    # Shape: (N*N, N*N)
    # Memory: 4096 * 4096 * 4 bytes = 64 MB. Acceptable.
    dx = all_points[:, 0].reshape(-1, 1) - all_points[:, 0].reshape(1, -1)
    dy = all_points[:, 1].reshape(-1, 1) - all_points[:, 1].reshape(1, -1)
    dist_matrix = (dx * dx + dy * dy).astype(np.int32)
    
    best_set = []
    best_size = 0
    
    # Helper to check if a candidate point can be added to a current set
    # current_set: list of point indices
    # dist_sets: list of sets of squared distances from each point in current_set to others
    # Returns: (bool, list_of_new_dist_sets) if valid, else (False, None)
    def can_add(candidate_idx, current_set, dist_sets):
        if not current_set:
            return True, [set()]
        
        # 1. Distances from candidate to all existing points must be distinct
        dists_from_cand = dist_matrix[candidate_idx, current_set]
        
        # Check for duplicates in dists_from_cand
        if len(set(dists_from_cand)) != len(dists_from_cand):
            return False, None
        
        # 2. For each existing point q, distance from q to candidate must not be in dist_sets[q]
        for i, q_idx in enumerate(current_set):
            d_sq = dists_from_cand[i]
            if d_sq in dist_sets[i]:
                return False, None
        
        # If we reach here, it's valid. Construct the new dist_sets
        new_dist_sets = [set(s) for s in dist_sets]
        
        # Update existing sets
        for i, q_idx in enumerate(current_set):
            new_dist_sets[i].add(dists_from_cand[i])
        
        # Create set for candidate
        cand_set = set(dists_from_cand)
        new_dist_sets.append(cand_set)
        
        return True, new_dist_sets

    # Strategy:
    # 1. Deterministic structured initialization (modular lattice)
    # 2. Greedy fill
    # 3. Adaptive local search (Add vs Swap) with dynamic action selection
    
    # Modular lattice initialization:
    # Points (x, y) such that (x * a + y * b) % m == 0
    # This creates a structured set of points.
    # We try a few different modular parameters.
    
    modular_params = [
        (1, 1, 2), (1, 1, 3), (1, 2, 5), (2, 1, 5), (1, 3, 7), (3, 1, 7),
        (1, 1, 4), (1, 2, 4), (2, 3, 7), (3, 2, 7), (1, 4, 9), (4, 1, 9)
    ]
    
    for a, b, m in modular_params:
        if time.time() > deadline:
            break
        
        # Generate initial set based on modular constraint
        current_set = []
        for x in range(N):
            for y in range(N):
                if (x * a + y * b) % m == 0:
                    current_set.append(x * N + y)
        
        # Build dist_sets for this initial set
        if not current_set:
            continue
            
        dist_sets = []
        for i in range(len(current_set)):
            s = set()
            for j in range(len(current_set)):
                if i != j:
                    s.add(dist_matrix[current_set[i], current_set[j]])
            dist_sets.append(s)
        
        # Greedy phase: try to add more points
        # Shuffle all points and try to add
        indices = np.arange(N * N)
        np.random.shuffle(indices)
        
        # Track which points are already in the set for quick check
        in_set = set(current_set)
        
        for idx in indices:
            if time.time() > deadline:
                break
            if int(idx) in in_set:
                continue
            valid, new_ds = can_add(int(idx), current_set, dist_sets)
            if valid:
                current_set.append(int(idx))
                in_set.add(int(idx))
                dist_sets = new_ds
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        # Adaptive Local Search Phase
        # Dynamic action selection based on recent success/failure
        # Actions: 'add' (try to add a point), 'swap' (remove one, add one)
        # We track success rates to balance exploration.
        
        add_success_count = 0
        swap_success_count = 0
        total_add_attempts = 0
        total_swap_attempts = 0
        
        # We run a fixed number of local search steps or until time runs out
        max_local_steps = 5000
        for step in range(max_local_steps):
            if time.time() > deadline:
                break
            
            # Dynamic action selection
            # If recent adds have been failing, try swaps.
            # If recent swaps have been failing, try adds.
            recent_add_rate = add_success_count / max(1, total_add_attempts)
            recent_swap_rate = swap_success_count / max(1, total_swap_attempts)
            
            # Probability of choosing 'add' vs 'swap'
            # If add_rate is high, prefer add. If swap_rate is high, prefer swap.
            p_add = 0.5
            if total_add_attempts > 10 or total_swap_attempts > 10:
                # Normalize rates
                total_rate = recent_add_rate + recent_swap_rate
                if total_rate > 0:
                    p_add = recent_add_rate / total_rate
                else:
                    p_add = 0.5
            
            action = 'add' if random.random() < p_add else 'swap'
            
            if action == 'add':
                total_add_attempts += 1
                # Pick a random candidate point not in the set
                # Sample a few candidates to increase chance of success
                found_valid = False
                for _ in range(20):  # Increased candidate sampling
                    cand = random.randint(0, N * N - 1)
                    if cand in in_set:
                        continue
                    valid, new_ds = can_add(cand, current_set, dist_sets)
                    if valid:
                        current_set.append(cand)
                        in_set.add(cand)
                        dist_sets = new_ds
                        add_success_count += 1
                        found_valid = True
                        break
                if not found_valid:
                    # Reset success count if failed? Or keep it?
                    # Keeping it provides a longer memory of failure.
                    pass
            else:
                total_swap_attempts += 1
                # Pick a random point to remove
                if len(current_set) < 2:
                    continue
                remove_idx = random.choice(current_set)
                
                # Save state for rollback
                old_set = current_set[:]
                old_dist_sets = [set(s) for s in dist_sets]
                old_in_set = set(in_set)
                
                # Remove point
                current_set.remove(remove_idx)
                in_set.remove(remove_idx)
                
                # Rebuild dist_sets for the reduced set (O(k^2))
                dist_sets = []
                for i in range(len(current_set)):
                    s = set()
                    for j in range(len(current_set)):
                        if i != j:
                            s.add(dist_matrix[current_set[i], current_set[j]])
                    dist_sets.append(s)
                
                # Try to add a new point
                found_valid = False
                for _ in range(20):
                    cand = random.randint(0, N * N - 1)
                    if cand in in_set:
                        continue
                    valid, new_ds = can_add(cand, current_set, dist_sets)
                    if valid:
                        current_set.append(cand)
                        in_set.add(cand)
                        dist_sets = new_ds
                        swap_success_count += 1
                        found_valid = True
                        break
                
                if not found_valid:
                    # Rollback
                    current_set = old_set
                    dist_sets = old_dist_sets
                    in_set = old_in_set
                
                # After swap, try to add more points (capitalize on freed space)
                for _ in range(5):
                    cand = random.randint(0, N * N - 1)
                    if cand in in_set:
                        continue
                    valid, new_ds = can_add(cand, current_set, dist_sets)
                    if valid:
                        current_set.append(cand)
                        in_set.add(cand)
                        dist_sets = new_ds
                        add_success_count += 1
                        total_add_attempts += 1
                        break
                    else:
                        total_add_attempts += 1

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]

    # Convert indices back to (x, y) coordinates
    final_points = [tuple(all_points[i]) for i in best_set]
    return final_points
# EVOLVE-BLOCK-END
