"""No-isosceles set: find max points in [0,N)^2 with no three forming an isosceles triangle."""
import time
import random
import math
import numpy as np

N = 64
TIME_LIMIT = 280  # Safety margin below 330s wall-clock limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Precompute all squared distances between pairs of grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    num_points = N * N
    
    # For each point, precompute which pairs of other points would form an isosceles triangle with it
    # conflicts[i] = set of j such that adding point i to a set containing both endpoints of some edge in conflicts[i] creates isosceles
    # Actually, let's think differently: for a set S, a point p can be added iff for all pairs (a,b) in S, 
    # the triangle (p,a,b) is not isosceles.
    
    # Better approach: precompute for each point p, and for each pair (a,b), whether (p,a,b) is isosceles.
    # This is too large (4096 * 4096^2). Instead, use the distance-based check efficiently.
    
    # For SA, we need fast evaluation of whether a point can be added to a set.
    # Let's precompute distance matrices.
    
    pts_arr = np.array(all_points, dtype=np.int32)
    
    # Precompute squared distances from each point to all others
    # dist2[i,j] = squared distance from point i to point j
    # This is 4096x4096 = 16M entries, ~64MB as int32 - acceptable
    
    dx = pts_arr[:, 0:1] - pts_arr[1:0, 0]  # This won't work, let me do it properly
    # Actually:
    xs = pts_arr[:, 0]
    ys = pts_arr[:, 1]
    dx_mat = xs[:, None] - xs[None, :]
    dy_mat = ys[:, None] - ys[None, :]
    dist2 = (dx_mat ** 2 + dy_mat ** 2).astype(np.int32)
    del dx_mat, dy_mat
    
    def check_isosceles(p_idx, a_idx, b_idx):
        """Check if points p, a, b form an isosceles triangle."""
        d_pa = dist2[p_idx, a_idx]
        d_pb = dist2[p_idx, b_idx]
        d_ab = dist2[a_idx, b_idx]
        return d_pa == d_pb or d_pa == d_ab or d_pb == d_ab
    
    def can_add_fast(p_idx, selected_indices):
        """Check if point p_idx can be added to the set of selected indices."""
        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                if check_isosceles(p_idx, selected_indices[i], selected_indices[j]):
                    return False
        return True
    
    def count_conflicts(p_idx, selected_indices):
        """Count how many pairs in selected_indices would form isosceles with p_idx."""
        count = 0
        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                if check_isosceles(p_idx, selected_indices[i], selected_indices[j]):
                    count += 1
        return count
    
    # Simulated Annealing
    def simulated_annealing(initial_set, time_budget):
        """Run SA starting from initial_set (list of point indices)."""
        current = set(initial_set)
        current_size = len(current)
        best = set(current)
        best_size = current_size
        
        # Temperature schedule
        t_start = time.time()
        t_end = t_start + time_budget
        T_initial = 10.0
        T_final = 0.1
        cooling_rate = 0.995
        
        T = T_initial
        rng_sa = random.Random(12345)
        
        while T > T_final and time.time() < t_end:
            # Generate a move: add, remove, or swap a point
            move_type = rng_sa.random()
            
            if move_type < 0.3:
                # Add a random point
                candidates = [i for i in range(num_points) if i not in current]
                if not candidates:
                    T *= cooling_rate
                    continue
                p = rng_sa.choice(candidates)
                new_set = current | {p}
                # Cost: number of isosceles triangles created
                conflicts = count_conflicts(p, list(current))
                delta_cost = conflicts  # positive means worse
                delta_size = 1
            elif move_type < 0.6:
                # Remove a random point
                if not current:
                    T *= cooling_rate
                    continue
                p = rng_sa.choice(list(current))
                new_set = current - {p}
                # Removing a point can only reduce conflicts, so delta_cost <= 0
                # Count how many isosceles triangles involving p are removed
                other_points = list(current - {p})
                conflicts_removed = 0
                for i in range(len(other_points)):
                    for j in range(i + 1, len(other_points)):
                        if check_isosceles(p, other_points[i], other_points[j]):
                            conflicts_removed += 1
                delta_cost = -conflicts_removed
                delta_size = -1
            else:
                # Swap: remove one, add one
                if not current:
                    T *= cooling_rate
                    continue
                p_remove = rng_sa.choice(list(current))
                candidates = [i for i in range(num_points) if i not in current]
                if not candidates:
                    T *= cooling_rate
                    continue
                p_add = rng_sa.choice(candidates)
                remaining = list(current - {p_remove})
                # Conflicts introduced by p_add with pairs in remaining
                conflicts_added = 0
                for i in range(len(remaining)):
                    for j in range(i + 1, len(remaining)):
                        if check_isosceles(p_add, remaining[i], remaining[j]):
                            conflicts_added += 1
                # Conflicts removed by removing p_remove
                conflicts_removed = 0
                for i in range(len(remaining)):
                    for j in range(i + 1, len(remaining)):
                        if check_isosceles(p_remove, remaining[i], remaining[j]):
                            conflicts_removed += 1
                delta_cost = conflicts_added - conflicts_removed
                delta_size = 0
            
            # Acceptance probability based on cost change and temperature
            # We want to minimize conflicts (ideally 0) and maximize size
            # Combined energy: cost = conflicts - lambda * size, we want to minimize
            lambda_val = 5.0  # weight for size
            energy_change = delta_cost - lambda_val * delta_size
            
            if energy_change <= 0:
                # Improvement or neutral, always accept
                if move_type < 0.3:
                    current.add(p)
                elif move_type < 0.6:
                    current.discard(p)
                else:
                    current.discard(p_remove)
                    current.add(p_add)
            else:
                # Accept with probability exp(-energy_change / T)
                if rng_sa.random() < math.exp(-energy_change / T):
                    if move_type < 0.3:
                        current.add(p)
                    elif move_type < 0.6:
                        current.discard(p)
                    else:
                        current.discard(p_remove)
                        current.add(p_add)
            
            # Update best
            current_size = len(current)
            if current_size > best_size:
                best = set(current)
                best_size = current_size
            
            T *= cooling_rate
        
        return list(best), best_size
    
    # Initial solution: greedy construction
    def greedy_initial():
        """Build an initial valid set using greedy."""
        selected = []
        selected_set = set()
        candidates = list(range(num_points))
        rng.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > start_time + 30:
                break
            # Check if this point can be added
            valid = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if check_isosceles(idx, selected[i], selected[j]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.append(idx)
                selected_set.add(idx)
        
        return selected
    
    # Run multiple SA restarts
    best_pts = []
    best_len = 0
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > start_time + TIME_LIMIT - 10:
            break
        
        # Get initial solution
        initial = greedy_initial()
        if len(initial) > best_len:
            best_pts = [all_points[i] for i in initial]
            best_len = len(initial)
        
        # Run SA with time budget
        sa_time_budget = min(40, TIME_LIMIT - 10 - (time.time() - start_time))
        if sa_time_budget < 5:
            break
        
        sa_result, sa_size = simulated_annealing(initial, sa_time_budget)
        if sa_size > best_len:
            best_pts = [all_points[i] for i in sa_result]
            best_len = sa_size
    
    return [(x, y) for x, y in best_pts]
# EVOLVE-BLOCK-END
