"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import itertools

# Precompute all points
ALL_POINTS = list(itertools.product(range(N), repeat=2))
POINT_INDEX = {p: i for i, p in enumerate(ALL_POINTS)}
NUM_POINTS = len(ALL_POINTS)

# Precompute squared distances between all pairs of points
# dist2[i][j] = squared distance between ALL_POINTS[i] and ALL_POINTS[j]
# This is a 256x256 matrix.
_points_arr = np.array(ALL_POINTS)
# Compute pairwise squared distances
# diff = _points_arr[:, np.newaxis, :] - _points_arr[np.newaxis, :, :]
# dist2 = np.sum(diff**2, axis=2)
# This might use too much memory or time? 256*256*2 floats = 524k floats. Fine.
diff = _points_arr[:, np.newaxis, :] - _points_arr[np.newaxis, :, :]
DIST2 = np.sum(diff**2, axis=2)

def _can_add_index(p_idx, chosen_indices):
    """
    Check if point p_idx can be added to the set of chosen_indices without creating an isosceles triangle.
    chosen_indices: list of indices into ALL_POINTS.
    """
    if len(chosen_indices) < 2:
        return True
    
    p_idx = int(p_idx)
    # Distances from p to all chosen points
    # d_p_c[i] = DIST2[p_idx, chosen_indices[i]]
    d_p_c = DIST2[p_idx, chosen_indices]
    
    # We need to check if there exist i, j in chosen_indices such that:
    # 1. d_p_c[i] == d_p_c[j] (p is apex)
    # 2. d_p_c[i] == DIST2[chosen_indices[i], chosen_indices[j]] (chosen[i] is apex)
    # 3. d_p_c[j] == DIST2[chosen_indices[i], chosen_indices[j]] (chosen[j] is apex)
    
    k = len(chosen_indices)
    # 1. Check for duplicate distances from p
    # If any two distances are equal, it's isosceles.
    # Since k is small (~10-20), we can just check all pairs or use a set.
    seen_dists = set()
    for d in d_p_c:
        if d in seen_dists:
            return False
        seen_dists.add(d)
        
    # 2 & 3. Check against distances between chosen points
    # For each pair i, j in chosen_indices:
    # d_ij = DIST2[chosen_indices[i], chosen_indices[j]]
    # if d_p_c[i] == d_ij or d_p_c[j] == d_ij: return False
    
    for i in range(k):
        ci = chosen_indices[i]
        dpi = d_p_c[i]
        for j in range(i + 1, k):
            cj = chosen_indices[j]
            dij = DIST2[ci, cj]
            if dpi == dij:
                return False
            dpj = d_p_c[j]
            if dpj == dij:
                return False
                
    return True

def _is_maximal(chosen_indices):
    """
    Check if the set of chosen_indices is maximal.
    A set is maximal if no other point can be added.
    """
    chosen_set = set(chosen_indices)
    for p_idx in range(NUM_POINTS):
        if p_idx not in chosen_set:
            if _can_add_index(p_idx, chosen_indices):
                return False
    return True

def _greedy_construct(order_indices):
    """
    Greedy construction given an order of point indices.
    Returns a list of indices.
    """
    chosen = []
    for p_idx in order_indices:
        if _can_add_index(p_idx, chosen):
            chosen.append(p_idx)
    return chosen

def _make_maximal(chosen_indices, all_indices):
    """
    Ensure the set is maximal by adding points.
    all_indices: list of all point indices.
    """
    chosen_set = set(chosen_indices)
    # We want to add points to make it maximal.
    # To minimize size, we should add points that "block" the most others?
    # Or just add in a random order?
    # Actually, if we start from a valid set, we just need to add points until maximal.
    # The order of addition affects the final size.
    # Strategy: Add points that are "far" from existing points? Or random?
    # Let's try adding points in an order that maximizes the number of blocked points.
    # But that's expensive.
    # Simple heuristic: Shuffle the remaining points and add if possible.
    
    remaining = [p for p in all_indices if p not in chosen_set]
    random.shuffle(remaining)
    
    new_chosen = list(chosen_indices)
    new_chosen_set = set(new_chosen)
    
    for p_idx in remaining:
        if p_idx not in new_chosen_set:
            if _can_add_index(p_idx, new_chosen):
                new_chosen.append(p_idx)
                new_chosen_set.add(p_idx)
                
    return new_chosen

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    random.seed(42)
    
    best_solution_indices = None
    best_size = float('inf')
    
    # Initial greedy with random order
    # Try a few different random orders to get a good starting point
    for _ in range(5):
        if time.time() > deadline:
            break
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        current = _greedy_construct(order)
        current = _make_maximal(current, list(range(NUM_POINTS)))
        if len(current) < best_size:
            best_size = len(current)
            best_solution_indices = current[:]
            
    # Local Search
    # Moves:
    # 1. Remove a point if the set remains maximal.
    # 2. Swap a point with another if the new set is valid and maximal.
    
    if best_solution_indices is None:
        # Fallback
        best_solution_indices = _make_maximal(_greedy_construct(list(range(NUM_POINTS))), list(range(NUM_POINTS)))
        best_size = len(best_solution_indices)
        
    current = best_solution_indices[:]
    current_size = best_size
    
    # Simulated Annealing / Random Local Search
    # We don't really need temperature if we only accept improvements (size decrease) or neutral (size same, but different config might lead to improvement later).
    # But since we want MINIMUM size, we only accept if size decreases.
    # Swaps keep size same. Removals decrease size.
    # So we should accept:
    # - Removals that keep maximality (size decreases)
    # - Swaps that keep maximality (size same, but might enable future removals)
    
    # To avoid getting stuck, we can use a simple random walk with restarts or just random moves.
    # Since the space is small, random swaps might be enough.
    
    num_iterations = 20000
    for i in range(num_iterations):
        if time.time() > deadline - 10:
            break
            
        if current_size < best_size:
            best_size = current_size
            best_solution_indices = current[:]
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Try Removal
            if len(current) == 0:
                continue
            p_idx = random.choice(current)
            # Check if current without p_idx is maximal
            # We need to check if any point can be added to current \ {p_idx}
            # This is expensive.
            # Optimization: Check if p_idx was "essential".
            # If removing p_idx makes the set non-maximal, it means there exists q not in current such that q can be added to current \ {p_idx}.
            # This implies q could NOT be added to current (because p_idx was blocking it? No, p_idx was in current).
            # Actually, if current is maximal, no q can be added to current.
            # If we remove p_idx, maybe some q can be added.
            # We need to check all q not in current \ {p_idx}.
            
            new_current = [x for x in current if x != p_idx]
            if _is_maximal(new_current):
                current = new_current
                current_size -= 1
            else:
                # Not maximal, so we might try to add a point back?
                # But that increases size or keeps it same.
                # We only care about minimizing size.
                # So we reject this removal.
                pass
        else:
            # Try Swap
            if len(current) == 0:
                continue
            p_idx = random.choice(current)
            # Choose q not in current
            # To make it efficient, pick a random index from all points, if it's in current, pick another.
            # Or maintain a list of non-chosen points.
            # For now, just pick random q.
            
            # Efficient way to pick q not in current:
            # If current is small, just pick random from 0..255 and check.
            q_idx = random.randint(0, NUM_POINTS - 1)
            if q_idx in current:
                # Try a few times
                for _ in range(10):
                    q_idx = random.randint(0, NUM_POINTS - 1)
                    if q_idx not in current:
                        break
                else:
                    continue
            
            # Check if swap is valid
            # new_current = (current \ {p_idx}) U {q_idx}
            # 1. Check if q_idx can be added to current \ {p_idx}
            # 2. Check if new_current is maximal
            
            temp_current = [x for x in current if x != p_idx]
            
            if _can_add_index(q_idx, temp_current):
                # Valid set
                new_current = temp_current + [q_idx]
                # Check maximality
                if _is_maximal(new_current):
                    current = new_current
                    # Size remains same
                    # We accept this move to explore the space
                    # We don't update best_size here as it didn't improve
                    pass
            else:
                pass
                
    # Final check
    if best_solution_indices is None:
        best_solution_indices = current
        
    return [ALL_POINTS[i] for i in best_solution_indices]

# EVOLVE-BLOCK-END
