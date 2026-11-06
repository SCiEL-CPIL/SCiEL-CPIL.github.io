"""Vectorized approach for minimum maximal no-isosceles set using precomputed constraints."""
import time
import numpy as np
import random

N = 16
DEADLINE = 325.0  # safety margin before 330s kill


def _precompute_isosceles_constraints(n):
    """Precompute which pairs of points cannot coexist with any third point."""
    # Generate all grid points
    points = np.array([(x, y) for x in range(n) for y in range(n)], dtype=np.int32)
    n_points = len(points)
    
    # Compute all pairwise squared distances
    # diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    
    # For each pair (i,j), check if there exists a k such that (i,j,k) forms isosceles
    # Isosceles means: dist(i,j) == dist(i,k) OR dist(i,j) == dist(j,k) OR dist(i,k) == dist(j,k)
    
    # blocked[i,j] = True if adding both i and j to a set would create an isosceles triangle with some k
    # Actually, we need: for a set S, no three points in S form isosceles.
    # So when checking if we can add point p to set S, we need to check all pairs (a,b) in S
    # to see if (p,a,b) is isosceles.
    
    # Better approach: precompute for each point p, which pairs (a,b) would form isosceles with p.
    # Then when checking if p can be added to S, check if any pair in S is in conflict[p].
    
    # Let's precompute conflict[p] = set of pairs (a,b) where a,b != p and (p,a,b) is isosceles
    # This is O(n^3) which is 16^3 = 4096, manageable.
    
    conflict = [set() for _ in range(n_points)]
    
    # For each triple (i,j,k) with i<j<k, check if isosceles
    # If isosceles, then for point i, the pair (j,k) is conflicting
    # For point j, the pair (i,k) is conflicting  
    # For point k, the pair (i,j) is conflicting
    
    for i in range(n_points):
        for j in range(i+1, n_points):
            for k in range(j+1, n_points):
                d_ij = dist2[i, j]
                d_ik = dist2[i, k]
                d_jk = dist2[j, k]
                
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    # Isosceles triangle
                    conflict[i].add((j, k))
                    conflict[j].add((i, k))
                    conflict[k].add((i, j))
    
    return points, conflict


def _can_add_point(p_idx, chosen_indices, conflict):
    """Check if point p_idx can be added to the set of chosen indices."""
    chosen_set = set(chosen_indices)
    for a_idx in chosen_set:
        for b_idx in chosen_set:
            if a_idx >= b_idx:
                continue
            if (a_idx, b_idx) in conflict[p_idx]:
                return False
    return True


def _is_maximal(chosen_indices, all_indices, conflict):
    """Check if no other point can be added to the set."""
    chosen_set = set(chosen_indices)
    for p_idx in all_indices:
        if p_idx in chosen_set:
            continue
        if _can_add_point(p_idx, chosen_indices, conflict):
            return False
    return True


def _greedy_construct(order, all_indices, conflict):
    """Greedy construction given an order of points."""
    chosen = []
    for p_idx in order:
        if _can_add_point(p_idx, chosen, conflict):
            chosen.append(p_idx)
    return chosen


def construct():
    start_time = time.time()
    
    # Precompute constraints
    points, conflict = _precompute_isosceles_constraints(N)
    n_points = len(points)
    all_indices = list(range(n_points))
    
    # Initial greedy in row-major order
    best_set = _greedy_construct(all_indices, all_indices, conflict)
    
    # Simulated annealing to minimize set size while maintaining maximality
    rng = random.Random(42)
    current_set = best_set[:]
    current_size = len(current_set)
    best_size = current_size
    
    temperature = 1.0
    cooling_rate = 0.999
    min_temperature = 0.01
    
    while temperature > min_temperature:
        if time.time() - start_time > DEADLINE:
            break
        
        # Random move: try to remove a point or swap
        move_type = rng.random()
        
        if move_type < 0.3 and current_size > 1:
            # Try removing a random point
            remove_idx = rng.choice(current_set)
            new_set = [x for x in current_set if x != remove_idx]
            
            # Check if still maximal
            if _is_maximal(new_set, all_indices, conflict):
                new_size = len(new_set)
                delta = new_size - current_size
                if delta < 0 or rng.random() < np.exp(-delta / temperature):
                    current_set = new_set
                    current_size = new_size
                    if current_size < best_size:
                        best_size = current_size
                        best_set = current_set[:]
        elif move_type < 0.7 and current_size > 1:
            # Try swapping: remove one, add another
            remove_idx = rng.choice(current_set)
            candidates = [x for x in all_indices if x not in current_set]
            if candidates:
                add_idx = rng.choice(candidates)
                new_set = [x for x in current_set if x != remove_idx] + [add_idx]
                
                # Check if valid (no isosceles)
                valid = True
                new_set_set = set(new_set)
                for p_idx in new_set:
                    for a_idx in new_set_set:
                        for b_idx in new_set_set:
                            if a_idx >= b_idx:
                                continue
                            if (a_idx, b_idx) in conflict[p_idx]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    new_size = len(new_set)
                    delta = new_size - current_size
                    if delta < 0 or rng.random() < np.exp(-delta / temperature):
                        current_set = new_set
                        current_size = new_size
                        if current_size < best_size:
                            best_size = current_size
                            best_set = current_set[:]
        else:
            # Try adding a point (if not maximal)
            candidates = [x for x in all_indices if x not in current_set and _can_add_point(x, current_set, conflict)]
            if candidates:
                add_idx = rng.choice(candidates)
                new_set = current_set + [add_idx]
                new_size = len(new_set)
                delta = new_size - current_size
                if rng.random() < np.exp(-delta / temperature):
                    current_set = new_set
                    current_size = new_size
        
        temperature *= cooling_rate
    
    # Convert indices back to grid points
    best_points = [points[idx] for idx in best_set]
    
    return best_points
# EVOLVE-BLOCK-END
