"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import itertools

def _check_isosceles(p, chosen):
    """Check if adding point p to chosen set creates any isosceles triangle."""
    if len(chosen) < 2:
        return False
    # Vectorized: compute all pairwise distances from p to chosen points
    dx = chosen[:, 0] - p[0]
    dy = chosen[:, 1] - p[1]
    d2 = dx * dx + dy * dy  # squared distances from p to each chosen point
    
    # For each pair (i, j) in chosen, check if d(p,i) == d(p,j) or
    # d(p,i) == d(i,j) or d(p,j) == d(i,j)
    # This is O(k^2) where k = len(chosen)
    k = len(chosen)
    if k < 2:
        return False
    
    # d2[i] = dist(p, chosen[i])^2
    # dist(chosen[i], chosen[j])^2
    # We need to check all triples (p, chosen[i], chosen[j])
    
    # For efficiency, check if any two distances from p are equal
    # (that would make p the apex of an isosceles)
    # Or if dist(p,i) == dist(i,j) for some j (i is apex)
    # Or if dist(p,j) == dist(i,j) for some i (j is apex)
    
    # Simple O(k^2) check
    for i in range(k):
        for j in range(i + 1, k):
            dij2 = (chosen[i, 0] - chosen[j, 0])**2 + (chosen[i, 1] - chosen[j, 1])**2
            # Check three conditions for isosceles
            if d2[i] == d2[j]:  # p is apex
                return True
            if d2[i] == dij2:  # chosen[i] is apex
                return True
            if d2[j] == dij2:  # chosen[j] is apex
                return True
    return False


def _can_add(p, chosen):
    """Check if point p can be added to the set without creating an isosceles triangle."""
    return not _check_isosceles(p, chosen)


def _is_maximal(chosen, all_points):
    """Check if the set is maximal - no point can be added."""
    chosen_set = set(map(tuple, chosen))
    for p in all_points:
        if p not in chosen_set:
            if _can_add(p, chosen):
                return False
    return True


def _greedy_construct(order):
    """Greedy construction given a specific point ordering."""
    chosen = []
    for p in order:
        if _can_add(p, np.array(chosen) if chosen else np.empty((0, 2), dtype=int)):
            chosen.append(p)
    return chosen


def _simulated_annealing(initial_order, max_iters=5000, start_temp=10.0, end_temp=0.1):
    """Simulated annealing to minimize set size while maintaining maximality."""
    all_points = list(itertools.product(range(N), repeat=2))
    
    # Start with greedy construction
    current = _greedy_construct(initial_order)
    current_set = set(map(tuple, current))
    current_cost = len(current)
    
    best = list(current)
    best_cost = current_cost
    
    T = start_temp
    T_factor = (end_temp / start_temp) ** (1.0 / max_iters)
    
    for it in range(max_iters):
        T *= T_factor
        
        # Random move: try to remove a random point and re-add others
        if len(current) <= 1:
            break
        
        # Pick a random point to remove
        idx = random.randint(0, len(current) - 1)
        removed = current[idx]
        
        # Try to remove it
        new_set = current[:idx] + current[idx+1:]
        
        # Check if removing this point still keeps the set valid (no isosceles)
        # The set is still valid if no isosceles exists among remaining points
        # Since we only removed a point, the remaining set is still no-isosceles
        
        # Now check maximality: can we add back any points?
        # We want to see if the smaller set is still maximal
        # If it's maximal, great - we found a smaller set
        # If not maximal, we might be able to add some points back but not all
        
        # Actually, let's think differently: we want to find a minimal maximal set
        # So we should try to remove points and see if the result is still maximal
        # If not maximal, we need to add back some points to make it maximal again
        
        # Simpler approach: try removing a point, then greedily try to add back
        # points in a different order to get a maximal set
        
        new_chosen = list(new_set)
        new_set_tuples = set(map(tuple, new_chosen))
        
        # Try to add back points that can be added
        # Use a random order for adding back
        remaining = [p for p in all_points if p not in new_set_tuples]
        random.shuffle(remaining)
        
        for p in remaining:
            if _can_add(p, np.array(new_chosen) if new_chosen else np.empty((0, 2), dtype=int)):
                new_chosen.append(p)
        
        new_cost = len(new_chosen)
        new_set = set(map(tuple, new_chosen))
        
        # Check if this is maximal
        if new_cost < best_cost:
            best = list(new_chosen)
            best_cost = new_cost
        
        # Accept or reject based on temperature
        delta = new_cost - len(new_set)  # cost difference
        if delta <= 0 or random.random() < np.exp(-delta / T):
            current = new_chosen
        # else: keep current
        
    return best, best_cost


def construct():
    start_time = time.time()
    deadline = start_time + 300  # 300 second deadline
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    best_solution = None
    best_size = float('inf')
    
    # First, get a baseline with row-major greedy
    baseline = _greedy_construct(all_points)
    best_solution = baseline
    best_size = len(baseline)
    
    elapsed = time.time() - start_time
    if elapsed > 250:
        return best_solution
    
    # Try multiple random restarts with simulated annealing
    random.seed(42)
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Create a random ordering
        order = all_points[:]
        random.shuffle(order)
        
        # Run simulated annealing
        try:
            solution, size = _simulated_annealing(
                order,
                max_iters=3000,
                start_temp=5.0,
                end_temp=0.01
            )
            
            if size < best_size:
                best_solution = solution
                best_size = size
        except Exception:
            continue
    
    return best_solution
# EVOLVE-BLOCK-END
