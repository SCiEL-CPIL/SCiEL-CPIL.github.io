"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import random
import time
import itertools

# EVOLVE-BLOCK-START

def _sq_dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def _isosceles(a, b, c):
    d1 = _sq_dist(a, b)
    d2 = _sq_dist(a, c)
    d3 = _sq_dist(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(p, chosen):
    """Check if p can be added to chosen without forming isosceles with any pair."""
    for i in range(len(chosen)):
        for j in range(i+1, len(chosen)):
            if _isosceles(p, chosen[i], chosen[j]):
                return False
    return True

def _is_maximal(chosen, all_points):
    """Check if the set is maximal: no point outside can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True

def _greedy_build(order, all_points):
    """Build a maximal set by greedily adding points in given order."""
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    # Ensure maximality: after greedy, the set might not be maximal if we skipped points.
    # Actually greedy in a fixed order: we add if possible. The result is maximal 
    # only if no remaining point can be added. Let's verify and fix.
    # But greedy doesn't guarantee maximality in general for this problem.
    # We need to ensure maximality. Let's do a post-pass: try to add any point that can be added.
    chosen_set = set(chosen)
    for p in all_points:
        if p not in chosen_set and _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _minimize_by_deletion(chosen, all_points):
    """Repeatedly remove points if the set remains maximal after removal."""
    chosen = list(chosen)
    changed = True
    while changed:
        changed = False
        for i in range(len(chosen)):
            # Try removing chosen[i]
            reduced = chosen[:i] + chosen[i+1:]
            if _is_maximal(reduced, all_points):
                chosen = reduced
                changed = True
                break  # restart loop since list changed
    return chosen

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    best = None
    best_size = len(all_points) + 1
    start = time.time()
    deadline = start + 300  # safety margin before 330s kill
    
    random.seed(42)
    
    # First: get a quick baseline with row-major greedy + deletion
    baseline_order = all_points[:]
    base_chosen = _greedy_build(baseline_order, all_points)
    base_chosen = _minimize_by_deletion(base_chosen, all_points)
    best = base_chosen
    best_size = len(base_chosen)
    
    # Now try many random orderings
    for attempt in range(20000):
        if time.time() > deadline - 10:
            break
        
        order = all_points[:]
        random.shuffle(order)
        
        chosen = _greedy_build(order, all_points)
        chosen = _minimize_by_deletion(chosen, all_points)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen
            if best_size <= 8:
                break
        elif time.time() > deadline - 30 and attempt % 500 == 0:
            # Periodic check: if we've been going a while and not improving,
            # try a few more aggressive deletion orders
            pass
    
    # Final polish: try deletion in different orders on the best set
    if best is not None and time.time() < deadline - 20:
        # Try multiple random deletion orders
        for _ in range(50):
            if time.time() > deadline - 20:
                break
            trial = list(best)
            # Randomly shuffle the deletion check order by permuting indices
            indices = list(range(len(trial)))
            random.shuffle(indices)
            # Re-run deletion with this order
            changed = True
            while changed:
                changed = False
                for idx in indices:
                    if idx >= len(trial):
                        continue
                    reduced = trial[:idx] + trial[idx+1:]
                    if _is_maximal(reduced, all_points):
                        trial = reduced
                        indices = [i for i in indices if i != idx]
                        for k in range(len(indices)):
                            if indices[k] > idx:
                                indices[k] -= 1
                        changed = True
                        break
            if len(trial) < best_size:
                best_size = len(trial)
                best = trial
    
    return best

# EVOLVE-BLOCK-END
