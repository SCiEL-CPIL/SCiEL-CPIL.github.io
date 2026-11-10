import itertools
import random
import time

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist[i][j] = d
        dist[j][i] = d

def can_add(p_idx, chosen_indices, dist_sets):
    """
    Check if p_idx can be added to the set of chosen_indices.
    dist_sets is a list of sets, dist_sets[i] contains distances from chosen_indices[i] to other chosen points.
    """
    if not chosen_indices:
        return True
    
    # 1. Check distances from p to all chosen are distinct
    dists_from_p = set()
    for idx in chosen_indices:
        d = dist[p_idx][idx]
        if d in dists_from_p:
            return False
        dists_from_p.add(d)
        
    # 2. Check if adding p creates duplicate distance for any existing point
    for idx in chosen_indices:
        d = dist[idx][p_idx]
        if d in dist_sets[idx]:
            return False
            
    return True

def get_dist_sets(chosen_indices):
    """Compute dist_sets for a given set of indices."""
    dist_sets = [set() for _ in range(len(chosen_indices))]
    for i in range(len(chosen_indices)):
        pi = chosen_indices[i]
        for j in range(len(chosen_indices)):
            if i == j: continue
            pj = chosen_indices[j]
            dist_sets[i].add(dist[pi][pj])
    return dist_sets

def is_maximal(chosen_indices):
    """Check if no point can be added."""
    chosen_set = set(chosen_indices)
    dist_sets = get_dist_sets(chosen_indices)
    
    for p in range(NUM_POINTS):
        if p in chosen_set:
            continue
        if can_add(p, chosen_indices, dist_sets):
            return False
    return True

def greedy_construct(order, deadline):
    chosen = []
    # We don't maintain dist_sets dynamically in greedy for simplicity of code structure, 
    # but we can. Let's maintain it for speed.
    dist_sets = []
    
    for p in order:
        if time.time() > deadline:
            return None
        if can_add(p, chosen, dist_sets):
            # Add p
            chosen.append(p)
            # Update dist_sets
            # 1. Add distance from p to existing points to existing dist_sets
            new_set = set()
            for i, idx in enumerate(chosen[:-1]):
                d = dist[idx][p]
                dist_sets[i].add(d)
                new_set.add(d)
            dist_sets.append(new_set)
            
    return chosen

def prune(chosen, deadline):
    """Iteratively remove points if the set remains maximal."""
    chosen = list(chosen)
    changed = True
    while changed:
        changed = False
        for i in range(len(chosen)):
            if time.time() > deadline:
                return chosen
            candidate = chosen[:i] + chosen[i+1:]
            if is_maximal(candidate):
                chosen = candidate
                changed = True
                break
    return chosen

def construct():
    deadline = time.time() + 290
    best = None
    best_size = float('inf')
    
    # Initial deterministic try
    order = list(range(NUM_POINTS))
    res = greedy_construct(order, deadline)
    if res:
        res = prune(res, deadline)
        best = res
        best_size = len(res)
        
    # Randomized restarts
    rng = random.Random(42)
    trials = 0
    while time.time() < deadline:
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        res = greedy_construct(order, deadline)
        if res is None:
            break
        res = prune(res, deadline)
        if len(res) < best_size:
            best = res
            best_size = len(res)
            if best_size <= 5: # Likely lower bound
                break
        trials += 1
        
    if best is None:
        return []
        
    return [POINTS[i] for i in best]
