import time
import random
import itertools
import numpy as np

N = 16
NUM = N * N
DEADLINE = time.time() + 285.0  # Safety margin before 330s kill

# Precompute points
POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
dist2 = np.zeros((NUM, NUM), dtype=np.int32)
for i in range(NUM):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM):
        x2, y2 = POINTS[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist2[i, j] = d
        dist2[j, i] = d

# Precompute isosceles conflict: conflict[i] is a bitmask of points j such that
# there exists k where {i, j, k} is isosceles. But that's not quite right.
# We need: for a set S, point p can be added iff for all pairs (a,b) in S, {p,a,b} is not isosceles.
# So p is blocked by S iff there exists a pair (a,b) in S such that {p,a,b} is isosceles.
# Let's precompute for each point p, the set of "witness pairs" - but that's complex.
# Instead, let's precompute: iso_conflict[p] = set of points q such that p and q are "in conflict"
# meaning there exists some r such that {p, q, r} is isosceles.
# Actually, the condition is: p can be added to S iff for every pair (a,b) in S, 
# {p, a, b} is NOT isosceles.
# So p is blocked iff there exist a, b in S such that {p, a, b} IS isosceles.
# This means: for each p, we need to know which pairs (a,b) would create an isosceles with p.
# Let's precompute: for each p, blocked_pairs[p] = set of (a,b) pairs where {p,a,b} is isosceles.
# But storing pairs is expensive. Instead, for each p and each a, we can compute which b's conflict.

# Better approach: For each point p, precompute a bitmask conflict_mask[p] where bit j is set
# if there exists some k such that {p, j, k} is isosceles. But that's still not quite right
# because we need pairs.

# Let me think differently. The condition "p can be added to S" means:
# For all pairs (a,b) in S: {p,a,b} is not isosceles.
# Equivalently: there is NO pair (a,b) in S such that {p,a,b} is isosceles.
# 
# Let's precompute for each p: iso_with[p] = set of all points q such that p and q can be 
# part of an isosceles triangle with some third point. But that's not sufficient either.
#
# Actually, let's just precompute a 3D structure or use a different approach.
# For speed, let's precompute: for each p, and for each a, the set of b such that {p,a,b} is isosceles.
# We can store this as: iso_b[p][a] = bitmask of b's where {p,a,b} is isosceles.

# iso_b[p][a] is a bitmask (Python int or numpy array) of points b where {p,a,b} forms isosceles
iso_b = [[0] * NUM for _ in range(NUM)]
for p in range(NUM):
    for a in range(NUM):
        if p == a:
            continue
        mask = 0
        for b in range(NUM):
            if b == p or b == a:
                continue
            d1 = dist2[p, a]
            d2 = dist2[p, b]
            d3 = dist2[a, b]
            if d1 == d2 or d1 == d3 or d2 == d3:
                mask |= (1 << b)
        iso_b[p][a] = mask

def can_add(p, chosen_mask):
    """Check if point p can be added to set represented by chosen_mask (bitmask)."""
    # p can be added iff for all a in chosen, there is no b in chosen such that {p,a,b} is isosceles
    # i.e., for all a in chosen: (iso_b[p][a] & chosen_mask) == 0
    for a in range(NUM):
        if (chosen_mask >> a) & 1:
            if iso_b[p][a] & chosen_mask:
                return False
    return True

def is_maximal(chosen_mask):
    """Check if the set is maximal."""
    for p in range(NUM):
        if not ((chosen_mask >> p) & 1):
            if can_add(p, chosen_mask):
                return False
    return True

def mask_to_list(mask):
    """Convert bitmask to list of point indices."""
    return [i for i in range(NUM) if (mask >> i) & 1]

def greedy_randomized(rng):
    """Construct a maximal no-isosceles set using randomized greedy."""
    order = list(range(NUM))
    rng.shuffle(order)
    
    chosen_mask = 0
    for p in order:
        if can_add(p, chosen_mask):
            chosen_mask |= (1 << p)
    return chosen_mask

def simulated_annealing(chosen_mask, rng, max_iters=200):
    """Use simulated annealing to minimize the set while maintaining maximality.
    
    Penalty function: size + lambda * (number of unblocked points)
    We want to minimize size, but the set must be maximal (0 unblocked points).
    
    Moves:
    1. Remove a random point from the set
    2. Add a random point not in the set
    3. Swap: remove one, add another
    """
    current_mask = chosen_mask
    current_size = bin(current_mask).count('1')
    
    # Count unblocked points
    def count_unblocked(mask):
        count = 0
        for p in range(NUM):
            if not ((mask >> p) & 1):
                if can_add(p, mask):
                    count += 1
        return count
    
    current_unblocked = count_unblocked(current_mask)
    current_penalty = current_size + 10.0 * current_unblocked
    
    best_mask = current_mask
    best_penalty = current_penalty
    best_size = current_size
    
    T = 50.0
    cooling = 0.995
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Choose a move
        move_type = rng.random()
        trial_mask = current_mask
        
        if move_type < 0.4:
            # Remove a random point
            bits = [i for i in range(NUM) if (current_mask >> i) & 1]
            if bits:
                p = rng.choice(bits)
                trial_mask = current_mask & ~(1 << p)
        elif move_type < 0.7:
            # Add a random point not in set
            not_in = [i for i in range(NUM) if not ((current_mask >> i) & 1)]
            if not_in:
                p = rng.choice(not_in)
                trial_mask = current_mask | (1 << p)
        else:
            # Swap
            bits = [i for i in range(NUM) if (current_mask >> i) & 1]
            not_in = [i for i in range(NUM) if not ((current_mask >> i) & 1)]
            if bits and not_in:
                p_remove = rng.choice(bits)
                p_add = rng.choice(not_in)
                trial_mask = (current_mask & ~(1 << p_remove)) | (1 << p_add)
        
        if trial_mask == current_mask:
            continue
        
        # Check validity: no isosceles triangles
        # Verify the set has no isosceles triangles
        valid = True
        trial_indices = [i for i in range(NUM) if (trial_mask >> i) & 1]
        for idx in range(len(trial_indices)):
            a = trial_indices[idx]
            for jdx in range(idx + 1, len(trial_indices)):
                b = trial_indices[jdx]
                # Check if adding any other point in the set creates isosceles
                # Actually we need to verify no three points in the set form isosceles
                pass
        # Faster check: for each pair in trial, verify no third point in trial creates isosceles
        for idx in range(len(trial_indices)):
            a = trial_indices[idx]
            for jdx in range(idx + 1, len(trial_indices)):
                b = trial_indices[jdx]
                # Check if there exists c in trial such that {a,b,c} is isosceles
                # Use precomputed: for pair (a,b), which c's create isosceles?
                # We can check: for each c in trial, is {a,b,c} isosceles?
                # This is O(n^3) in the worst case but n is small
                pass
        
        # Let's do a proper validity check
        valid = True
        n_trial = len(trial_indices)
        for i in range(n_trial):
            a = trial_indices[i]
            for j in range(i + 1, n_trial):
                b = trial_indices[j]
                for k in range(j + 1, n_trial):
                    c = trial_indices[k]
                    d1 = dist2[a, b]
                    d2 = dist2[a, c]
                    d3 = dist2[b, c]
                    if d1 == d2 or d1 == d3 or d2 == d3:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Invalid move, skip
            continue
        
        trial_size = bin(trial_mask).count('1')
        trial_unblocked = count_unblocked(trial_mask)
        trial_penalty = trial_size + 10.0 * trial_unblocked
        
        delta = trial_penalty - current_penalty
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_mask = trial_mask
            current_size = trial_size
            current_unblocked = trial_unblocked
            current_penalty = trial_penalty
            
            if trial_penalty < best_penalty:
                best_penalty = trial_penalty
                best_mask = trial_mask
                best_size = trial_size
        
        T *= cooling
    
    return best_mask

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_mask = None
    best_size = NUM + 1
    
    # Phase 1: Multiple greedy restarts to find good initial solutions
    for trial in range(50):
        if time.time() > DEADLINE - 60:
            break
        
        mask = greedy_randomized(rng)
        size = bin(mask).count('1')
        
        if size < best_size:
            best_size = size
            best_mask = mask
    
    # Phase 2: Simulated annealing on the best and top few candidates
    # Keep track of top candidates
    candidates = []
    for trial in range(100):
        if time.time() > DEADLINE - 30:
            break
        
        mask = greedy_randomized(rng)
        size = bin(mask).count('1')
        candidates.append((size, mask))
        
        if size < best_size:
            best_size = size
            best_mask = mask
    
    # Sort by size and take top 5
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:5]
    
    # Apply simulated annealing to top candidates
    for size, mask in top_candidates:
        if time.time() > DEADLINE - 10:
            break
        
        improved = simulated_annealing(mask, rng, max_iters=100)
        improved_size = bin(improved).count('1')
        
        if improved_size < best_size:
            best_size = improved_size
            best_mask = improved
    
    # Final check: ensure the best is actually maximal and valid
    if best_mask is None:
        # Fallback
        best_mask = greedy_randomized(rng)
    
    return [POINTS[i] for i in mask_to_list(best_mask)]
