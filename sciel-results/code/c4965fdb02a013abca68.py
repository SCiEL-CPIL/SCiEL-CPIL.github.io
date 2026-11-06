"""
Strategy: Simulated Annealing on the set of points.
Objective: Minimize size of S, subject to S being no-isosceles and maximal.
Approach:
1. Represent S as a set of indices.
2. Define a cost function: size(S) + heavy_penalty * (violations).
   Violations:
   - Number of isosceles triangles in S.
   - Number of points p not in S that can be added to S (non-maximality).
3. Use SA to explore the space, accepting worse cost with probability exp(-delta_cost/T).
4. Track the best valid set found.
"""
import numpy as np
import time
import random

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
xs = np.array([p[0] for p in POINTS])
ys = np.array([p[1] for p in POINTS])
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST2 = dx**2 + dy**2

def count_isosceles_triangles(S_indices):
    """
    Counts the number of isosceles triangles in the set S.
    A triangle (a,b,c) is isosceles if d(a,b)==d(a,c) or d(a,b)==d(b,c) or d(a,c)==d(b,c).
    We count each pair of equal distances from a vertex as one "violation unit".
    For a fixed vertex a, if there are k points at distance d, there are k*(k-1)/2 pairs.
    Summing over all vertices a in S counts each isosceles triangle 1 or 3 times.
    This is sufficient for a penalty term.
    """
    if len(S_indices) < 3:
        return 0
    
    S_arr = np.array(S_indices)
    # Distances from each point in S to all other points in S
    # sub_dist2[i, j] = DIST2[S[i], S[j]]
    sub_dist2 = DIST2[np.ix_(S_arr, S_arr)]
    
    # For each i, count duplicates in sub_dist2[i, :] excluding self (0)
    # We can use a loop over S, which is small
    total_violations = 0
    for i in range(len(S_arr)):
        # Get distances to other points
        dists = sub_dist2[i, :]
        # Exclude self (dists[i] is 0)
        # We only care about non-zero distances
        nonzero_dists = dists[dists > 0]
        if len(nonzero_dists) < 2:
            continue
        # Count occurrences of each distance
        # Using np.unique with return_counts is efficient for small arrays
        unique_d, counts = np.unique(nonzero_dists, return_counts=True)
        # Sum k*(k-1)/2 for each count k
        violations = np.sum(counts * (counts - 1) // 2)
        total_violations += violations
        
    return total_violations

def count_addable_points(S_indices):
    """
    Counts how many points p not in S can be added to S without creating an isosceles triangle.
    """
    S_set = set(S_indices)
    S_arr = np.array(S_indices)
    addable_count = 0
    
    # If S is empty or very small, many points are addable
    # We check each point p not in S
    for p_idx in range(NUM_POINTS):
        if p_idx in S_set:
            continue
            
        # Check if p can be added
        # Condition 1: Distances from p to all a in S must be unique.
        # Condition 2: For all a in S, d(p,a) != d(a,b) for any b in S.
        
        if len(S_arr) == 0:
            addable_count += 1
            continue
            
        dists_p_to_S = DIST2[p_idx, S_arr]
        
        # Check 1: Uniqueness
        if len(dists_p_to_S) != len(np.unique(dists_p_to_S)):
            continue # Cannot add
            
        # Check 2: No d(p,a) == d(a,b)
        # sub_dist2_S is the distance matrix of S
        sub_dist2_S = DIST2[np.ix_(S_arr, S_arr)]
        
        # For each a_i in S, check if dists_p_to_S[i] exists in sub_dist2_S[i, :]
        # dists_p_to_S[i] is d(p, S[i])
        # We need to check if any sub_dist2_S[i, j] == dists_p_to_S[i]
        # Note: sub_dist2_S[i, i] is 0. dists_p_to_S[i] > 0.
        
        # Vectorized check:
        # match[i, j] = (sub_dist2_S[i, j] == dists_p_to_S[i])
        # If any match is True, then d(p, S[i]) == d(S[i], S[j]), which forms an isosceles triangle (p, S[i], S[j])
        # with d(p, S[i]) == d(S[i], S[j]).
        
        # We can do this by expanding dists_p_to_S to column and comparing
        D_col = dists_p_to_S[:, None]
        matches = (sub_dist2_S == D_col)
        
        if np.any(matches):
            continue # Cannot add
            
        addable_count += 1
        
    return addable_count

def evaluate(S_indices):
    """
    Returns (cost, is_valid, size)
    """
    size = len(S_indices)
    
    # Quick check for validity components
    # If size is 0, it's not maximal (unless grid is empty, which it isn't)
    # If size is small, likely not maximal.
    
    iso_pen = count_isosceles_triangles(S_indices)
    add_pen = count_addable_points(S_indices)
    
    # We want to minimize size, but strictly avoid invalid states.
    # Cost function:
    # If invalid, cost is high.
    # If valid, cost is size.
    
    # To allow SA to move through invalid states, we use a weighted sum.
    # Weight for isosceles: High.
    # Weight for addable: High.
    
    w_iso = 1000
    w_add = 1000
    
    cost = size + w_iso * iso_pen + w_add * add_pen
    is_valid = (iso_pen == 0 and add_pen == 0)
    
    return cost, is_valid, size

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # Initial solution: Use a greedy construction to get a valid starting point
    # This ensures we start in a valid region, which is safer.
    
    def greedy_construct():
        S = []
        # Random order
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        for p in order:
            # Check if p can be added
            if not S:
                S.append(p)
                continue
            # Reuse logic from count_addable_points but for single point
            S_arr = np.array(S)
            dists_p_to_S = DIST2[p, S_arr]
            if len(dists_p_to_S) != len(np.unique(dists_p_to_S)):
                continue
            sub_dist2_S = DIST2[np.ix_(S_arr, S_arr)]
            D_col = dists_p_to_S[:, None]
            matches = (sub_dist2_S == D_col)
            if np.any(matches):
                continue
            S.append(p)
        return S
        
    # Get an initial valid set
    current_S = greedy_construct()
    current_cost, current_valid, current_size = evaluate(current_S)
    
    best_S = list(current_S)
    best_cost = current_cost
    best_size = current_size
    
    if not current_valid:
        # Fallback if greedy failed (unlikely)
        # Try a few more greedy restarts
        for _ in range(10):
            trial_S = greedy_construct()
            t_cost, t_valid, t_size = evaluate(trial_S)
            if t_valid and t_size < best_size:
                current_S = trial_S
                current_cost = t_cost
                best_S = list(trial_S)
                best_cost = t_cost
                best_size = t_size
                break
    
    # Simulated Annealing
    T = 50.0 # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    max_steps = 20000
    
    # We will use a list for S for fast set operations? 
    # Actually, for SA moves, we need to add/remove/swap.
    # Using a set is good for membership.
    # But for evaluation, we convert to array.
    
    current_S_set = set(current_S)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random point not in S
            candidates = [p for p in range(NUM_POINTS) if p not in current_S_set]
            if not candidates:
                continue
            p_add = random.choice(candidates)
            new_S_set = set(current_S_set)
            new_S_set.add(p_add)
            move_desc = f"add {p_add}"
            
        elif move_type < 0.6:
            # Remove a random point from S
            if not current_S_set:
                continue
            p_rem = random.choice(list(current_S_set))
            new_S_set = set(current_S_set)
            new_S_set.remove(p_rem)
            move_desc = f"rem {p_rem}"
            
        else:
            # Swap: Remove one, add one
            if not current_S_set:
                continue
            p_rem = random.choice(list(current_S_set))
            candidates = [p for p in range(NUM_POINTS) if p not in current_S_set]
            if not candidates:
                continue
            p_add = random.choice(candidates)
            new_S_set = set(current_S_set)
            new_S_set.remove(p_rem)
            new_S_set.add(p_add)
            move_desc = f"swap {p_rem}->{p_add}"
            
        # Evaluate new state
        new_S_list = list(new_S_set)
        new_cost, new_valid, new_size = evaluate(new_S_list)
        
        # Metropolis Criterion
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0:
            # Accept
            current_S_set = new_S_set
            current_cost = new_cost
            current_size = new_size
            current_valid = new_valid
            
            if new_valid and new_size < best_size:
                best_S = list(new_S_set)
                best_size = new_size
                best_cost = new_cost
        else:
            # Accept with probability
            if T > T_min:
                prob = np.exp(-delta_cost / T)
                if random.random() < prob:
                    current_S_set = new_S_set
                    current_cost = new_cost
                    current_size = new_size
                    current_valid = new_valid
                    # Don't update best here, only if valid
                    if new_valid and new_size < best_size:
                        best_S = list(new_S_set)
                        best_size = new_size
                        best_cost = new_cost
        
        # Cool down
        T *= cooling_rate
        
        # Optional: If we found a very small set, we can restart SA from there
        # or just continue.
        
    # Final check: Ensure best_S is valid
    # If best_S is not valid (should not happen if we only update on valid), 
    # we need to return a valid set.
    # The greedy initial set was valid.
    
    # Verify best_S one last time
    b_cost, b_valid, b_size = evaluate(best_S)
    if not b_valid:
        # Fallback to current if current is valid and smaller?
        # Or just return the greedy one.
        # Actually, best_S is only updated when valid.
        # So it should be valid.
        pass
        
    return [POINTS[i] for i in best_S]

# EVOLVE-BLOCK-END
