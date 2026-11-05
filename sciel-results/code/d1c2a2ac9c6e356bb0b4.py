import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def edge_vertices(idx):
    i = 0
    while i * N - i * (i + 1) // 2 + (N - i - 1) <= idx:
        i += 1
    i -= 1
    offset = idx - (i * N - i * (i + 1) // 2)
    j = i + 1 + offset
    return i, j

def construct():
    deadline = time.time() + 55  # safety margin
    
    # Precompute all 4-cycles
    # For each 4-set {a,b,c,d}, there are 3 distinct 4-cycles (cyclic orderings)
    # Cycle 1: a-b-c-d-a (edges ab, bc, cd, da)
    # Cycle 2: a-b-d-c-a (edges ab, bd, dc, ca)
    # Cycle 3: a-c-b-d-a (edges ac, cb, bd, da)
    
    cycles = []
    vertices = list(range(N))
    
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        e1 = (edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a))
        e2 = (edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a))
        e3 = (edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a))
        
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
        
    # Precompute for each edge, the list of cycle indices containing it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    # State: colors array
    # Objective: Maximize distinct colors, Minimize invalid cycles
    # Cost function: -100 * (num_distinct_colors) + (num_invalid_cycles)
    # We want to minimize cost.
    
    def get_cost(colors):
        distinct = len(set(colors))
        invalid_count = 0
        for cycle in cycles:
            ec = [colors[e] for e in cycle]
            if len(set(ec)) == 4:
                invalid_count += 1
        return -100 * distinct + invalid_count

    def is_fully_valid(colors):
        for cycle in cycles:
            ec = [colors[e] for e in cycle]
            if len(set(ec)) == 4:
                return False
        return True

    # Initialization: 
    # A good starting point is often a proper edge coloring or a structured coloring.
    # Let's try a coloring based on (i+j) % K for various K, or random.
    # Actually, a known good structure for "no rainbow 4-cycle" is often related to
    # decompositions into triangles or bipartite structures.
    # Let's start with a random valid-ish state or a simple modular one.
    
    # Strategy: Start with a random assignment of 0..119, then repair/optimize.
    # Or better: Start with a low-color state and expand.
    # Let's use a heuristic initialization: 
    # Color edge (i,j) with (i+j) % 5. This uses 5 colors.
    # Then use SA to increase colors.
    
    colors = [0] * NUM_EDGES
    for idx in range(NUM_EDGES):
        i, j = edge_vertices(idx)
        colors[idx] = (i + j) % 5
        
    # Ensure we have a valid start? (i+j)%5 might have rainbow 4-cycles.
    # If invalid, SA will fix it.
    
    best_colors = colors[:]
    best_cost = get_cost(best_colors)
    
    # Simulated Annealing
    rng = random.Random(42)
    T = 10.0  # Initial temperature
    cooling_rate = 0.9999
    min_T = 0.01
    
    # We will run multiple restarts if time permits
    # But first, a long SA run.
    
    current_cost = best_cost
    current_colors = best_colors[:]
    
    # Track distinct colors for quick check
    current_distinct = len(set(current_colors))
    
    # Precompute cycle edges as lists for faster access? 
    # cycles is a list of tuples.
    
    steps = 0
    max_steps = 2000000 # Cap steps, but time is the real limit
    
    while time.time() < deadline and T > min_T:
        # Pick a random edge
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        
        # Pick a new color
        # Bias towards colors that are already present to maintain density?
        # Or bias towards unused colors to increase diversity?
        # Let's try: 50% random 0..119, 50% one of the currently used colors.
        if rng.random() < 0.5:
            new_color = rng.randint(0, 119)
        else:
            used_colors = list(set(current_colors))
            new_color = rng.choice(used_colors)
            
        if new_color == old_color:
            continue
            
        # Calculate delta cost
        # Cost = -100 * distinct + invalid
        # Changing edge e affects all cycles in edge_to_cycles[e]
        
        delta_invalid = 0
        affected_cycles = edge_to_cycles[e]
        
        # Check each affected cycle
        # We need to know if the cycle was invalid before and after
        # And if the distinct count changes
        
        # Optimization: 
        # 1. Check if distinct count changes.
        #    If new_color is not in current_colors, distinct increases by 1 (delta_distinct = +1)
        #    If new_color is in current_colors, distinct stays same UNLESS old_color was the only one of its kind.
        #    Let's count occurrences of old_color.
        
        # To speed up, we can maintain a count array.
        # But for simplicity in this loop, let's just calculate the change.
        # However, counting occurrences every step is O(N). 
        # Let's maintain a frequency array `freq`.
        
        # We need to update freq outside the inner loop logic or pass it.
        # Let's restructure to maintain freq.
        
        # Since we are inside a tight loop, let's assume we have freq.
        # I will implement freq maintenance in the main loop.
        
        # For now, let's just use a simpler cost delta calculation if possible, 
        # or just recalculate cost if the move is accepted? No, that's slow.
        
        # Let's use the standard SA approach: 
        # 1. Apply move
        # 2. Calculate new cost
        # 3. Accept/Reject
        # 4. Revert if rejected
        
        # To make this fast, we need fast cost calculation.
        # The cost is dominated by the -100 * distinct term.
        # If we change color to a new one, distinct increases -> Cost decreases (good).
        # If we change color to an existing one, distinct might decrease if we remove the last of old_color.
        
        # Let's maintain `freq` array.
        
        # Re-initializing freq for the first time
        # Actually, let's just write the loop with freq maintenance.
        
        # Break out of this thought block and write the actual code with freq.
        pass

    # Re-implementing the loop properly with freq maintenance
    
    # Reset state for the actual run
    current_colors = [0] * NUM_EDGES
    for idx in range(NUM_EDGES):
        i, j = edge_vertices(idx)
        current_colors[idx] = (i + j) % 5
        
    freq = [0] * 120
    for c in current_colors:
        freq[c] += 1
        
    current_distinct = sum(1 for c in freq if c > 0)
    
    # Calculate initial invalid count
    initial_invalid = 0
    for cycle in cycles:
        ec = [current_colors[e] for e in cycle]
        if len(set(ec)) == 4:
            initial_invalid += 1
            
    current_cost = -100 * current_distinct + initial_invalid
    best_cost = current_cost
    best_colors = current_colors[:]
    
    T = 50.0
    cooling_rate = 0.99995
    min_T = 0.1
    
    # Precompute set of colors in each cycle? No, too memory heavy.
    
    while time.time() < deadline and T > min_T:
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        
        # Choose new color
        # Strategy: 30% random 0..119, 70% pick from a small set of "interesting" colors
        # "Interesting" = currently used, or close to used.
        if rng.random() < 0.3:
            new_color = rng.randint(0, 119)
        else:
            # Pick a random used color
            used = [c for c in range(120) if freq[c] > 0]
            if not used:
                new_color = 0
            else:
                new_color = rng.choice(used)
                
        if new_color == old_color:
            continue
            
        # Calculate change in distinct count
        old_freq = freq[old_color]
        new_freq = freq[new_color]
        
        delta_distinct = 0
        if old_freq == 1:
            delta_distinct -= 1 # Losing a color
        if new_freq == 0:
            delta_distinct += 1 # Gaining a color
            
        # Calculate change in invalid cycles
        # A cycle is invalid if it has 4 distinct colors.
        # Changing edge e in cycle C:
        # Old state: colors C_old. Invalid if len(set(C_old)) == 4.
        # New state: colors C_new (e replaced). Invalid if len(set(C_new)) == 4.
        
        delta_invalid = 0
        for ci in edge_to_cycles[e]:
            cycle = cycles[ci]
            # Get other 3 edges
            e1, e2, e3 = [x for x in cycle if x != e]
            
            # Old colors
            c1 = current_colors[e1]
            c2 = current_colors[e2]
            c3 = current_colors[e3]
            c_old = old_color
            
            # Check if old set had 4 distinct
            # set({c1, c2, c3, c_old})
            # We can check without creating a set object for speed?
            # len({c1, c2, c3, c_old}) == 4
            
            old_set = {c1, c2, c3, c_old}
            was_invalid = (len(old_set) == 4)
            
            # New colors
            c_new = new_color
            new_set = {c1, c2, c3, c_new}
            is_invalid = (len(new_set) == 4)
            
            if was_invalid and not is_invalid:
                delta_invalid -= 1
            elif not was_invalid and is_invalid:
                delta_invalid += 1
        
        delta_cost = -100 * delta_distinct + delta_invalid
        
        # Acceptance probability
        if delta_cost < 0:
            accept = True
        else:
            # exp(-delta_cost / T)
            # If T is small, this is 0.
            if T < 0.01:
                accept = False
            else:
                prob = np.exp(-delta_cost / T)
                accept = (rng.random() < prob)
                
        if accept:
            # Apply move
            current_colors[e] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            current_distinct += delta_distinct
            current_cost += delta_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = current_colors[:]
                
        else:
            # Reject (do nothing, state unchanged)
            pass
            
        T *= cooling_rate
        
    # Final check
    if is_fully_valid(best_colors):
        return best_colors
    else:
        # If SA didn't find a valid one, return a known valid fallback
        # All 0s is valid.
        # Or try to repair?
        # For safety, return the best found if it's valid, else all 0s.
        # But we want to maximize score. If best is invalid, it scores 0.
        # So we should ensure we return a valid one.
        # Let's do a quick local search to repair if invalid.
        # Or just return all 0s if invalid.
        # Given the high weight on distinct colors, SA should find valid ones easily.
        
        # Let's try a greedy repair on best_colors
        repaired = best_colors[:]
        for _ in range(1000):
            if is_fully_valid(repaired):
                break
            # Find an invalid cycle
            valid = False
            for cycle in cycles:
                ec = [repaired[e] for e in cycle]
                if len(set(ec)) == 4:
                    # Change one edge in this cycle to match another
                    # Pick random edge in cycle
                    e_to_change = rng.choice(cycle)
                    # Change to color of another edge in cycle
                    other_edges = [x for x in cycle if x != e_to_change]
                    target_color = repaired[rng.choice(other_edges)]
                    repaired[e_to_change] = target_color
                    valid = True
                    break
            if not valid:
                break
                
        if is_fully_valid(repaired):
            return repaired
        else:
            return [0] * NUM_EDGES
