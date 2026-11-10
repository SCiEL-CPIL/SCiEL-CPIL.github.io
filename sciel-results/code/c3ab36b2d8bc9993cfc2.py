# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large square-difference-free set modulo 205.
    
    Strategy:
    1. Construct the "conflict graph" where nodes are residues 0..204 and edges
       exist if the difference is a nonzero square modulo 205.
    2. The goal is to find the Maximum Independent Set (MIS) in this graph.
    3. Use Simulated Annealing (SA) to search for large independent sets.
       - Start with a random valid set or a greedy seed.
       - Move: Add a random valid node, or swap a selected node with an unselected one 
         (if it improves size or is accepted via Boltzmann probability).
    4. Use multiple random restarts and a time budget to explore the space.
    """
    m = 205
    deadline = time.time() + 590.0  # Leave 10s buffer for 600s limit
    
    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add((x * x) % m)
    forbidden.discard(0)
    
    # 2. Precompute conflict matrix for fast lookup
    # conflict[i][j] is True if (i-j) % m is in forbidden
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if (i - j) % m in forbidden:
                conflict[i][j] = True
    
    # Helper: Check if a node can be added to a set represented by a boolean mask
    def can_add(node, mask):
        # mask[i] is True if i is in the set
        # We need to ensure no existing node in mask conflicts with 'node'
        # conflict[node, :] is a boolean array of length m
        # If any True in mask intersects with True in conflict[node, :], it's invalid.
        # Since conflict is symmetric (if d is square, -d is square? No, not always for composite m, 
        # but (a-b) in F implies (b-a) in F is NOT necessarily true for all squares mod m?
        # Wait, if x^2 = d, then (-x)^2 = d. So if d is a square, d is a square.
        # (a-b) % m = d. (b-a) % m = m-d.
        # Is m-d a square if d is a square?
        # Yes, because if x^2 = d, then (m-x)^2 = m^2 - 2mx + x^2 = x^2 = d mod m? No.
        # (m-x)^2 = m^2 - 2mx + x^2. Mod m, this is x^2.
        # So the set of squares is symmetric under negation?
        # Let S = {x^2 mod m}. If d in S, is -d in S?
        # -d = m-d.
        # If x^2 = d, then (m-x)^2 = d. This doesn't help with -d.
        # However, for any square d, there is a y such that y^2 = -d mod m?
        # Not necessarily. But the problem statement says:
        # "neither (a - b) mod 205 nor (b - a) mod 205 may be in F"
        # This implies the condition is directional or F is symmetric?
        # If F is the set of squares, is -1 a square mod 205?
        # 205 = 5 * 41.
        # -1 mod 5 = 4 = 2^2. -1 mod 41 = 40. Is 40 a square mod 41?
        # 41 = 1 mod 4, so -1 is a square mod 41.
        # Thus -1 is a square mod 205.
        # If -1 is a square, then if d is a square, -d = d * (-1) is a product of squares, hence a square.
        # So F is symmetric: d in F iff -d in F.
        # Therefore conflict[i][j] == conflict[j][i].
        
        return not np.any(conflict[node, :] & mask)

    # 3. Greedy Seed Generator
    def greedy_seed():
        # Randomize order for diversity
        order = list(range(m))
        random.shuffle(order)
        mask = np.zeros(m, dtype=bool)
        for node in order:
            if can_add(node, mask):
                mask[node] = True
        return mask

    # 4. Simulated Annealing
    def simulated_annealing(start_mask, time_budget):
        current_mask = start_mask.copy()
        current_size = np.sum(current_mask)
        best_mask = current_mask.copy()
        best_size = current_size
        
        # Temperature parameters
        T_start = 1.0
        T_end = 0.01
        cooling_rate = 0.999
        T = T_start
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            # Generate a move
            # Option A: Add a random node (if valid)
            # Option B: Remove a random node
            # Option C: Swap a selected node with an unselected one
            
            move_type = random.random()
            new_mask = current_mask.copy()
            new_size = current_size
            
            if move_type < 0.4:
                # Try to add a random unselected node
                unselected = np.where(~current_mask)[0]
                if len(unselected) > 0:
                    node = unselected[random.randint(0, len(unselected)-1)]
                    if can_add(node, current_mask):
                        new_mask[node] = True
                        new_size += 1
                    else:
                        # Skip this move if invalid? Or accept with penalty?
                        # In SA for combinatorial optimization, we usually restrict to valid states 
                        # or use penalties. Since we must return a valid set, let's stick to valid moves 
                        # or use a repair step. 
                        # Simpler: If add fails, try a different move or skip.
                        # To maintain flow, let's just skip this iteration if the primary move fails.
                        continue
            elif move_type < 0.7:
                # Try to remove a random selected node
                selected = np.where(current_mask)[0]
                if len(selected) > 0:
                    node = selected[random.randint(0, len(selected)-1)]
                    new_mask[node] = False
                    new_size -= 1
            else:
                # Swap: Remove one selected, try to add one unselected
                selected = np.where(current_mask)[0]
                unselected = np.where(~current_mask)[0]
                if len(selected) > 0 and len(unselected) > 0:
                    rem_node = selected[random.randint(0, len(selected)-1)]
                    add_node = unselected[random.randint(0, len(unselected)-1)]
                    
                    # Temporarily remove rem_node
                    temp_mask = current_mask.copy()
                    temp_mask[rem_node] = False
                    
                    if can_add(add_node, temp_mask):
                        new_mask = temp_mask
                        new_mask[add_node] = True
                        # Size stays the same
                    else:
                        continue # Invalid swap
            
            # Evaluate move
            delta = new_size - current_size
            
            if delta > 0:
                # Always accept improvement
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask.copy()
            else:
                # Accept with probability exp(delta / T)
                # delta is negative or 0
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current_mask = new_mask
                    current_size = new_size
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
        
        return best_mask

    # 5. Main Search Loop with Restarts
    best_overall_mask = np.zeros(m, dtype=bool)
    best_overall_size = 0
    
    # Initial greedy seed to ensure we start with a decent baseline
    init_mask = greedy_seed()
    best_overall_mask = init_mask
    best_overall_size = np.sum(init_mask)
    
    # Time budget for SA runs
    # We will run SA for several seconds each, with different random seeds/starting points
    # Total time available is ~590s. Let's use 580s for search.
    search_deadline = time.time() + 580.0
    
    restart_count = 0
    while time.time() < search_deadline:
        # Generate a fresh random start or perturb the best found so far
        if random.random() < 0.5:
            start_mask = greedy_seed()
        else:
            # Perturb the best solution found so far
            start_mask = best_overall_mask.copy()
            # Remove a few random elements to allow re-optimization
            selected = np.where(start_mask)[0]
            if len(selected) > 3:
                for _ in range(3):
                    if len(selected) > 0:
                        idx = random.randint(0, len(selected)-1)
                        start_mask[selected[idx]] = False
                        selected = np.where(start_mask)[0]
        
        # Determine time slice for this SA run
        # Shorter runs for more diversity, longer runs for deep search
        # Let's do runs of ~10 seconds each
        slice_time = 10.0
        if time.time() + slice_time > search_deadline:
            slice_time = max(1.0, search_deadline - time.time())
            
        result_mask = simulated_annealing(start_mask, slice_time)
        result_size = np.sum(result_mask)
        
        if result_size > best_overall_size:
            best_overall_size = result_size
            best_overall_mask = result_mask
        
        restart_count += 1
        # Check if we've hit the absolute limit
        if time.time() > search_deadline:
            break

    # Extract final set
    final_set = np.where(best_overall_mask)[0]
    
    # Final validation (safety check)
    # Ensure no forbidden pairs
    valid = True
    F_set = forbidden
    for i in range(len(final_set)):
        for j in range(i+1, len(final_set)):
            a = final_set[i]
            b = final_set[j]
            if (a - b) % m in F_set or (b - a) % m in F_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known small valid set if something went wrong (should not happen)
        return np.array([0], dtype=np.int64)

    return final_set.astype(np.int64)

# EVOLVE-BLOCK-END
