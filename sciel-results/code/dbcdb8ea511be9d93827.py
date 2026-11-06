import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Generate all edges (i, j) with i < j
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Generate all K4s (sets of 4 vertices)
    k4s = []
    k4_edge_indices = []
    for quad in combinations(range(n), 4):
        k4s.append(quad)
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(quad[i], quad[j])])
        k4_edge_indices.append(e_indices)
        
    num_k4s = len(k4s)
    
    # Map edge to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Precompute edge positions within each K4 for faster mask updates
    # k4_edge_pos[k4_idx] is a dict: edge_idx -> position (0-5)
    k4_edge_pos = [{} for _ in range(num_k4s)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for pos, ei in enumerate(e_indices):
            k4_edge_pos[k4_idx][ei] = pos

    # Helper: Get distinct count from mask
    def mask_count(mask):
        return mask.bit_count()

    # Helper: Calculate cost (number of violated K4s)
    def get_cost(colors):
        cost = 0
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            mask = 0
            for ei in e_indices:
                mask |= (1 << colors[ei])
            if mask_count(mask) < 5:
                cost += 1
        return cost

    # Helper: Calculate delta cost for changing edge e_idx to new_color
    def delta_cost(e_idx, new_color, colors, k4_masks):
        """
        k4_masks: list of current bitmasks for each K4.
        Returns (delta, updated_masks_for_affected_k4s)
        """
        old_color = colors[e_idx]
        if old_color == new_color:
            return 0, {}
        
        delta = 0
        updates = {}
        
        for k4_idx in edge_to_k4s[e_idx]:
            old_mask = k4_masks[k4_idx]
            old_distinct = mask_count(old_mask)
            old_bad = 1 if old_distinct < 5 else 0
            
            # Calculate new mask
            # We need to remove old_color if it's not present in other edges, 
            # and add new_color.
            # Since we don't have counts, we can't just bit_op. 
            # But we can recompute the mask for this specific K4 (6 edges).
            e_indices = k4_edge_indices[k4_idx]
            new_mask = 0
            for ei in e_indices:
                if ei == e_idx:
                    new_mask |= (1 << new_color)
                else:
                    new_mask |= (1 << colors[ei])
            
            new_distinct = mask_count(new_mask)
            new_bad = 1 if new_distinct < 5 else 0
            
            delta += new_bad - old_bad
            updates[k4_idx] = new_mask
            
        return delta, updates

    # Main Search Loop
    best_solution = None
    best_num_colors = float('inf')
    
    # We want to find the minimum number of colors.
    # Theoretical min is 13.
    # Let's try to find a valid solution with 13, 14, 15 colors.
    
    # Phase 1: Greedy Initialization
    def greedy_init(num_colors):
        colors = [0] * num_edges
        # Initialize k4_masks
        k4_masks = [0] * num_k4s
        
        # Update masks for initial state (all 0)
        for k4_idx in range(num_k4s):
            # All edges are 0, so mask is 1
            k4_masks[k4_idx] = 1
            
        # Iterate through edges in random order to break symmetry? 
        # Or fixed order. Fixed order is deterministic and fast.
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            best_c = 0
            best_delta = float('inf')
            
            # Try all colors
            for c in range(num_colors):
                if c == colors[e_idx]:
                    continue
                # Calculate delta
                d, _ = delta_cost(e_idx, c, colors, k4_masks)
                if d < best_delta:
                    best_delta = d
                    best_c = c
            
            # Apply best color
            if best_c != colors[e_idx]:
                _, updates = delta_cost(e_idx, best_c, colors, k4_masks)
                colors[e_idx] = best_c
                for k4_idx, new_mask in updates.items():
                    k4_masks[k4_idx] = new_mask
        
        return colors, k4_masks

    # Phase 2: Simulated Annealing
    def simulated_annealing(initial_colors, initial_masks, num_colors, max_time):
        colors = initial_colors[:]
        k4_masks = initial_masks[:]
        
        current_cost = get_cost(colors)
        
        if current_cost == 0:
            return colors, current_cost
        
        T = 5.0
        cool_rate = 0.999
        min_T = 0.1
        no_improve = 0
        max_no_improve = 5000
        
        best_colors = colors[:]
        best_cost = current_cost
        
        while time.time() < max_time:
            if current_cost == 0:
                best_colors = colors[:]
                best_cost = 0
                break
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            new_color = random.randint(0, num_colors - 1)
            
            if new_color == colors[e_idx]:
                continue
            
            d, updates = delta_cost(e_idx, new_color, colors, k4_masks)
            
            # Acceptance criterion
            if d <= 0 or random.random() < np.exp(-d / T):
                # Apply move
                colors[e_idx] = new_color
                for k4_idx, new_mask in updates.items():
                    k4_masks[k4_idx] = new_mask
                current_cost += d
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > max_no_improve:
                    # Restart or cool down
                    T *= 0.5
                    no_improve = 0
                    # Optionally revert to best
                    if random.random() < 0.5:
                        colors = best_colors[:]
                        # Recalculate masks for best_colors
                        for k4_idx in range(num_k4s):
                            e_indices = k4_edge_indices[k4_idx]
                            mask = 0
                            for ei in e_indices:
                                mask |= (1 << colors[ei])
                            k4_masks[k4_idx] = mask
                        current_cost = best_cost
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    T *= 0.5
                    no_improve = 0
            
            T *= cool_rate
            if T < min_T:
                break
                
        return best_colors, best_cost

    # Try to find solution with 13 colors
    found_valid = False
    best_solution_13 = None
    
    if time.time() < deadline - 10:
        # Greedy init
        init_colors, init_masks = greedy_init(13)
        init_cost = get_cost(init_colors)
        
        if init_cost == 0:
            best_solution_13 = init_colors
            found_valid = True
        else:
            # SA
            remaining = deadline - time.time()
            if remaining > 5:
                sa_colors, sa_cost = simulated_annealing(init_colors, init_masks, 13, time.time() + min(remaining - 2, 60))
                if sa_cost == 0:
                    best_solution_13 = sa_colors
                    found_valid = True
    
    if found_valid and best_solution_13:
        # Minimize distinct colors used (though we constrained to 13, we might use fewer)
        # Relabel to 0..k-1
        used = set(best_solution_13)
        if len(used) < 13:
            mapping = {c: i for i, c in enumerate(sorted(used))}
            best_solution = [mapping[c] for c in best_solution_13]
            best_num_colors = len(used)
        else:
            best_solution = best_solution_13
            best_num_colors = 13
        
        return best_solution

    # If 13 failed, try 14
    if time.time() < deadline - 5:
        init_colors, init_masks = greedy_init(14)
        init_cost = get_cost(init_colors)
        
        if init_cost == 0:
            best_solution = init_colors
            # Minimize distinct
            used = set(best_solution)
            if len(used) < 14:
                mapping = {c: i for i, c in enumerate(sorted(used))}
                best_solution = [mapping[c] for c in best_solution]
            return best_solution
        else:
            remaining = deadline - time.time()
            if remaining > 5:
                sa_colors, sa_cost = simulated_annealing(init_colors, init_masks, 14, time.time() + min(remaining - 2, 40))
                if sa_cost == 0:
                    best_solution = sa_colors
                    used = set(best_solution)
                    if len(used) < 14:
                        mapping = {c: i for i, c in enumerate(sorted(used))}
                        best_solution = [mapping[c] for c in best_solution]
                    return best_solution

    # Fallback: Return seed-like 17 color solution if all else fails
    # This ensures validity (assuming seed is valid) and returns a score.
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
