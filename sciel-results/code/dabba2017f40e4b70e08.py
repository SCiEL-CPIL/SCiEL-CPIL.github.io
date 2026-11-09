"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples for fast violation checking
    # A 4-AP is (a, a+d, a+2d, a+3d) with a>=1, a+3d<=N
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute 3-AP tuples for counting
    # A 3-AP is (a, a+d, a+2d) with a>=1, a+2d<=N
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))
    
    # Precompute bitmask representations
    # For each 4-AP, store the bitmask of its elements
    four_ap_bitmasks = []
    for (a, b, c, d_val) in four_ap_tuples:
        mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (d_val-1))
        four_ap_bitmasks.append(mask)
    
    # For each 3-AP, store the bitmask of its elements
    three_ap_bitmasks = []
    for (a, b, c) in three_ap_tuples:
        mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
        three_ap_bitmasks.append(mask)
    
    # Precompute for each element x, which 3-APs it participates in
    # and which 4-APs it participates in
    elem_to_3aps = {x: [] for x in range(1, N + 1)}
    for i, (a, b, c) in enumerate(three_ap_tuples):
        elem_to_3aps[a].append(i)
        elem_to_3aps[b].append(i)
        elem_to_3aps[c].append(i)
    
    elem_to_4aps = {x: [] for x in range(1, N + 1)}
    for i, (a, b, c, d_val) in enumerate(four_ap_tuples):
        elem_to_4aps[a].append(i)
        elem_to_4aps[b].append(i)
        elem_to_4aps[c].append(i)
        elem_to_4aps[d_val].append(i)
    
    def count_3aps_bitmask(selected_mask):
        """Count distinct 3-APs in selected_mask using bit operations."""
        count = 0
        for mask in three_ap_bitmasks:
            if (selected_mask & mask) == mask:
                count += 1
        return count
    
    def has_4ap_bitmask(selected_mask):
        """Check if selected_mask contains any 4-AP using bit operations."""
        for mask in four_ap_bitmasks:
            if (selected_mask & mask) == mask:
                return True
        return False
    
    def can_add_without_4ap_bitmask(selected_mask, x):
        """Check if adding x to selected_mask would create a 4-AP."""
        x_bit = 1 << (x - 1)
        new_mask = selected_mask | x_bit
        # Only need to check 4-APs that include x
        for i in elem_to_4aps[x]:
            mask = four_ap_bitmasks[i]
            if (new_mask & mask) == mask:
                return False
        return True
    
    # Weighted greedy construction using bitmasks
    # Score each candidate by the number of 3-APs it would complete
    selected_mask = 0
    available_elements = set(range(1, N + 1))
    
    while available_elements and time.time() < deadline - 5.0:
        best_elem = None
        best_score = -1
        best_new_3ap_count = -1
        
        for x in available_elements:
            if not can_add_without_4ap_bitmask(selected_mask, x):
                continue
            
            x_bit = 1 << (x - 1)
            new_mask = selected_mask | x_bit
            
            # Count how many new 3-APs are created by adding x
            # A 3-AP is new if it includes x and the other two elements are already selected
            new_3aps = 0
            for i in elem_to_3aps[x]:
                mask = three_ap_bitmasks[i]
                # Check if the other two elements are in selected_mask
                # Remove x from mask, check if remaining is subset of selected_mask
                remaining = mask & ~x_bit
                if (selected_mask & remaining) == remaining:
                    new_3aps += 1
            
            total_3aps = count_3aps_bitmask(new_mask)
            
            if total_3aps > best_score or (total_3aps == best_score and new_3aps > best_new_3ap_count):
                best_score = total_3aps
                best_new_3ap_count = new_3aps
                best_elem = x
        
        if best_elem is None:
            break
        
        selected_mask |= (1 << (best_elem - 1))
        available_elements.remove(best_elem)
    
    # Convert to set for local search
    current_solution = set()
    for x in range(1, N + 1):
        if (selected_mask >> (x - 1)) & 1:
            current_solution.add(x)
    
    current_score = count_3aps_bitmask(selected_mask)
    best_solution = set(current_solution)
    best_score = current_score
    best_mask = selected_mask
    
    # Simulated annealing refinement with bitmask operations
    if time.time() < deadline - 2.0:
        T = 50.0
        T_min = 0.1
        cooling_rate = 0.999
        
        while time.time() < deadline - 2.0 and T > T_min:
            if random.random() < 0.5 and len(current_solution) > 0:
                # Remove a random element
                elem_to_remove = random.choice(list(current_solution))
                elem_bit = 1 << (elem_to_remove - 1)
                new_mask = selected_mask & ~elem_bit
                new_score = count_3aps_bitmask(new_mask)
                
                if new_score >= current_score or random.random() < math.exp((new_score - current_score) / T):
                    selected_mask = new_mask
                    current_solution.remove(elem_to_remove)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = set(current_solution)
                        best_mask = selected_mask
            else:
                # Try to add a random element not in current_solution
                available = [x for x in range(1, N + 1) if x not in current_solution]
                if available:
                    elem_to_add = random.choice(available)
                    elem_bit = 1 << (elem_to_add - 1)
                    new_mask = selected_mask | elem_bit
                    if can_add_without_4ap_bitmask(selected_mask, elem_to_add):
                        new_score = count_3aps_bitmask(new_mask)
                        if new_score >= current_score or random.random() < math.exp((new_score - current_score) / T):
                            selected_mask = new_mask
                            current_solution.add(elem_to_add)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_solution = set(current_solution)
                                best_mask = selected_mask
            
            T *= cooling_rate
    
    return sorted(best_solution)
# EVOLVE-BLOCK-END
