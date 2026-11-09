"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 256
    MASK_SIZE = N + 1  # bits 1..N

    # Precompute 4-AP tuples
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))

    # Precompute 3-AP tuples
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))

    # For each element x, precompute list of 4-AP indices that include x
    elem_to_4aps = {x: [] for x in range(1, N + 1)}
    for i, (a, b, c, d) in enumerate(four_ap_tuples):
        for x in (a, b, c, d):
            elem_to_4aps[x].append(i)

    # For each element x, precompute list of 3-AP indices that include x
    # and the position (0,1,2) so we can check which other two elements are needed
    elem_to_3aps = {x: [] for x in range(1, N + 1)}
    for i, (a, b, c) in enumerate(three_ap_tuples):
        elem_to_3aps[a].append((i, 0))
        elem_to_3aps[b].append((i, 1))
        elem_to_3aps[c].append((i, 2))

    # Use a list as a boolean array for membership (faster than set for this size)
    # We'll use a Python int as a bitmask for O(1) membership via bit operations
    
    def make_mask():
        return 0  # bit x is set if x is in the set
    
    def in_mask(mask, x):
        return (mask >> x) & 1
    
    def set_bit(mask, x):
        return mask | (1 << x)
    
    def clear_bit(mask, x):
        return mask & ~(1 << x)

    def count_3aps_from_mask(mask):
        """Count distinct 3-APs in mask."""
        count = 0
        for (a, b, c) in three_ap_tuples:
            if in_mask(mask, a) and in_mask(mask, b) and in_mask(mask, c):
                count += 1
        return count

    def has_4ap_from_mask(mask):
        """Check if mask contains any 4-AP."""
        for (a, b, c, d) in four_ap_tuples:
            if in_mask(mask, a) and in_mask(mask, b) and in_mask(mask, c) and in_mask(mask, d):
                return True
        return False

    # Greedy construction using bitmask
    mask = 0
    elements = list(range(1, N + 1))
    random.shuffle(elements)  # Randomize order for diversity

    while elements and time.time() < deadline - 20.0:
        best_elem = None
        best_marginal = -1
        
        for x in elements:
            new_mask = set_bit(mask, x)
            
            # Quick check: does adding x create a 4-AP?
            creates_4ap = False
            for ap_idx in elem_to_4aps[x]:
                a, b, c, d = four_ap_tuples[ap_idx]
                # Check if the other three elements are already in mask
                others = [y for y in (a, b, c, d) if y != x]
                if all(in_mask(mask, y) for y in others):
                    creates_4ap = True
                    break
            
            if creates_4ap:
                continue
            
            # Count marginal 3-AP gain
            marginal = 0
            for (ap_idx, pos) in elem_to_3aps[x]:
                a, b, c = three_ap_tuples[ap_idx]
                others = [y for y in (a, b, c) if y != x]
                if all(in_mask(mask, y) for y in others):
                    marginal += 1
            
            if marginal > best_marginal:
                best_marginal = marginal
                best_elem = x
        
        if best_elem is None:
            break
        
        mask = set_bit(mask, best_elem)
        elements.remove(best_elem)

    # Convert mask to set for SA
    current_set = set(x for x in range(1, N + 1) if in_mask(mask, x))
    current_score = count_3aps_from_mask(mask)
    best_solution = set(current_set)
    best_score = current_score

    # Simulated annealing refinement
    if time.time() < deadline - 5.0:
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        while time.time() < deadline - 5.0 and T > T_min:
            # Choose move type
            r = random.random()
            
            if r < 0.3 and len(current_set) > 0:
                # Remove a random element
                elem_to_remove = random.choice(list(current_set))
                new_set = current_set - {elem_to_remove}
                new_mask = clear_bit(mask, elem_to_remove)
                new_score = count_3aps_from_mask(new_mask)
                
                if new_score >= current_score or random.random() < math.exp((new_score - current_score) / T):
                    current_set = new_set
                    mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = set(current_set)
                        
            elif r < 0.6:
                # Add a random element
                available = [x for x in range(1, N + 1) if x not in current_set]
                if available:
                    elem_to_add = random.choice(available)
                    new_mask = set_bit(mask, elem_to_add)
                    
                    # Check 4-AP
                    creates_4ap = False
                    for ap_idx in elem_to_4aps[elem_to_add]:
                        a, b, c, d = four_ap_tuples[ap_idx]
                        others = [y for y in (a, b, c, d) if y != elem_to_add]
                        if all(in_mask(mask, y) for y in others):
                            creates_4ap = True
                            break
                    
                    if not creates_4ap:
                        new_set = current_set | {elem_to_add}
                        new_score = count_3aps_from_mask(new_mask)
                        
                        if new_score >= current_score or random.random() < math.exp((new_score - current_score) / T):
                            current_set = new_set
                            mask = new_mask
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_solution = set(current_set)
                            
            else:
                # Swap: remove one, add another
                if len(current_set) > 0:
                    elem_to_remove = random.choice(list(current_set))
                    available = [x for x in range(1, N + 1) if x not in current_set]
                    if available:
                        elem_to_add = random.choice(available)
                        new_mask = clear_bit(mask, elem_to_remove)
                        new_mask = set_bit(new_mask, elem_to_add)
                        
                        # Check 4-AP for the added element in the new context
                        creates_4ap = False
                        for ap_idx in elem_to_4aps[elem_to_add]:
                            a, b, c, d = four_ap_tuples[ap_idx]
                            others = [y for y in (a, b, c, d) if y != elem_to_add]
                            if all(in_mask(new_mask, y) for y in others):
                                creates_4ap = True
                                break
                        
                        if not creates_4ap:
                            new_set = (current_set - {elem_to_remove}) | {elem_to_add}
                            new_score = count_3aps_from_mask(new_mask)
                            
                            if new_score >= current_score or random.random() < math.exp((new_score - current_score) / T):
                                current_set = new_set
                                mask = new_mask
                                current_score = new_score
                                if current_score > best_score:
                                    best_score = current_score
                                    best_solution = set(current_set)
            
            T *= cooling_rate

    return sorted(best_solution)
# EVOLVE-BLOCK-END
