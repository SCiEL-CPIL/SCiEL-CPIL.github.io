import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability: for each mask, which other masks are comparable
    # comp_list[a] = list of b where a and b are comparable (a strict subset of b or vice versa)
    # We'll use bitsets for fast delta computation
    comp_bitsets = np.zeros(num_masks, dtype=np.uint64)  # Not enough for 4096 bits
    # Use a list of Python ints as bitsets (arbitrary precision)
    comp_bits = [0] * num_masks
    for a in range(num_masks):
        bits = 0
        for b in range(num_masks):
            if a != b and ((a & b) == a or (a & b) == b):
                bits |= (1 << b)
        comp_bits[a] = bits
    
    # Precompute comparability count between each pair for fast delta
    # comp_count[a][b] = 1 if comparable, else 0
    # Use a flat array for speed
    comp_flat = np.zeros((num_masks, num_masks), dtype=np.int8)
    for a in range(num_masks):
        bits = comp_bits[a]
        # Extract set bits
        b = 0
        while bits:
            if bits & 1:
                comp_flat[a, b] = 1
            bits >>= 1
            b += 1
    
    def calc_score_from_selection(selected):
        """Calculate score given a list of selected mask indices"""
        s = 0
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                s += comp_flat[selected[i], selected[j]]
        return s
    
    # 1. Greedy Initialization
    current_selection = [0, full_mask]  # Start with empty and full sets
    used = {0, full_mask}
    
    # Add remaining m-2 sets greedily
    for _ in range(m - 2):
        best_mask = -1
        best_score = -1
        for mask in range(1, full_mask):
            if mask in used:
                continue
            score = 0
            for cm in current_selection:
                score += comp_flat[mask, cm]
            if score > best_score:
                best_score = score
                best_mask = mask
        
        if best_mask != -1:
            current_selection.append(best_mask)
            used.add(best_mask)
    
    best_score = calc_score_from_selection(current_selection)
    best_solution = list(current_selection)
    
    # 2. Simulated Annealing on the selection
    random.seed(42)
    
    T = 5.0
    cooling_rate = 0.9999
    min_T = 0.001
    
    restarts = 0
    max_restarts = 12
    
    while time.time() < deadline and restarts < max_restarts:
        current_sel = list(best_solution)
        used = set(current_sel)
        current_score = best_score
        
        T = 5.0
        no_improve_steps = 0
        max_no_improve = 15000
        
        while time.time() < deadline and T > min_T:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_sel[idx]
            
            # Generate new mask
            r = random.random()
            if r < 0.4:
                # Flip a bit
                if old_mask == 0:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask | (1 << bit)
                elif old_mask == full_mask:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask ^ (1 << bit)
                else:
                    if random.random() < 0.5:
                        bits = [i for i in range(n) if (old_mask >> i) & 1]
                        bit = random.choice(bits)
                        new_mask = old_mask ^ (1 << bit)
                    else:
                        bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                        bit = random.choice(bits)
                        new_mask = old_mask | (1 << bit)
            elif r < 0.7:
                # Swap with another selected mask
                jdx = random.randint(0, m - 1)
                if jdx == idx:
                    continue
                new_mask = current_sel[jdx]
                # Remove old_mask, add new_mask to jdx position
                # Actually, let's do a proper swap
                # For simplicity, just pick a random unused mask
                new_mask = random.randint(0, full_mask)
            else:
                # Random mask
                new_mask = random.randint(0, full_mask)
            
            if new_mask == old_mask or new_mask in used:
                continue
            
            # Calculate delta using precomputed matrix
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                delta += comp_flat[new_mask, current_sel[j]] - comp_flat[old_mask, current_sel[j]]
            
            # Acceptance
            if delta > 0 or random.random() < math.exp(delta / T):
                used.remove(old_mask)
                used.add(new_mask)
                current_sel[idx] = new_mask
                current_score += delta
                no_improve_steps = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_sel)
            else:
                no_improve_steps += 1
            
            T *= cooling_rate
            
            if no_improve_steps > max_no_improve:
                break
        
        restarts += 1
    
    # 3. Final polish: local search with 2-swap moves
    if time.time() < deadline - 5:
        current_sel = list(best_solution)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(m):
                for j in range(i + 1, m):
                    # Try swapping i and j with other masks
                    for k in range(m):
                        if k == i or k == j:
                            continue
                        # Try replacing i with some unused mask
                        # For speed, only try a few random replacements
                        for _ in range(3):
                            new_mask = random.randint(0, full_mask)
                            if new_mask in current_sel:
                                continue
                            # Calculate delta
                            delta = 0
                            for l in range(m):
                                if l == i:
                                    continue
                                delta += comp_flat[new_mask, current_sel[l]] - comp_flat[current_sel[i], current_sel[l]]
                            if delta > 0:
                                current_sel[i] = new_mask
                                current_score += delta
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_solution = list(current_sel)
                                break
    
    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
