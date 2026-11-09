import time
import random
import numpy as np

N = 512
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Precompute all 4-term arithmetic progressions in [1, 512]
    # Each AP is a 4-tuple (a, a+d, a+2d, a+3d) with d > 0
    # We'll store them as lists of 4-tuples
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # For each element x (1-indexed, so index x-1), precompute which APs contain it
    # And for each AP, store its 4 elements (0-indexed)
    ap_elements = []
    elem_to_aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(aps):
        ap_elements.append(tuple(x - 1 for x in ap))  # 0-indexed
        for x in ap:
            elem_to_aps[x].append(i)
    
    # Bitmask representation: bit i is set if element i+1 is in the set
    # We'll use Python integers for bitmasks (arbitrary precision)
    
    def count_violations(mask):
        """Count how many 4-APs are fully contained in the set represented by mask"""
        count = 0
        for ap_idx in range(num_aps):
            e0, e1, e2, e3 = ap_elements[ap_idx]
            if (mask >> e0) & 1 and (mask >> e1) & 1 and (mask >> e2) & 1 and (mask >> e3) & 1:
                count += 1
        return count
    
    def is_valid(mask):
        """Check if the set is 4-AP-free"""
        return count_violations(mask) == 0
    
    def maximal_fill(mask):
        """Try to add more elements to the set without creating new violations (greedy)"""
        # Only consider adding elements that don't create new 4-APs
        # We'll iterate through elements in random order
        elements = list(range(N))
        random.shuffle(elements)
        
        for elem in elements:
            if (mask >> elem) & 1:
                continue  # Already in set
            
            new_mask = mask | (1 << elem)
            # Check if adding this element creates any new 4-AP
            # Only need to check APs containing elem
            valid = True
            for ap_idx in elem_to_aps[elem + 1]:
                e0, e1, e2, e3 = ap_elements[ap_idx]
                # Check if all other 3 elements are in the new set
                others = [e0, e1, e2, e3]
                others.remove(elem)
                all_in = True
                for o in others:
                    if not (new_mask >> o) & 1:
                        all_in = False
                        break
                if all_in:
                    valid = False
                    break
            
            if valid:
                mask = new_mask
        
        return mask
    
    def walksat(mask, steps):
        """WalkSAT-style local search to minimize violations"""
        current_mask = mask
        current_violations = count_violations(current_mask)
        best_mask = current_mask
        best_violations = current_violations
        
        for step in range(steps):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            if current_violations == 0:
                break
            
            # Choose a random violated AP
            violated_aps = []
            for ap_idx in range(num_aps):
                e0, e1, e2, e3 = ap_elements[ap_idx]
                if ((current_mask >> e0) & 1 and (current_mask >> e1) & 1 and 
                    (current_mask >> e2) & 1 and (current_mask >> e3) & 1):
                    violated_aps.append(ap_idx)
            
            if not violated_aps:
                break
            
            ap_idx = random.choice(violated_aps)
            e0, e1, e2, e3 = ap_elements[ap_idx]
            
            # Choose a random element from this AP to flip
            # Weighted towards removing elements that are in many violated APs
            candidates = [e0, e1, e2, e3]
            elem = random.choice(candidates)
            
            # Flip the element
            if (current_mask >> elem) & 1:
                new_mask = current_mask & ~(1 << elem)  # Remove
            else:
                new_mask = current_mask | (1 << elem)  # Add
            
            new_violations = count_violations(new_mask)
            
            # Accept if it improves or with some probability (simulated annealing-like)
            if new_violations <= current_violations or random.random() < 0.1:
                current_mask = new_mask
                current_violations = new_violations
                
                if current_violations < best_violations:
                    best_mask = current_mask
                    best_violations = current_violations
        
        return best_mask, best_violations
    
    def get_size(mask):
        """Count bits in mask"""
        return bin(mask).count('1')
    
    # Strategy:
    # 1. Start with a good initial set (base-3 method)
    # 2. Use WalkSAT to minimize violations
    # 3. Apply maximal fill
    # 4. Repeat with different random seeds
    
    best_mask = 0
    best_size = 0
    
    # Initial set: base-3 method (numbers with only 0s and 1s in base 3)
    base3_mask = 0
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid:
            base3_mask |= (1 << (i - 1))
    
    # Try multiple approaches
    for restart in range(30):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        if restart == 0:
            # Start with base-3 set
            current_mask = base3_mask
        else:
            # Random restart: start with a random subset of size ~100-150
            k = random.randint(80, 150)
            elements = random.sample(range(N), k)
            current_mask = 0
            for elem in elements:
                current_mask |= (1 << elem)
        
        # Apply maximal fill first
        current_mask = maximal_fill(current_mask)
        
        # Then WalkSAT
        steps = max(100, min(5000, int((TIME_LIMIT - 10 - (time.time() - start_time)) * 50)))
        current_mask, violations = walksat(current_mask, steps)
        
        # Final maximal fill
        current_mask = maximal_fill(current_mask)
        
        size = get_size(current_mask)
        if size > best_size:
            best_size = size
            best_mask = current_mask
    
    # Convert mask to sorted list
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    return sorted(result)
