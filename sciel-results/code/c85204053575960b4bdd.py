import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s safety margin
    
    n = 512
    
    # Precompute all 4-APs using bitmasks for fast conflict checking
    # For each element x, store a list of bitmasks representing the other 3 elements
    # of each 4-AP containing x
    conflict_masks = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs (a, a+d, a+2d, a+3d)
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            ap = (a, b, c, term4)
            
            # For each term in the AP, store a mask of the other 3 terms
            for i, term in enumerate(ap):
                mask = 0
                for j, other in enumerate(ap):
                    if i != j:
                        mask |= (1 << (other - 1))
                conflict_masks[term].append(mask)
    
    def can_add_mask(x, S_mask):
        """Check if x can be added to S (represented as bitmask) without forming a 4-AP."""
        for mask in conflict_masks[x]:
            # Check if all 3 other elements are in S_mask
            if (mask & S_mask) == mask:
                return False
        return True
    
    def popcount(x):
        """Count bits in x"""
        return bin(x).count('1')
    
    best_set = set()
    best_size = 0
    best_mask = 0
    
    # Multi-start local search
    while time.time() < deadline:
        # 1. Greedy Construction with bitmask
        S_mask = 0
        
        # Try multiple random orderings
        for _ in range(3):
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            
            trial_mask = 0
            for x in candidates:
                if can_add_mask(x, trial_mask):
                    trial_mask |= (1 << (x - 1))
            
            if popcount(trial_mask) > popcount(S_mask):
                S_mask = trial_mask
        
        current_size = popcount(S_mask)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            best_set = set(i for i in range(1, n + 1) if (S_mask >> (i - 1)) & 1)
        
        # 2. Local Search (Swaps) with bitmask
        # Try to improve by swapping elements
        steps_in_local_search = 0
        max_local_steps = 3000  # Limit per restart
        
        while steps_in_local_search < max_local_steps and time.time() < deadline:
            steps_in_local_search += 1
            
            # Pick a random element x not in S
            x = random.randint(1, n)
            if (S_mask >> (x - 1)) & 1:
                continue
                
            # Check if x can be added directly
            if can_add_mask(x, S_mask):
                S_mask |= (1 << (x - 1))
                current_size = popcount(S_mask)
                if current_size > best_size:
                    best_size = current_size
                    best_mask = S_mask
                    best_set = set(i for i in range(1, n + 1) if (S_mask >> (i - 1)) & 1)
                continue
                
            # If x cannot be added, find a conflicting mask
            conflicting_mask = None
            for mask in conflict_masks[x]:
                if (mask & S_mask) == mask:
                    conflicting_mask = mask
                    break
                    
            if not conflicting_mask:
                continue
                
            # Pick a random element y from the conflicting AP to remove
            # Extract the 3 bits from conflicting_mask
            y_candidates = [i for i in range(1, n + 1) if (conflicting_mask >> (i - 1)) & 1]
            if not y_candidates:
                continue
            y = random.choice(y_candidates)
            
            # Try to swap: remove y, add x
            S_mask_without_y = S_mask & ~(1 << (y - 1))
            
            if can_add_mask(x, S_mask_without_y):
                S_mask = S_mask_without_y | (1 << (x - 1))
                # Size unchanged, but configuration changed
                # This might enable future additions
            else:
                # Swap failed, no change needed (S_mask unchanged)
                pass
        
        # Final check after local search
        current_size = popcount(S_mask)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            best_set = set(i for i in range(1, n + 1) if (S_mask >> (i - 1)) & 1)
        
        # Safety: if we've been running for too long, break
        if time.time() > deadline - 2:
            break
    
    # Convert best_mask to sorted list
    return sorted(i for i in range(1, n + 1) if (best_mask >> (i - 1)) & 1)
