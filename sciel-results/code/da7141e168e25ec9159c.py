import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = []
    best_size = 0
    
    candidates = list(range(1, 257))
    
    def try_add(sums_by_k, x, current_size):
        """
        Returns (is_valid, new_sums_by_k)
        sums_by_k: list of ints (bitsets)
        x: int
        current_size: int
        """
        # Calculate total union of all existing sums
        total_union = 0
        for s in sums_by_k:
            total_union |= s
            
        # Check for collisions
        # New sums for size k are (sums_by_k[k-1] << x)
        # These must not intersect with total_union
        for k in range(1, current_size + 2):
            added = sums_by_k[k-1] << x
            if added & total_union:
                return False, None
                
        # If valid, construct new sums_by_k
        new_sums_by_k = [0] * (current_size + 2)
        new_sums_by_k[0] = 1
        for k in range(1, current_size + 2):
            new_sums_by_k[k] = sums_by_k[k] | (sums_by_k[k-1] << x)
            
        return True, new_sums_by_k

    # Randomized Greedy
    while time.time() < deadline:
        random.shuffle(candidates)
        current_set = []
        sums_by_k = [1]  # Bit 0 set for sum 0
        
        for x in candidates:
            valid, new_sums_by_k = try_add(sums_by_k, x, len(current_set))
            if valid:
                current_set.append(x)
                sums_by_k = new_sums_by_k
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            if best_size == 60:
                break

    return [int(x) for x in best_set]
