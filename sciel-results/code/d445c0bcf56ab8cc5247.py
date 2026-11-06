import random
import time

def construct():
    n = 4096
    
    # Generate candidates based on base-4 digits 0/1
    # Max value must be <= 4096
    # 4^6 = 4096. Sum of digits 1 * (4^6 - 1) / 3 = 1365.
    # We can use 7 digits (0..6). Max value 1 * (4^7 - 1) / 3 = 3495.
    # 2^7 = 128 candidates.
    
    candidates = []
    for mask in range(1, 1 << 7): # 1 to 127 (exclude 0 to ensure positive, though +1 handles it)
        val = 0
        m = mask
        for i in range(7):
            if m & 1:
                val += (1 << (2 * i)) # 4^i
            m >>= 1
        val += 1 # Make positive
        if val <= n:
            candidates.append(val)
            
    # Remove duplicates just in case
    candidates = list(set(candidates))
    
    best_set = []
    best_size = 0
    
    # Helper to check if x can be added
    def can_add(x, current_set, existing_sums):
        new_sums = []
        # (x, x, x)
        s = 3 * x
        if s in existing_sums:
            return False, []
        new_sums.append(s)
        
        # (x, x, y)
        for y in current_set:
            s = 2 * x + y
            if s in existing_sums:
                return False, []
            new_sums.append(s)
            
        # (x, y, z)
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                y = current_set[i]
                z = current_set[j]
                s = x + y + z
                if s in existing_sums:
                    return False, []
                new_sums.append(s)
                
        return True, new_sums

    def run_greedy(order):
        current_set = []
        existing_sums = set()
        for x in order:
            ok, new_sums = can_add(x, current_set, existing_sums)
            if ok:
                current_set.append(x)
                existing_sums.update(new_sums)
        return current_set

    # Try multiple random shuffles
    start_time = time.time()
    while time.time() - start_time < 50: # Leave 10s margin
        random.shuffle(candidates)
        res = run_greedy(candidates)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    # Local improvement: try to add remaining candidates to best_set
    # This is a simple hill climbing step
    remaining = [x for x in candidates if x not in best_set]
    changed = True
    while changed:
        changed = False
        for x in remaining[:]:
            ok, new_sums = can_add(x, best_set, set()) # Need to rebuild sums? 
            # Rebuilding sums is O(k^2). k is small.
            # Let's just rebuild the sum set for best_set once
            pass
            
    # Rebuild sum set for best_set to do local search properly
    if best_set:
        existing_sums = set()
        # Rebuild
        for i in range(len(best_set)):
            for j in range(i, len(best_set)):
                for k in range(j, len(best_set)):
                    existing_sums.add(best_set[i] + best_set[j] + best_set[k])
                    
        remaining = [x for x in candidates if x not in best_set]
        improved = True
        while improved:
            improved = False
            for x in remaining[:]:
                ok, new_sums = can_add(x, best_set, existing_sums)
                if ok:
                    best_set.append(x)
                    existing_sums.update(new_sums)
                    remaining.remove(x)
                    improved = True
                    break # Restart loop to check others with new state

    return best_set
