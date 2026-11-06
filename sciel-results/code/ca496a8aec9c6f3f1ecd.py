import random
import time

def _is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A):
    n = len(A)
    best = 0
    # Iterate all subsets
    for mask in range(1, 1 << n):
        sub = [A[i] for i in range(n) if (mask >> i) & 1]
        if len(sub) <= best:
            continue
        if _is_sidon(sub):
            best = len(sub)
            if best == n:
                break
    return best

def _find_sidon_subset_of_size(A, k):
    n = len(A)
    # Iterate combinations of size k
    # Simple way: iterate masks
    for mask in range(1, 1 << n):
        if bin(mask).count('1') != k:
            continue
        sub = [A[i] for i in range(n) if (mask >> i) & 1]
        if _is_sidon(sub):
            return sub
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # 80s limit to be safe
    
    # Initialize with a random set
    random.seed(42)
    current_set = sorted(random.sample(range(129), 12))
    current_val = _max_sidon_size(current_set)
    
    best_set = current_set[:]
    best_val = current_val
    
    while time.time() < deadline:
        if current_val <= 3:
            # We found a set with max Sidon size 3.
            # This is the theoretical minimum (as size 2 is impossible for N=12).
            # We can return immediately.
            return current_set
        
        # Find a Sidon subset of size current_val
        # We want to break it.
        # If current_val > 4, we can try to break a size 4 subset, or just the max one.
        # Breaking a size 4 subset is sufficient to reduce max size if it was the only one?
        # No, there might be others.
        # So we should find ANY Sidon subset of size 4 (or current_val).
        # If we break one, the max size might stay the same if another exists.
        # So we loop.
        
        target_size = 4 # We want to ensure no size 4 exists.
        # If current_val > 4, there is a size 4 subset inside it.
        # So finding a size 4 subset is always a valid step to break.
        
        sidon_sub = _find_sidon_subset_of_size(current_set, target_size)
        
        if sidon_sub is None:
            # No size 4 Sidon subset found?
            # But current_val > 3 implies there is one.
            # Maybe _find_sidon_subset_of_size is slow or buggy?
            # Or current_val is 3? But we checked that.
            # If current_val > 3, there MUST be a size 4 subset.
            # So this should not happen.
            # Break to avoid infinite loop if something is wrong.
            break
        
        # Pick an element to replace from sidon_sub
        # Try replacing each element to see if we can find a valid y
        replaced = False
        for idx, elem_to_replace in enumerate(sidon_sub):
            # The other 3 elements
            others = [x for x in sidon_sub if x != elem_to_replace]
            
            # Find a y in [0, 128] such that y not in current_set
            # and {others, y} is NOT Sidon
            found_y = None
            for y in range(129):
                if y in current_set:
                    continue
                new_sub = others + [y]
                if not _is_sidon(new_sub):
                    found_y = y
                    break
            
            if found_y is not None:
                # Replace elem_to_replace with found_y
                # Find index in current_set
                idx_in_set = current_set.index(elem_to_replace)
                current_set[idx_in_set] = found_y
                current_set.sort()
                
                # Recalculate max Sidon size
                new_val = _max_sidon_size(current_set)
                
                # Accept the move (greedy)
                current_val = new_val
                
                if current_val < best_val:
                    best_val = current_val
                    best_set = current_set[:]
                
                replaced = True
                break # Break out of element loop, restart main loop
        
        if not replaced:
            # Could not find a y to break this specific subset?
            # Try random restart or perturbation
            # Perturb: pick a random element in current_set, replace with random
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            new_val_elem = random.choice(range(129))
            if new_val_elem not in current_set:
                current_set[idx] = new_val_elem
                current_set.sort()
                current_val = _max_sidon_size(current_set)
                if current_val < best_val:
                    best_val = current_val
                    best_set = current_set[:]
            else:
                # Swap with another element
                idx2 = random.randint(0, 11)
                current_set[idx], current_set[idx2] = current_set[idx2], current_set[idx]
                current_set.sort()
                current_val = _max_sidon_size(current_set)
                if current_val < best_val:
                    best_val = current_val
                    best_set = current_set[:]

    return best_set
