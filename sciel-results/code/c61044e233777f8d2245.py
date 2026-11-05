import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    def is_valid(seq):
        """Check if all non-empty consecutive block sums are distinct."""
        if not seq:
            return True
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_increasing():
        """Standard greedy: add smallest possible next (append to end)."""
        values = []
        sums = set()
        for x in range(1, 257):
            # New sums if we append x:
            # x, x + v_k, x + v_k + v_{k-1}, ...
            tails = []
            total = x
            tails.append(total)
            # We need suffix sums of current values
            # Since values is sorted, we can compute suffix sums on the fly
            # But we need to check intersection with `sums`
            # Let's just generate the set of new sums
            
            new_sums = {x}
            current_suffix = 0
            for v in reversed(values):
                current_suffix += v
                new_sums.add(x + current_suffix)
                
            if not sums.intersection(new_sums):
                values.append(x)
                sums.update(new_sums)
        return values

    def greedy_decreasing():
        """Greedy: add largest possible next (prepend to start)."""
        values = []
        sums = set()
        for x in range(256, 0, -1):
            # New sums if we prepend x:
            # x, x + v_1, x + v_1 + v_2, ...
            new_sums = {x}
            current_prefix = 0
            for v in values: # values is sorted ascending
                current_prefix += v
                new_sums.add(x + current_prefix)
                
            if not sums.intersection(new_sums):
                # Insert x at the beginning (since x < all values in values)
                values.insert(0, x)
                sums.update(new_sums)
        return values

    # 1. Get baseline from greedy
    seq_inc = greedy_increasing()
    seq_dec = greedy_decreasing()
    
    best_seq = seq_inc if len(seq_inc) >= len(seq_dec) else seq_dec
    best_len = len(best_seq)
    current_seq = list(best_seq)
    current_set = set(current_seq)
    
    # 2. Local Search
    # We will try to improve by adding, removing, or swapping.
    # Since the sequence must be strictly increasing, the order is determined by values.
    # "Adding" means inserting a new value into the sorted list.
    # "Removing" means removing a value.
    # "Swapping" means removing one and adding another.
    
    # Pre-compute all candidates
    all_candidates = list(range(1, 257))
    
    # To speed up, we can bias towards adding elements that are "small" or "large" 
    # or just random. Random is safer.
    
    # Optimization: If we just added an element, we might try to add another immediately.
    # If we failed, we might try to remove something.
    
    while time.time() < deadline:
        # Randomly decide move type
        # Bias towards adding to maximize length
        r = random.random()
        
        if r < 0.5: # Add
            # Pick a random candidate not in current_seq
            available = [x for x in all_candidates if x not in current_set]
            if available:
                x = random.choice(available)
                # Insert x into current_seq to maintain sorted order
                new_seq = list(current_seq)
                # Binary search or simple insert
                # Since N is small, simple insert is fine
                # Find insertion point
                lo, hi = 0, len(new_seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if new_seq[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                new_seq.insert(lo, x)
                
                if is_valid(new_seq):
                    current_seq = new_seq
                    current_set.add(x)
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = list(current_seq)
        
        elif r < 0.7: # Remove
            if current_seq:
                x = random.choice(current_seq)
                current_seq.remove(x)
                current_set.remove(x)
                # Note: We don't update best on remove, but we stay in this state
                # to potentially add a different element next.
                
        else: # Swap
            if current_seq:
                current_set_copy = set(current_set) # Just in case, though we modify carefully
                available = [x for x in all_candidates if x not in current_set]
                if available:
                    x_rem = random.choice(current_seq)
                    x_add = random.choice(available)
                    
                    # Perform swap
                    new_seq = [x for x in current_seq if x != x_rem]
                    # Insert x_add
                    lo, hi = 0, len(new_seq)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if new_seq[mid] < x_add:
                            lo = mid + 1
                        else:
                            hi = mid
                    new_seq.insert(lo, x_add)
                    
                    if is_valid(new_seq):
                        current_seq = new_seq
                        current_set.remove(x_rem)
                        current_set.add(x_add)
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = list(current_seq)

    return sorted(best_seq)
