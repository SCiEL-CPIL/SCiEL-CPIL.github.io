import time
import random

def construct():
    start = time.time()
    deadline = start + 88.0  # 2s safety margin before 90s kill

    def is_valid(sums):
        """Check if any two distinct sums divide each other."""
        if len(sums) < 2:
            return True
        sum_list = sorted(sums)
        max_sum = sum_list[-1]
        sums_set = sums
        for u in sum_list:
            if u * 2 > max_sum:
                break
            for v in range(u * 2, max_sum + 1, u):
                if v in sums_set:
                    return False
        return True

    def try_add_element(current_sums, c):
        """Try adding element c. Returns new_sums if valid, else None."""
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + c)
        new_sums.add(c)
        
        if len(new_sums) != 2 * len(current_sums) + 1:
            return None
            
        if not is_valid(new_sums):
            return None
            
        return new_sums

    best_set = []
    best_size = 0

    # Pre-compute candidate ranges, sorted in descending order for greedy passes
    # Large numbers first: fewer multiples in range, less likely to cause divisibility
    large_desc = list(range(256, 127, -1))
    mid_desc = list(range(256, 63, -1))
    all_desc = list(range(256, 0, -1))
    
    # Strategy 1: Deterministic greedy with descending order (large numbers first)
    # This is fast and often finds a good starting point
    for candidates in [large_desc, mid_desc, all_desc]:
        if time.time() > deadline - 3.0:
            break
        current = []
        current_sums = set()
        for c in candidates:
            if len(current) >= 30:
                break
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                current.append(c)
                current_sums = new_sums
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 30:
                break

    # Strategy 2: Randomized restarts with mixed orderings
    # Use different strategies for candidate ordering
    ranges = [
        list(range(256, 127, -1)),  # large only, descending
        list(range(256, 100, -1)),  # large-mid, descending
        list(range(256, 50, -1)),   # broader, descending
        list(range(256, 1, -1)),    # all, descending
    ]
    
    for attempt in range(5000):
        if time.time() > deadline - 3.0:
            break
        
        # Choose range: bias toward larger ranges early, then mix
        if attempt < 500:
            r_idx = 0  # large only
        elif attempt < 1500:
            r_idx = random.choice([0, 1])  # large or large-mid
        elif attempt < 3000:
            r_idx = random.randint(0, 2)  # mix
        else:
            r_idx = random.randint(0, 3)  # all ranges
        
        candidates = ranges[r_idx][:]
        # Randomize but keep a bias: partially shuffle
        # Use a "biased shuffle" - keep some structure
        if random.random() < 0.3:
            random.shuffle(candidates)
        else:
            # Partial shuffle: reverse chunks
            chunk_size = random.randint(5, 20)
            for i in range(0, len(candidates), chunk_size):
                chunk = candidates[i:i+chunk_size]
                random.shuffle(chunk)
                candidates[i:i+chunk_size] = chunk
        
        current = []
        current_sums = set()
        
        for c in candidates:
            if len(current) >= 30:
                break
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                current.append(c)
                current_sums = new_sums
                
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 30:
                break

    # Strategy 3: Local search - try to swap out elements from the best set
    # to find larger sets
    if best_size > 0 and time.time() < deadline - 5.0:
        # Try to extend the best set by removing one element and adding a better one
        for iteration in range(200):
            if time.time() > deadline - 3.0:
                break
            
            base_set = best_set[:]
            base_sums = set()
            # Recompute sums for base_set
            n = len(base_set)
            for mask in range(1, 1 << n):
                s = 0
                for i in range(n):
                    if mask & (1 << i):
                        s += base_set[i]
                base_sums.add(s)
            
            # Try removing each element and adding a new one
            for rem_idx in range(len(base_set)):
                if time.time() > deadline - 3.0:
                    break
                
                reduced_set = base_set[:rem_idx] + base_set[rem_idx+1:]
                # Recompute sums for reduced set
                n_red = len(reduced_set)
                red_sums = set()
                for mask in range(1, 1 << n_red):
                    s = 0
                    for i in range(n_red):
                        if mask & (1 << i):
                            s += reduced_set[i]
                    red_sums.add(s)
                
                # Try adding each candidate not in reduced_set
                for c in range(256, 0, -1):
                    if c in reduced_set:
                        continue
                    new_sums = try_add_element(red_sums, c)
                    if new_sums is not None:
                        new_set = reduced_set + [c]
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = new_set[:]
                        break  # Only try first valid candidate for this removal
    
    # Ensure types are correct
    return [int(x) for x in best_set]
