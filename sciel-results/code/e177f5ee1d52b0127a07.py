import time
import random
import sys

def _check_divisibility_conflict(new_sums, existing_sums):
    """Check if any sum in new_sums divides or is divisible by any sum in existing_sums."""
    # For each new sum, check against existing sums
    for ns in new_sums:
        for es in existing_sums:
            if ns == es:
                continue
            if es % ns == 0 or ns % es == 0:
                return False
    return True

def _build_sums(elements):
    """Build all non-empty subset sums from a list of elements."""
    sums = set()
    for x in elements:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        sums.update(new_sums)
    return sums

def _is_valid(elements):
    """Check if a set of elements is valid (all subset sums pairwise incomparable under divisibility)."""
    if not elements:
        return True
    if len(set(elements)) != len(elements):
        return False
    if any(x < 1 or x > 256 for x in elements):
        return False
    
    sums = _build_sums(elements)
    if not sums:
        return True
    
    max_sum = max(sums)
    present = [False] * (max_sum + 1)
    for s in sums:
        present[s] = True
    
    for s in sums:
        if s > max_sum // 2:
            continue
        multiple = 2 * s
        while multiple <= max_sum:
            if present[multiple]:
                return False
            multiple += s
    return True

def _backtrack_search(pool, current, current_sums, deadline, best, best_size):
    """
    Backtracking search to find a valid set of maximum size.
    pool: list of candidate elements to try
    current: list of elements in current solution
    current_sums: set of all non-empty subset sums of current
    deadline: time limit
    best: best solution found so far
    best_size: size of best solution
    """
    if time.time() > deadline:
        return best, best_size
    
    # Try to add more elements from pool
    for i in range(len(pool)):
        if time.time() > deadline:
            break
        
        x = pool[i]
        new_current = current + [x]
        
        # Compute new subset sums
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        new_sums.add(x)
        
        # Check for divisibility conflicts
        valid = True
        for ns in new_sums:
            for es in current_sums:
                if ns == es:
                    continue
                if es % ns == 0 or ns % es == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Also check among new_sums themselves
        for a in new_sums:
            for b in new_sums:
                if a == b:
                    continue
                if b % a == 0 or a % b == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Check if new sums conflict with each other via divisibility
        # (already checked above)
        
        all_sums = current_sums | new_sums
        
        # Check divisibility among all sums
        max_s = max(all_sums)
        present = [False] * (max_s + 1)
        for s in all_sums:
            present[s] = True
        
        conflict = False
        for s in all_sums:
            if s > max_s // 2:
                continue
            m = 2 * s
            while m <= max_s:
                if present[m]:
                    conflict = True
                    break
                m += s
            if conflict:
                break
        
        if conflict:
            continue
        
        # Valid extension
        if len(new_current) > best_size:
            best = list(new_current)
            best_size = len(new_current)
        
        # Recurse with remaining pool (excluding x)
        remaining = pool[:i] + pool[i+1:]
        best, best_size = _backtrack_search(remaining, new_current, all_sums, deadline, best, best_size)
    
    return best, best_size

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s margin before 90s kill
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Backtracking with different pool orderings
    
    # Pool 1: Descending order (large numbers first, fewer divisibility issues)
    pool1 = list(range(256, 0, -1))
    best, best_size = _backtrack_search(pool1, [], set(), deadline, best_set, best_size)
    
    if time.time() > deadline - 10.0:
        return [int(x) for x in best_set] if best_set else [11, 13, 17]
    
    # Pool 2: Randomized
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Create a biased random pool
        pool = list(range(1, 257))
        random.shuffle(pool)
        
        # Try backtracking with this pool
        best, best_size = _backtrack_search(pool, [], set(), deadline, best_set, best_size)
    
    if time.time() > deadline - 2.0:
        return [int(x) for x in best_set] if best_set else [11, 13, 17]
    
    # Strategy 2: Simulated annealing to try to improve
    
    if best_set and best_size < 10:
        current = list(best_set)
        current_sums = _build_sums(current)
        best_sa = list(current)
        best_sa_size = best_size
        
        # SA parameters
        temp = 10.0
        cooling = 0.95
        
        while time.time() < deadline - 1.0 and temp > 0.1:
            # Try to add a random element
            candidates = [x for x in range(1, 257) if x not in current]
            if not candidates:
                break
            
            x = random.choice(candidates)
            new_current = current + [x]
            
            # Compute new sums
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + x)
            new_sums.add(x)
            
            all_sums = current_sums | new_sums
            
            # Check validity
            valid = True
            max_s = max(all_sums)
            present = [False] * (max_s + 1)
            for s in all_sums:
                present[s] = True
            
            for s in all_sums:
                if s > max_s // 2:
                    continue
                m = 2 * s
                while m <= max_s:
                    if present[m]:
                        valid = False
                        break
                    m += s
                if not valid:
                    break
            
            if valid:
                current = new_current
                current_sums = all_sums
                if len(current) > best_sa_size:
                    best_sa_size = len(current)
                    best_sa = list(current)
            else:
                # Try removing a random element instead
                if len(current) > 1:
                    remove_idx = random.randint(0, len(current) - 1)
                    current = current[:remove_idx] + current[remove_idx+1:]
                    current_sums = _build_sums(current)
            
            temp *= cooling
        
        if best_sa_size > best_size:
            best_size = best_sa_size
            best_set = best_sa
    
    if best_set:
        return [int(x) for x in best_set]
    else:
        return [11, 13, 17]

