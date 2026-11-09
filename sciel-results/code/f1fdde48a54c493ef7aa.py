"""Program combining incremental checking with diverse ordering strategies for B_3[1] sets."""
# EVOLVE-BLOCK-START
import time
import random

def _check_valid(selected):
    """Return True if all unordered triple sums (with replacement) are unique."""
    n = len(selected)
    seen = set()
    for a in range(n):
        sa = selected[a]
        for b in range(a, n):
            sb = selected[b]
            sa_b = sa + sb
            for c in range(b, n):
                s = sa_b + selected[c]
                if s in seen:
                    return False
                seen.add(s)
    return True

def _greedy_build_order(order, limit, deadline):
    """Greedy construction following a given order, using incremental sum tracking."""
    vals = []
    existing_sums = set()
    
    for x in order:
        if time.time() > deadline:
            break
        if len(vals) >= 80:
            break
        if x > limit:
            break
        
        k = len(vals)
        conflict = False
        new_sums = []
        
        # Check (x,x,x)
        s = 3 * x
        if s in existing_sums:
            conflict = True
        else:
            new_sums.append(s)
        
        # Check (v, x, x) for v in vals
        if not conflict:
            for v in vals:
                s = v + 2 * x
                if s in existing_sums or s in new_sums:
                    conflict = True
                    break
                new_sums.append(s)
        
        # Check (v1, v2, x) for v1 <= v2 in vals
        if not conflict:
            for i in range(k):
                v1 = vals[i]
                for j in range(i, k):
                    v2 = vals[j]
                    s = v1 + v2 + x
                    if s in existing_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
                if conflict:
                    break
        
        if not conflict:
            vals.append(x)
            existing_sums.update(new_sums)
    
    return vals

def _greedy_from_start(start_list, limit, deadline):
    """Greedy construction starting from a given list, adding smallest valid numbers."""
    base = sorted(start_list)
    if not base:
        base = [1]
    
    # Compute initial sums
    sums = set()
    n = len(base)
    for a in range(n):
        sa = base[a]
        for b in range(a, n):
            sb = base[b]
            sa_b = sa + sb
            for c in range(b, n):
                s = sa_b + base[c]
                if s in sums:
                    return base  # Invalid start
                sums.add(s)
    
    while len(base) < 80:
        if time.time() > deadline:
            break
        next_min = max(base) + 1
        found = False
        for x in range(next_min, limit + 1):
            if x > limit:
                break
            k = len(base)
            conflict = False
            new_sums = []
            
            s = 3 * x
            if s in sums:
                conflict = True
            else:
                new_sums.append(s)
            
            if not conflict:
                for v in base:
                    s = v + 2 * x
                    if s in sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
            
            if not conflict:
                for i in range(k):
                    v1 = base[i]
                    for j in range(i, k):
                        v2 = base[j]
                        s = v1 + v2 + x
                        if s in sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                    if conflict:
                        break
            
            if not conflict:
                base.append(x)
                sums.update(new_sums)
                found = True
                break
        if not found:
            break
    return base

def _local_search_jump(base, limit, deadline, max_iter=50):
    """Try to improve by removing one element and greedily adding more.
    Optimized: precompute sums, use incremental removal check."""
    base = sorted(base)
    if not base:
        return base
    
    best = list(base)
    best_size = len(base)
    
    # Precompute all sums for the current base
    def _compute_sums(lst):
        sums = set()
        n = len(lst)
        for a in range(n):
            sa = lst[a]
            for b in range(a, n):
                sb = lst[b]
                sa_b = sa + sb
                for c in range(b, n):
                    s = sa_b + lst[c]
                    if s in sums:
                        return None  # Invalid
                    sums.add(s)
        return sums
    
    current_sums = _compute_sums(base)
    if current_sums is None:
        return best
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing each element and see if we can add more
        for remove_idx in range(len(base)):
            if time.time() > deadline:
                break
            
            candidate = [x for i, x in enumerate(base) if i != remove_idx]
            
            # Recompute sums for candidate
            cand_sums = _compute_sums(candidate)
            if cand_sums is None:
                continue
            
            # Try to add elements greedily
            temp_base = list(candidate)
            temp_sums = cand_sums
            next_min = max(temp_base) + 1 if temp_base else 1
            
            for x in range(next_min, limit + 1):
                if time.time() > deadline:
                    break
                if len(temp_base) >= 80:
                    break
                
                k = len(temp_base)
                conflict = False
                new_sums = []
                
                s = 3 * x
                if s in temp_sums:
                    conflict = True
                else:
                    new_sums.append(s)
                
                if not conflict:
                    for v in temp_base:
                        s = v + 2 * x
                        if s in temp_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                
                if not conflict:
                    for i in range(k):
                        v1 = temp_base[i]
                        for j in range(i, k):
                            v2 = temp_base[j]
                            s = v1 + v2 + x
                            if s in temp_sums or s in new_sums:
                                conflict = True
                                break
                            new_sums.append(s)
                        if conflict:
                            break
                
                if not conflict:
                    temp_base.append(x)
                    temp_sums.update(new_sums)
            
            if len(temp_base) > best_size:
                best = temp_base
                best_size = len(best)
                base = temp_base
                current_sums = _compute_sums(base)
                improved = True
                break
        
        if not improved:
            break
    
    return best

def _greedy_from_seed(seed_vals, limit, deadline):
    """Greedy construction starting from given seed values, adding smallest valid numbers."""
    base = sorted(seed_vals)
    if not base:
        base = [1]
    
    # Compute initial sums
    sums = set()
    n = len(base)
    valid = True
    for a in range(n):
        sa = base[a]
        for b in range(a, n):
            sb = base[b]
            sa_b = sa + sb
            for c in range(b, n):
                s = sa_b + base[c]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return base
    
    while len(base) < 80:
        if time.time() > deadline:
            break
        next_min = max(base) + 1
        found = False
        for x in range(next_min, limit + 1):
            if x > limit:
                break
            k = len(base)
            conflict = False
            new_sums = []
            
            s = 3 * x
            if s in sums:
                conflict = True
            else:
                new_sums.append(s)
            
            if not conflict:
                for v in base:
                    s = v + 2 * x
                    if s in sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
            
            if not conflict:
                for i in range(k):
                    v1 = base[i]
                    for j in range(i, k):
                        v2 = base[j]
                        s = v1 + v2 + x
                        if s in sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                    if conflict:
                        break
            
            if not conflict:
                base.append(x)
                sums.update(new_sums)
                found = True
                break
        if not found:
            break
    return base

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    limit = 4096
    best_result = []
    
    # Phase 1: Standard greedy from 1 upward
    order1 = list(range(1, limit + 1))
    result = _greedy_build_order(order1, limit, deadline)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 2: Local search jump on the best result
    result = _local_search_jump(best_result, limit, deadline, max_iter=40)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 3: Diverse ordering strategies with more attempts
    rng = random.Random(42)
    
    for attempt in range(300):
        if time.time() > deadline - 5:  # Leave 5s buffer
            break
        
        strategy = attempt % 6
        
        if strategy == 0:
            # Random shuffle
            order = list(range(1, limit + 1))
            rng.shuffle(order)
        elif strategy == 1:
            # Random offset: start from offset, then wrap around
            offset = rng.randint(0, 1000)
            order = list(range(1 + offset, limit + 1)) + list(range(1, 1 + offset))
        elif strategy == 2:
            # Two-phase: small numbers shuffled, then large numbers shuffled
            small = list(range(1, 500))
            large = list(range(500, limit + 1))
            rng.shuffle(small)
            rng.shuffle(large)
            order = small + large
        elif strategy == 3:
            # Exponential outward from random center
            center = rng.randint(1, limit)
            order = [center]
            for d in range(1, limit):
                if center - d >= 1:
                    order.append(center - d)
                if center + d <= limit:
                    order.append(center + d)
        elif strategy == 4:
            # Reverse order: large to small
            order = list(range(limit, 0, -1))
        else:
            # Skip-based: take every kth number, then fill gaps
            k = rng.randint(2, 5)
            order = list(range(1, limit + 1, k))
            remaining = [x for x in range(1, limit + 1) if x % k != 1]
            rng.shuffle(remaining)
            order.extend(remaining)
        
        result = _greedy_build_order(order, limit, deadline)
        if len(result) > len(best_result):
            best_result = result
            
            # If we found a better result, try local search
            if time.time() < deadline - 15:
                result2 = _local_search_jump(best_result, limit, deadline, max_iter=25)
                if len(result2) > len(best_result):
                    best_result = result2
    
    # Phase 4: Try seeding with random subsets and greedy completion
    for seed_attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        seed_size = rng.randint(1, 5)
        seed = rng.sample(range(1, limit + 1), seed_size)
        result = _greedy_from_seed(seed, limit, deadline)
        if len(result) > len(best_result):
            best_result = result
            if time.time() < deadline - 10:
                result2 = _local_search_jump(best_result, limit, deadline, max_iter=20)
                if len(result2) > len(best_result):
                    best_result = result2
    
    return best_result

# EVOLVE-BLOCK-END
