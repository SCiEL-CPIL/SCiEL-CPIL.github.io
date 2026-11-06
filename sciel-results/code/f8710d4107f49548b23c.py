import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s limit
    
    # Greedy seed: process from largest to smallest, track reachable sums up to 200
    selected = []
    reachable = {0}
    for x in range(100, 0, -1):
        new_sums = {s + x for s in reachable if s + x <= 200}
        if 200 not in new_sums:
            selected.append(x)
            reachable |= new_sums
    
    best = list(selected)
    best_len = len(best)
    
    if time.time() > deadline - 5.0:
        return best if best else [100]
    
    # Try to improve by swapping elements
    # For each element in best, try replacing it with a smaller element not in best
    best_set = set(best)
    
    def is_valid(subset):
        """Check if no subset of subset sums to 200."""
        if not subset:
            return True
        # Use bitset-like approach with a set of reachable sums
        sums = {0}
        for x in subset:
            new_sums = {s + x for s in sums if s + x <= 200}
            if 200 in new_sums:
                return False
            sums |= new_sums
        return True
    
    def try_improve(current, current_len, deadline):
        """Try to find a larger valid subset by local search."""
        cur = list(current)
        cur_set = set(cur)
        cur_len = len(cur)
        reachable = {0}
        for x in cur:
            reachable |= {s + x for s in reachable if s + x <= 200}
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding elements not in cur
            for x in range(100, 0, -1):
                if x in cur_set:
                    continue
                # Check if adding x keeps validity
                new_sums = {s + x for s in reachable if s + x <= 200}
                if 200 not in new_sums:
                    cur.append(x)
                    cur_set.add(x)
                    cur_len += 1
                    reachable |= new_sums
                    improved = True
                    break
            
            if time.time() > deadline - 2.0:
                break
            
            # Try removing and re-adding to find better configurations
            for i in range(len(cur)):
                if time.time() > deadline - 2.0:
                    break
                removed = cur.pop(i)
                cur_set.discard(removed)
                cur_len -= 1
                # Recompute reachable without removed
                reachable = {0}
                for x in cur:
                    reachable |= {s + x for s in reachable if s + x <= 200}
                
                # Try adding larger elements
                best_add = None
                for x in range(100, removed, -1):
                    if x in cur_set:
                        continue
                    new_sums = {s + x for s in reachable if s + x <= 200}
                    if 200 not in new_sums:
                        best_add = x
                        break
                
                if best_add is not None:
                    cur.append(best_add)
                    cur_set.add(best_add)
                    cur_len += 1
                    reachable |= {s + best_add for s in reachable if s + best_add <= 200}
                    improved = True
                else:
                    # Put it back
                    cur.insert(i, removed)
                    cur_set.add(removed)
                    cur_len += 1
                    reachable = {0}
                    for x in cur:
                        reachable |= {s + x for s in reachable if s + x <= 200}
        
        return cur, cur_len
    
    # Run local search from the greedy seed
    result, result_len = try_improve(best, best_len, deadline)
    if result_len > best_len:
        best = result
        best_len = result_len
    
    if time.time() > deadline - 5.0:
        return best if best else [100]
    
    # Try different initial orderings
    for start_val in [50, 75, 25]:
        if time.time() > deadline - 10.0:
            break
        
        # Try a different greedy: include elements > start_val first, then fill with small
        selected2 = []
        reachable2 = {0}
        # First add all large elements
        for x in range(100, start_val, -1):
            new_sums = {s + x for s in reachable2 if s + x <= 200}
            if 200 not in new_sums:
                selected2.append(x)
                reachable2 |= new_sums
        
        # Then fill with smaller elements
        for x in range(start_val, 0, -1):
            new_sums = {s + x for s in reachable2 if s + x <= 200}
            if 200 not in new_sums:
                selected2.append(x)
                reachable2 |= new_sums
        
        if len(selected2) > best_len:
            best = selected2
            best_len = len(selected2)
        
        # Local search from this seed
        result2, result2_len = try_improve(selected2, len(selected2), deadline)
        if result2_len > best_len:
            best = result2
            best_len = result2_len
    
    if time.time() > deadline - 5.0:
        return best if best else [100]
    
    # Try a randomized local search with multiple restarts
    import random
    rng = random.Random(42)
    
    for _ in range(5):
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization: pick a random subset and try to grow
        candidates = list(range(1, 101))
        rng.shuffle(candidates)
        selected3 = []
        reachable3 = {0}
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            new_sums = {s + x for s in reachable3 if s + x <= 200}
            if 200 not in new_sums:
                selected3.append(x)
                reachable3 |= new_sums
        
        result3, result3_len = try_improve(selected3, len(selected3), deadline)
        if result3_len > best_len:
            best = result3
            best_len = result3_len
    
    return best if best else [100]
