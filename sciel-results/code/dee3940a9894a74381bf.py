"""Search-based solver for Sidon set extension containing {1, 2, 5, 11} in [1, 284]."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s budget, 10s safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    # Pre-compute initial sums
    initial_sums = set()
    for i, x in enumerate(prescribed):
        for y in prescribed[i:]:
            initial_sums.add(x + y)
    
    # Best found so far
    best_set = set(prescribed)
    best_size = len(best_set)
    best_sums = set(initial_sums)
    
    def build_sums(s):
        """Build the set of all pairwise sums a+b for a<=b in s."""
        sl = sorted(s)
        sums = set()
        for i, x in enumerate(sl):
            for y in sl[i:]:
                sums.add(x + y)
        return sums

    # Helper to check if adding x to a set with given sums is valid
    def can_add(x, current_set, current_sums):
        new_sums = {x + y for y in current_set} | {2 * x}
        if len(new_sums) == len(current_set) + 1 and not (new_sums & current_sums):
            return True
        return False

    def get_new_sums(x, current_set, current_sums):
        return {x + y for y in current_set} | {2 * x}

    # Strategy 1: Greedy forward construction with randomization
    remaining = [x for x in range(12, 285) if x not in prescribed]
    
    def greedy_with_random_order(order, time_limit=5.0):
        nonlocal best_set, best_size, best_sums
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        start = time.time()
        
        for x in order:
            if time.time() - start > time_limit:
                break
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            best_sums = set(current_sums)
        
        return len(current_set)

    # Try a few quick greedy passes with different random seeds
    rng = random.Random(42)
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        order = remaining[:]
        rng.shuffle(order)
        greedy_with_random_order(order, time_limit=0.5)
    
    # Strategy 2: Fast local search with efficient conflict tracking
    def fast_local_search(start_set, time_limit=60.0):
        nonlocal best_set, best_size, best_sums
        current_set = set(start_set)
        current_sums = build_sums(current_set)
        start = time.time()
        
        # Precompute available candidates
        all_candidates = list(range(1, 285))
        
        # Conflict count: for each candidate x, count how many of its new sums
        # (x+y for y in current_set, plus 2x) collide with existing sums
        # If conflict_count[x] == 0, x can be added
        def compute_conflict_counts(current_set, current_sums):
            """For each candidate not in current_set, compute number of sum conflicts."""
            conflicts = {}
            for x in all_candidates:
                if x in current_set:
                    continue
                cnt = 0
                # Check each new sum
                for y in current_set:
                    if (x + y) in current_sums:
                        cnt += 1
                if (2 * x) in current_sums:
                    cnt += 1
                # Also check internal conflicts among new sums
                new_sums_set = set()
                for y in current_set:
                    s = x + y
                    if s in new_sums_set:
                        cnt += 1
                    new_sums_set.add(s)
                s2 = 2 * x
                if s2 in new_sums_set:
                    cnt += 1
                conflicts[x] = cnt
            return conflicts
        
        # Initial conflict computation
        conflict_counts = compute_conflict_counts(current_set, current_sums)
        
        # Zero-conflict candidates
        zero_conflict = [x for x, c in conflict_counts.items() if c == 0]
        
        for iteration in range(500000):
            if time.time() - start > time_limit or time.time() > deadline - 5:
                break
            
            # Try to add a zero-conflict element
            if zero_conflict:
                x = rng.choice(zero_conflict)
                if x not in current_set:
                    new_sums = get_new_sums(x, current_set, current_sums)
                    current_set.add(x)
                    current_sums.update(new_sums)
                    
                    # Update conflict counts efficiently
                    # For each other candidate z, check if new sums create conflicts
                    # This is O(n * m) where n = candidates, m = new sums
                    for z in all_candidates:
                        if z in current_set:
                            continue
                        # Check if any of the new sums conflict with z's potential sums
                        for ns in new_sums:
                            # z + y or 2z equals ns for some y in current_set?
                            # Actually we need to recompute: does z have new conflicts?
                            pass
                        # Simpler: recompute conflict for z
                        old_cnt = conflict_counts.get(z, 0)
                        new_cnt = old_cnt
                        # Check if any new sum equals z+y or 2z for y in current_set
                        for y in current_set:
                            if (z + y) == ns or (2 * z) == ns:
                                new_cnt += 1
                        # This is getting complex; just recompute for simplicity
                        # but only for candidates that might be affected
                        pass
                    
                    # For now, recompute all conflicts (slower but correct)
                    # Optimization: only recompute for candidates affected by new sums
                    affected = set()
                    for ns in new_sums:
                        # Candidates z where z+y = ns or 2z = ns
                        for y in current_set:
                            z = ns - y
                            if 1 <= z <= 284 and z not in current_set:
                                affected.add(z)
                        z2 = ns // 2
                        if 2 * z2 == ns and 1 <= z2 <= 284 and z2 not in current_set:
                            affected.add(z2)
                    
                    for z in affected:
                        if z in current_set:
                            continue
                        cnt = 0
                        for y in current_set:
                            if (z + y) in current_sums:
                                cnt += 1
                        if (2 * z) in current_sums:
                            cnt += 1
                        new_sums_set = set()
                        for y in current_set:
                            s = z + y
                            if s in new_sums_set:
                                cnt += 1
                            new_sums_set.add(s)
                        s2 = 2 * z
                        if s2 in new_sums_set:
                            cnt += 1
                        conflict_counts[z] = cnt
                        if cnt == 0:
                            if z not in zero_conflict:
                                zero_conflict.append(z)
                        else:
                            if z in zero_conflict:
                                zero_conflict.remove(z)
                    
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
                        best_sums = set(current_sums)
                    continue
            
            # No zero-conflict candidates: try a swap
            removable = [x for x in current_set if x not in prescribed_set]
            if removable:
                x_remove = rng.choice(removable)
                remaining_set = current_set - {x_remove}
                reduced_sums = build_sums(remaining_set)
                
                # Find candidates that can be added to remaining_set
                valid_adds = []
                for x_add in all_candidates:
                    if x_add in remaining_set:
                        continue
                    new_sums_test = {x_add + y for y in remaining_set} | {2 * x_add}
                    if len(new_sums_test) == len(remaining_set) + 1 and not (new_sums_test & reduced_sums):
                        valid_adds.append(x_add)
                
                if valid_adds:
                    x_add = rng.choice(valid_adds)
                    current_set = remaining_set | {x_add}
                    current_sums = reduced_sums | {x_add + y for y in remaining_set} | {2 * x_add}
                    
                    # Recompute conflict counts
                    conflict_counts = compute_conflict_counts(current_set, current_sums)
                    zero_conflict = [x for x, c in conflict_counts.items() if c == 0]
                    
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
                        best_sums = set(current_sums)
        
        return len(current_set)

    # Run fast local search from best found
    if best_size > len(prescribed):
        fast_local_search(best_set, time_limit=50.0)
    
    # Strategy 3: Systematic greedy strategies
    # Greedy that picks the smallest valid element each time
    if time.time() < deadline - 15:
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        for x in range(12, 285):
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            best_sums = set(current_sums)
    
    # Greedy that picks the largest valid element each time
    if time.time() < deadline - 15:
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        for x in range(284, 11, -1):
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            best_sums = set(current_sums)
    
    # Strategy 4: Greedy with lookahead - try multiple candidates at each step
    if time.time() < deadline - 15:
        def greedy_with_lookahead(time_limit=5.0):
            nonlocal best_set, best_size, best_sums
            current_set = set(prescribed)
            current_sums = set(initial_sums)
            start = time.time()
            
            while time.time() - start < time_limit:
                # Find all valid additions
                valid_adds = []
                for x in range(1, 285):
                    if x not in current_set and can_add(x, current_set, current_sums):
                        valid_adds.append(x)
                
                if not valid_adds:
                    break
                
                # Pick randomly among valid additions
                x = rng.choice(valid_adds)
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
            
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
                best_sums = set(current_sums)
            return len(current_set)
        
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            greedy_with_lookahead(time_limit=2.0)
    
    # More randomized greedy passes
    if time.time() < deadline - 10:
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            order = remaining[:]
            rng.shuffle(order)
            greedy_with_random_order(order, time_limit=1.0)
    
    # Final fast local search with more time
    if time.time() < deadline - 5 and best_size > len(prescribed):
        fast_local_search(best_set, time_limit=30.0)
    
    # Final check: ensure we return a valid set
    result = sorted(best_set)
    
    # Verify validity
    sums = set()
    valid = True
    for i, x in enumerate(result):
        for y in result[i:]:
            s = x + y
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    
    if not valid:
        # Fall back to prescribed only
        return prescribed
    
    return result

# For testing
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Result: {result}")
    sums = set()
    valid = True
    for i, x in enumerate(result):
        for y in result[i:]:
            s = x + y
            if s in sums:
                valid = False
                break
            sums.add(s)
    print(f"Valid: {valid}")
