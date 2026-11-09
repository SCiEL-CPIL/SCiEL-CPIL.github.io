import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Precompute conflict structure using bitsets
    # For each a, precompute which pairs (b,c) with b>=a+1, c>=b cause conflict
    # conflict_mask[a] is a set of (b,c) pairs where a divides b+c
    
    # Actually, let's precompute for each a, the set of values v such that there exist b,c in S with b+c=v and a|v
    # Better: for each a, precompute which pairs (b,c) are conflicting
    
    # Use a different approach: for each a, precompute the "forbidden sums"
    # If a is in S, then for any b,c in S with b>a, c>=b, we need (b+c) % a != 0
    
    # Let's precompute conflict pairs for each a
    # conflict_with[a] = set of (b,c) pairs where a|b+c and a<b<=c
    
    # More efficient: for each a, precompute which c values conflict with which b values
    # For fixed a and b, c conflicts if c ≡ -b (mod a) and c >= b
    
    # Let's use a bitset approach
    # For each a, create a 256-bit integer where bit i is set if i is in the set
    # Then check conflicts efficiently
    
    def full_validate(S):
        S_sorted = sorted(S)
        k = len(S_sorted)
        # Use bitset for fast membership
        S_set = set(S_sorted)
        for i in range(k):
            a = S_sorted[i]
            for j in range(i+1, k):
                b = S_sorted[j]
                # For each c >= b, check if (b+c) % a == 0
                # c must satisfy c ≡ -b (mod a)
                remainder = (-b) % a
                # Find all c in S with c >= b and c % a == remainder
                for l in range(j, k):
                    c = S_sorted[l]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def find_conflict_fast(S):
        S_sorted = sorted(S)
        k = len(S_sorted)
        S_set = set(S_sorted)
        for i in range(k):
            a = S_sorted[i]
            for j in range(i+1, k):
                b = S_sorted[j]
                remainder = (-b) % a
                # Find c in S with c >= b and c % a == remainder
                for l in range(j, k):
                    c = S_sorted[l]
                    if (b + c) % a == 0:
                        return (a, b, c)
        return None
    
    # Precompute for each a, the list of (b, c) pairs that conflict
    # This allows faster validation
    
    # Start with the known best interval
    best_set = set(range(171, 257))
    best_size = 86
    
    # Try to improve by local search
    # Strategy: Try to add elements and repair conflicts
    
    # First, try systematic addition from the top
    S = set(best_set)
    for x in range(170, 0, -1):
        if time.time() > deadline - 10:
            break
        if x in S:
            continue
        S.add(x)
        conflict = find_conflict_fast(S)
        if conflict is None:
            best_set = set(S)
            best_size = len(S)
        else:
            S.remove(x)
    
    # Now try random restarts with repair
    for trial in range(2000):
        if time.time() > deadline - 5:
            break
        
        # Random initialization: start with a random subset biased towards larger numbers
        # Mix of random and structured
        if trial % 3 == 0:
            # Start from best_set and perturb
            S = set(best_set)
            # Remove 1-3 random elements
            to_remove = random.sample(list(S), min(random.randint(1, 3), len(S)))
            for x in to_remove:
                S.remove(x)
            # Add 1-3 random elements from 1..170
            candidates = [x for x in range(1, 171) if x not in S]
            if candidates:
                to_add = random.sample(candidates, min(random.randint(1, 3), len(candidates)))
                for x in to_add:
                    S.add(x)
        elif trial % 3 == 1:
            # Start with upper tail of varying size
            start = random.randint(160, 200)
            S = set(range(start, 257))
            # Add some random smaller elements
            candidates = [x for x in range(1, start) if x not in S]
            if candidates:
                to_add = random.sample(candidates, min(random.randint(0, 5), len(candidates)))
                for x in to_add:
                    S.add(x)
        else:
            # Pure random with bias towards larger numbers
            S = set()
            candidates = list(range(1, n+1))
            random.shuffle(candidates)
            # Take more from the upper half
            upper = [x for x in candidates if x > 128]
            lower = [x for x in candidates if x <= 128]
            random.shuffle(upper)
            random.shuffle(lower)
            S = set(upper[:random.randint(40, 60)] + lower[:random.randint(10, 30)])
        
        # Repair: iteratively find and resolve conflicts
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            conflict = find_conflict_fast(S)
            if conflict:
                a, b, c = conflict
                # Heuristic: remove the element that causes the most conflicts
                # Try removing each of a, b, c and see which leads to fewer remaining conflicts
                best_to_remove = None
                min_conflicts = float('inf')
                
                for candidate in [a, b, c]:
                    S_test = set(S)
                    S_test.remove(candidate)
                    # Count conflicts in S_test (approximate)
                    conflict_count = 0
                    S_test_sorted = sorted(S_test)
                    k_test = len(S_test_sorted)
                    for i in range(min(k_test, 20)):  # Limit for speed
                        ai = S_test_sorted[i]
                        for j in range(i+1, min(k_test, 20)):
                            bi = S_test_sorted[j]
                            for l in range(j, min(k_test, 20)):
                                ci = S_test_sorted[l]
                                if (bi + ci) % ai == 0:
                                    conflict_count += 1
                                    if conflict_count > min_conflicts:
                                        break
                            if conflict_count > min_conflicts:
                                break
                        if conflict_count > min_conflicts:
                            break
                    
                    if conflict_count < min_conflicts:
                        min_conflicts = conflict_count
                        best_to_remove = candidate
                
                if best_to_remove is not None:
                    S.remove(best_to_remove)
                    improved = True
        
        if len(S) > best_size and full_validate(S):
            best_set = set(S)
            best_size = len(S)
            if best_size >= 95:
                break
    
    # Final greedy addition pass
    if time.time() < deadline - 5:
        S = set(best_set)
        for x in range(1, n+1):
            if time.time() > deadline - 3:
                break
            if x in S:
                continue
            S.add(x)
            if find_conflict_fast(S) is None:
                best_set = set(S)
                best_size = len(S)
            else:
                S.remove(x)
    
    return list(best_set)
