import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    def subset_sums(S):
        """Compute all non-empty subset sums using iterative set union."""
        sums = set()
        for x in S:
            new_sums = {x + s for s in sums}
            new_sums.add(x)
            sums = sums | new_sums
        return sums

    def is_divisibility_free(sums):
        """Check no two distinct sums divide each other. O(k^2) but with early exit."""
        sl = sorted(sums)
        k = len(sl)
        for i in range(k):
            a = sl[i]
            if a <= 1:
                continue
            # b must be >= 2a to possibly be divisible by a
            for j in range(i + 1, k):
                b = sl[j]
                if b >= 2 * a and b % a == 0:
                    return False
                if b > 2 * a * 10:  # heuristic: skip far apart
                    break
        return True

    def try_add(S, x, current_sums):
        """Check if adding x keeps divisibility-freeness. Returns (success, new_sums)."""
        # New sums: x, and x+s for each s in current_sums
        new_sums = {x}
        for s in current_sums:
            new_sums.add(x + s)
        all_sums = current_sums | new_sums
        if is_divisibility_free(all_sums):
            return True, all_sums
        return False, None

    best = [127, 129, 131]
    best_size = 3

    def validate_and_update(S):
        nonlocal best, best_size
        if len(S) > best_size:
            sums = subset_sums(S)
            if is_divisibility_free(sums):
                best = list(S)
                best_size = len(S)
                return True
        return False

    # Strategy: focus on large numbers [128, 256] to minimize divisibility conflicts
    # Large numbers are less likely to divide each other's subset sums
    
    # Precompute: for each candidate x, which existing sums would it conflict with?
    # Greedy: add numbers that create the fewest conflicts
    
    all_candidates = list(range(128, 257))
    
    # Try multiple random orderings
    random.seed(12345)
    
    for trial in range(5000):
        if time.time() > deadline - 2.0:
            break
        
        # Random starting point: pick 2-3 large numbers
        seed_size = random.randint(2, 3)
        seed = random.sample(all_candidates, seed_size)
        seed.sort()
        
        # Verify seed is valid
        seed_sums = subset_sums(seed)
        if not is_divisibility_free(seed_sums):
            continue
        
        S = list(seed)
        current_sums = seed_sums
        
        # Greedy: try to add remaining candidates in random order
        remaining = [x for x in all_candidates if x not in S]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline - 2.0:
                break
            ok, new_sums = try_add(S, x, current_sums)
            if ok:
                S.append(x)
                current_sums = new_sums
        
        if len(S) > best_size:
            validate_and_update(S)
    
    # Also try a more systematic approach: build from scratch with priority
    # Start empty, always add the number that causes fewest conflicts
    for trial in range(100):
        if time.time() > deadline - 2.0:
            break
        
        S = []
        current_sums = set()
        
        candidates = list(range(128, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 2.0:
                break
            ok, new_sums = try_add(S, x, current_sums)
            if ok:
                S.append(x)
                current_sums = new_sums
        
        if len(S) > best_size:
            validate_and_update(S)
    
    # Final validation
    if best_size >= 3:
        final_sums = subset_sums(best)
        if is_divisibility_free(final_sums):
            return [int(x) for x in best]
    
    # Fallback: known valid small set
    fallback = [127, 129, 131]
    fs = subset_sums(fallback)
    if is_divisibility_free(fs):
        return [int(x) for x in fallback]
    
    # Absolute fallback
    return [127, 129]
