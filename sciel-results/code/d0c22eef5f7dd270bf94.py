import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    def check_sidon(subset):
        """Check if subset is a Sidon set including diagonal sums."""
        n = len(subset)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def max_sidon_size(ambient):
        """Find the maximum size of a Sidon subset in ambient."""
        n = len(ambient)
        # We are interested in small max sizes, likely 3, 4, or 5.
        # Check from 6 down to 1.
        # If we find a Sidon subset of size k, return k.
        # Optimization: If we only care about minimizing, and we suspect 3 or 4,
        # we can check 6, 5, 4 first.
        
        # Pre-compute combinations for speed? 
        # C(12, 6) = 924
        # C(12, 5) = 792
        # C(12, 4) = 495
        # C(12, 3) = 220
        
        # Let's check 6, 5, 4. If none, return 3 (assuming 3 exists, which is almost always true for random sets).
        # Actually, if no 4-subset is Sidon, max is 3 (or less).
        
        for k in range(6, 3, -1):
            for subset in itertools.combinations(ambient, k):
                if check_sidon(subset):
                    return k
        # If no subset of size 4, 5, or 6 is Sidon, check 3.
        # If no 3-subset is Sidon, return 2.
        for subset in itertools.combinations(ambient, 3):
            if check_sidon(subset):
                return 3
        return 2

    def count_sidon_subsets(ambient, k):
        """Count number of Sidon subsets of size k."""
        count = 0
        for subset in itertools.combinations(ambient, k):
            if check_sidon(subset):
                count += 1
        return count

    # Initial candidates
    # 1. Dense range
    # 2. Random
    rng = random.Random(42)
    
    candidates = []
    candidates.append(list(range(12)))
    candidates.append(list(range(1, 13)))
    candidates.append(sorted(rng.sample(range(129), 12)))
    candidates.append(sorted(rng.sample(range(129), 12)))
    
    best_ambient = None
    best_size = 10
    best_count_4 = 1000
    
    for cand in candidates:
        size = max_sidon_size(cand)
        if size < best_size:
            best_size = size
            best_ambient = list(cand)
            if size == 3:
                # Try to verify if we can get 2? Unlikely.
                # But we should minimize count of size 4 if size is 4, 
                # or count of size 3 if size is 3?
                # The metric is just the max size.
                # However, to guide search towards 3, we can use count_4 as a tie-breaker.
                pass
        elif size == best_size:
            # Tie-breaker: minimize number of Sidon subsets of size best_size
            # This helps escape plateaus.
            if best_size == 4:
                c4 = count_sidon_subsets(cand, 4)
                if c4 < best_count_4:
                    best_count_4 = c4
                    best_ambient = list(cand)
            elif best_size == 3:
                # If size is 3, we are likely done, but let's see if we can find 2?
                # Max Sidon size 2 means no 3 elements are Sidon.
                # This implies every 3 elements form a non-Sidon set.
                # Non-Sidon 3-set: $a+c = 2b$ (Arithmetic Progression).
                # So we need every 3-subset to be an AP. This is impossible for 12 distinct integers.
                # So min size is 3.
                pass

    # Local Search
    current = list(best_ambient)
    current_size = max_sidon_size(current)
    current_count_4 = count_sidon_subsets(current, 4) if current_size >= 4 else 0
    
    # If current_size is 3, we are at the theoretical minimum (likely).
    # If current_size is 4, we want to reduce count_4 to 0.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        i = rng.randrange(12)
        old_val = current[i]
        
        # Try a random new value
        # Bias towards small values to maintain density?
        # Or random in [0, 128]?
        # Small values cause more collisions.
        # Let's try random in [0, 30] first, then [0, 128].
        if rng.random() < 0.5:
            new_val = rng.randrange(30)
        else:
            new_val = rng.randrange(129)
            
        if new_val == old_val:
            continue
            
        if new_val in current:
            continue
            
        current[i] = new_val
        
        # Evaluate
        new_size = max_sidon_size(current)
        
        if new_size < current_size:
            current_size = new_size
            if current_size == 4:
                current_count_4 = count_sidon_subsets(current, 4)
            best_ambient = list(current)
            best_size = current_size
        elif new_size == current_size:
            # Tie-breaker
            if current_size == 4:
                new_count_4 = count_sidon_subsets(current, 4)
                if new_count_4 < current_count_4:
                    current_count_4 = new_count_4
                    best_ambient = list(current)
                    # best_size remains 4
                else:
                    # Revert if not better
                    current[i] = old_val
            elif current_size == 3:
                # If size is 3, we can't get better.
                # Just keep the best one found?
                # Or try to find if 2 is possible (it's not).
                pass
            else:
                current[i] = old_val
        else:
            # Revert
            current[i] = old_val

    return best_ambient
