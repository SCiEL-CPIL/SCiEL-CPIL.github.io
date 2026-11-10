"""Rewritten program: find 12 integers in [0,128] minimizing largest Sidon subset size."""
import time
from itertools import combinations

_DEADLINE = 90.0
_SAFETY_MARGIN = 5.0

def _is_sidon(subset):
    """Check if all unordered pair sums (i<=j) are unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient_set, max_check=5):
    """Find the largest Sidon subset size in ambient_set, checking down from max_check."""
    n = len(ambient_set)
    for size in range(min(max_check, n), 1, -1):
        for combo in combinations(ambient_set, size):
            if _is_sidon(list(combo)):
                return size
    return 2

def _has_sidon_4(ambient_set):
    """Quick check: does ambient_set contain a Sidon subset of size 4?"""
    for combo in combinations(ambient_set, 4):
        if _is_sidon(list(combo)):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY_MARGIN
    
    best_set = None
    best_size = 12  # worst case
    
    # Strategy: We want to minimize the largest Sidon subset.
    # The minimum possible is 3 (since any 3 distinct integers a<b<c have sums
    # a+a, a+b, a+c, b+b, b+c, c+c. For these to be all distinct we need:
    # 2a != a+b (a!=b), 2a != a+c, 2a != 2b, 2a != b+c, a+b != a+c, a+b != 2b,
    # a+b != b+c, a+c != 2b, a+c != b+c, 2b != b+c. Most are automatically satisfied
    # for distinct a<b<c. The critical one is 2a != b+c and a+b != 2b (i.e., a!=b, ok)
    # and 2a+b != 2b+c etc. Actually for 3 elements, the sums are 2a, a+b, a+c, 2b, b+c, 2c.
    # These are 6 sums. They can collide. E.g., a=0,b=1,c=2: sums=0,1,2,2,3,4 -> 2b=2=a+c. Not Sidon.
    # a=0,b=1,c=3: sums=0,1,3,2,4,6 -> all distinct. So Sidon 3-subsets exist.
    # So minimum possible max Sidon subset size is 3.
    
    # We search for sets where NO 4-element subset is Sidon.
    # That means every 4-element subset has a collision.
    
    # Approach: Random search + local improvement
    import random
    
    # Try to find a set with max Sidon size = 3
    # Use random sampling with bias toward "dense" sum structures
    
    random.seed(12345)
    
    # Phase 1: Random search for sets with no Sidon-4
    attempts = 0
    found_size3 = False
    
    while time.time() < deadline and not found_size3:
        attempts += 1
        
        # Generate a random set of 12 distinct integers in [0, 128]
        # Bias: use numbers that are close together to create more sum collisions
        # Or use arithmetic progressions which create many collisions
        
        strategy = random.random()
        
        if strategy < 0.3:
            # Arithmetic progression with small difference
            a = random.randint(0, 100)
            d = random.randint(1, 5)
            if a + 11 * d > 128:
                d = max(1, (128 - a) // 11)
            if d == 0:
                continue
            candidate = [a + i * d for i in range(12)]
            if len(set(candidate)) != 12:
                continue
        elif strategy < 0.5:
            # Two interleaved progressions
            a1 = random.randint(0, 50)
            d1 = random.randint(2, 8)
            a2 = random.randint(0, 50)
            d2 = random.randint(2, 8)
            candidate = []
            for i in range(6):
                v1 = a1 + i * d1
                v2 = a2 + i * d2
                if v1 <= 128 and v2 <= 128:
                    candidate.extend([v1, v2])
            candidate = list(set(candidate))
            random.shuffle(candidate)
            candidate = candidate[:12]
            if len(candidate) < 12:
                # Fill in
                used = set(candidate)
                while len(candidate) < 12:
                    v = random.randint(0, 128)
                    if v not in used:
                        candidate.append(v)
                        used.add(v)
        elif strategy < 0.7:
            # Random but clustered
            center = random.randint(10, 118)
            spread = random.randint(5, 20)
            candidate = []
            used = set()
            while len(candidate) < 12:
                v = center + random.randint(-spread, spread)
                v = max(0, min(128, v))
                if v not in used:
                    candidate.append(v)
                    used.add(v)
        else:
            # Fully random
            candidate = random.sample(range(129), 12)
        
        candidate.sort()
        
        # Check if this set has no Sidon-4 subset
        if not _has_sidon_4(candidate):
            # Verify: max Sidon size should be 3
            ms = _max_sidon_size(candidate, 4)
            if ms <= 3:
                best_set = list(candidate)
                best_size = ms
                found_size3 = True
                break
        
        # Also track best found so far
        ms = _max_sidon_size(candidate, 4)
        if ms < best_size:
            best_size = ms
            best_set = list(candidate)
    
    # Phase 2: Local search to try to reduce max Sidon size
    if best_set is not None and time.time() < deadline:
        current = list(best_set)
        current_size = best_size
        
        # Try swapping elements to reduce Sidon subset size
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                for new_val in range(129):
                    if new_val in current and new_val != current[i]:
                        continue
                    trial = list(current)
                    trial[i] = new_val
                    trial.sort()
                    
                    # Quick check: does it have Sidon-4?
                    if current_size <= 3:
                        # Already at minimum, try to maintain
                        if not _has_sidon_4(trial):
                            # Good, keep it
                            pass
                        else:
                            continue
                    else:
                        ms_trial = _max_sidon_size(trial, 4)
                        if ms_trial < current_size:
                            current = trial
                            current_size = ms_trial
                            best_set = list(current)
                            best_size = current_size
                            improved = True
                            break
                if improved:
                    break
        
        if current_size < best_size:
            best_set = current
            best_size = current_size
    
    # Final verification
    if best_set is None:
        # Fallback: a known set
        best_set = [0, 1, 3, 7, 12, 18, 26, 35, 45, 56, 68, 81]
    
    # Ensure exactly 12 distinct integers in [0, 128]
    best_set = sorted(set(best_set))
    while len(best_set) < 12:
        for v in range(129):
            if v not in best_set:
                best_set.append(v)
                break
        best_set.sort()
    best_set = best_set[:12]
    
    return best_set
