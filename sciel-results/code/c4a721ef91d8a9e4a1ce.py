# EVOLVE-BLOCK-START
import random
import time
import itertools
import numpy as np

def _is_sidon_fast(subset):
    """Fast Sidon check with early termination using a set of sums."""
    n = len(subset)
    if n <= 1:
        return True
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size_exact(ambient, time_limit=None, start_time=None):
    """Find largest Sidon subset exactly by trying all combinations from large to small.
    Returns the maximum k such that some k-subset is Sidon."""
    if start_time is None:
        start_time = time.time()
    n = len(ambient)
    ambient = list(ambient)
    
    # Check if full set is Sidon
    if _is_sidon_fast(ambient):
        return n
    
    # Try decreasing k
    for k in range(n - 1, 1, -1):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        found = False
        for combo in itertools.combinations(ambient, k):
            if _is_sidon_fast(list(combo)):
                return k
            if time_limit and (time.time() - start_time) > time_limit:
                break
        if time_limit and (time.time() - start_time) > time_limit:
            break
    return 2

def _no_5_sidon_check(ambient):
    """Check if NO 5-element subset of ambient is Sidon.
    Returns True if max Sidon subset size <= 4."""
    n = len(ambient)
    # Check all 5-element subsets
    for combo in itertools.combinations(ambient, 5):
        if _is_sidon_fast(list(combo)):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    random.seed(42)
    
    best_set = None
    best_sidon_size = 999
    
    evaluated_sets = set()
    
    def evaluate(candidate):
        nonlocal best_set, best_sidon_size
        if len(candidate) != 12 or len(set(candidate)) != 12:
            return
        if any(x < 0 or x > 128 for x in candidate):
            return
        key = tuple(sorted(candidate))
        if key in evaluated_sets:
            return
        evaluated_sets.add(key)
        
        sidon_size = _max_sidon_size_exact(candidate, time_limit=2.0, start_time=start_time)
        if sidon_size < best_sidon_size:
            best_sidon_size = sidon_size
            best_set = list(candidate)
    
    # Strategy 1: Dense consecutive sets (most collisions)
    for start in range(0, 120):
        if time.time() > deadline:
            break
        candidate = list(range(start, start + 12))
        if candidate[-1] <= 128:
            evaluate(candidate)
    
    # Strategy 2: Arithmetic progressions with small step
    for step in range(1, 6):
        for start in range(0, 129 - 11 * step + 1):
            if time.time() > deadline:
                break
            candidate = [start + i * step for i in range(12)]
            evaluate(candidate)
    
    # Strategy 3: Targeted construction - pairs that share sums
    # Create 6 pairs where multiple pairs have the same sum
    # This forces collisions in any 5-element subset that picks from 3+ pairs
    for _ in range(2000):
        if time.time() > deadline:
            break
        
        # Choose a base sum S
        S = random.randint(20, 200)
        
        # Create pairs summing to S or S+1 (to create multiple collision points)
        candidate = []
        used = set()
        
        # First, create 3-4 pairs with the same sum S
        num_same_sum = random.randint(3, 5)
        for _ in range(num_same_sum):
            a = random.randint(0, min(128, S - 1))
            b = S - a
            if a != b and a not in used and b not in used and 0 <= a <= 128 and 0 <= b <= 128:
                candidate.append(a)
                candidate.append(b)
                used.add(a)
                used.add(b)
        
        # Fill remaining slots with values that create more collisions
        while len(candidate) < 12:
            # Try to add pairs that collide with existing sums
            existing_sums = set()
            for i in range(len(candidate)):
                for j in range(i, len(candidate)):
                    existing_sums.add(candidate[i] + candidate[j])
            
            added = False
            for _ in range(20):
                a = random.randint(0, 128)
                b = random.randint(0, 128)
                if a != b and a not in used and b not in used and a <= 128 and b <= 128:
                    # Prefer if a+b is already in existing_sums
                    if a + b in existing_sums or random.random() < 0.3:
                        candidate.append(a)
                        candidate.append(b)
                        used.add(a)
                        used.add(b)
                        added = True
                        break
            
            if not added:
                # Just pick random unused values
                while len(candidate) < 12:
                    v = random.randint(0, 128)
                    if v not in used:
                        candidate.append(v)
                        used.add(v)
                    else:
                        break
        
        if len(candidate) == 12:
            evaluate(candidate)
    
    # Strategy 4: Local search focused on reducing max Sidon size
    if best_set is not None and best_sidon_size > 4:
        current = list(best_set)
        current_sidon = best_sidon_size
        
        for iteration in range(10000):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 11)
            old_val = current[idx]
            
            # Try values that create more sum collisions
            new_val = None
            for _ in range(50):
                candidate_val = random.randint(0, 128)
                if candidate_val == old_val or candidate_val in current:
                    continue
                
                # Check if this creates more collisions
                test_set = list(current)
                test_set[idx] = candidate_val
                
                # Count sum collisions
                sum_count = {}
                for i in range(12):
                    for j in range(i, 12):
                        s = test_set[i] + test_set[j]
                        sum_count[s] = sum_count.get(s, 0) + 1
                
                collision_score = sum(v * (v - 1) // 2 for v in sum_count.values())
                
                if new_val is None:
                    new_val = candidate_val
                    best_collision = collision_score
                elif collision_score > best_collision:
                    new_val = candidate_val
                    best_collision = collision_score
            
            if new_val is None or new_val == old_val:
                continue
            
            current[idx] = new_val
            sidon_size = _max_sidon_size_exact(current, time_limit=1.0, start_time=start_time)
            
            if sidon_size < best_sidon_size:
                best_sidon_size = sidon_size
                best_set = list(current)
                current_sidon = sidon_size
            else:
                current[idx] = old_val
    
    # Strategy 5: Structured sets with multiple collision points
    # Use a pattern where values cluster around specific sum targets
    for base in range(0, 100):
        if time.time() > deadline:
            break
        
        # Create clusters around base
        for spread in range(1, 8):
            candidate = []
            used = set()
            
            # Add values in a pattern that creates sum collisions
            for i in range(6):
                # Pair i: (base + i*spread, base + (5-i)*spread)
                a = base + i * spread
                b = base + (5 - i) * spread
                if a != b and a not in used and b not in used and 0 <= a <= 128 and 0 <= b <= 128:
                    candidate.append(a)
                    candidate.append(b)
                    used.add(a)
                    used.add(b)
            
            if len(candidate) == 12:
                evaluate(candidate)
    
    # Final fallback
    if best_set is None or len(set(best_set)) != 12 or any(x < 0 or x > 128 for x in best_set):
        return list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
