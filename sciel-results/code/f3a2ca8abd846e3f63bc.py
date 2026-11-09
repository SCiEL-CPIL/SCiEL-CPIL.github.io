import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0
    N = 4096
    MAX_SIZE = 80

    def fast_check_add(arr, x, triple_sums):
        """Check if x can be added to arr without creating duplicate triple sums."""
        n = len(arr)
        # (x, x, x)
        if 3 * x in triple_sums:
            return False
        # (x, x, y) for each y in arr
        for i in range(n):
            if 2 * x + arr[i] in triple_sums:
                return False
        # (x, y, z) for y <= z in arr
        for i in range(n):
            for j in range(i, n):
                if x + arr[i] + arr[j] in triple_sums:
                    return False
        return True

    def compute_triple_sums(arr):
        """Compute the set of all unordered triple sums for arr."""
        sums = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(arr[i] + arr[j] + arr[k])
        return sums

    def build_greedy(order, max_size=MAX_SIZE, deadline=None):
        """Greedy: iterate through order, add element if triple sums remain unique."""
        selected = []
        triple_sums = set()
        
        for x in order:
            if len(selected) >= max_size:
                break
            if deadline is not None and time.time() > deadline:
                break
            
            valid = True
            new_sums = set()
            
            # (x, x, x)
            s = 3 * x
            if s in triple_sums:
                valid = False
            else:
                new_sums.add(s)
            
            if valid:
                # (x, x, y) for each y in selected
                for y in selected:
                    s = 2 * x + y
                    if s in triple_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            
            if valid:
                # (x, y, z) for y <= z in selected
                n_sel = len(selected)
                for i in range(n_sel):
                    for j in range(i, n_sel):
                        s = x + selected[i] + selected[j]
                        if s in triple_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected

    def local_improvement(arr, max_size=MAX_SIZE, deadline=None):
        """Try to improve by adding new elements."""
        if not arr:
            return arr
        
        arr = list(arr)
        arr_set = set(arr)
        
        # Precompute existing triple sums
        triple_sums = compute_triple_sums(arr)
        
        # Try to add more elements - scan multiple times
        for _round in range(3):
            if len(arr) >= max_size:
                break
            if deadline is not None and time.time() > deadline:
                break
            
            for x in range(1, N + 1):
                if len(arr) >= max_size:
                    break
                if deadline is not None and time.time() > deadline:
                    break
                if x in arr_set:
                    continue
                
                if fast_check_add(arr, x, triple_sums):
                    n_old = len(arr)
                    arr.append(x)
                    arr_set.add(x)
                    # Update triple sums
                    for i in range(n_old):
                        for j in range(i, n_old):
                            triple_sums.add(x + arr[i] + arr[j])
                    for i in range(n_old):
                        triple_sums.add(2 * x + arr[i])
                    triple_sums.add(3 * x)
        
        return arr

    # Strategy 1: Greedy with base-3 sparsity ordering
    def base3_sparsity_score(x):
        digits = []
        while x > 0:
            digits.append(x % 3)
            x //= 3
        if not digits:
            return 0
        non_zero_positions = [i for i, d in enumerate(digits) if d != 0]
        if len(non_zero_positions) <= 1:
            return 100 - len(digits)
        max_gap = 0
        for i in range(1, len(non_zero_positions)):
            gap = non_zero_positions[i] - non_zero_positions[i-1] - 1
            max_gap = max(max_gap, gap)
        return max_gap * 10 - len(digits)
    
    order1 = sorted(range(1, N + 1), key=lambda x: -base3_sparsity_score(x))
    best = build_greedy(order1, deadline=DEADLINE - 20)
    best_size = len(best)
    
    # Strategy 2: Simple greedy from 1
    if time.time() < DEADLINE - 20:
        candidate = build_greedy(range(1, N + 1), deadline=DEADLINE - 20)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Strategy 3: Greedy with log-spacing preference
    if time.time() < DEADLINE - 20:
        def log_spacing_score(x):
            best_dist = float('inf')
            for i in range(1, 13):
                p = 3 ** i
                if p <= N:
                    best_dist = min(best_dist, abs(x - p))
            return -best_dist
        
        order3 = sorted(range(1, N + 1), key=lambda x: log_spacing_score(x))
        candidate = build_greedy(order3, deadline=DEADLINE - 20)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Strategy 4: Local improvement on best
    if time.time() < DEADLINE - 15:
        improved = local_improvement(best, deadline=DEADLINE - 15)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Strategy 5: Multiple random restarts
    if time.time() < DEADLINE - 10:
        random.seed(42)
        for trial in range(15):
            if time.time() > DEADLINE - 10:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            candidate = build_greedy(order, deadline=DEADLINE - 10)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
    
    # Final local improvement
    if time.time() < DEADLINE - 5:
        improved = local_improvement(best, deadline=DEADLINE - 5)
        if len(improved) > best_size:
            best = improved
    
    # Another round of random restarts with different seed
    if time.time() < DEADLINE - 3:
        random.seed(123)
        for trial in range(10):
            if time.time() > DEADLINE - 3:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            # Bias: interleave small and large
            small_half = [x for x in order if x <= N // 2]
            large_half = [x for x in order if x > N // 2]
            interleaved = []
            i = j = 0
            while i < len(small_half) or j < len(large_half):
                if i < len(small_half):
                    interleaved.append(small_half[i])
                    i += 1
                if j < len(large_half):
                    interleaved.append(large_half[j])
                    j += 1
            candidate = build_greedy(interleaved, deadline=DEADLINE - 3)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
    
    return best
