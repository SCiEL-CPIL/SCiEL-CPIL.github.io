"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys
import numpy as np
from collections import defaultdict

def _triple_sums_involving_x(x, S):
    """Return the set of triple sums that involve x, given x is NOT in S."""
    sums = set()
    n = len(S)
    sums.add(3 * x)
    for y in S:
        sums.add(2 * x + y)
        sums.add(x + 2 * y)
    for i in range(n):
        for j in range(i + 1, n):
            sums.add(x + S[i] + S[j])
    return sums

def _greedy_build_bitset(deadline, seed=0, order=None):
    """Greedy build using a numpy bitset for fast collision checks."""
    rng = random.Random(seed)
    max_val = 4096
    # Bitset for existing sums: max triple sum is 3*4096 = 12288
    max_sum = 3 * max_val
    existing = np.zeros(max_sum + 1, dtype=np.uint8)
    S = []
    
    if order is None:
        order = list(range(1, max_val + 1))
    
    for x in order:
        if time.time() > deadline:
            break
        
        n = len(S)
        if n == 0:
            S.append(x)
            existing[3 * x] = 1
            continue
        
        # Compute new sums as a numpy array for fast intersection
        # Sums: 3x, 2x+y for y in S, x+2y for y in S, x+S[i]+S[j] for i<j
        new_sums_list = [3 * x]
        for y in S:
            new_sums_list.append(2 * x + y)
            new_sums_list.append(x + 2 * y)
        S_arr = np.array(S, dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                new_sums_list.append(x + S[i] + S[j])
        
        new_sums_arr = np.array(new_sums_list, dtype=np.int32)
        
        # Check for internal collisions (within new_sums)
        if len(new_sums_arr) != len(np.unique(new_sums_arr)):
            continue
        
        # Check against existing sums using bitset
        # Only check indices that are in range
        valid_mask = (new_sums_arr >= 0) & (new_sums_arr <= max_sum)
        if not np.all(valid_mask):
            continue
        
        # Check intersection with existing
        if np.any(existing[new_sums_arr] == 1):
            continue
        
        # Accept x
        existing[new_sums_arr] = 1
        S.append(x)
    
    return S

def _compute_all_triple_sums(S):
    """Compute all triple sums for set S. Returns a dict mapping sum -> list of triples."""
    sums_dict = defaultdict(list)
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                sums_dict[s].append((i, j, k))
    return sums_dict

def _greedy_build(deadline, seed=0):
    """Optimized greedy build. Returns the largest valid set found."""
    S = []
    existing_sums = set()
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        
        n = len(S)
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
            new_sums.add(x + 2 * y)
        for i in range(n):
            for j in range(i + 1, n):
                new_sums.add(x + S[i] + S[j])
        
        expected = 1 + n + n + n * (n - 1) // 2
        if len(new_sums) < expected:
            continue
        if new_sums & existing_sums:
            continue
        
        S.append(x)
        existing_sums |= new_sums
    
    return S

def _greedy_with_backtracking(deadline, seed=0, max_depth=3):
    """Greedy with limited backtracking: try alternative choices at each step."""
    rng = random.Random(seed)
    best = []
    
    def _greedy_from(S, existing_sums, start_x, remaining_time, depth):
        """Greedy from state (S, existing_sums) starting at x=start_x."""
        if time.time() > remaining_time:
            return S[:]
        
        current_best = S[:]
        current_existing = existing_sums.copy()
        
        for x in range(start_x, 4097):
            if time.time() > remaining_time:
                break
            
            n = len(current_best)
            new_sums = set()
            new_sums.add(3 * x)
            for y in current_best:
                new_sums.add(2 * x + y)
                new_sums.add(x + 2 * y)
            for i in range(n):
                for j in range(i + 1, n):
                    new_sums.add(x + current_best[i] + current_best[j])
            
            expected = 1 + n + n + n * (n - 1) // 2
            if len(new_sums) < expected:
                continue
            if new_sums & current_existing:
                continue
            
            current_best.append(x)
            current_existing |= new_sums
            
            # Try to continue greedy from here
            if depth > 0:
                extended = _greedy_from(current_best[:], current_existing, x + 1, remaining_time, depth - 1)
                if len(extended) > len(best):
                    best.extend([])  # clear
                    best[:] = extended
            
            if len(current_best) > len(best):
                best[:] = current_best[:]
        
        return current_best
    
    _greedy_from([], set(), 1, deadline, max_depth)
    return best

def _local_search(S, deadline, seed=0):
    """Hill-climbing: try to add elements or swap to increase cardinality."""
    rng = random.Random(seed)
    best = S[:]
    best_size = len(S)
    
    # Build current triple sums set
    current_sums = set()
    for x in best:
        current_sums |= _triple_sums_involving_x(x, best)
    
    # Try adding elements
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if x in set(best):
            continue
        
        new_sums = _triple_sums_involving_x(x, best)
        expected = 1 + len(best) + len(best) + len(best) * (len(best) - 1) // 2
        if len(new_sums) < expected:
            continue
        if new_sums & current_sums:
            continue
        
        # Try adding x
        new_best = best + [x]
        new_sums_set = current_sums | new_sums
        # Verify validity
        all_sums = set()
        for y in new_best:
            all_sums |= _triple_sums_involving_x(y, [z for z in new_best if z != y] or new_best)
        
        # Simpler: just check no collisions
        if len(new_sums_set) == len(current_sums) + len(new_sums - current_sums):
            best = new_best
            current_sums = new_sums_set
            best_size = len(best)
    
    # Try swaps: remove one element, add another
    for _ in range(200):
        if time.time() > deadline:
            break
        
        if not best:
            break
        
        # Pick a random element to remove
        idx = rng.randint(0, len(best) - 1)
        removed = best[idx]
        S_without = best[:idx] + best[idx+1:]
        
        # Compute sums without removed element
        sums_without = set()
        for y in S_without:
            sums_without |= _triple_sums_involving_x(y, S_without)
        
        # Try to add a new element
        for x in range(1, 4097):
            if time.time() > deadline:
                break
            if x in set(S_without):
                continue
            
            new_sums = _triple_sums_involving_x(x, S_without)
            expected = 1 + len(S_without) + len(S_without) + len(S_without) * (len(S_without) - 1) // 2
            if len(new_sums) < expected:
                continue
            if new_sums & sums_without:
                continue
            
            new_best = S_without + [x]
            if len(new_best) > best_size:
                best = new_best
                best_size = len(best)
                break
            elif len(new_best) == best_size and rng.random() < 0.1:
                # Accept with small probability for diversity
                best = new_best
                break
    
    return best

def _greedy_with_randomization(deadline, seed=0):
    """Greedy with randomized ordering using bitset for speed."""
    rng = random.Random(seed)
    best = []
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Different randomization strategies
        strategy = attempt % 5
        if strategy == 0:
            # Full shuffle
            order = list(range(1, 4097))
            rng.shuffle(order)
        elif strategy == 1:
            # Small first, then shuffled
            order = list(range(1, 4097))
            small = order[:50]
            large = order[50:]
            rng.shuffle(large)
            order = small + large
        elif strategy == 2:
            # Random start point, then increasing
            start = rng.randint(1, 500)
            order = list(range(start, 4097))
            remaining = list(range(1, start))
            rng.shuffle(remaining)
            order = remaining + order
        elif strategy == 3:
            # Weighted: prefer medium values
            order = list(range(1, 4097))
            # Sort by distance from 2048, shuffle within bands
            bands = {}
            for x in order:
                band = abs(x - 2048) // 200
                bands.setdefault(band, []).append(x)
            new_order = []
            for band in sorted(bands.keys()):
                rng.shuffle(bands[band])
                new_order.extend(bands[band])
            order = new_order
        else:
            # Pure random
            order = list(range(1, 4097))
            rng.shuffle(order)
        
        S = _greedy_build_bitset(deadline, seed=seed + attempt, order=order)
        
        if len(S) > len(best):
            best = S[:]
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best = []
    
    # Strategy 1: Greedy in increasing order (bitset version)
    S1 = _greedy_build_bitset(deadline, seed=42)
    if len(S1) > len(best):
        best = S1[:]
    
    # Strategy 2: Randomized greedy with diverse orderings
    if time.time() < deadline:
        S2 = _greedy_with_randomization(deadline, seed=456)
        if len(S2) > len(best):
            best = S2[:]
    
    # Strategy 3: Local search to improve
    if time.time() < deadline:
        S3 = _local_search(best, deadline, seed=123)
        if len(S3) > len(best):
            best = S3[:]
    
    # Strategy 4: More randomized greedy attempts with time remaining
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        S4 = _greedy_with_randomization(deadline, seed=789)
        if len(S4) > len(best):
            best = S4[:]
    
    # Final local search on best
    if time.time() < deadline:
        S5 = _local_search(best, deadline, seed=321)
        if len(S5) > len(best):
            best = S5[:]
    
    return best
