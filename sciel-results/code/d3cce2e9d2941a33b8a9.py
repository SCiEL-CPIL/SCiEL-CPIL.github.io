import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a given subset has all unique pair sums including diagonals."""
    n = len(subset)
    sums = set()
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size_fast(candidate_set):
    """Find the size of the largest Sidon subset using greedy + local search."""
    n = len(candidate_set)
    cs = sorted(candidate_set)
    
    # First, try to find a large Sidon subset using greedy approach
    # Then try to extend it
    
    # Greedy: add elements that don't create collisions
    def greedy_sidon_start(cs):
        """Greedy construction of a Sidon subset."""
        subset = []
        sums = set()
        for x in cs:
            ok = True
            for y in subset:
                s = x + y
                if s in sums:
                    ok = False
                    break
            if ok:
                # Check diagonal
                s = x + x
                if s in sums:
                    ok = False
                else:
                    subset.append(x)
                    sums.add(s)
                    for y in subset[:-1]:
                        sums.add(x + y)
        return subset
    
    # Try multiple greedy starts with different orderings
    best_subset = []
    
    # Standard greedy
    g1 = greedy_sidon_start(cs)
    if len(g1) > len(best_subset):
        best_subset = g1
    
    # Greedy from largest
    g2 = greedy_sidon_start(list(reversed(cs)))
    if len(g2) > len(best_subset):
        best_subset = g2
    
    # Try to extend best_subset by adding elements
    def try_extend(subset, cs):
        """Try to add more elements to the subset."""
        sums = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                sums.add(subset[i] + subset[j])
        
        current = list(subset)
        for x in cs:
            if x in current:
                continue
            ok = True
            for y in current:
                s = x + y
                if s in sums:
                    ok = False
                    break
            if ok:
                s = x + x
                if s in sums:
                    ok = False
                else:
                    current.append(x)
                    sums.add(s)
                    for y in current[:-1]:
                        sums.add(x + y)
        return current
    
    extended = try_extend(best_subset, cs)
    if len(extended) > len(best_subset):
        best_subset = extended
    
    # Now try local search: swap elements to potentially get a larger Sidon subset
    # This is a heuristic - we try to find a Sidon subset of size > len(best_subset)
    base_size = len(best_subset)
    
    # If we already have 6+, check if we can do better with targeted search
    # For small sizes, brute force from that size down
    if base_size >= 5:
        # Try to find if there's a Sidon subset of size base_size + 1
        # Use a more targeted approach
        for _ in range(20):
            # Randomly perturb and try greedy
            random.shuffle(cs)
            g = greedy_sidon_start(cs)
            if len(g) > len(best_subset):
                best_subset = g
                base_size = len(g)
                break
    
    # Final verification: check if best_subset is actually Sidon
    if not is_sidon(best_subset):
        # Fallback to brute force from smaller sizes
        for k in range(base_size - 1, 0, -1):
            for subset in combinations(cs, k):
                if is_sidon(subset):
                    return k
        return base_size - 1
    
    return base_size

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_set = list(range(12))
    best_size = max_sidon_size_fast(best_set)
    
    # Structured candidates designed to force pair-sum collisions
    # Key insight: arithmetic progressions and dense clusters create many sum collisions
    candidates = [
        list(range(12)),  # Consecutive: many collisions
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32],  # Block structure
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],  # Tighter blocks
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],  # Two blocks
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],  # Two blocks
        [0, 1, 3, 7, 12, 18, 25, 33, 42, 52, 63, 75],
        [0, 1, 4, 10, 20, 40, 80, 2, 5, 15, 35, 75],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # Even spaced
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17],  # Periodic blocks
        # Additional structured candidates
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],  # One outlier
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
        [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15],
        # New candidates with more collision potential
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # baseline
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14],  # slight shift
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],  # gap in middle
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],  # gap near end
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13],  # larger gap
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],  # even larger gap
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],  # two gaps
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13],  # three elements shifted
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],  # cluster at end
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13],  # mixed gaps
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],  # larger gaps
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14],  # spread out end
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13],  # alternating gaps
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14],  # more spread
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13],  # gap in middle
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13],  # gap after 4
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],  # gap after 3
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # gap after 2
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # gap after 1
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # gap after 0
    ]
    
    for c in candidates:
        if len(set(c)) != 12: continue
        if any(x < 0 or x > 128 for x in c): continue
        sz = max_sidon_size_fast(c)
        if sz < best_size:
            best_size = sz
            best_set = list(c)
    
    # Local search with improved mutation strategies
    random.seed(42)
    current_set = list(best_set)
    current_size = best_size
    
    def try_swap(s):
        """Try swapping elements to create more sum collisions."""
        best_s = list(s)
        best_sz = max_sidon_size_fast(s)
        s_set = set(s)
        for _ in range(30):
            new_s = list(s)
            i, j = random.sample(range(12), 2)
            # Try values that are close to existing elements to create collisions
            # Or values that would make sum collisions with other pairs
            new_val = random.randint(0, 128)
            if new_val in s_set:
                continue
            new_s[j] = new_val
            new_sz = max_sidon_size_fast(new_s)
            if new_sz < best_sz:
                best_sz = new_sz
                best_s = new_s
                s_set = set(new_s)
        return best_s, best_sz
    
    def try_targeted_collision(s):
        """Try to create additional sum collisions by targeting specific pairs."""
        best_s = list(s)
        best_sz = max_sidon_size_fast(s)
        s_set = set(s)
        for _ in range(20):
            new_s = list(s)
            idx = random.randint(0, 11)
            # Try to create a sum collision: find two pairs (a,b) and (c,d) where a+b=c+d
            # We can try to adjust one element to create such a collision
            other_idx = random.randint(0, 11)
            if other_idx == idx:
                continue
            third_idx = random.randint(0, 11)
            if third_idx in (idx, other_idx):
                continue
            
            # Try to make s[idx] + s[other_idx] = s[third_idx] + s[third_idx]
            target_sum = s[third_idx] + s[third_idx]
            candidate = target_sum - s[other_idx]
            
            if candidate != s[idx] and 0 <= candidate <= 128 and candidate not in s_set:
                new_s[idx] = candidate
                new_sz = max_sidon_size_fast(new_s)
                if new_sz < best_sz:
                    best_sz = new_sz
                    best_s = new_s
                    s_set = set(new_s)
        return best_s, best_sz
    
    # Simulated annealing style local search
    while time.time() < deadline:
        # Try multiple mutation rounds
        for _ in range(15):
            if time.time() >= deadline:
                break
            
            # Alternate between different mutation strategies
            strategy = random.random()
            if strategy < 0.4:
                mutated, mut_sz = try_swap(current_set)
            elif strategy < 0.7:
                mutated, mut_sz = try_targeted_collision(current_set)
            else:
                # Random restart from current
                mutated = random.sample(range(129), 12)
                mut_sz = max_sidon_size_fast(mutated)
            
            if mut_sz < current_size:
                current_set = mutated
                current_size = mut_sz
                if current_size < best_size:
                    best_size = current_size
                    best_set = list(current_set)
                    if best_size <= 3:
                        break
        if best_size <= 3:
            break
    
    # Final random restarts if time remains
    if time.time() < deadline - 5:
        while time.time() < deadline - 2:
            s = random.sample(range(129), 12)
            sz = max_sidon_size_fast(s)
            if sz < best_size:
                best_size = sz
                best_set = s
                if best_size <= 3:
                    break
    
    return best_set
