"""
Construct an ambient set of 12 distinct integers in [0,128] that minimizes
the size of the largest Sidon subset (including diagonal sums).
"""
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # safety margin before 60s limit

    def check_sidon(subset):
        """Check if subset is a Sidon set including diagonal sums.
        A Sidon set with diagonals: all pair sums a_i + a_j for i <= j are unique.
        """
        sums = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def max_sidon_size(ambient):
        """Find the maximum size of a Sidon subset (with diagonals) in ambient set."""
        n = len(ambient)
        # Try from largest possible size down
        # Max possible Sidon size: for a set of size k with diagonals,
        # we need k(k+1)/2 distinct sums, each in [2*min, 2*max]
        # For ambient in [0,128], max sum = 256, so k(k+1)/2 <= 257 -> k <= 22
        # But we're looking for small sizes, so try from some reasonable max down
        # Actually, let's be smarter: try sizes from 6 down to 4 first
        
        # We want to minimize the max Sidon subset size, so we check
        # whether there exists a Sidon subset of size k for k from high to low
        # and return the largest such k
        
        # First check if the whole ambient is Sidon
        if check_sidon(ambient):
            return n
        
        # Try decreasing sizes
        # For 12 elements, the max Sidon size is at most... let's just iterate
        for k in range(n, 3, -1):
            # Check all subsets of size k
            found = False
            for subset in itertools.combinations(range(n), k):
                vals = [ambient[i] for i in subset]
                if check_sidon(vals):
                    found = True
                    break
            if found:
                return k
        return 3  # any 3 elements might not be Sidon, but 2 always are
    
    # Actually, for 12 elements, checking all C(12,k) subsets is feasible
    # C(12,6) = 924, C(12,5) = 792, C(12,4) = 495, etc.
    
    # Strategy: use randomized local search
    # Start with a candidate, evaluate its max Sidon size, then try swaps
    
    def evaluate(ambient):
        """Return the max Sidon subset size for this ambient set."""
        n = len(ambient)
        # Check from size 6 down (unlikely to be larger for 12 elements in [0,128])
        # Actually C(12,7) = 792, C(12,8) = 495, still manageable
        for k in range(n, 1, -1):
            count = 0
            for subset in itertools.combinations(range(n), k):
                vals = [ambient[i] for i in subset]
                if check_sidon(vals):
                    return k
        return 2
    
    # Better evaluation: we want to MINIMIZE the max Sidon subset size
    # So a good ambient set has NO large Sidon subsets
    # The minimum possible max Sidon size for 12 elements is at least 4
    # (any 4 elements can potentially be Sidon)
    # Actually, can we force max Sidon size to be 3 or less?
    # If no 4-element subset is Sidon, then max is 3 or less.
    
    def no_four_sidon(ambient):
        """Check if no 4-element subset is Sidon."""
        n = len(ambient)
        for subset in itertools.combinations(range(n), 4):
            vals = [ambient[i] for i in subset]
            if check_sidon(vals):
                return False
        return True
    
    # Target: find 12 integers in [0,128] such that NO 4-element subset is Sidon
    # This would give max Sidon size <= 3
    
    # If we can't achieve that, try to maximize the number of 4-element non-Sidon subsets
    
    def count_four_sidon(ambient):
        """Count how many 4-element subsets ARE Sidon."""
        n = len(ambient)
        count = 0
        for subset in itertools.combinations(range(n), 4):
            vals = [ambient[i] for i in subset]
            if check_sidon(vals):
                count += 1
        return count
    
    # Start with a random or structured initial set
    # A structured approach: use arithmetic-like progressions with perturbations
    # to create many sum collisions
    
    # Key insight: if we use elements that are close together, many pairs will
    # have overlapping sums. For example, if all elements are in a small range,
    # sums will collide.
    
    # Try: elements clustered in two groups to force sum collisions
    # Group 1: {a, a+1, a+2, ..., a+5}
    # Group 2: {b, b+1, b+2, ..., b+5} where b - a is small
    
    # Let's do a local search
    best_ambient = None
    best_score = 999
    
    rng = random.Random(42)
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: tight cluster
    candidates.append(list(range(0, 12)))
    
    # Candidate 2: two tight clusters
    candidates.append([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    
    # Candidate 3: even tighter
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Candidate 4: mixed
    candidates.append([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
    
    # Candidate 5: arithmetic with step 1 but spread
    candidates.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    
    # Random candidates
    for _ in range(20):
        candidates.append(sorted(rng.sample(range(129), 12)))
    
    for cand in candidates:
        if time.time() > deadline:
            break
        score = count_four_sidon(cand)
        if score < best_score:
            best_score = score
            best_ambient = list(cand)
    
    # Local search: try swapping elements
    if best_ambient is not None and time.time() < deadline:
        current = list(best_ambient)
        current_score = best_score
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            # Try random swaps
            i = rng.randrange(12)
            j = rng.randrange(12)
            if i == j:
                continue
            
            old_val = current[j]
            new_val = rng.randrange(129)
            
            # Check if new_val is already in the set
            if new_val in current:
                continue
            
            current[j] = new_val
            score = count_four_sidon(current)
            
            if score < current_score:
                current_score = score
                best_score = score
                best_ambient = list(current)
            else:
                current[j] = old_val
            
            # Also try moving an element to a random position
            if rng.random() < 0.5:
                new_val2 = rng.randrange(129)
                if new_val2 in current:
                    continue
                current[j] = new_val2
                score2 = count_four_sidon(current)
                if score2 < current_score:
                    current_score = score2
                    best_score = score2
                    best_ambient = list(current)
                else:
                    current[j] = old_val
    
    # If we found a set with no 4-Sidon subsets, great.
    # Otherwise, try to improve further.
    
    # Final check: verify the result
    result = best_ambient if best_ambient is not None else list(range(12))
    
    # Ensure distinct integers in [0, 128]
    assert len(result) == 12
    assert len(set(result)) == 12
    assert all(0 <= x <= 128 for x in result)
    
    return result
