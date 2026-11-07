"""Rewritten program: cardinality-separated subset sums, maximize set size."""
import random
import time

def construct():
    """Return a list of distinct ints in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize size."""
    deadline = time.time() + 55  # 55s safety margin within 60s limit
    rng = random.Random(42)

    # Known valid set of size 9 (powers of 2)
    base9 = [1, 2, 4, 8, 16, 32, 64, 128, 2]  # need 9 distinct; use 1,2,4,8,16,32,64,128 + another
    # Actually powers of 2 up to 2^7 = 128, that's 8 elements.
    # 1,2,4,8,16,32,64,128 -> 8 elements. Add a 9th: need distinct, in 1..256.
    # Let's use: 1, 2, 4, 8, 16, 32, 64, 128, 3
    # Check: subset sums for different cardinalities must differ.
    # With powers of 2, each subset sum uniquely identifies the subset, so all
    # cardinalities have distinct sums automatically. But adding 3 breaks uniqueness.
    # Let me just use 9 elements: 1,2,4,8,16,32,64,128, and one more.
    # Actually for the base, let me use a verified valid set.
    # The simplest: {1, 3, 9, 27, 81, 243} is size 6 with all subset sums unique (base 3).
    # For larger sizes, we need to search.
    
    # Let me use a different approach: start with a greedy construction.
    # Use the property: if we pick elements greedily such that each new element
    # is larger than the sum of all previous elements, then all subset sums are unique.
    # 1, 2, 4, 8, 16, 32, 64, 128 -> 8 elements, next would be 256 (in range!).
    # So {1,2,4,8,16,32,64,128,256} is size 9 with all subset sums unique.
    # That's 9 elements!
    
    base_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Verify: all subset sums are unique (superincreasing sequence).
    # So any two subsets with different cardinalities have different sums. ✓
    
    best_size = len(base_set)
    best_set = list(base_set)
    
    # Try to find a valid set of size 10
    # Strategy: local search with collision counting
    # A set is valid iff: for every pair of subsets S1, S2 with |S1| != |S2|, sum(S1) != sum(S2)
    # Equivalent: for each cardinality k, the set of subset sums of size k is disjoint from
    # the set of subset sums of size k' for k' != k.
    
    def check_valid(arr):
        """Check if arr satisfies the cardinality-separated subset sum property."""
        n = len(arr)
        # Compute subset sums by cardinality
        # sums[k] = set of sums of subsets of size k
        sums = [set() for _ in range(n + 1)]
        sums[0].add(0)
        for x in arr:
            # Iterate in reverse to avoid using x multiple times
            for k in range(n, 0, -1):
                for s in list(sums[k - 1]):
                    sums[k].add(s + x)
        # Check disjointness across different cardinalities
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums[i] & sums[j]:
                    return False
        return True
    
    def collision_count(arr):
        """Count number of cross-cardinality sum collisions. 0 means valid."""
        n = len(arr)
        sums = [set() for _ in range(n + 1)]
        sums[0].add(0)
        for x in arr:
            for k in range(n, 0, -1):
                for s in list(sums[k - 1]):
                    sums[k].add(s + x)
        count = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                count += len(sums[i] & sums[j])
        return count
    
    # Try local search for size 10
    def local_search_for_10():
        """Try to find a valid set of size 10."""
        # Start with base 9 + one random element
        used = set(base_set)
        candidates = [v for v in range(1, 257) if v not in used]
        rng.shuffle(candidates)
        
        for extra in candidates[:50]:
            arr = base_set + [extra]
            if check_valid(arr):
                return arr
            # Try local search from this starting point
            if time.time() > deadline:
                return None
            # Simulated annealing style
            current = arr[:]
            current_coll = collision_count(current)
            if current_coll == 0:
                return current
            T = 50.0
            used_vals = set(current)
            for step in range(5000):
                if time.time() > deadline:
                    return None
                i = rng.randint(0, 9)
                # Pick a new value not in current
                avail = [v for v in range(1, 257) if v not in used_vals]
                if not avail:
                    break
                new_val = rng.choice(avail)
                old_val = current[i]
                current[i] = new_val
                used_vals.remove(old_val)
                used_vals.add(new_val)
                new_coll = collision_count(current)
                if new_coll == 0:
                    return current
                delta = new_coll - current_coll
                if delta < 0 or rng.random() < (10 ** (-delta / max(T, 0.1))):
                    current_coll = new_coll
                else:
                    # Revert
                    current[i] = old_val
                    used_vals.remove(new_val)
                    used_vals.add(old_val)
                T *= 0.99
        return None
    
    result_10 = local_search_for_10()
    if result_10 is not None:
        best_set = result_10
        best_size = 10
    
    # If we still have time, try for size 11
    if best_size == 10 and time.time() < deadline - 5:
        # Start from the size-10 solution, try to add an 11th element
        used = set(best_set)
        for v in range(1, 257):
            if v not in used:
                arr11 = best_set + [v]
                if check_valid(arr11):
                    best_set = arr11
                    best_size = 11
                    break
    
    # Ensure all values are Python int
    best_set = [int(x) for x in best_set]
    return best_set
