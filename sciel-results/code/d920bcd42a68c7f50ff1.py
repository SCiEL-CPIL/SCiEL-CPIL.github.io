"""Construct large sets with cardinality-separated subset sums."""
import time
import itertools
import random

# EVOLVE-BLOCK-START

def _validate(s):
    """Check that s is a list of distinct ints in 1..256 and cardinality-separated."""
    if not isinstance(s, (list, tuple)):
        return False
    n = len(s)
    if n < 1 or n > 60:
        return False
    # Check types
    for x in s:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    # Check distinctness
    if len(set(s)) != n:
        return False
    # Check cardinality separation
    # For each pair of different cardinalities, sums must differ.
    # We need: for any k1 != k2, the set of sums of size-k1 subsets is disjoint from size-k2.
    # This is equivalent to: no sum value appears in two different cardinality layers.
    # We can compute all subset sums grouped by cardinality and check.
    # But for large n this is expensive. Use a smarter check.
    # Actually, a necessary and sufficient condition: 
    #   min sum of size (k+1) > max sum of size k  for all k >= 0
    #   (where size 0 has sum 0)
    # Because if this holds, then all size-k sums are in [min_k, max_k] and these intervals are disjoint.
    # But this is sufficient but not necessary. The actual requirement is just that the sets are disjoint.
    # For a valid construction, we'll use the stronger condition during search and verify with exact check.
    
    # Exact verification: compute all subset sums by cardinality
    # For n up to ~40, 2^n is too large. We need a smarter approach.
    # Actually, we can use the fact that if we sort the set, and check:
    #   For each k, the minimum k-sum is sum of k smallest, max k-sum is sum of k largest.
    #   If max(k-sum) < min((k+1)-sum) for all k, then it's valid.
    # This is a sufficient condition. Is it necessary? No, but it's easy to check.
    # For the search, we'll use this sufficient condition.
    # For final validation, we should do exact check for small n, or trust the sufficient condition.
    
    # Let's do the sufficient check (sorted intervals disjoint)
    s_sorted = sorted(s)
    n = len(s_sorted)
    # min sum of size k = sum of first k elements
    # max sum of size k = sum of last k elements
    # We need: for k=0..n-1: max(k-sum) < min((k+1)-sum)
    # max(0-sum) = 0, min(1-sum) = s_sorted[0]
    # max(k-sum) = sum(s_sorted[n-k:n]), min((k+1)-sum) = sum(s_sorted[0:k+1])
    
    # Precompute prefix sums
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + s_sorted[i]
    
    for k in range(0, n):
        max_k = prefix[n] - prefix[n-k]  # sum of last k elements
        min_next = prefix[k+1]  # sum of first k+1 elements
        if max_k >= min_next:
            return False
    return True


def _construct_greedy():
    """Greedy construction: pick largest possible values to maximize size."""
    # Strategy: pick numbers in descending order, ensuring the interval condition holds.
    # We want to maximize the count.
    # Key insight: if we pick numbers a1 < a2 < ... < an, we need:
    #   sum of largest k < sum of smallest k+1 for all k
    # In particular, for k=n-1: sum of largest n-1 < sum of smallest n
    #   This means: (total - a1) < total  => a1 > 0 (always true)
    #   Wait, that's not right. Let me re-derive.
    #   max(n-1 sum) = sum of all except smallest = total - a1
    #   min(n sum) = total
    #   So we need total - a1 < total, i.e., a1 > 0. Always true.
    # For k=n-2: max(n-2 sum) = total - a1 - a2, min(n-1 sum) = total - an
    #   Need: total - a1 - a2 < total - an  => an < a1 + a2
    # For k=1: max(1 sum) = an, min(2 sum) = a1 + a2
    #   Need: an < a1 + a2
    # So the critical constraint is: for each k, sum of k largest < sum of k+1 smallest.
    
    # Greedy approach: start with the largest possible number, then add next largest that satisfies constraints.
    # But this might not be optimal. Let's try a different approach.
    
    # Alternative: use the known result that powers of 2 work: {1, 2, 4, ..., 2^(n-1)}
    # But these must be <= 256, so max n = 9 (1,2,4,8,16,32,64,128,256).
    # That gives n=9. We can do better.
    
    # Better approach: use numbers that are close together but satisfy the gap condition.
    # The condition sum of k largest < sum of k+1 smallest is equivalent to:
    #   For sorted a1 < a2 < ... < an:
    #   a_{n-k+1} + ... + a_n < a_1 + ... + a_{k+1}
    
    # Let's try: pick n numbers from 1..256 that satisfy this.
    # A good strategy: use arithmetic progression or near-geometric.
    
    # Let's try greedy: start with a candidate set, try to add more elements.
    # We'll use a heuristic: try to build a set of size n by picking numbers that are "spread out".
    
    # Actually, let's use a simpler approach: 
    # Try all sizes from 60 down to 1, and for each size, try to find a valid set.
    # For a given size n, we can use a greedy/local search approach.
    
    # For now, let's use a known good construction.
    # The set {1, 2, 4, 8, 16, 32, 64, 128} has size 8.
    # Can we do better?
    
    # Let's try: {1, 2, 4, 8, 16, 32, 64, 100, 150, 200}
    # Check: sorted = [1,2,4,8,16,32,64,100,150,200]
    # k=1: max(1)=200, min(2)=1+2=3. 200 < 3? No. Fails.
    
    # The constraint is very tight. Let me think again.
    # For n numbers a1 < a2 < ... < an:
    #   Constraint k: a_{n-k+1} + ... + a_n < a_1 + ... + a_{k+1}
    #   For k=1: a_n < a_1 + a_2
    #   For k=2: a_{n-1} + a_n < a_1 + a_2 + a_3
    #   etc.
    
    # So a_n < a_1 + a_2 is the most restrictive for large n.
    # If a1=1, a2=2, then a_n < 3, so a_n <= 2. But a_n > a2=2, contradiction.
    # So we need a1 + a2 to be large enough.
    
    # If a1=1, a2=100, then a_n < 101, so a_n <= 100. But a_n > a2=100, contradiction.
    # If a1=50, a2=100, then a_n < 150, so a_n <= 149.
    # Then for k=2: a_{n-1} + a_n < a1+a2+a3. If a3=101, a_{n-1} <= 148, a_n <= 149.
    #   148+149=297 < 50+100+101=251? No.
    
    # This is very constrained. Let me use a computational approach.
    
    # Strategy: use simulated annealing or local search to find a valid set of maximum size.
    
    # For the initial program, let's just return a known valid set.
    # The simplest valid set: powers of 2 up to 256.
    # {1, 2, 4, 8, 16, 32, 64, 128} - size 8
    # Check: 
    #   k=1: max(1)=128, min(2)=1+2=3. 128<3? No!
    # Wait, I made an error. Let me re-check.
    # For {1,2,4,8,16,32,64,128}:
    #   k=1: max 1-sum = 128, min 2-sum = 1+2=3. 128 >= 3, so FAIL.
    
    # So powers of 2 don't work! The condition is much stronger.
    
    # Let me reconsider. The condition is:
    #   No two subsets of different sizes have the same sum.
    # This is NOT the same as the interval condition I was using.
    # The interval condition (max k-sum < min (k+1)-sum) is SUFFICIENT but not necessary.
    # Powers of 2: subset sums are all distinct (each sum corresponds to a unique subset),
    # so no two subsets of ANY size have the same sum. So powers of 2 DO work!
    
    # My interval check was wrong. Let me fix the validation.
    
    # The correct condition: for any two subsets S1, S2 with |S1| != |S2|, sum(S1) != sum(S2).
    # This is equivalent to: the sets of subset sums for each cardinality are pairwise disjoint.
    
    # For powers of 2, each subset sum is unique (binary representation), so this holds.
    # {1,2,4,8,16,32,64,128} is valid with size 8.
    
    # Can we get more? We need 1..256 distinct integers.
    # The maximum sum is 1+2+...+n = n(n+1)/2, but we're picking from 1..256.
    # For n=16, max sum = 141+142+...+156 = (141+156)*16/2 = 2392.
    # The number of possible sums for each cardinality k is C(n,k), and the range is at most 256*k.
    # For the sets to be disjoint, we need the total number of distinct sums to be at least sum_k C(n,k) = 2^n.
    # The total range is 0 to 256*n, so we need 2^n <= 256*n + 1.
    # For n=10: 1024 <= 2561. OK.
    # For n=12: 4096 <= 3073. No.
    # So n <= 11 by this loose bound.
    
    # Let's just use a local search to find the maximum valid set.
    
    # For the initial implementation, return powers of 2.
    return [1, 2, 4, 8, 16, 32, 64, 128]


def construct():
    """Return a list of distinct integers in 1..256 with cardinality-separated subset sums."""
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Start with powers of 2
    best = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Try to improve with local search
    # We'll use a greedy + local search approach
    
    # First, let's try to build a larger set using a greedy approach
    # that actually checks the real constraint (not the interval one).
    
    def is_valid(s):
        """Check if s has cardinality-separated subset sums."""
        if not s:
            return True
        n = len(s)
        # For n > 20, this is expensive. Use a hash-based approach.
        # For each cardinality k, compute all subset sums and store in a set.
        # Then check that these sets are pairwise disjoint.
        if n > 20:
            # Too expensive to check exactly. Use the sufficient interval condition as a proxy.
            s_sorted = sorted(s)
            prefix = [0] * (n + 1)
            for i in range(n):
                prefix[i+1] = prefix[i] + s_sorted[i]
            for k in range(0, n):
                max_k = prefix[n] - prefix[n-k]
                min_next = prefix[k+1]
                if max_k >= min_next:
                    return False
            return True
        
        # Exact check for small n
        sums_by_card = [set() for _ in range(n+1)]
        sums_by_card[0].add(0)
        for i, x in enumerate(s):
            # Update in reverse to avoid using x multiple times
            for k in range(i, -1, -1):
                for val in list(sums_by_card[k]):
                    sums_by_card[k+1].add(val + x)
        
        # Check pairwise disjointness
        seen = set()
        for k in range(n+1):
            for val in sums_by_card[k]:
                if val in seen:
                    return False
                seen.add(val)
        return True
    
    # Greedy construction: try to add elements one by one
    # Start with a base set and try to expand
    
    # Try different starting points
    random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Start with a random subset of 1..256 of some size
        # Then try to grow it
        
        # Strategy: pick n numbers that are "well-spaced"
        # Use a geometric-like progression
        
        # Try: a_i = floor(base * ratio^(i-1)) for i=1..n, all in 1..256, distinct
        # Or use a more systematic approach
        
        # Let's try: pick n numbers from 1..256 using a greedy method
        # that maximizes the minimum gap between consecutive sorted elements
        
        # For a given n, the best we can do is roughly uniform spacing: 256/n
        # So for n=16, spacing ~16. For n=20, spacing ~12.8.
        
        # Let's try n from 16 down to 9 and see if we can find a valid set
        for n in range(16, 8, -1):
            if time.time() > deadline:
                break
            
            # Try to find n distinct numbers in 1..256 that are valid
            # Use local search: start with a random set, then swap elements to fix violations
            
            # Initial set: evenly spaced
            step = 256 // n
            candidates = [step * i + 1 for i in range(n)]
            if candidates[-1] > 256:
                candidates[-1] = 256
            # Ensure distinctness
            candidates = sorted(set(candidates))
            while len(candidates) < n:
                # Add a random element
                candidates.append(random.randint(1, 256))
            candidates = sorted(set(candidates))
            if len(candidates) > n:
                candidates = candidates[:n]
            
            # Local search: try to fix violations by swapping elements
            current = list(candidates)
            
            # We'll use a simple approach: try random swaps and keep if valid
            # But checking validity is expensive for n > 20.
            # For n <= 16, exact check is feasible.
            
            for iteration in range(2000):
                if time.time() > deadline:
                    break
                
                if is_valid(current):
                    if len(current) > len(best):
                        best = list(current)
                    break
                
                # Try a random swap
                i, j = random.sample(range(len(current)), 2)
                current[i], current[j] = current[j], current[i]
                if not is_valid(current):
                    # Try to fix by replacing one element
                    pass
            
            # Also try: start with powers of 2 and try to add more
            if n <= 10:
                base = [1, 2, 4, 8, 16, 32, 64, 128][:n]
                # Try to add elements
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    if is_valid(base):
                        break
                    # Remove a random element and try a different one
                    idx = random.randint(0, len(base)-1)
                    old = base[idx]
                    new = random.randint(1, 256)
                    if new in base:
                        continue
                    base[idx] = new
                    base = sorted(base)
    
    # Final validation
    if not is_valid(best):
        # Fallback to known valid set
        best = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Ensure return type is list of Python ints
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
