"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def _largest_sidon_subset_size(S):
    """Compute the largest subset of S with all unordered pair sums unique (including diagonals).
    
    A subset T of S is a Sidon set (B2 set) if all sums a+b where a<=b (a,b in T) are distinct.
    We use a branch-and-bound / backtracking approach to find the maximum such subset.
    """
    n = len(S)
    if n <= 1:
        return n
    
    # Precompute all pair sums for the full set to help with pruning
    # Actually, let's do a smarter backtracking
    
    # Sort the set
    S_sorted = sorted(S)
    
    # We'll use a recursive backtracking that tries to build the largest Sidon subset
    # Key insight: for a Sidon set, if we have elements a < b < c < d, then
    # a+d != b+c (since a+d and b+c would be equal sums from different pairs)
    # More generally, all sums a_i + a_j (i<=j) must be distinct.
    
    # Use a greedy + local search approach for the maximum, then verify with exact method
    # Actually for n=12, we can do exact search with bitmasks
    
    # For n=12, the number of subsets is 2^12 = 4096, so we can enumerate all subsets
    # and check which are Sidon, then find the maximum size.
    
    best = 0
    # Enumerate all subsets by size, starting from largest
    for size in range(n, 0, -1):
        found = False
        for subset in itertools.combinations(range(n), size):
            # Check if this subset forms a Sidon set
            elements = [S_sorted[i] for i in subset]
            sums = set()
            valid = True
            for i in range(len(elements)):
                for j in range(i, len(elements)):
                    s = elements[i] + elements[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                best = size
                found = True
                break
        if found:
            break
    
    return best

def _count_sum_collisions(S):
    """Count the number of sum collisions in the set S.
    
    A collision occurs when two different pairs (i,j) and (k,l) with i<=j, k<=l,
    (i,j) != (k,l), have the same sum S[i]+S[j] = S[k]+S[l].
    """
    n = len(S)
    sum_count = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            sum_count[S[i] + S[j]] += 1
    
    # Count excess collisions: for each sum that appears c times, there are c-1 collisions
    collisions = 0
    for c in sum_count.values():
        if c > 1:
            collisions += c - 1
    return collisions

def _is_good_candidate(S):
    """Quick check: does this set have relatively few sum collisions?"""
    return _count_sum_collisions(S)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline for safety
    
    # Strategy: Use simulated annealing to minimize the largest Sidon subset size.
    # The objective is to make it hard to find large Sidon subsets.
    # We can use the number of sum collisions as a proxy - more collisions means
    # it's harder to find large Sidon subsets.
    
    # Actually, let's think more carefully. We want to MINIMIZE the largest Sidon subset.
    # A set with many sum collisions will have smaller maximum Sidon subsets.
    
    # Start with a random set and use simulated annealing
    # State: a set of 12 distinct integers in [0, 128]
    # Move: pick two elements and swap one with a random other value
    
    random.seed(42)
    
    # Initial candidate: try to create a set with many sum collisions
    # Start with a structured set that has lots of arithmetic progressions
    # e.g., {0, 1, 2, ..., 11} has many collisions
    
    # Actually, let's try multiple initial candidates and pick the best
    candidates = []
    
    # Candidate 1: consecutive integers (many collisions)
    candidates.append(list(range(12)))
    
    # Candidate 2: powers of 2 (few collisions, but let's see)
    candidates.append([1, 2, 4, 8, 16, 32, 64, 5, 10, 20, 40, 80])
    
    # Candidate 3: random
    candidates.append(random.sample(range(129), 12))
    
    # Candidate 4: two interleaved arithmetic progressions
    candidates.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    
    # Candidate 5: clustered
    candidates.append([0, 1, 2, 3, 4, 5, 60, 61, 62, 63, 64, 65])
    
    # Evaluate each candidate
    best_candidate = None
    best_sidon_size = float('inf')
    best_collisions = -1
    
    for cand in candidates:
        # Ensure all values are in [0, 128] and distinct
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        
        sidon_size = _largest_sidon_subset_size(cand)
        collisions = _count_sum_collisions(cand)
        
        if sidon_size < best_sidon_size:
            best_sidon_size = sidon_size
            best_candidate = cand[:]
            best_collisions = collisions
        elif sidon_size == best_sidon_size and collisions > best_collisions:
            best_candidate = cand[:]
            best_collisions = collisions
    
    # Now use simulated annealing to improve
    current = best_candidate[:]
    current_sidon_size = best_sidon_size
    current_collisions = best_collisions
    
    # Temperature schedule
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    iterations = 0
    max_iterations = 50000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Generate a neighbor: pick a random element and change it
        idx = random.randint(0, 11)
        old_val = current[idx]
        new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val == old_val:
            continue
        if new_val in current and new_val != old_val:
            # Check if new_val is already in the set at another position
            if new_val in current:
                continue
        
        current[idx] = new_val
        
        # Check validity
        if len(set(current)) != 12:
            current[idx] = old_val
            continue
        if any(x < 0 or x > 128 for x in current):
            current[idx] = old_val
            continue
        
        new_sidon_size = _largest_sidon_subset_size(current)
        new_collisions = _count_sum_collisions(current)
        
        # Accept if better or with some probability
        delta_sidon = new_sidon_size - current_sidon_size
        delta_collisions = new_collisions - current_collisions
        
        # We want to minimize sidon_size, and maximize collisions as tiebreaker
        if delta_sidon < 0 or (delta_sidon == 0 and delta_collisions > 0):
            current_sidon_size = new_sidon_size
            current_collisions = new_collisions
            if new_sidon_size < best_sidon_size:
                best_sidon_size = new_sidon_size
                best_candidate = current[:]
                best_collisions = new_collisions
        else:
            # Accept with probability based on temperature
            # Energy = sidon_size * 1000 - collisions (we want to minimize energy)
            energy_delta = delta_sidon * 1000 - delta_collisions
            if energy_delta < 0:
                current_sidon_size = new_sidon_size
                current_collisions = new_collisions
                if new_sidon_size < best_sidon_size:
                    best_sidon_size = new_sidon_size
                    best_candidate = current[:]
                    best_collisions = new_collisions
            else:
                import math
                prob = math.exp(-energy_delta / T)
                if random.random() < prob:
                    current_sidon_size = new_sidon_size
                    current_collisions = new_collisions
                    if new_sidon_size < best_sidon_size:
                        best_sidon_size = new_sidon_size
                        best_candidate = current[:]
                        best_collisions = new_collisions
                else:
                    current[idx] = old_val
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final check
    if best_sidon_size < float('inf') and len(set(best_candidate)) == 12:
        return best_candidate
    else:
        return list(range(12))

# EVOLVE-BLOCK-END
