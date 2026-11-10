"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict

P = {'n': 256, 'maximum_size': 60}

def compute_dp_with_tracking(S):
    """
    Compute the set of achievable sums for each cardinality using bitsets,
    and track which elements contribute to each (cardinality, sum) pair.
    
    Returns:
        dp: list of bitsets for each cardinality
        contributors: dict mapping (cardinality, sum) -> set of element indices that can form this sum
    """
    m = len(S)
    if m == 0:
        return [1], {}
    
    dp = [0] * (m + 1)
    dp[0] = 1  # bit 0 is set for cardinality 0
    
    # contributors[k][s] = set of indices i such that element i is part of some subset of size k with sum s
    # We'll track this incrementally
    contributors = [defaultdict(set) for _ in range(m + 1)]
    contributors[0][0].add(-1)  # sentinel for empty set
    
    for idx, x in enumerate(S):
        # Iterate backwards to ensure each element is used at most once
        for k in range(m, 0, -1):
            # Shift the bitset for cardinality k-1 by x and OR into cardinality k
            shifted = dp[k-1] << x
            new_bits = shifted & ~dp[k]
            
            if new_bits:
                # Find which sums are newly added
                s = 0
                temp = new_bits
                while temp:
                    if temp & 1:
                        # Sum s is newly achievable at cardinality k using element idx
                        contributors[k][s].add(idx)
                    s += 1
                    temp >>= 1
            
            dp[k] |= shifted
            
    return dp, contributors

def count_collisions(dp):
    """
    Count the total number of sum collisions across different cardinalities.
    """
    m = len(dp) - 1
    collisions = 0
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            common = dp[i] & dp[j]
            if common:
                collisions += common.bit_count()
    return collisions

def find_colliding_elements(dp, contributors, S):
    """
    Find elements that are involved in cross-cardinality sum collisions.
    Returns a dict: element_index -> count of collisions it's involved in.
    """
    m = len(S)
    collision_count = defaultdict(int)
    
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            common = dp[i] & dp[j]
            if common:
                # For each common sum, find elements involved
                s = 0
                temp = common
                while temp:
                    if temp & 1:
                        # Sum s is achievable at both cardinality i and j
                        # Get elements involved in forming sum s at cardinality i
                        elems_i = contributors[i].get(s, set())
                        elems_j = contributors[j].get(s, set())
                        
                        # All elements in both sets are involved in this collision
                        for e in elems_i:
                            if e >= 0:
                                collision_count[e] += 1
                        for e in elems_j:
                            if e >= 0:
                                collision_count[e] += 1
                    s += 1
                    temp >>= 1
    
    return collision_count

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums must differ.
    
    Strategy:
    1. Start with a random set of size N.
    2. Compute global collision count and identify elements in most collisions.
    3. Remove the element involved in the most cross-cardinality collisions.
    4. Repeat until valid or size is small.
    5. Use multiple random restarts with different initialization strategies.
    """
    max_size = 60
    max_val = 256
    best_set = []
    best_score = -1
    start_time = time.time()
    time_limit = 55  # Leave 5s buffer
    
    # Try multiple random restarts with different strategies
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            # Pure random
            S = random.sample(range(1, max_val + 1), max_size)
        elif restart % 3 == 1:
            # Biased toward larger values (less likely to collide)
            S = random.sample(range(64, max_val + 1), min(max_size, max_val - 63))
            while len(S) < max_size:
                remaining = [x for x in range(1, 64) if x not in S]
                if remaining:
                    S.append(random.choice(remaining))
                else:
                    break
        else:
            # Spread out values
            step = max_val // max_size
            base = [i * step + 1 for i in range(max_size) if i * step + 1 <= max_val]
            S = base[:]
            random.shuffle(S)
            # Add some randomness
            if len(S) < max_size:
                remaining = [x for x in range(1, max_val + 1) if x not in S]
                S.extend(random.sample(remaining, min(max_size - len(S), len(remaining))))
        
        current_size = len(S)
        
        # Local search loop with global conflict guidance
        while current_size > 0 and time.time() - start_time < time_limit:
            dp, contributors = compute_dp_with_tracking(S)
            colls = count_collisions(dp)
            
            if colls == 0:
                # Found a valid set
                if len(S) > len(best_set):
                    best_set = S.copy()
                    best_score = len(S)
                break
            
            # Find elements involved in most collisions
            collision_counts = find_colliding_elements(dp, contributors, S)
            
            # Find the element with the highest collision count
            if not collision_counts:
                # No specific collisions tracked, just remove a random element
                idx_to_remove = random.randint(0, current_size - 1)
            else:
                max_coll = max(collision_counts.values())
                # Among elements with max collisions, pick one randomly for diversity
                candidates = [idx for idx, cnt in collision_counts.items() if cnt == max_coll]
                idx_to_remove = random.choice(candidates)
            
            # Remove the element
            S.pop(idx_to_remove)
            current_size -= 1
        
        # After each restart, check if we improved
        if len(S) > len(best_set):
            dp_check, _ = compute_dp_with_tracking(S)
            if count_collisions(dp_check) == 0:
                best_set = S.copy()
                best_score = len(S)
    
    # Ensure we return a list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
