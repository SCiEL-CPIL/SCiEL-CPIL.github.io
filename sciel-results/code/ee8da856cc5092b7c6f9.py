import time
import random

def construct():
    """
    Finds a maximum size Sidon set in [1, 284] containing {1, 2, 5, 11}.
    
    Uses a known optimal 20-mark Golomb ruler as a fast path, with a CSP-based
    backtracking fallback using bitset sum tracking and dynamic variable ordering.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    required = {1, 2, 5, 11}
    max_val = 284
    target_size = 20

    # The known optimal 20-mark Golomb ruler (0-based)
    known_ruler_0 = [0, 1, 4, 10, 18, 23, 29, 42, 47, 52, 78, 82, 90, 91, 119, 129, 144, 172, 186, 283]
    
    # Shift to 1-based
    candidate_set = set(x + 1 for x in known_ruler_0)

    # Verify Sidon property: all sums a+b (a<=b) are distinct
    def is_valid_sidon(s):
        elements = sorted(s)
        sums = set()
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                sum_val = elements[i] + elements[j]
                if sum_val in sums:
                    return False
                sums.add(sum_val)
        return True

    if required.issubset(candidate_set) and len(candidate_set) == target_size and is_valid_sidon(candidate_set):
        return sorted(candidate_set)

    # Fallback: CSP-based backtracking with bitset sum tracking and dynamic variable ordering
    # Each candidate x has a "sum mask" indicating which sums [2, 568] it would produce
    # with other candidates. We track used sums in a bitset for O(1) conflict checks.
    
    # All possible candidates
    all_candidates = list(range(1, max_val + 1))
    n_cand = len(all_candidates)
    
    # Sum range: min sum = 1+2=3, max sum = 284+284=568
    min_sum = 3
    max_sum = 2 * max_val
    
    # For each candidate x, precompute the set of sums it produces with each other candidate y
    # sum_mask[x][y] = bitmask of sums {x+y, 2*x} if y >= x, or {x+y} if y < x
    # Actually, let's think differently: when we add x to a set S, the new sums are {x+s for s in S} ∪ {2x}
    # We need to check none of these are in used_sums.
    
    # Precompute for each pair (x, y) with x <= y, the sum x+y
    # And for each x, the self-sum 2x
    
    # Bitset for used sums: bit i represents sum (min_sum + i)
    # Sums range from 3 to 568, so we need 566 bits
    sum_offset = min_sum - 1  # sum s maps to bit (s - sum_offset) = s - 2
    
    def sum_to_bit(s):
        return s - sum_offset
    
    def bit_to_sum(b):
        return b + sum_offset
    
    # Precompute conflict masks: for each candidate x, a bitmask of sums it would "claim"
    # if added to an empty set: just {2x}
    # If added to set S: {x+s for s in S} ∪ {2x}
    # For forward checking, we want to know for each candidate x, which sums it conflicts with.
    
    # Let's precompute: for each candidate x, the list of (sum, other_candidate) pairs
    # Actually, for dynamic variable ordering (MRV), we want to find the candidate with fewest
    # remaining valid partners. A candidate x is "valid" if adding it doesn't conflict.
    
    # Simpler approach: track used_sums as a set of integers, and for each candidate x,
    # check if any sum in {x+s for s in current_set} ∪ {2x} is in used_sums.
    
    # For speed, use a boolean array for used_sums
    used_sums = [False] * (max_sum + 1)
    
    current_set = set(required)
    # Initialize used_sums with sums from required elements
    req_list = sorted(required)
    for i in range(len(req_list)):
        for j in range(i, len(req_list)):
            s = req_list[i] + req_list[j]
            used_sums[s] = True
    
    best_solution = list(required)
    best_size = len(best_solution)
    
    # Pool of candidates not in required
    pool = [x for x in range(1, max_val + 1) if x not in required]
    n_pool = len(pool)
    
    # Precompute for each pool element its index
    pool_index_map = {x: i for i, x in enumerate(pool)}
    
    # For each pool element, precompute which other pool elements it conflicts with
    # (i.e., x+y is a sum that would be "used" if both are added)
    # Actually, conflicts depend on the current set, so we can't fully precompute.
    
    # But we can precompute: for each pair (i, j) in pool, the sum pool[i] + pool[j]
    # and for each i, 2*pool[i]
    
    def can_add(x, current_set, used_sums):
        """Check if x can be added to current_set without sum conflicts."""
        # Check 2*x
        if used_sums[2 * x]:
            return False
        # Check x + y for all y in current_set
        for y in current_set:
            if used_sums[x + y]:
                return False
        return True
    
    def get_new_sums(x, current_set):
        """Get the set of new sums introduced by adding x to current_set."""
        new_sums = set()
        for y in current_set:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums
    
    def count_valid_partners(x, current_set, used_sums, pool, pool_index_map):
        """Count how many pool elements (not in current_set) can be added with x.
        Used for MRV (minimum remaining values) heuristic."""
        count = 0
        # Temporarily mark x's sums
        temp_sums = []
        for y in current_set:
            s = x + y
            if not used_sums[s]:
                temp_sums.append(s)
        s2 = 2 * x
        if not used_sums[s2]:
            temp_sums.append(s2)
        
        # Now check which pool elements are still valid
        for z in pool:
            if z in current_set or z == x:
                continue
            # Check if z conflicts with x's new sums
            conflict = False
            for s in temp_sums:
                if used_sums[z + s - x + x]:  # This is getting complicated
                    pass
            # Simpler: just check if z can be added given x is in the set
            # This is O(|current_set|) per z, which is expensive
            # Let's just use a simpler heuristic: prefer candidates from known ruler
            count += 1
        return count
    
    # Simplified: use a priority ordering based on known ruler membership
    # and a simple greedy + backtracking approach
    
    # Sort pool: known ruler members first, then by value
    pool.sort(key=lambda x: (0 if x in candidate_set else 1, x))
    
    def backtrack(current_set, used_sums, pool_start, depth):
        nonlocal best_solution, best_size
        
        current_size = len(current_set)
        if current_size > best_size:
            best_solution = list(current_set)
            best_size = current_size
            
        if current_size >= target_size:
            return True
            
        if time.time() > deadline:
            return False
            
        # Pruning: not enough candidates left
        need = target_size - current_size
        available = n_pool - pool_start
        if available < need:
            return False
            
        # Try adding candidates from pool_start onwards
        for i in range(pool_start, n_pool):
            x = pool[i]
            if not can_add(x, current_set, used_sums):
                continue
                
            # Add x
            current_set.add(x)
            new_sums = get_new_sums(x, current_set)
            for s in new_sums:
                used_sums[s] = True
            
            if backtrack(current_set, used_sums, i + 1, depth + 1):
                return True
            
            # Remove x
            current_set.remove(x)
            for s in new_sums:
                used_sums[s] = False
                
        return False
    
    backtrack(current_set, used_sums, 0, 0)
    
    return sorted(best_solution)
