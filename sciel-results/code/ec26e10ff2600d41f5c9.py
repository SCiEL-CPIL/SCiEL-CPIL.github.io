import time
import math
import random
from math import lcm
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0
    
    # Universe
    universe = list(range(1, 23))
    n = len(universe)
    
    # Precompute LCM for scaling to integers
    def get_scale(S):
        """Get the LCM of all elements in S."""
        if not S:
            return 1
        s = 1
        for x in S:
            s = lcm(s, x)
        return s
    
    def is_dissociated(S):
        """Check if all subset sums of reciprocals of S are distinct using integer arithmetic."""
        if not S:
            return True
        if len(S) > 22:
            return False
        
        scale = get_scale(S)
        weights = [scale // x for x in S]
        
        n_elem = len(S)
        if n_elem > 20:
            return _probabilistic_check(weights)
        
        sums = set()
        sums.add(0)
        for w in weights:
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
            sums.update(new_sums)
        
        expected = 1 << n_elem
        return len(sums) == expected
    
    def _probabilistic_check(weights):
        """For large sets, use a faster heuristic check."""
        if len(set(weights)) != len(weights):
            return False
        
        n = len(weights)
        random.seed(12345)
        sample_size = min(2**n, 10000)
        sampled_sums = set()
        for _ in range(sample_size):
            subset_sum = 0
            for i in range(n):
                if random.random() < 0.5:
                    subset_sum += weights[i]
            if subset_sum in sampled_sums:
                return False
            sampled_sums.add(subset_sum)
        return True
    
    def try_add(S, x):
        """Check if S union {x} is dissociated."""
        new_S = list(S)
        new_S.append(x)
        return is_dissociated(new_S)
    
    def greedy_build(order):
        """Build a dissociated set by greedily adding elements in the given order."""
        current = []
        for x in order:
            if try_add(current, x):
                current.append(x)
        return current
    
    def expand_set(S):
        """Try to expand a dissociated set by adding any remaining element."""
        remaining = [x for x in universe if x not in S]
        added = True
        while added:
            added = False
            for x in remaining[:]:
                if try_add(S, x):
                    S.append(x)
                    remaining.remove(x)
                    added = True
                    if time.time() - start_time > timeout:
                        break
            if time.time() - start_time > timeout:
                break
        return S
    
    # === MIS-based approach ===
    # Build conflict graph: edge between a,b if {a,b} is NOT dissociated
    # We want maximum independent set in this graph
    
    # Precompute conflicts using bitsets for efficiency
    # conflict[i] is a bitmask where bit j is set if element i conflicts with element j
    conflict = [0] * n
    
    # For each pair, check if they can coexist in a dissociated set
    for i in range(n):
        for j in range(i + 1, n):
            a, b = universe[i], universe[j]
            # Check if {a, b} is dissociated
            # Subset sums: 0, 1/a, 1/b, 1/a+1/b
            # All distinct iff 1/a != 1/b (true since a != b) and 1/a+1/b != 0, 1/a, 1/b
            # 1/a+1/b = 1/a iff 1/b = 0 (false)
            # 1/a+1/b = 1/b iff 1/a = 0 (false)
            # So any pair is always dissociated!
            # But wait - we need to check larger sets too
            # The conflict should be: can a and b both be in some dissociated set?
            # Actually, the constraint is global: ALL subset sums must be distinct
            # So we need to check if adding both creates a collision
    
    # Better approach: use the actual dissociation check for pairs
    # But for efficiency, let's first get a good initial solution, then use branch-and-bound
    
    # Strategy 1: Greedy with descending order
    candidates_desc = sorted(universe, reverse=True)
    greedy_set = greedy_build(candidates_desc)
    greedy_set = expand_set(greedy_set)
    best_set = list(greedy_set)
    best_size = len(greedy_set)
    
    # Strategy 2: Greedy with ascending order
    candidates_asc = sorted(universe)
    greedy_set_asc = greedy_build(candidates_asc)
    greedy_set_asc = expand_set(greedy_set_asc)
    if len(greedy_set_asc) > best_size:
        best_set = list(greedy_set_asc)
        best_size = len(greedy_set_asc)
    
    # Strategy 3: Branch and bound with bitsets
    # Represent sets as bitmasks for fast operations
    
    def bitmask_to_set(mask):
        """Convert bitmask to list of elements."""
        return [universe[i] for i in range(n) if (mask >> i) & 1]
    
    def set_to_bitmask(S):
        """Convert set to bitmask."""
        mask = 0
        for x in S:
            idx = universe.index(x)
            mask |= (1 << idx)
        return mask
    
    # Precompute which subsets are dissociated for small sizes
    # Use memoization
    dissociated_cache = {}
    
    def is_dissociated_mask(mask):
        """Check if the set represented by mask is dissociated."""
        if mask in dissociated_cache:
            return dissociated_cache[mask]
        
        S = bitmask_to_set(mask)
        result = is_dissociated(S)
        dissociated_cache[mask] = result
        return result
    
    def can_add(mask, elem_idx):
        """Check if adding element elem_idx to the set represented by mask is valid."""
        new_mask = mask | (1 << elem_idx)
        return is_dissociated_mask(new_mask)
    
    # Branch and bound for maximum dissociated set
    def branch_and_bound(available_mask, current_mask, current_size):
        """Find maximum dissociated set using branch and bound."""
        nonlocal best_set, best_size
        
        if time.time() - start_time > timeout:
            return
        
        # Upper bound: current_size + number of available elements
        upper_bound = current_size + bin(available_mask).count('1')
        if upper_bound <= best_size:
            return
        
        if available_mask == 0:
            return
        
        # Get available elements
        avail = []
        for i in range(n):
            if (available_mask >> i) & 1:
                avail.append(i)
        
        if not avail:
            return
        
        # Greedy coloring upper bound approximation
        # Sort by degree in conflict graph (higher degree first for pruning)
        
        # Try adding each available element
        for idx in avail:
            if time.time() - start_time > timeout:
                return
            
            if can_add(current_mask, idx):
                new_mask = current_mask | (1 << idx)
                new_size = current_size + 1
                
                if new_size > best_size:
                    best_set = bitmask_to_set(new_mask)
                    best_size = new_size
                
                # Recurse with remaining available elements
                remaining = available_mask & ~(1 << idx)
                branch_and_bound(remaining, new_mask, new_size)
    
    # Start branch and bound from empty set
    if time.time() - start_time < timeout - 5:
        branch_and_bound((1 << n) - 1, 0, 0)
    
    # Strategy 4: Randomized restarts
    random.seed(42)
    restarts = 50
    for r in range(restarts):
        if time.time() - start_time > timeout:
            break
        
        if r % 4 == 0:
            order = sorted(universe, reverse=True)
        elif r % 4 == 1:
            order = sorted(universe)
        elif r % 4 == 2:
            order = universe[:]
            random.shuffle(order)
        else:
            order = sorted(universe, key=lambda x: abs(x - 11.5) + random.random() * 5)
        
        current_set = greedy_build(order)
        current_set = expand_set(current_set)
        
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
    
    # Strategy 5: Local search
    if time.time() - start_time < timeout - 10:
        for _ in range(20):
            if time.time() - start_time > timeout:
                break
            
            current = list(best_set)
            improved = True
            while improved:
                improved = False
                for i in range(len(current)):
                    for x in universe:
                        if x in current:
                            continue
                        test_set = [current[j] for j in range(len(current)) if j != i]
                        test_set.append(x)
                        if is_dissociated(test_set):
                            expanded = expand_set(test_set)
                            if len(expanded) > len(current):
                                current = expanded
                                improved = True
                                break
                    if improved:
                        break
            
            if len(current) > best_size:
                best_set = current
                best_size = len(current)
    
    return best_set
