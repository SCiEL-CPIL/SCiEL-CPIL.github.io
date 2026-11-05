import time
import math
import random
from math import lcm

def construct():
    start_time = time.time()
    timeout = 55.0
    
    # Universe
    universe = list(range(1, 23))
    
    # Precompute LCM for scaling to integers
    # For a set S, scale factor = lcm(S)
    # Then 1/x * scale = scale/x, which is an integer
    
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
        
        # Convert to integer weights: w_i = scale / x_i
        weights = [scale // x for x in S]
        
        # Generate all subset sums using bit manipulation or iterative method
        n = len(S)
        if n > 20:
            # Too many subsets, use probabilistic check
            return _probabilistic_check(weights)
        
        # Use a set of all 2^n subset sums
        sums = set()
        sums.add(0)
        for i, w in enumerate(weights):
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
            sums.update(new_sums)
        
        expected = 1 << n
        return len(sums) == expected
    
    def _probabilistic_check(weights):
        """For large sets, use a faster heuristic check."""
        # Check if any two elements have the same weight (immediate collision)
        if len(set(weights)) != len(weights):
            return False
        
        # For large sets, do a partial check
        n = len(weights)
        # Sample some random subsets to check for collisions
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
    
    # Strategy 1: Greedy with descending order (largest elements first)
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
    
    # Strategy 3: Backtracking search for maximum cardinality
    # Try to find a set of size k by backtracking
    def backtrack_search(target_size, current_set, start_idx, candidates):
        """Recursively try to build a set of target_size."""
        if time.time() - start_time > timeout:
            return None
        if len(current_set) == target_size:
            return list(current_set)
        
        remaining_needed = target_size - len(current_set)
        available = candidates[start_idx:]
        if len(available) < remaining_needed:
            return None
        
        for i, x in enumerate(available):
            if try_add(current_set, x):
                current_set.append(x)
                result = backtrack_search(target_size, current_set, start_idx + i + 1, candidates)
                if result is not None:
                    return result
                current_set.pop()
        
        return None
    
    # Try to find sets of increasing size
    for target_size in range(best_size + 1, 13):
        if time.time() - start_time > timeout:
            break
        
        # Use a good ordering for backtracking
        # Sort by descending value (smaller reciprocal) first
        ordered_candidates = sorted(universe, reverse=True)
        result = backtrack_search(target_size, [], 0, ordered_candidates)
        if result is not None:
            best_set = result
            best_size = target_size
            # Continue trying for larger sizes
            continue
    
    # Strategy 4: Randomized restarts with various heuristics
    random.seed(42)
    restarts = 100
    for r in range(restarts):
        if time.time() - start_time > timeout:
            break
        
        # Different ordering strategies
        if r % 4 == 0:
            order = sorted(universe, reverse=True)
        elif r % 4 == 1:
            order = sorted(universe)
        elif r % 4 == 2:
            order = universe[:]
            random.shuffle(order)
        else:
            # Weighted random: favor middle values
            order = sorted(universe, key=lambda x: abs(x - 11.5) + random.random() * 5)
        
        current_set = greedy_build(order)
        current_set = expand_set(current_set)
        
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
    
    # Strategy 5: Local search - try swapping elements to improve
    if time.time() - start_time < timeout - 10:
        for _ in range(20):
            if time.time() - start_time > timeout:
                break
            
            current = list(best_set)
            improved = True
            while improved:
                improved = False
                # Try removing one element and adding two others
                for i in range(len(current)):
                    for x in universe:
                        if x in current:
                            continue
                        # Try swapping: remove current[i], add x
                        test_set = [current[j] for j in range(len(current)) if j != i]
                        test_set.append(x)
                        if is_dissociated(test_set):
                            # Check if this allows us to add more
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
