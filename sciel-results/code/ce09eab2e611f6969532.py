"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_sum_free(S, strategy='positive_large'):
        """Greedy construction of a sum-free subset."""
        S_sorted = sorted(S)
        
        if strategy == 'positive_large':
            # Take largest elements first (positive large values tend to be sum-free)
            candidates = [x for x in S_sorted if x > 0]
            candidates.sort(reverse=True)
        elif strategy == 'negative_large':
            # Take most negative elements
            candidates = [x for x in S_sorted if x < 0]
            candidates.sort()
        elif strategy == 'all_large_abs':
            # Take elements with largest absolute value
            candidates = sorted(S_sorted, key=lambda x: -abs(x))
        elif strategy == 'odd':
            # Odd numbers are sum-free
            candidates = [x for x in S_sorted if x % 2 != 0]
        elif strategy == 'even':
            # Even numbers: sum of two evens is even, so NOT sum-free in general
            # But if we restrict to a range, it might work
            candidates = [x for x in S_sorted if x % 2 == 0]
        else:
            candidates = S_sorted[:]
        
        result = []
        result_set = set()
        for x in candidates:
            # Check if adding x keeps the set sum-free
            valid = True
            for y in result:
                if x + y in result_set or (x + y) == x or (x + y) == y:
                    # x + y in result_set means conflict
                    if x + y in result_set:
                        valid = False
                        break
                # Also check if y + x is in result (same as above)
                # Check if x + x is in result
            # More careful check:
            valid = True
            for y in result:
                if x + y in result_set:
                    valid = False
                    break
                if x + x in result_set:
                    valid = False
                    break
            if valid:
                result.append(x)
                result_set.add(x)
        
        return result

    def local_optimize(S, initial_subset, max_iterations=100):
        """Try to improve the subset by swapping elements."""
        current = list(initial_subset)
        current_set = set(current)
        S_set = set(S)
        candidates = [x for x in S if x not in current_set]
        
        best_size = len(current)
        best_subset = current[:]
        
        for _ in range(max_iterations):
            if time.time() > DEADLINE - 5:
                break
            
            improved = False
            # Try adding elements
            for x in candidates:
                new_set = current_set | {x}
                # Check sum-free
                valid = True
                for a in new_set:
                    for b in new_set:
                        if a + b in new_set:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    current.append(x)
                    current_set.add(x)
                    candidates.remove(x)
                    best_size = len(current)
                    best_subset = current[:]
                    improved = True
                    break
            
            # Try removing and replacing
            if not improved:
                for i, x in enumerate(current):
                    for y in candidates:
                        new_current = current[:i] + current[i+1:] + [y]
                        new_set = set(new_current)
                        valid = True
                        for a in new_set:
                            for b in new_set:
                                if a + b in new_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(new_current) > best_size:
                            current = new_current
                            current_set = new_set
                            candidates.remove(y)
                            candidates.append(x)
                            best_size = len(current)
                            best_subset = current[:]
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_subset

    def find_best_subset(S):
        """Find the best sum-free subset using multiple strategies."""
        strategies = ['positive_large', 'negative_large', 'all_large_abs', 'odd']
        best_result = []
        best_size = 0
        
        for strategy in strategies:
            subset = greedy_sum_free(S, strategy)
            if len(subset) > best_size:
                best_size = len(subset)
                best_result = subset
        
        # Try to improve with local optimization
        if best_result and time.time() < DEADLINE - 10:
            improved = local_optimize(S, best_result)
            if len(improved) > best_size:
                best_result = improved
                best_size = len(best_result)
        
        # Fallback: odd numbers always work
        if not best_result:
            best_result = [x for x in S if x % 2 != 0]
        
        return best_result

    results = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        if time.time() > DEADLINE - 10:
            # Not enough time, use quick fallback
            S = instance(size, seed)
            results.append([x for x in S if x % 2 != 0])
            continue
        
        S = instance(size, seed)
        subset = find_best_subset(S)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
