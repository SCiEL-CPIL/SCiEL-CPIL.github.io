"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def count_violations(subset):
        s = set(subset)
        count = 0
        for a in s:
            for b in s:
                if a + b in s:
                    count += 1
        return count

    def greedy_remove(arr):
        """Remove elements causing violations, starting with those involved in most violations."""
        s = set(arr)
        if not s:
            return list(s)
        
        # Count how many violations each element is involved in
        violation_count = {x: 0 for x in s}
        for a in s:
            for b in s:
                if a + b in s:
                    violation_count[a] += 1
                    violation_count[b] += 1
                    violation_count[a + b] += 1
        
        # Iteratively remove the element with the most violations
        current = list(s)
        current_set = set(s)
        
        while True:
            if time.time() > deadline:
                break
            # Find element with max violations
            if not current_set:
                break
            max_elem = max(current_set, key=lambda x: violation_count.get(x, 0))
            if violation_count.get(max_elem, 0) == 0:
                break
            current_set.remove(max_elem)
            current.remove(max_elem)
            
            # Recalculate violations after removal
            violation_count = {x: 0 for x in current_set}
            for a in current_set:
                for b in current_set:
                    if a + b in current_set:
                        violation_count[a] += 1
                        violation_count[b] += 1
                        violation_count[a + b] += 1
        
        return list(current_set)

    def local_search(arr, max_iterations=500):
        """Try to add back removed elements or swap to improve size while maintaining sum-free."""
        s = set(arr)
        all_elements = set(instance_arr)  # will be set per call
        removed = all_elements - s
        
        best = list(s)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            # Try adding each removed element
            best_improvement = None
            best_size = len(best)
            
            for elem in removed:
                if time.time() > deadline:
                    break
                new_s = s | {elem}
                if is_sum_free(new_s):
                    if len(new_s) > best_size:
                        best_size = len(new_s)
                        best_improvement = elem
            
            if best_improvement is not None:
                s.add(best_improvement)
                best = list(s)
                removed.discard(best_improvement)
            else:
                break
        
        return best

    def local_search_swap(arr, max_iterations=200):
        """Try swapping elements to potentially get a larger sum-free set."""
        s = set(arr)
        all_elements = set(instance_arr)
        
        best = list(s)
        best_size = len(best)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False
            for elem_in in list(s):
                if time.time() > deadline:
                    break
                for elem_out in (all_elements - s):
                    if time.time() > deadline:
                        break
                    new_s = (s - {elem_in}) | {elem_out}
                    if is_sum_free(new_s) and len(new_s) > best_size:
                        s = new_s
                        best = list(s)
                        best_size = len(s)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return best

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Return best effort with whatever we have
            results.append([])
            continue
        
        arr = instance(size, seed)
        global instance_arr
        instance_arr = arr
        
        # Start with odd numbers (sum-free: odd + odd = even, not odd)
        odd_subset = [x for x in arr if x % 2 != 0]
        
        # Try to improve: can we add any even numbers?
        even_numbers = [x for x in arr if x % 2 == 0]
        
        # Greedy approach: start with odds, try adding evens one by one
        current = set(odd_subset)
        
        for even in even_numbers:
            if time.time() > deadline:
                break
            test_set = current | {even}
            if is_sum_free(test_set):
                current.add(even)
        
        # Now try to add back some odds that were excluded? No, all odds are in.
        # Try local search: can we swap to get more elements?
        # Also try starting from scratch with different strategies
        
        # Strategy 2: Try to find a large sum-free subset using a different approach
        # Sort by absolute value, try to include large numbers (harder to form sums)
        
        # Try greedy from largest absolute value
        sorted_by_abs = sorted(arr, key=lambda x: -abs(x))
        current2 = set()
        for x in sorted_by_abs:
            if time.time() > deadline:
                break
            test_set = current2 | {x}
            if is_sum_free(test_set):
                current2.add(x)
        
        # Choose the better of the two strategies
        if len(current2) > len(current):
            best_subset = list(current2)
        else:
            best_subset = list(current)
        
        # Local search to improve
        improved = local_search(best_subset, max_iterations=300)
        if len(improved) > len(best_subset):
            best_subset = improved
        
        # Try swap-based local search
        swapped = local_search_swap(best_subset, max_iterations=100)
        if len(swapped) > len(best_subset):
            best_subset = swapped
        
        # Final verification
        if not is_sum_free(best_subset):
            # Fallback to greedy_remove
            best_subset = greedy_remove(best_subset)
            if not is_sum_free(best_subset):
                # Last resort: just use odds
                best_subset = odd_subset
        
        results.append(sorted(best_subset))
    
    return results
# EVOLVE-BLOCK-END
