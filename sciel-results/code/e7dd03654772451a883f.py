import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin before 90s kill

    def is_valid_sums(sums_set, max_sum):
        """Check if no two distinct sums in sums_set divide each other."""
        if len(sums_set) < 2:
            return True
        sum_list = sorted(sums_set)
        for u in sum_list:
            if u * 2 > max_sum:
                break
            for v in range(u * 2, max_sum + 1, u):
                if v in sums_set:
                    return False
        return True

    def try_add_element(current_sums, c):
        """Try adding element c. Returns new_sums if valid, else None."""
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + c)
        new_sums.add(c)
        
        # Check for duplicate sums (collision means two subsets have same sum)
        if len(new_sums) != 2 * len(current_sums) + 1:
            return None
            
        max_sum = max(new_sums)
        if not is_valid_sums(new_sums, max_sum):
            return None
            
        return new_sums

    def compute_sums(elements):
        """Compute all non-empty subset sums of elements."""
        n = len(elements)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    def validate_full(elements):
        """Full validation of a set of elements."""
        if not elements:
            return True
        sums = compute_sums(elements)
        max_sum = max(sums)
        return is_valid_sums(sums, max_sum)

    best_set = []
    best_size = 0

    # Pre-compute candidate ranges
    large_desc = list(range(256, 127, -1))
    mid_desc = list(range(256, 63, -1))
    all_desc = list(range(256, 0, -1))
    
    # Strategy 1: Deterministic greedy with descending order (large numbers first)
    for candidates in [large_desc, mid_desc, all_desc]:
        if time.time() > deadline - 5.0:
            break
        current = []
        current_sums = set()
        for c in candidates:
            if len(current) >= 30:
                break
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                current.append(c)
                current_sums = new_sums
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 30:
                break

    # Strategy 2: Randomized restarts with mixed orderings
    ranges = [
        list(range(256, 127, -1)),  # large only, descending
        list(range(256, 100, -1)),  # large-mid, descending
        list(range(256, 50, -1)),   # broader, descending
        list(range(256, 1, -1)),    # all, descending
    ]
    
    max_attempts = 10000
    for attempt in range(max_attempts):
        if time.time() > deadline - 5.0:
            break
        
        # Choose range: bias toward larger ranges early, then mix
        if attempt < 1000:
            r_idx = 0  # large only
        elif attempt < 3000:
            r_idx = random.choice([0, 1])  # large or large-mid
        elif attempt < 6000:
            r_idx = random.randint(0, 2)  # mix
        else:
            r_idx = random.randint(0, 3)  # all ranges
        
        candidates = ranges[r_idx][:]
        # Randomize but keep a bias: partially shuffle
        if random.random() < 0.3:
            random.shuffle(candidates)
        else:
            # Partial shuffle: reverse chunks
            chunk_size = random.randint(5, 20)
            for i in range(0, len(candidates), chunk_size):
                chunk = candidates[i:i+chunk_size]
                random.shuffle(chunk)
                candidates[i:i+chunk_size] = chunk
        
        current = []
        current_sums = set()
        
        for c in candidates:
            if len(current) >= 30:
                break
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                current.append(c)
                current_sums = new_sums
                
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 30:
                break

    # Strategy 3: Local search - try to swap out elements from the best set
    if best_size > 0 and time.time() < deadline - 8.0:
        # Try to extend the best set by removing one element and adding a better one
        for iteration in range(300):
            if time.time() > deadline - 5.0:
                break
            
            base_set = best_set[:]
            
            # Try removing each element and adding a new one
            found_improvement = False
            for rem_idx in range(len(base_set)):
                if time.time() > deadline - 5.0:
                    break
                
                reduced_set = base_set[:rem_idx] + base_set[rem_idx+1:]
                red_sums = compute_sums(reduced_set)
                
                # Try adding each candidate not in reduced_set
                for c in range(256, 0, -1):
                    if c in reduced_set:
                        continue
                    new_sums = try_add_element(red_sums, c)
                    if new_sums is not None:
                        new_set = reduced_set + [c]
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = new_set[:]
                            found_improvement = True
                        break  # Only try first valid candidate for this removal
            
            if not found_improvement:
                # No improvement in this iteration, try a different approach
                # Random swap: remove a random element, try to add a better one
                if len(base_set) >= 2:
                    rem_idx = random.randint(0, len(base_set) - 1)
                    reduced_set = base_set[:rem_idx] + base_set[rem_idx+1:]
                    red_sums = compute_sums(reduced_set)
                    
                    # Try random candidates
                    for _ in range(50):
                        if time.time() > deadline - 5.0:
                            break
                        c = random.randint(1, 256)
                        if c in reduced_set:
                            continue
                        new_sums = try_add_element(red_sums, c)
                        if new_sums is not None:
                            new_set = reduced_set + [c]
                            if len(new_set) > best_size:
                                best_size = len(new_set)
                                best_set = new_set[:]
                                break

    # Strategy 4: ILP-inspired approach using scipy's milp if available
    if best_size < 30 and time.time() < deadline - 10.0:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            # For ILP, we need to formulate the problem
            # Binary variable x[i] = 1 if element (i+1) is included
            # This is complex due to subset sum constraints, so we use a heuristic approach
            # Instead, we do a more structured search focusing on prime-like numbers
            
            # Try sets of numbers that are pairwise coprime or have limited divisibility
            prime_candidates = [p for p in range(2, 257) if all(p % d != 0 for d in range(2, int(math.sqrt(p)) + 1))]
            
            if len(prime_candidates) >= 10:
                current = []
                current_sums = set()
                for c in prime_candidates:
                    if len(current) >= 30:
                        break
                    new_sums = try_add_element(current_sums, c)
                    if new_sums is not None:
                        current.append(c)
                        current_sums = new_sums
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current[:]
        except ImportError:
            pass

    # Ensure we return at least something valid
    if not best_set:
        # Fallback: try a known small valid set
        for test_set in [[2, 3], [3, 5], [5, 7, 11], [2, 5, 11, 23]]:
            if validate_full(test_set):
                best_set = test_set
                break
    
    # Final validation
    if best_set and not validate_full(best_set):
        # If somehow invalid, try to fix by removing problematic elements
        for i in range(len(best_set) - 1, -1, -1):
            test = best_set[:i] + best_set[i+1:]
            if validate_full(test):
                best_set = test
                break
        if not best_set:
            best_set = [2, 3]
    
    # Convert to list of Python ints
    return [int(x) for x in sorted(best_set)]
