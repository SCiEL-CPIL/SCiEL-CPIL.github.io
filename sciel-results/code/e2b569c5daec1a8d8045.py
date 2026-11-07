"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 256, 'maximum_size': 30}

def _all_subset_sums(nums):
    """Compute all non-empty subset sums as a set."""
    sums = set()
    n = len(nums)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sums.add(s)
    return sums

def _check_valid(nums):
    """Check if the set satisfies all constraints:
    1. All non-empty subset sums are distinct (no two subsets have the same sum)
    2. No two distinct subset sums divide each other
    """
    # Check distinct subset sums
    n = len(nums)
    sum_to_count = {}
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sum_to_count[s] = sum_to_count.get(s, 0) + 1
        if sum_to_count[s] > 1:
            return False
    
    # Check divisibility
    sums = list(sum_to_count.keys())
    sums.sort()
    for i in range(len(sums)):
        for j in range(i + 1, len(sums)):
            if sums[j] % sums[i] == 0:
                return False
    return True

def _check_valid_fast(nums):
    """Faster validity check using incremental subset sum tracking."""
    n = len(nums)
    # First check: all subset sums must be unique
    # This is equivalent to: the set is "sum-free" in the sense that no two subsets have the same sum
    # For positive integers, this means the set must be such that all 2^n - 1 subset sums are distinct
    # A necessary condition: sum of any subset < sum of complement is not needed, but we need all sums distinct
    
    # Efficient check: compute all subset sums and check for duplicates
    sums = [0] * (1 << n)
    sums[0] = 0
    for mask in range(1, 1 << n):
        # Find lowest set bit
        lb = mask & (-mask)
        i = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + nums[i]
    
    # Check for duplicate sums among non-empty subsets
    seen = set()
    for mask in range(1, 1 << n):
        s = sums[mask]
        if s in seen:
            return False
        seen.add(s)
    
    # Check divisibility
    sum_list = sorted(seen)
    for i in range(len(sum_list)):
        s_i = sum_list[i]
        for j in range(i + 1, len(sum_list)):
            s_j = sum_list[j]
            if s_j % s_i == 0:
                return False
    return True

def _greedy_build():
    """Build a valid set using greedy approach."""
    result = []
    for candidate in range(256, 0, -1):
        if candidate in result:
            continue
        test_set = result + [candidate]
        if len(test_set) <= 20:  # Only do full check for small sets
            if _check_valid_fast(test_set):
                result.append(candidate)
        else:
            # For larger sets, do a quick check
            result.append(candidate)
            if not _check_valid_fast(result):
                result.pop()
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    best_set = [31, 32, 34]
    best_size = 3
    
    # Try greedy from different starting points
    random.seed(42)
    
    # First, try a simple greedy with large numbers
    greedy_set = _greedy_build()
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return list(best_set)
    
    # Try simulated annealing / local search
    # Start from the best set found so far
    current_set = list(best_set)
    current_size = len(current_set)
    
    # Candidate pool: numbers not in current set
    used = set(current_set)
    candidates = [i for i in range(256, 0, -1) if i not in used]
    
    best_found = list(current_set)
    best_found_size = current_size
    
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        
        # Try to add a new element
        if candidates:
            idx = random.randint(0, len(candidates) - 1)
            candidate = candidates[idx]
            test_set = current_set + [candidate]
            
            if len(test_set) <= 18:
                if _check_valid_fast(test_set):
                    current_set = test_set
                    used.add(candidate)
                    candidates.remove(candidate)
                    current_size += 1
                    if current_size > best_found_size:
                        best_found = list(current_set)
                        best_found_size = current_size
            else:
                # Quick reject for large sets
                pass
        
        # Try swap: remove one, add one
        if len(current_set) >= 2 and candidates:
            if random.random() < 0.3:
                remove_idx = random.randint(0, len(current_set) - 1)
                remove_val = current_set[remove_idx]
                add_idx = random.randint(0, len(candidates) - 1)
                add_val = candidates[add_idx]
                
                new_set = current_set[:remove_idx] + current_set[remove_idx+1:] + [add_val]
                
                if len(new_set) <= 18:
                    if _check_valid_fast(new_set):
                        current_set = new_set
                        used.remove(remove_val)
                        used.add(add_val)
                        candidates.remove(add_val)
                        candidates.append(remove_val)
                        if len(current_set) > best_found_size:
                            best_found = list(current_set)
                            best_found_size = len(current_set)
        
        # Occasionally try to remove an element to allow different additions
        if len(current_set) > 3 and random.random() < 0.1:
            remove_idx = random.randint(0, len(current_set) - 1)
            remove_val = current_set[remove_idx]
            current_set = current_set[:remove_idx] + current_set[remove_idx+1:]
            used.remove(remove_val)
            candidates.append(remove_val)
    
    # Final validation of best found
    if _check_valid_fast(best_found):
        return list(best_found)
    else:
        return [31, 32, 34]
# EVOLVE-BLOCK-END
