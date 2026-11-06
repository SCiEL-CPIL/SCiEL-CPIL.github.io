<<<<<<< SEARCH
import time

def construct():
    """
    Returns a list of 12 distinct positive integers with distinct subset sums,
    minimizing the largest integer.
    
    The theoretical minimum for the largest element in a set of n distinct
    positive integers with distinct subset sums is 2^(n-1), achieved uniquely
    by the powers of 2: {1, 2, 4, ..., 2^(n-1)}.
    
    For n=12, this gives max element = 2^11 = 2048.
    """
    n = 12
    best = [1 << i for i in range(n)]
    
    # Verify correctness: all 2^12 = 4096 subset sums must be distinct
    sums = set()
    for mask in range(1 << n):
        s = 0
        for j in range(n):
            if mask & (1 << j):
                s += best[j]
        if s in sums:
            # Should never happen for powers of 2, but guard just in case
            return best
        sums.add(s)
    
    return best
=======
import time
import random

def _check_distinct_sums(arr, n):
    """Check if all 2^n subset sums are distinct. Returns True if valid."""
    sums = set()
    for mask in range(1 << n):
        s = 0
        for j in range(n):
            if mask & (1 << j):
                s += arr[j]
        if s in sums:
            return False
        sums.add(s)
    return True

def _try_reduce(arr, n, deadline):
    """Try to reduce the max element via local search. Returns improved list or None."""
    best = list(arr)
    best_max = max(best)
    best_valid = True
    
    # Sort ascending for consistent ordering
    best.sort()
    
    # Try to reduce the largest element by swapping it with smaller values
    # and adjusting other elements
    for i in range(5000):
        if time.time() > deadline:
            break
        
        # Try replacing the largest element with a smaller candidate
        largest_idx = best.index(max(best))
        current_max = best[largest_idx]
        
        # Try candidates from current_max - 1 down to 1
        found = False
        for candidate in range(current_max - 1, 0, -1):
            if candidate in best:
                continue
            
            new_arr = list(best)
            new_arr[largest_idx] = candidate
            new_arr.sort()
            
            if not _check_distinct_sums(new_arr, n):
                continue
            
            # Valid! Update best
            best = new_arr
            best_max = candidate
            found = True
            break
        
        if not found:
            # Try more complex moves: swap two elements
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    
                    # Try to reduce element j by 1
                    if best[j] > 1 and best[j] - 1 not in best:
                        new_arr = list(best)
                        new_arr[j] = best[j] - 1
                        new_arr.sort()
                        if _check_distinct_sums(new_arr, n):
                            best = new_arr
                            best_max = max(best)
                            break
                    if time.time() > deadline:
                        break
                if best_max < max(best):
                    break
                if best_max == max(best) and best[j] - 1 not in best:
                    continue
                # Check if we made progress
                if max(best) < best_max:
                    best_max = max(best)
                    break
                if time.time() > deadline:
                    break
            if best_max < max(best):
                best_max = max(best)
            if time.time() > deadline:
                break
        
        # If we can't reduce further, try random perturbations
        if not found and random.random() < 0.1:
            idx = random.randint(0, n - 1)
            delta = random.choice([-2, -1, 1, 2])
            new_val = best[idx] + delta
            if new_val > 0 and new_val not in best:
                new_arr = list(best)
                new_arr[idx] = new_val
                new_arr.sort()
                if _check_distinct_sums(new_arr, n):
                    best = new_arr
                    best_max = max(best)
    
    return best if best_valid else None

def construct():
    """
    Returns a list of 12 distinct positive integers with distinct subset sums,
    minimizing the largest integer.
    
    Starts from powers of 2 (max = 2^11 = 2048) and uses local search
    to try to reduce the largest element while maintaining distinct subset sums.
    """
    n = 12
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Start with powers of 2
    best = [1 << i for i in range(n)]
    
    # Try to improve
    improved = _try_reduce(best, n, deadline)
    if improved is not None and max(improved) < max(best):
        best = improved
    
    # Final verification
    if not _check_distinct_sums(best, n):
        # Fallback to known valid solution
        best = [1 << i for i in range(n)]
    
    return best
>>>>>>> REPLACE
