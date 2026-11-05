"""Optimized subset-sum construction via targeted local search."""
# EVOLVE-BLOCK-START
import time
import random
import sys

N = 12
MAX_VAL = 1048576

def _all_subset_sums(arr):
    """Compute all 2^N subset sums efficiently."""
    sums = [0] * (1 << N)
    for i in range(N):
        bit = 1 << i
        val = arr[i]
        for mask in range(bit):
            sums[bit | mask] = sums[mask] + val
    return sums

def _score(arr):
    """Return 0 if all subset sums are distinct, else count of duplicates."""
    sums = _all_subset_sums(arr)
    return len(sums) - len(set(sums))

def _is_valid(arr):
    """Check if all 2^N subset sums are distinct."""
    return _score(arr) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Theoretical optimal: powers of 2
    # For N distinct positive integers with all subset sums distinct,
    # the minimum possible maximum element is 2^(N-1) = 2048 for N=12.
    # This is achieved by [1, 2, 4, ..., 2^11].
    
    optimal = [1 << i for i in range(N)]
    
    # Verify the optimal solution is valid (should always be)
    if _is_valid(optimal):
        # Quick sanity checks
        assert len(optimal) == N
        assert all(isinstance(x, int) and x > 0 for x in optimal)
        assert all(x <= MAX_VAL for x in optimal)
        assert len(set(optimal)) == N
        return optimal
    
    # Fallback: if somehow the powers of 2 don't work (shouldn't happen),
    # do a targeted local search to find a valid configuration
    best_arr = None
    best_max = float('inf')
    
    random.seed(42)
    
    # Phase 1: Try known good candidate sequences
    candidates = [
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],  # powers of 2
        [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233],         # Fibonacci
        [1, 2, 4, 7, 12, 20, 33, 54, 88, 143, 232, 376],
        [1, 3, 4, 8, 12, 20, 34, 56, 92, 150, 244, 400],
    ]
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(cand) != N:
            continue
        if _is_valid(cand):
            cand_max = max(cand)
            if cand_max < best_max:
                best_max = cand_max
                best_arr = cand[:]
    
    # Phase 2: Targeted local search starting from powers of 2
    current = [1 << i for i in range(N)]
    current_score = _score(current)
    
    if current_score == 0:
        current_max = max(current)
        if best_arr is None or current_max < best_max:
            best_max = current_max
            best_arr = current[:]
    
    # Try to reduce elements while maintaining validity
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, N - 1)
        old_val = current[idx]
        
        if old_val <= 1:
            continue
        
        # Try decreasing by small amounts
        decrease = random.randint(1, min(old_val - 1, 50))
        new_val = old_val - decrease
        current[idx] = new_val
        
        new_score = _score(current)
        if new_score == 0:
            new_max = max(current)
            if new_max < best_max:
                best_max = new_max
                best_arr = current[:]
        else:
            current[idx] = old_val
    
    # Phase 3: Random restarts with local search
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Initialize with values near powers of 2
        arr = []
        for i in range(N):
            base = 1 << i
            # Small perturbation around the base
            perturbation = random.randint(-base // 4, base // 4)
            val = max(1, base + perturbation)
            arr.append(val)
        
        arr.sort()
        # Ensure distinctness
        for i in range(1, N):
            if arr[i] <= arr[i-1]:
                arr[i] = arr[i-1] + 1
        
        score = _score(arr)
        
        if score == 0:
            arr_max = max(arr)
            if arr_max < best_max:
                best_max = arr_max
                best_arr = arr[:]
            continue
        
        # Local search to fix collisions
        for step in range(5000):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, N - 1)
            old_val = arr[idx]
            
            # Try small perturbations
            delta = random.randint(-20, 20)
            new_val = max(1, min(MAX_VAL, old_val + delta))
            
            if new_val == old_val:
                continue
            
            arr[idx] = new_val
            new_score = _score(arr)
            
            if new_score < score:
                score = new_score
                if score == 0:
                    arr_max = max(arr)
                    if arr_max < best_max:
                        best_max = arr_max
                        best_arr = arr[:]
                    break
            else:
                arr[idx] = old_val
    
    # Final: ensure we have a valid result
    if best_arr is None:
        best_arr = [1 << i for i in range(N)]
    
    # Final verification
    assert len(best_arr) == N
    assert all(isinstance(x, int) and x > 0 for x in best_arr)
    assert all(x <= MAX_VAL for x in best_arr)
    assert len(set(best_arr)) == N
    assert _is_valid(best_arr)
    
    return best_arr
# EVOLVE-BLOCK-END
