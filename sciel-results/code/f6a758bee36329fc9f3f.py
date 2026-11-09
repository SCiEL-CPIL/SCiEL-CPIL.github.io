import time
import sys
import random

def _check_validity_bitset(A):
    """Check validity using bitsets for subset sums."""
    n = len(A)
    if n == 0:
        return True
    
    B = [0] * (n + 1)
    B[0] = 1
    
    for a in A:
        for k in range(n - 1, -1, -1):
            if B[k]:
                B[k + 1] |= (B[k] << a)
    
    total = 0
    sum_bits = 0
    for k in range(n + 1):
        total |= B[k]
        sum_bits += B[k]
        
    return total == sum_bits

def _check_validity_fast(A):
    """Faster validity check with early termination."""
    n = len(A)
    if n == 0:
        return True
    
    B = [0] * (n + 1)
    B[0] = 1
    
    for a in A:
        for k in range(n - 1, -1, -1):
            if B[k]:
                B[k + 1] |= (B[k] << a)
    
    # Check for conflicts: no sum should appear in two different cardinality bitsets
    # If any bit is set in more than one B[k], invalid
    combined = 0
    for k in range(n + 1):
        if combined & B[k]:
            return False
        combined |= B[k]
    return True

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_A = []
    
    def time_left():
        return deadline - (time.time() - start_time)

    # Strategy 1: Greedy Ascending with Exact Check
    A = []
    for candidate in range(1, 257):
        if time_left() < 5:
            break
        new_A = A + [candidate]
        if _check_validity_fast(new_A):
            A = new_A
    if len(A) > len(best_A):
        best_A = A[:]

    # Strategy 2: Greedy Descending with Exact Check
    if time_left() > 5:
        A = []
        for candidate in range(256, 0, -1):
            if time_left() < 5:
                break
            new_A = A + [candidate]
            if _check_validity_fast(new_A):
                A = new_A
        if len(A) > len(best_A):
            best_A = A[:]

    # Strategy 3: Start with Powers of 2 and extend
    if time_left() > 5:
        A = [1, 2, 4, 8, 16, 32, 64, 128]
        for candidate in range(1, 257):
            if time_left() < 5:
                break
            if candidate in A:
                continue
            new_A = A + [candidate]
            if _check_validity_fast(new_A):
                A = new_A
        if len(A) > len(best_A):
            best_A = A[:]

    # Strategy 4: Randomized greedy with many restarts
    random.seed(42)
    num_restarts = 50
    for r in range(num_restarts):
        if time_left() < 3:
            break
        
        # Different heuristics for different restarts
        candidates = list(range(1, 257))
        
        if r % 3 == 0:
            random.shuffle(candidates)
        elif r % 3 == 1:
            # Prefer larger numbers
            candidates.sort(reverse=True)
            random.shuffle(candidates[:50])
        else:
            # Prefer medium numbers
            candidates.sort(key=lambda x: abs(x - 128))
            random.shuffle(candidates)
        
        A = []
        for candidate in candidates:
            if time_left() < 3:
                break
            new_A = A + [candidate]
            if _check_validity_fast(new_A):
                A = new_A
        if len(A) > len(best_A):
            best_A = A[:]

    # Strategy 5: Local search - try to add one element at a time
    if time_left() > 3:
        A = best_A[:]
        improved = True
        while improved and time_left() > 1:
            improved = False
            n = len(A)
            
            # Try to add one element
            for candidate in range(1, 257):
                if time_left() < 1:
                    break
                if candidate in A:
                    continue
                new_A = A + [candidate]
                if _check_validity_fast(new_A):
                    A = new_A
                    improved = True
                    break
            
            if len(A) > len(best_A):
                best_A = A[:]
            if not improved:
                break

    # Strategy 6: Try replacing elements to enable adding more
    if time_left() > 5:
        A = best_A[:]
        n = len(A)
        
        # Try removing one element and adding two different ones
        for i in range(n):
            if time_left() < 3:
                break
            base = A[:i] + A[i+1:]
            
            # Try to add two elements to base
            for c1 in range(1, 257):
                if time_left() < 2:
                    break
                if c1 in base:
                    continue
                mid = base + [c1]
                if not _check_validity_fast(mid):
                    continue
                
                for c2 in range(1, 257):
                    if time_left() < 1:
                        break
                    if c2 in mid:
                        continue
                    new_A = mid + [c2]
                    if _check_validity_fast(new_A):
                        if len(new_A) > len(best_A):
                            best_A = new_A[:]
                        A = new_A[:]
                        n = len(A)
                        break

    return [int(x) for x in best_A]
