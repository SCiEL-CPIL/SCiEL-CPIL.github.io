import time
import heapq

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Generate all (value, divisor_index) pairs
    events = []
    for i, d in enumerate(DIVISORS):
        for multiple in range(d, MAX_VAL + 1, d):
            events.append((multiple, i))
    
    # 2. Sort by value
    events.sort(key=lambda x: x[0])
    
    # 3. Sliding window to find top K minimal spans
    # We use a heap to keep the top 100 smallest windows
    # Heap elements: (width, L, R)
    # Since heapq is min-heap, we can't easily keep top-K smallest by pushing all.
    # We can push all, then sort. 
    # Or we can just keep the best one, and if we need more, we can't easily get the 2nd best from a single pass 
    # without storing more info.
    # Storing all is simplest.
    
    windows = []
    
    counts = [0] * N
    formed = 0
    left = 0
    
    for right in range(len(events)):
        val_r, idx_r = events[right]
        counts[idx_r] += 1
        if counts[idx_r] == 1:
            formed += 1
            
        while formed == N:
            val_l, idx_l = events[left]
            cur_width = val_r - val_l
            
            windows.append((cur_width, val_l, val_r))
            
            counts[idx_l] -= 1
            if counts[idx_l] == 0:
                formed -= 1
            left += 1
            
        if time.time() > deadline:
            break

    if not windows:
        return [d for d in DIVISORS]

    # Sort windows by width
    windows.sort(key=lambda x: x[0])
    
    # Extract top 50 unique windows
    seen = set()
    unique_windows = []
    for w, l, r in windows:
        key = (l, r)
        if key not in seen:
            seen.add(key)
            unique_windows.append((w, l, r))
        if len(unique_windows) >= 50:
            break

    # 4. Try to find a valid assignment in these windows
    for w, L, R in unique_windows:
        candidates = [[] for _ in range(N)]
        
        for i, d in enumerate(DIVISORS):
            # First multiple >= L
            k = (L + d - 1) // d
            val = k * d
            while val <= R:
                candidates[i].append(val)
                val += d
        
        # Backtracking
        assignment = [None] * N
        used = set()
        order = sorted(range(N), key=lambda i: len(candidates[i]))
        
        def backtrack(idx):
            if idx == N:
                return True
            i = order[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
            return False

        if backtrack(0):
            return assignment

    # Fallback
    return [d for d in DIVISORS]
