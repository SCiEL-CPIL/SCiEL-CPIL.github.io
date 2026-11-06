import time
import random

def construct():
    n = 12
    deadline = time.time() + 80.0
    best_result = [1 << i for i in range(n)]
    best_max = max(best_result)
    
    # Local search: try to reduce the largest element by swapping or replacing
    # We maintain the property that all subset sums are distinct.
    
    def is_valid(arr):
        sums = set()
        # Fast subset sum check using bit manipulation or iterative addition
        # For n=12, 4096 sums is small.
        # We can compute sums iteratively.
        current_sums = {0}
        for x in arr:
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + x)
            current_sums = current_sums | new_sums
            if len(current_sums) != 2 ** len([1 for _ in range(len(arr))]):
                # This check is inefficient, better to just check size at end or during
                pass
        
        # Simpler: just generate all sums
        sums = set()
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += arr[i]
            if s in sums:
                return False
            sums.add(s)
        return True

    def try_improve(arr, max_time):
        arr = arr[:]
        current_max = max(arr)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Sort to keep track, but order doesn't matter for validity
            # Try to reduce the largest element
            idx_max = arr.index(current_max)
            # Try replacing arr[idx_max] with a smaller value
            # We need to try values from 1 to current_max - 1
            # But we must ensure distinctness and validity.
            # To speed up, we can try random values first, then systematic.
            
            # Heuristic: The largest element is often the one that can be reduced.
            # Let's try to reduce it.
            # We must ensure the new value is not in the set (distinctness).
            existing = set(arr)
            
            # Try random candidates first
            candidates = list(range(1, current_max))
            random.shuffle(candidates)
            
            for new_val in candidates:
                if new_val in existing:
                    continue
                new_arr = arr[:]
                new_arr[idx_max] = new_val
                if is_valid(new_arr):
                    arr = new_arr
                    current_max = max(arr)
                    improved = True
                    break
            # If no random improvement, we might need systematic search or accept current
            # For now, if no improvement in this round, break to avoid infinite loop
            # unless we did improve.
        
        return arr

    # Start with powers of 2
    current = best_result[:]
    
    # Run local search
    # We might run multiple restarts or just one deep search
    # Given the time limit, one deep search is safer.
    
    improved_arr = try_improve(current, deadline)
    if max(improved_arr) < best_max:
        best_result = improved_arr
        best_max = max(best_result)
    
    return best_result
