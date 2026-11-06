"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80
    
    def is_valid(arr):
        """Check if all subset sums are distinct."""
        n = len(arr)
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

    def try_reduce(arr, idx, max_val):
        """Try to reduce arr[idx] to a smaller positive integer while maintaining validity."""
        others = arr[:idx] + arr[idx+1:]
        used = set(others)
        
        for val in range(1, max_val):
            if val in used:
                continue
            new_arr = others[:idx] + [val] + others[idx:]
            if is_valid(new_arr):
                return val
        return None

    # Start with powers of 2 (known valid)
    arr = [1 << i for i in range(12)]
    
    # Try to reduce each element from smallest to largest
    for _pass in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        for idx in range(12):
            if time.time() > deadline - 5:
                break
            current = arr[idx]
            # Try to find a smaller value
            new_val = try_reduce(arr, idx, current)
            if new_val is not None and new_val < current:
                arr[idx] = new_val
                improved = True
        if not improved:
            break
    
    return arr
# EVOLVE-BLOCK-END
