import time
from math import gcd

N = 512
DEADLINE = 55.0

def construct():
    start_time = time.time()
    
    # Precompute compatibility
    # compat[i][j] is True if lcm(i+1, j+1) <= 512
    # Using 0-indexed for lists, but values are 1..512
    # compat_mask[i] is a bitset of j such that lcm(i+1, j+1) <= 512
    
    # Actually, let's just use a list of sets or lists for simplicity first, 
    # but bitsets are faster for "compatible with all in S".
    
    # Let's use a list of integers representing bitmasks.
    # 512 bits fits in 8 Python integers of 64 bits, or just use Python's arbitrary precision ints.
    # Python ints are arbitrary precision, so bit operations are fast.
    
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if (i // gcd(i, j)) * j <= N:
                mask |= (1 << j)
        compat[i] = mask
        
    # Helper to check if x is compatible with all in set_mask
    def is_compat_with_set(x, set_mask):
        # x is 1..512
        # set_mask has bits set for elements in S
        # We need (compat[x] & set_mask) == set_mask
        return (compat[x] & set_mask) == set_mask

    # Helper to get set size
    def popcount(mask):
        return bin(mask).count('1')

    best_set_mask = 0
    best_size = 0
    
    # Seeds
    seeds = []
    
    # Seed 1: 1..22
    mask = 0
    for i in range(1, 23):
        mask |= (1 << i)
    seeds.append(mask)
    
    # Seed 2: Divisors of 360
    mask = 0
    for i in range(1, N + 1):
        if 360 % i == 0:
            mask |= (1 << i)
    seeds.append(mask)
    
    # Seed 3: Divisors of 420
    mask = 0
    for i in range(1, N + 1):
        if 420 % i == 0:
            mask |= (1 << i)
    seeds.append(mask)
    
    # Seed 4: Divisors of 480
    mask = 0
    for i in range(1, N + 1):
        if 480 % i == 0:
            mask |= (1 << i)
    seeds.append(mask)
    
    # Seed 5: Divisors of 504
    mask = 0
    for i in range(1, N + 1):
        if 504 % i == 0:
            mask |= (1 << i)
    seeds.append(mask)
    
    # Seed 6: Divisors of 252
    mask = 0
    for i in range(1, N + 1):
        if 252 % i == 0:
            mask |= (1 << i)
    seeds.append(mask)
    
    for seed in seeds:
        current_mask = seed
        current_size = popcount(current_mask)
        
        # Greedy Add
        improved = True
        while improved:
            improved = False
            # Try to add any compatible number
            # Iterate through 1..N
            for i in range(1, N + 1):
                if not (current_mask & (1 << i)):
                    if is_compat_with_set(i, current_mask):
                        current_mask |= (1 << i)
                        current_size += 1
                        improved = True
                        # Don't break, keep adding if possible
                        # But re-checking all is O(N^2). N=512, so 260k ops. Fast.
        
        if current_size > best_size:
            best_size = current_size
            best_set_mask = current_mask
            
        # Local Search: Swap
        # Try to remove one and add one (or more)
        # To keep it simple and fast, try 1-swap that allows immediate addition of another?
        # Or just 1-swap if it increases size? No, 1-swap keeps size same.
        # We need a move that increases size.
        # Strategy: Try removing u. Then greedily add as many as possible.
        # If final size > current_size, accept.
        
        # Limit local search time per seed
        local_start = time.time()
        
        # Try removing each element
        for u in range(1, N + 1):
            if time.time() - start_time > DEADLINE:
                return list(i for i in range(1, N + 1) if (best_set_mask >> i) & 1)
            
            if not (current_mask & (1 << u)):
                continue
                
            # Remove u
            trial_mask = current_mask ^ (1 << u)
            trial_size = current_size - 1
            
            # Greedy add to trial_mask
            improved = True
            while improved:
                improved = False
                for i in range(1, N + 1):
                    if not (trial_mask & (1 << i)):
                        if is_compat_with_set(i, trial_mask):
                            trial_mask |= (1 << i)
                            trial_size += 1
                            improved = True
            
            if trial_size > current_size:
                current_mask = trial_mask
                current_size = trial_size
                if current_size > best_size:
                    best_size = current_size
                    best_set_mask = current_mask
                # Restart the removal loop? Or just continue?
                # Restarting is safer to find new local optima.
                # But might be slow. Let's just break and restart the whole seed loop?
                # No, just break the inner loop and restart the "for u" loop?
                # Actually, if we improved, we should restart the local search from the new state.
                # Let's use a flag.
                break 
        
        # Check time
        if time.time() - start_time > DEADLINE:
            break

    # Convert best_set_mask to list
    result = [i for i in range(1, N + 1) if (best_set_mask >> i) & 1]
    
    # Final check for validity (safety)
    # If invalid, return seed
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if (result[i] // gcd(result[i], result[j])) * result[j] > N:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return list(range(1, 23))
        
    return result

# EVOLVE-BLOCK-END
