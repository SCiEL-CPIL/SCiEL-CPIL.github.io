"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s safety margin for 60s limit
    
    MAX_N = 1000000
    WINDOW_SIZE = 24
    MAX_VAL = MAX_N + WINDOW_SIZE
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    p_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute mask for each number
    # mask[i] is a bitmask where bit j is set if prime[j] divides i
    masks = [0] * (MAX_VAL + 1)
    
    for p_idx, p in enumerate(primes):
        bit = 1 << p_idx
        for multiple in range(p, MAX_VAL + 1, p):
            masks[multiple] |= bit
    
    # Sliding window to find minimum distinct factors
    # We want to minimize the popcount of (masks[n+1] | masks[n+2] | ... | masks[n+24])
    
    # Initial window [1, 24] corresponds to n=0, but n must be >= 1000
    # Let's start from n=1000
    # Window for n is [n+1, n+24]
    
    best_n = 1000
    best_score = float('inf')
    
    # We can optimize by checking windows in a way that allows early termination if we find a very low score
    # The minimum possible score is 0 (if all small parts are 1, which means no small primes divide them).
    # This is impossible for a window of 24 consecutive integers because among any 2 consecutive integers, one is even (divisible by 2).
    # So 2 is always a factor. Thus score >= 1.
    # Can we get score = 1? Only if all numbers are powers of 2 (times large primes).
    # Consecutive powers of 2 don't exist in a window of 24.
    # So score is likely >= 2 or 3.
    
    # Let's just scan.
    # To speed up, we can compute the union mask for the first window, then slide.
    # However, union is not easily updatable with sliding window (removing an element doesn't allow easy removal from union).
    # Given MAX_VAL is ~1e6, and window is 24, a simple loop is O(24 * 1e6) ~ 2.4e7 operations, which is fast in Python if optimized or in C (NumPy).
    # Let's use a simple loop first, as 2.4e7 iterations in Python might take ~2-5 seconds.
    
    # Optimization: 
    # If we find a window with score 1, we can stop.
    # If we find score 2, we can stop if we are confident 1 is impossible (it is).
    
    # Let's use NumPy for vectorized operations if possible, but simple Python loop with bit ops is often competitive for this size.
    # Let's try a Python loop with early break.
    
    # Calculate initial window for n = 1000
    # We need to check n from 1000 to MAX_N
    
    # A faster approach in Python:
    # Iterate n.
    # For each n, compute union of masks[n+1 ... n+24].
    
    # Since 2.4e7 iterations is borderline for 50s in pure Python, let's try to optimize.
    # We can check if the current best is 1. If not, keep searching.
    
    # Let's write a tight loop.
    current_union = 0
    # We can't easily maintain a sliding union.
    # But we can just recompute.
    
    # Alternative: 
    # The number of distinct primes is small (9).
    # We can count how many of the 9 primes appear in the window.
    # A prime p appears if any number in the window is divisible by p.
    # This is equivalent to: NOT (all numbers in window are NOT divisible by p).
    # So for each prime p, check if there is a multiple of p in [n+1, n+24].
    # This is a range query.
    # For a fixed p, the condition "no multiple of p in [L, R]" is equivalent to floor(R/p) == floor((L-1)/p).
    
    # This is much faster! O(9) per window.
    # Total ops: 1e6 * 9 = 9e6. Very fast.
    
    best_n = 1000
    best_count = 9 # Max possible
    
    for n in range(1000, MAX_N + 1):
        L = n + 1
        R = n + 24
        
        count = 0
        for p in primes:
            # Check if there is a multiple of p in [L, R]
            # Multiple exists if floor(R/p) > floor((L-1)/p)
            if R // p != (L - 1) // p:
                count += 1
        
        if count < best_count:
            best_count = count
            best_n = n
            
            # If we found the theoretical minimum (which is likely 2 or 3, but let's see), we can potentially stop early if we hit 1 (impossible) or if we have enough time.
            # Actually, 1 is impossible. 2 is possible?
            # If count is 2, it means only 2 primes divide the numbers.
            # Since 2 always divides at least one, and 3 divides at least one in any 3 consecutive, etc.
            # In 24 consecutive, 2, 3, 5, 7, 11, 13, 17, 19, 23...
            # It is very likely that many primes appear.
            # However, if we find a very low count, we should record it.
            
            # If best_count == 1, break (impossible)
            # If best_count == 2, it's a strong candidate.
            
        if time.time() > deadline:
            break
            
    return best_n

# EVOLVE-BLOCK-END
