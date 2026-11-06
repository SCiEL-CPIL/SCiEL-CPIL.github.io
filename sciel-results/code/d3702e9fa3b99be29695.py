"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    # Generate primes up to 1000
    def primes_up_to(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = primes_up_to(1000)
    n_primes = len(primes)
    
    # We need to evaluate L for x in 1..L_max.
    # The score is capped at 10^5, but reaching 1000 is the target.
    # Let's set a reasonable cap for evaluation speed. 
    # If we reach the cap, we can try to extend, but for SA, a fixed window is better for speed.
    # However, the score is the *largest* L. If we use a window of 2000, and all are covered, 
    # we don't know if 2001 is covered. 
    # To be safe and fast, we can evaluate up to a large number, but NumPy makes it fast.
    # Let's use a cap of 5000. If L > 5000, we return 5000 (or we could expand, but 1000 is the goal).
    # Actually, the problem says "capped at 10^5". 
    # Evaluating 10^5 integers with 168 primes is 16.8M operations. NumPy can do this in ~10-50ms.
    # So we can afford to evaluate up to 10,000 or even 100,000 if needed.
    # Let's stick to 10,000 for the SA loop to be fast, and if we exceed it, we return the cap.
    # But wait, the score is the raw value. If the true L is 100,000, we want to report that.
    # However, finding L=100,000 is extremely hard. L=1000 is the target.
    # Let's set eval_cap = 5000. If we hit 5000, we assume it's good enough for the search, 
    # but for the final return, we might want to check further.
    # Actually, let's just use a cap of 2000 for the SA loop to be very fast.
    # If we find a solution with L >= 2000, we can do a final check on a larger range.
    
    eval_cap = 2000
    
    # Initialize residues
    # A good initial guess is random, or based on the seed.
    # Seed: r_p = ((i + 2) // 2) % p
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    
    # Precompute prime array for vectorized ops
    p_arr = np.array(primes, dtype=np.int64)
    
    # Helper to compute L given residues
    def compute_L(res, cap):
        # Create a grid of indices 1..cap
        # For each prime p, the covered indices are r, r+p, r+2p...
        # We can use a boolean mask or a count array.
        # Count array is better because we need "at least two".
        
        # Efficient way:
        # coverage = np.zeros(cap + 1, dtype=np.int8)
        # for i, p in enumerate(primes):
        #     r = res[i]
        #     if r == 0:
        #         start = p
        #     else:
        #         start = r
        #     coverage[start::p] += 1
        
        # Vectorized approach using broadcasting is memory heavy for 168 * 2000.
        # 168 * 2000 = 336,000 elements. That's small.
        # But creating a full matrix is wasteful.
        
        # Let's use the loop but with NumPy slicing which is fast in C.
        coverage = np.zeros(cap + 1, dtype=np.int8)
        for i in range(len(primes)):
            p = p_arr[i]
            r = res[i]
            if r == 0:
                start = p
            else:
                start = r
            # This slice operation is very fast
            coverage[start::p] += 1
        
        # Find the largest L such that coverage[1:L+1] are all >= 2
        # We need the first index where coverage < 2.
        # coverage is 1-indexed effectively (index 0 is unused for x=0, but x starts at 1)
        # Wait, x=1..L.
        # coverage[1], coverage[2]...
        
        # Find first index i in 1..cap where coverage[i] < 2
        # If all are >= 2, L = cap.
        
        # Create a mask of "good" indices (coverage >= 2)
        # We need the prefix of 1s.
        # good = coverage >= 2
        # But we care about indices 1..cap.
        # If coverage[1] < 2, L=0.
        
        # Efficient way to find the break point:
        # Flip the array where coverage < 2.
        # Find the first True in the flipped array (starting from index 1).
        
        bad = coverage < 2
        # We only care about indices 1..cap
        # If bad[1] is True, L=0.
        # We can use np.argmax on a modified array.
        
        # Create a vector of indices 1..cap
        # If we find the first index j in 1..cap where coverage[j] < 2, then L = j - 1.
        # If no such j exists, L = cap.
        
        # Trick: 
        # bad[0] is always True (coverage[0] is 0, which is < 2). 
        # So np.argmax(bad) will always return 0. That's not helpful directly.
        
        # Let's just loop or use a cumulative product?
        # Cumulative product of (coverage >= 2) for indices 1..cap.
        # If the product becomes 0 at index k, then L = k-1.
        
        # mask = (coverage >= 2).astype(np.int8)
        # mask[0] = 1 # Ignore index 0
        # cumprod = np.cumprod(mask)
        # # Find the last index where cumprod is 1
        # # If cumprod[cap] is 1, L = cap
        # # Else, find the first 0.
        
        # This is O(cap). Cap is 2000. This is very fast.
        
        mask = (coverage >= 2).astype(np.int8)
        # We want to know if the prefix 1..L is all 1s.
        # Set mask[0] = 1 so it doesn't break the cumprod immediately if we included 0.
        # But we only care about 1..cap.
        
        # Let's just use a simple loop in Python? 2000 iterations is ~0.1ms.
        # Or np.argmax on the "bad" mask, but we need to handle the case where no bad exists.
        
        # Let's use the cumprod trick, it's robust.
        # We need to check indices 1 to cap.
        # Create a view or copy?
        # mask[1:] is the relevant part.
        
        # If all mask[1:cap+1] are 1, L=cap.
        # Else, find the first 0.
        
        # We can invert: bad_mask = (coverage < 2)
        # We want the first index in 1..cap where bad_mask is True.
        
        # To avoid argmax returning 0 (from index 0), we can set bad_mask[0] = False.
        bad_mask = (coverage < 2)
        bad_mask[0] = False
        
        first_bad = np.argmax(bad_mask)
        
        # If bad_mask[first_bad] is False, it means no bad was found (argmax returns 0 if all False)
        # But we set bad_mask[0] = False.
        # If all are False, argmax returns 0.
        # If bad_mask[5] is True, argmax returns 5.
        
        if bad_mask[first_bad]:
            # The first bad index is first_bad.
            # The covered prefix is 1..first_bad-1.
            L = first_bad - 1
        else:
            L = cap
            
        return L

    # Initial Score
    current_L = compute_L(residues, eval_cap)
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing
    # Temperature starts high, decays.
    # Move: pick a random prime, change its residue.
    # Or: swap residues of two primes? No, residues are modulo different primes.
    # Move: Pick random i, set residues[i] = random.randint(0, primes[i]-1)
    
    T = 1.0
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Generate a neighbor
        new_residues = best_residues.copy()
        idx = random.randint(0, n_primes - 1)
        new_residues[idx] = random.randint(0, primes[idx] - 1)
        
        new_L = compute_L(new_residues, eval_cap)
        
        delta = new_L - current_L
        
        # Accept if better, or with probability exp(delta/T) if worse
        if delta > 0 or (T > 0 and random.random() < np.exp(delta / T)):
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = new_residues.copy()
        
        T *= cooling
        
        # If we hit the cap, we might want to try to extend the cap?
        # For now, we assume the cap is sufficient for the search landscape.
        # If best_L >= eval_cap, we can break early if we are confident, 
        # but let's keep searching to see if we can find a "deeper" structure or just stabilize.
        # Actually, if we hit the cap, we are likely doing well.
        # We can increase the cap for the final check.

    # Final check with a larger cap to get the true score if we exceeded eval_cap
    if best_L >= eval_cap:
        # Try to evaluate on a larger range, say 10000
        large_cap = 10000
        # We need to re-compute L on the large cap for the best residues
        # Re-use compute_L logic but with larger cap
        coverage = np.zeros(large_cap + 1, dtype=np.int8)
        for i in range(len(primes)):
            p = p_arr[i]
            r = best_residues[i]
            if r == 0:
                start = p
            else:
                start = r
            coverage[start::p] += 1
        
        bad_mask = (coverage < 2)
        bad_mask[0] = False
        first_bad = np.argmax(bad_mask)
        if bad_mask[first_bad]:
            final_L = first_bad - 1
        else:
            final_L = large_cap
            
        # If we hit 10000, we might want to check even more? 
        # The problem says capped at 10^5.
        # If final_L >= 10000, we could try 50000. But 1000 is the target.
        # Let's just return what we found.
        best_L = final_L

    return best_residues.tolist()
# EVOLVE-BLOCK-END
