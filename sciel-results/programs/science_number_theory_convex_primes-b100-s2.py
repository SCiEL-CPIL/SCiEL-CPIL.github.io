"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def construct():
    deadline = time.time() + 50.0  # safety margin before 60s hard kill

    # ---- Sieve of primes up to 10000 ----
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = [False] * (((N - start) // step) + 1)
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    primes_arr = np.array(primes, dtype=np.int32)

    # ---- Phase 1: Greedy from each prime ----
    best = []

    def validate(chain):
        if not chain:
            return False
        if chain[-1] > N:
            return False
        for p in chain:
            if not is_prime[p]:
                return False
        if len(chain) < 2:
            return True
        for a, b in zip(chain, chain[1:]):
            if b <= a:
                return False
        for i in range(1, len(chain) - 1):
            if chain[i + 1] - chain[i] < chain[i] - chain[i - 1]:
                return False
        return True

    def greedy_from(start_idx):
        """Greedily extend from prime at start_idx."""
        if start_idx >= n:
            return [primes[start_idx]]
        chain = [primes[start_idx]]
        last_gap = 0
        for j in range(start_idx + 1, n):
            p = primes[j]
            gap = p - chain[-1]
            if gap >= last_gap:
                chain.append(p)
                last_gap = gap
        return chain

    for i in range(n):
        if time.time() > deadline:
            break
        chain = greedy_from(i)
        if validate(chain) and len(chain) > len(best):
            best = chain

    # ---- Phase 2: DP with numpy ----
    # dp[j][g] = max length of chain ending at prime j with last gap g
    # g ranges from 0 to max_gap
    max_gap = primes_arr[-1] - primes_arr[0]  # at most 9999
    # Use a 2D array: n x (max_gap+1), but this is ~1229 x 10000 = 12.3M, too big for float
    # Instead, use int16 or int8. Max chain length is small (< 20), so int8 is fine.
    
    # Actually, let's think about the max possible gap. The gap between consecutive primes
    # up to 10000 is at most about 200. But our gaps can be larger since we skip primes.
    # However, for a valid chain, gaps are nondecreasing, so the first gap is the smallest.
    # The first gap is at most the gap between the first two primes in the chain.
    # Since we're looking for long chains, we want small gaps.
    # Let's cap the gap search at, say, 500.
    
    max_g = 500
    dp = np.zeros((n, max_g + 1), dtype=np.int8)
    # Initialize: each prime alone has chain length 1, gap 0
    dp[:, 0] = 1

    # cum_max[j][g] = max(dp[j][0..g])
    # We'll compute this on the fly or maintain it.

    # For each prime j (from high to low is not needed; we go left to right)
    # dp[j][g] = 1 + max over i < j, primes[j] - primes[i] == g, and h <= g of dp[i][h]
    # Actually, the previous gap h must satisfy h <= g (nondecreasing gaps).
    # So dp[j][g] = 1 + max(0, max_{h=0..g} dp[i][h]) where i = prime_index[primes[j] - g]
    
    # Let's precompute for each g and j, whether there's an i such that primes[j] - primes[i] = g
    # This is expensive. Instead, for each j and g, compute i = prime_index.get(primes[j] - g)
    
    best_len = 1
    best_j = 0
    best_g = 0

    # Process primes in order
    for j in range(n):
        if time.time() > deadline:
            break
        p_j = primes[j]
        # For this j, what gaps g are possible?
        # g must be such that p_j - g is a prime (and < p_j)
        # g ranges from 1 to max_g, but also p_j - g >= 2
        min_g = 1
        max_g_for_j = min(max_g, p_j - 2)
        
        # We want to compute dp[j][g] for g in [min_g, max_g_for_j]
        # dp[j][g] = 1 + max_{h=0..g} dp[i][h] where i = prime_index[p_j - g]
        
        # To be efficient, we can iterate over possible previous primes i < j
        # and compute g = p_j - primes[i], then if g <= max_g, update dp[j][g]
        
        # For each i in range(j):
        #   g = p_j - primes[i]
        #   if g > max_g: break (since primes are increasing, g decreases as i increases... wait no)
        #   Actually as i increases, primes[i] increases, so g decreases.
        #   So for i from j-1 down to 0: g increases.
        #   We want g <= max_g, so we need primes[i] >= p_j - max_g
        #   Find the smallest i such that primes[i] >= p_j - max_g
        
        # Let's use binary search or just iterate
        # For efficiency, iterate i from j-1 down while primes[j] - primes[i] <= max_g
        
        # Actually, let's just iterate i from 0 to j-1 and check g
        # But that's O(n^2 * max_g) which is too slow.
        
        # Better: for each j, iterate over possible g values that are gaps to a prime
        # For g in range(1, max_g_for_j + 1):
        #   p_prev = p_j - g
        #   if p_prev is prime: i = prime_index[p_prev]
        #   dp[j][g] = 1 + cum_max[i][g]
        
        # We need cum_max[i][g] = max(dp[i][0..g])
        # We can maintain a running cumulative max for each row as we go.
        
        # Let's precompute cum_max for all rows up to the current j.
        # Actually, since we process j in order, when we're at j, all i < j have their dp computed.
        # We can compute cum_max[i] on the fly or maintain it.
        
        # Let's maintain a 2D cum_max array, or compute it per row.
        
        # For simplicity, let's compute cum_max for each i when needed.
        # But that's O(max_g) per query, and we have O(max_g) queries per j, so O(max_g^2) per j.
        # With max_g = 500, that's 250000 per j, times 1229 j's = 300M, too slow.
        
        # Better: precompute cum_max for all i < j before processing j.
        # Or maintain cum_max as a 2D array and update it.
        
        # Let's maintain cum_max as a 2D array of the same shape as dp.
        
        # Actually, let's just compute dp[j][g] by iterating over i and updating.
        # For each i < j:
        #   g = p_j - primes[i]
        #   if 1 <= g <= max_g:
        #       # We need max_{h=0..g} dp[i][h]
        #       # Let's precompute prefix max for row i
        #       # This is getting complex.
        
        # Alternative approach: for each j, iterate over i from j-1 down to 0
        # while p_j - primes[i] <= max_g.
        # For each such i, g = p_j - primes[i].
        # We want to update dp[j][g] = max(dp[j][g], 1 + max_{h=0..g} dp[i][h])
        
        # To get max_{h=0..g} dp[i][h] quickly, we can maintain a cumulative max array for each row.
        
        # Let's maintain cum_max[i] as a numpy array for each i, computed once.
        
        # Actually, let's just use a different approach: iterate over g for each j.
        
    # This is getting too complex. Let me use a simpler DP with a different structure.
    
    # New approach: dp[i][g] = max chain length ending at prime i with last gap g.
    # We process primes in order. For each prime j, for each possible gap g (1 to max_g),
    # if p_j - g is a prime at index i, then:
    #   dp[j][g] = 1 + max_{h=0..g} dp[i][h]
    
    # To compute max_{h=0..g} dp[i][h] quickly, we maintain a cumulative max array.
    
    # Let's allocate dp and cum_max as 2D arrays.
    # n x (max_g+1) with max_g=500 is 1229*501 = 615,729, which is fine.
    
    dp2 = np.zeros((n, max_g + 1), dtype=np.int16)
    cum_max2 = np.zeros((n, max_g + 1), dtype=np.int16)
    
    # Initialize
    for j in range(n):
        dp2[j, 0] = 1
        cum_max2[j, 0] = 1
        # For g > 0, initially 0
    
    best_len2 = 1
    best_j2 = 0
    best_g2 = 0
    
    for j in range(n):
        if time.time() > deadline:
            break
        p_j = primes[j]
        
        # Update cum_max2[j] to be the cumulative max of dp2[j]
        # Actually, we should compute cum_max2[j] after dp2[j] is fully computed.
        # But dp2[j] depends on previous rows, so we compute dp2[j] first, then cum_max2[j].
        
        # For each g from 1 to max_g:
        for g in range(1, max_g + 1):
            p_prev = p_j - g
            if p_prev < 2:
                break
            i = prime_index.get(p_prev)
            if i is None:
                continue
            if i >= j:
                continue
            # dp2[j][g] = 1 + cum_max2[i][g]
            val = 1 + cum_max2[i, g]
            if val > dp2[j, g]:
                dp2[j, g] = val
        
        # Now compute cum_max2[j] = cumulative max of dp2[j]
        if max_g > 0:
            cum_max2[j, :] = np.maximum.accumulate(dp2[j, :])
        
        # Update best
        for g in range(max_g + 1):
            if dp2[j, g] > best_len2:
                best_len2 = dp2[j, g]
                best_j2 = j
                best_g2 = g
    
    # Reconstruct chain from DP
    if best_len2 > len(best):
        chain = []
        j = best_j2
        g = best_g2
        while j is not None and g is not None:
            chain.append(primes[j])
            if g == 0:
                break
            p_prev = primes[j] - g
            i = prime_index.get(p_prev)
            if i is None:
                break
            # Find the h in [0, g] that maximized dp2[i][h]
            # cum_max2[i][g] = max(dp2[i][0..g])
            # We need to find which h achieved this max.
            sub = dp2[i, 0:g+1]
            h = int(np.argmax(sub))
            # But we need h such that dp2[i][h] == cum_max2[i][g]
            # argmax gives the first occurrence, which should be fine
            j = i
            g = h
        chain.reverse()
        if validate(chain) and len(chain) > len(best):
            best = chain
    
    # ---- Phase 3: Try to extend the best chain greedily ----
    if best and len(best) > 0:
        last = best[-1]
        if len(best) >= 2:
            last_gap = best[-1] - best[-2]
        else:
            last_gap = 0
        # Try to extend
        start_idx = prime_index.get(last)
        if start_idx is not None:
            for j in range(start_idx + 1, n):
                p = primes[j]
                gap = p - last
                if gap >= last_gap:
                    best.append(p)
                    last_gap = gap
                    last = p
    
    if not validate(best):
        best = [2, 3, 5]
    
    return best
# EVOLVE-BLOCK-END
