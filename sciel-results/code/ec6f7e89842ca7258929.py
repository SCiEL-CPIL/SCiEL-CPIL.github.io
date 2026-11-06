"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 256
    # Precompute all arithmetic progressions within [1, 256]
    # An AP is defined by start a, step d, and length k
    # The AP is {a, a+d, ..., a+(k-1)d}
    # We need to include all such APs contained in [1, 256]
    
    # For efficiency, we'll store the indices (0-based) for each AP
    # But since n=256, the number of APs might be large. Let's estimate.
    # d=1: starts 1..256, lengths 1..256-start+1 -> sum_{s=1}^{256} (256-s+1) = 256*257/2 = 32896
    # d=2: starts 1..255, lengths 1..floor((256-start)/2)+1
    # Total number of APs is roughly O(n^3) which for n=256 is about 16 million. That's too many to store explicitly in a dense matrix.
    
    # Alternative approach: Instead of precomputing all APs, we can compute the objective function on the fly.
    # For a given sign vector x of length 256, the maximum absolute sum over all APs can be computed by iterating over all (a, d) pairs and computing the sums.
    
    # Let's think about the complexity:
    # Number of (a, d) pairs: a in 1..256, d in 1..255, but a+(k-1)d <= 256 for some k>=1.
    # For each (a, d), the maximum length is floor((256-a)/d) + 1.
    # Total pairs: sum_{a=1}^{256} sum_{d=1}^{255} [a+d <= 256 or d < 256] ... actually for each a, d can go up to 255 but we need at least one element, so d can be anything as long as a <= 256.
    # Actually for each a in 1..256 and d in 1..255, if a <= 256, there's at least one element (the start). So total pairs is 256 * 255 = 65280.
    # For each pair, we compute the maximum absolute prefix sum of the subsequence.
    # The length of the subsequence for (a, d) is floor((256-a)/d) + 1.
    # So total work is sum over all (a,d) of the length, which is sum_{a=1}^{256} sum_{d=1}^{255} (floor((256-a)/d) + 1).
    # This is roughly O(n^2 log n) which for n=256 is manageable.
    
    # Let's use simulated annealing with this on-the-fly evaluation.
    
    def evaluate(x):
        """Compute the maximum absolute sum over all arithmetic progressions."""
        x = np.asarray(x)
        max_abs_sum = 0
        for a in range(1, n + 1):
            for d in range(1, n):
                # Start at a, step d
                # Indices: a-1, a-1+d, a-1+2d, ...
                idx = a - 1
                if idx >= n:
                    break
                # Compute prefix sums
                s = 0
                max_s = 0
                min_s = 0
                while idx < n:
                    s += x[idx]
                    if s > max_s:
                        max_s = s
                    if s < min_s:
                        min_s = s
                    idx += d
                # The maximum absolute sum is max(|max_s|, |min_s|)
                abs_max = max(abs(max_s), abs(min_s))
                if abs_max > max_abs_sum:
                    max_abs_sum = abs_max
        return max_abs_sum
    
    # Simulated Annealing
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    # Initialize
    x = np.ones(n, dtype=np.int8)
    # Random initialization
    for i in range(n):
        if rng.random() < 0.5:
            x[i] = -1
    
    current_x = x.copy()
    current_cost = evaluate(current_x)
    
    best_x = current_x.copy()
    best_cost = current_cost
    
    # SA parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.99
    max_iters = 5000
    iters_per_temp = 20
    
    T = T_start
    for it in range(max_iters):
        # Generate a neighbor by flipping one or a few positions
        # Flip a random position
        i = rng.randrange(n)
        neighbor = current_x.copy()
        neighbor[i] = -neighbor[i]
        
        new_cost = evaluate(neighbor)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
            current_x = neighbor
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = current_x.copy()
        
        T = max(T * alpha, T_end)
        
        # Early stopping if we've reached a good value
        if best_cost <= 2:
            break
    
    return [int(v) for v in best_x]

# EVOLVE-BLOCK-END
