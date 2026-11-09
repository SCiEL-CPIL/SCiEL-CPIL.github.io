import numpy as np
import time
import random

def construct():
    """
    Construct 256 signs (-1, +1) minimizing the maximum absolute sum over any arithmetic progression.
    
    Strategy:
    1. Start with random initialization
    2. Use local search (flipping individual signs) to minimize the maximum AP sum
    3. Use simulated annealing to escape local minima
    4. Track best solution found and return it before time limit
    """
    n = 256
    deadline = time.time() + 45  # 45 seconds to be safe
    
    # Precompute all arithmetic progressions that need to be checked
    # For each AP (start, step, length), compute the indices
    # To make this efficient, we'll focus on the most critical APs
    
    # Generate key APs to monitor:
    # - All singletons (trivial, sum = ±1)
    # - Small steps (1, 2, 3, ...) with various lengths
    # - We'll use a representative set
    
    # Actually, let's use a smarter approach:
    # Focus on APs with step <= some threshold and reasonable lengths
    # The worst-case APs are likely to have small steps
    
    # Precompute AP index lists for efficiency
    # For each step s and start i, find all APs of various lengths
    # We'll sample a subset to keep evaluation fast
    
    # Let's precompute a set of critical APs
    critical_aps = []
    
    # All APs with step = 1 (contiguous subarrays) - most critical
    # But there are O(n^2) of them, so we'll sample or use a windowed approach
    # For step = 1, we can use prefix sums to evaluate quickly
    
    # Strategy: Use a combination of:
    # 1. All APs with step <= 3 (dense)
    # 2. Sampled APs with larger steps
    # 3. Use prefix sum arrays for fast evaluation
    
    # Precompute prefix sums for each step
    # For step s, we can compute prefix sums for each residue class mod s
    
    def get_ap_sum(signs, start, step, length):
        """Compute sum of signs[start], signs[start+step], ..., signs[start+(length-1)*step] (0-indexed)"""
        end = start + (length - 1) * step
        if end >= n:
            return 0
        return sum(signs[start:end+1:step])
    
    # Better: Precompute for each step s, for each residue r in 0..s-1,
    # the prefix sums along that arithmetic progression
    
    def build_prefix_sums(signs, max_step):
        """Build prefix sum arrays for fast AP sum queries"""
        prefix_sums = {}
        for s in range(1, max_step + 1):
            prefix_sums[s] = {}
            for r in range(s):
                # Collect all indices with index % s == r
                indices = list(range(r, n, s))
                if not indices:
                    prefix_sums[s][r] = [0]
                    continue
                # Build prefix sum
                ps = [0]
                for idx in indices:
                    ps.append(ps[-1] + signs[idx])
                prefix_sums[s][r] = ps
        return prefix_sums
    
    def ap_sum_fast(prefix_sums, start, step, length):
        """Fast AP sum using precomputed prefix sums"""
        if step not in prefix_sums:
            return get_ap_sum([0]*n, start, step, length)  # fallback
        r = start % step
        ps = prefix_sums[step][r]
        # Find position of 'start' in the sequence of indices with residue r
        pos = start // step
        end_pos = pos + length - 1
        if end_pos >= len(ps) - 1:
            end_pos = len(ps) - 1
        return ps[end_pos + 1] - ps[pos]
    
    # Initialize with random signs
    rng = random.Random(42)
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    
    # Track best solution
    best_signs = signs.copy()
    best_max_sum = float('inf')
    
    # Precompute a set of APs to evaluate
    # We'll use all APs with step <= 5 and length >= 2, plus some with larger steps
    # But this could be expensive, so let's be selective
    
    # Generate a fixed set of APs to monitor
    monitor_aps = set()
    
    # All APs with step = 1 (contiguous), length >= 3
    for start in range(n):
        for length in range(3, min(20, n - start) + 1):
            monitor_aps.add((start, 1, length))
    
    # APs with step = 2, 3, 4, 5, length >= 2
    for step in range(2, 6):
        for start in range(n):
            max_len = (n - 1 - start) // step + 1
            for length in range(2, min(10, max_len) + 1):
                monitor_aps.add((start, step, length))
    
    # Some APs with larger steps
    for step in range(6, 16):
        for start in range(n):
            max_len = (n - 1 - start) // step + 1
            if max_len >= 2:
                for length in range(2, min(5, max_len) + 1):
                    monitor_aps.add((start, step, length))
    
    # Convert to list for faster iteration
    monitor_aps_list = list(monitor_aps)
    
    def evaluate_signs(s, prefix_sums=None):
        """Evaluate the maximum absolute AP sum for a given sign assignment"""
        if prefix_sums is None:
            prefix_sums = build_prefix_sums(s, 15)
        
        max_abs = 0
        for start, step, length in monitor_aps_list:
            val = ap_sum_fast(prefix_sums, start, step, length)
            abs_val = abs(val)
            if abs_val > max_abs:
                max_abs = abs_val
        return max_abs
    
    # Initial evaluation
    prefix_sums = build_prefix_sums(signs, 15)
    current_max = evaluate_signs(signs, prefix_sums)
    best_max_sum = current_max
    best_signs = signs.copy()
    
    # Local search with simulated annealing
    temperature = 10.0
    cooling_rate = 0.995
    min_temperature = 0.5
    
    while time.time() < deadline and temperature > min_temperature:
        # Pick a random position to flip
        pos = rng.randrange(n)
        
        # Try flipping
        signs[pos] = -signs[pos]
        
        # Rebuild prefix sums (expensive, but necessary for accurate evaluation)
        # Optimization: Only rebuild affected prefix sums
        # For now, rebuild all for correctness
        
        prefix_sums = build_prefix_sums(signs, 15)
        new_max = evaluate_signs(signs, prefix_sums)
        
        delta = new_max - current_max
        
        # Accept or reject
        if delta < 0 or rng.random() < np.exp(-delta / temperature):
            current_max = new_max
            if new_max < best_max_sum:
                best_max_sum = new_max
                best_signs = signs.copy()
        else:
            # Revert
            signs[pos] = -signs[pos]
            prefix_sums = build_prefix_sums(signs, 15)
        
        # Cool down
        temperature *= cooling_rate
    
    return best_signs
