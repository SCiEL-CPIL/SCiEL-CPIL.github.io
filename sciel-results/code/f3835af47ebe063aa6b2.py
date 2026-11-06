"""Rewritten program: spin-system subgradient descent for minimizing monochromatic 4-APs."""
import numpy as np
import time
import sys

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all 4-term APs as index tuples (0-based)
    # a, a+d, a+2d, a+3d where 1 <= a, a+3d <= N
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(aps)
    aps_array = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)

    def count_mono(color):
        """Count monochromatic 4-APs for a binary color vector (0-based indices)."""
        # color: numpy array of shape (N,) with values 0 or 1
        # For each AP (i, j, k, l), it's monochromatic if all four have same color
        # color[i] == color[j] == color[k] == color[l]
        # This is: (1 - |c_i - c_j|) * (1 - |c_j - c_k|) * (1 - |c_k - c_l|)
        # But easier: all equal means (c_i == c_j) and (c_j == c_k) and (k == l)
        # Using: indicator = 1 if all same, 0 otherwise
        # For binary: all same iff (c_i + c_j + c_k + c_l) is 0 or 4
        # So mono = 1 if sum is 0 or 4
        ci = color[aps_array[:, 0]]
        cj = color[aps_array[:, 1]]
        ck = color[aps_array[:, 2]]
        cl = color[aps_array[:, 3]]
        total = ci + cj + ck + cl
        return int(np.sum((total == 0) | (total == 4)))

    def delta_for_flip(color, pos):
        """
        Compute the change in monochromatic count if we flip color[pos].
        Returns the delta (new_count - old_count).
        We need to find all APs that include position 'pos'.
        """
        # Find all APs containing position pos
        # An AP (i, j, k, l) contains pos if pos is one of i, j, k, l
        mask = (aps_array[:, 0] == pos) | (aps_array[:, 1] == pos) | \
               (aps_array[:, 2] == pos) | (aps_array[:, 3] == pos)
        relevant_aps = aps_array[mask]
        
        if len(relevant_aps) == 0:
            return 0
        
        old_delta = 0
        new_delta = 0
        for ap in relevant_aps:
            i, j, k, l = ap
            # Get colors
            ci, cj, ck, cl = color[i], color[j], color[k], color[l]
            # Old: is this AP monochromatic?
            old_mono = 1 if (ci == cj == ck == cl) else 0
            # New: flip the color at pos
            new_ci = ci
            new_cj = cj
            new_ck = ck
            new_cl = cl
            if i == pos:
                new_ci = 1 - ci
            elif j == pos:
                new_cj = 1 - cj
            elif k == pos:
                new_ck = 1 - ck
            elif l == pos:
                new_cl = 1 - cl
            new_mono = 1 if (new_ci == new_cj == new_ck == new_cl) else 0
            old_delta += old_mono
            new_delta += new_mono
        
        return new_delta - old_delta

    # Initialize with a random coloring
    np.random.seed(42)
    color = np.random.randint(0, 2, N).astype(np.int32)
    best_color = color.copy()
    best_count = count_mono(best_color)

    # Simulated annealing with vectorized updates
    # We'll do multiple passes, each pass trying to flip bits
    # Use a temperature schedule
    
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    max_iterations = 50000
    
    iteration = 0
    while time.time() < deadline and iteration < max_iterations:
        # Try a batch of random flips
        batch_size = 100
        for _ in range(batch_size):
            if time.time() > deadline:
                break
            
            pos = np.random.randint(0, N)
            delta = delta_for_flip(color, pos)
            
            if delta <= 0:
                # Accept (improvement or neutral)
                color[pos] = 1 - color[pos]
                new_count = best_count  # We'll recompute properly
                # Actually, let's track count properly
                # For efficiency, let's just recompute periodically
            else:
                # Accept with probability exp(-delta / T)
                if np.random.random() < np.exp(-delta / T):
                    color[pos] = 1 - color[pos]
        
        # Periodically recompute the exact count and update best
        if iteration % 10 == 0:
            current_count = count_mono(color)
            if current_count < best_count:
                best_count = current_count
                best_color = color.copy()
        
        T = max(T * cooling_rate, T_min)
        iteration += 1

    # Final polish: greedy local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for pos in range(N):
            delta = delta_for_flip(color, pos)
            if delta < 0:
                color[pos] = 1 - color[pos]
                improved = True
    
    final_count = count_mono(color)
    if final_count < best_count:
        best_count = final_count
        best_color = color.copy()

    # Convert to list of 0s and 1s for positions 1..256
    result = [int(best_color[i]) for i in range(N)]
    return result
