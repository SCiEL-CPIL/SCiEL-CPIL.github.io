import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Precompute all 4-term arithmetic progressions
    # A 4-AP: a, a+d, a+2d, a+3d with a >= 1, a+3d <= N, d >= 1
    aps = []
    ap_indices = [[] for _ in range(N)]  # ap_indices[i] = list of AP IDs containing position i (0-indexed)
    
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx1 = a - 1
            idx2 = a + d - 1
            idx3 = a + 2 * d - 1
            idx4 = a + 3 * d - 1
            ap_id = len(aps)
            aps.append((idx1, idx2, idx3, idx4))
            for idx in (idx1, idx2, idx3, idx4):
                ap_indices[idx].append(ap_id)
    
    total_aps = len(aps)
    
    def count_mono(colors):
        """Count monochromatic 4-APs."""
        count = 0
        for ap in aps:
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            if c0 == c1 == c2 == c3:
                count += 1
        return count
    
    # Simulated annealing
    best_colors = None
    best_count = float('inf')
    
    # Multiple restarts
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: random coloring or alternating pattern
        if restart == 0:
            # Start with alternating pairs (same as seed)
            colors = [(i // 2) % 2 for i in range(N)]
        elif restart == 1:
            # Start with random
            colors = [random.randint(0, 1) for _ in range(N)]
        else:
            # Start with best found so far, perturbed
            if best_colors is not None:
                colors = best_colors[:]
                # Flip ~10% of bits
                for i in range(N):
                    if random.random() < 0.1:
                        colors[i] ^= 1
            else:
                colors = [random.randint(0, 1) for _ in range(N)]
        
        current_count = count_mono(colors)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
        
        # Simulated annealing parameters
        T = 5.0  # initial temperature
        T_min = 0.01
        cooling = 0.9995
        max_steps = 50000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick a random position to flip
            p = random.randint(0, N - 1)
            old_color = colors[p]
            
            # Compute delta in monochromatic count
            # Only APs containing p can change
            delta = 0
            for ap_id in ap_indices[p]:
                ap = aps[ap_id]
                # Count how many of the 4 positions have the same color as each other
                # Before flip
                colors_before = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                # After flip
                colors_after = colors_before[:]
                pos_in_ap = ap.index(p)
                colors_after[pos_in_ap] = 1 - old_color
                
                # Check if mono before
                mono_before = all(c == colors_before[0] for c in colors_before)
                # Check if mono after
                mono_after = all(c == colors_after[0] for c in colors_after)
                
                if mono_before and not mono_after:
                    delta -= 1
                elif not mono_before and mono_after:
                    delta += 1
            
            # Accept or reject
            if delta <= 0:
                # Improvement or neutral - always accept
                colors[p] = 1 - old_color
                current_count += delta
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors[:]
            else:
                # Worse - accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    colors[p] = 1 - old_color
                    current_count += delta
            
            T = max(T_min, T * cooling)
        
        # Local search (hill climbing) from current state
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for p in range(N):
                if time.time() > deadline:
                    break
                old_color = colors[p]
                delta = 0
                for ap_id in ap_indices[p]:
                    ap = aps[ap_id]
                    colors_before = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                    colors_after = colors_before[:]
                    pos_in_ap = ap.index(p)
                    colors_after[pos_in_ap] = 1 - old_color
                    mono_before = all(c == colors_before[0] for c in colors_before)
                    mono_after = all(c == colors_after[0] for c in colors_after)
                    if mono_before and not mono_after:
                        delta -= 1
                    elif not mono_before and mono_after:
                        delta += 1
                if delta < 0:
                    colors[p] = 1 - old_color
                    current_count += delta
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = colors[:]
    
    # Return best colors found
    if best_colors is not None:
        return best_colors
    else:
        return [(i // 2) % 2 for i in range(N)]

import math
