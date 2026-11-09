import time
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0  # Safety margin before 150s
    
    N = 256
    
    # Precompute all 4-APs
    ap_indices = []
    ap_list = [[] for _ in range(N)]
    others = [] # others[ap_id] = tuple of 3 indices
    ap_id = 0
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(N - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            ap_indices.append(ap)
            for idx in ap:
                ap_list[idx].append(ap_id)
            
            # Precompute others for fast delta check
            # others[ap_id] will be the 3 indices other than the one being flipped? 
            # No, 'others' depends on which index is flipped.
            # So we can't precompute a single 'others' list per AP.
            # We just use ap_indices[ap_id] and ignore the flipped index.
            ap_id += 1
            
    total_aps = len(ap_indices)
    
    # Initialize with Thue-Morse sequence
    def thue_morse(n):
        return bin(n).count('1') % 2
        
    colors = np.array([thue_morse(i) for i in range(N)], dtype=int)
    
    # Function to count mono 4-APs
    def count_mono_4aps(colors):
        count = 0
        # Vectorized count is hard without large memory. 
        # Loop is fast enough for initial and final checks.
        for d in range(1, (N - 3) // 3 + 1):
            for a in range(N - 3 * d):
                if colors[a] == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                    count += 1
        return count

    current_count = count_mono_4aps(colors)
    best_count = current_count
    best_colors = colors.copy()
    
    # Maintain ap_mono array for fast delta updates
    ap_mono = np.zeros(total_aps, dtype=int)
    for i in range(total_aps):
        ap = ap_indices[i]
        if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
            ap_mono[i] = 1
            
    # Simulated Annealing
    T_start = 10.0
    T_end = 0.01
    # We don't know exact number of steps, so we will just run until time limit.
    # We need a cooling schedule. 
    # Let's estimate steps we can do. 
    # Each step is ~100-200 ns? No, Python loop is slower. ~1-5 us.
    # 120s / 5us = 24M steps.
    # We can define a fixed number of steps or just loop while time < limit.
    
    # To make cooling work, we need to know total steps or use a fixed schedule.
    # Let's use a fixed number of iterations based on a guess, or just linear cooling over time.
    # Better: Fixed number of iterations per "phase" or just run until time limit with T decreasing.
    
    # Let's try 500,000 iterations as a baseline, or just loop.
    # If we loop, T needs to decrease. 
    # Let's assume we can do ~1,000,000 steps in 100s.
    # So T = T_start * (T_end/T_start) ** (step / max_steps)
    
    # Estimate max steps:
    # We will just run while time < limit.
    # We'll use a simple exponential decay based on time.
    # T(t) = T_start * exp(-lambda * t)
    
    lambda_cool = 0.05 # Tune this? 
    # Or just use a counter.
    
    # Let's use a counter to be safe and deterministic-ish.
    # We will run for a fixed number of iterations if we can, but time limit is hard.
    # Let's try 2,000,000 iterations. If it takes too long, we break.
    
    max_iterations = 2000000
    current_T = T_start
    
    for step in range(max_iterations):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Cool down
        # Linear cooling is simple: T = T_start * (1 - step/max_steps)
        # But we might break early.
        # Let's use time-based cooling.
        elapsed = time.time() - start_time
        # Assume total time is 120s.
        # T = T_start * (T_end/T_start) ** (elapsed / 120.0)
        # This ensures we reach T_end at 120s.
        current_T = T_start * (T_end / T_start) ** (elapsed / 120.0)
        
        # Pick random index to flip
        i = random.randint(0, N - 1)
        
        # Calculate delta
        delta = 0
        for ap_idx in ap_list[i]:
            ap = ap_indices[ap_idx]
            
            # Get other 3 indices
            # We can avoid list comprehension for speed
            o0, o1, o2, o3 = ap
            if o0 == i:
                others = (o1, o2, o3)
            elif o1 == i:
                others = (o0, o2, o3)
            elif o2 == i:
                others = (o0, o1, o3)
            else:
                others = (o0, o1, o2)
                
            # Check if AP is currently mono
            if ap_mono[ap_idx]:
                # It is mono. Flipping i makes it non-mono.
                delta -= 1
            else:
                # It is non-mono. 
                # Check if other 3 are mono
                if colors[others[0]] == colors[others[1]] == colors[others[2]]:
                    # Other 3 are same color. 
                    # If new color of i matches them, it becomes mono.
                    if (1 - colors[i]) == colors[others[0]]:
                        delta += 1
        
        # Acceptance probability
        if delta <= 0:
            accept = True
        else:
            # Metropolis criterion
            # exp(-delta / T)
            # If T is very small, this is ~0
            if current_T > 0.001: # Avoid division by zero or underflow issues
                prob = np.exp(-delta / current_T)
                accept = random.random() < prob
            else:
                accept = False
        
        if accept:
            # Flip
            colors[i] = 1 - colors[i]
            current_count += delta
            
            # Update ap_mono for affected APs
            for ap_idx in ap_list[i]:
                ap = ap_indices[ap_idx]
                if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                    ap_mono[ap_idx] = 1
                else:
                    ap_mono[ap_idx] = 0
            
            if current_count < best_count:
                best_count = current_count
                best_colors = colors.copy()
                
    # Final verification (optional, but good for safety)
    # best_count = count_mono_4aps(best_colors)
    
    return best_colors.tolist()
