import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 150s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 1 <= a <= 256
    # Indices are 0-based: a-1, a+d-1, a+2d-1, a+3d-1
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    
    # Precompute for each index, which APs it belongs to
    ap_list = [[] for _ in range(N)]
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            others = tuple(j for j in ap if j != idx)
            ap_list[idx].append((ap_id, others))

    def count_mono(colors):
        count = 0
        for ap in aps:
            c = colors[ap[0]]
            if c == colors[ap[1]] and c == colors[ap[2]] and c == colors[ap[3]]:
                count += 1
        return count

    def get_delta(colors, idx):
        delta = 0
        c = colors[idx]
        for ap_id, others in ap_list[idx]:
            o0 = colors[others[0]]
            if o0 == colors[others[1]] and o0 == colors[others[2]]:
                if c == o0:
                    delta -= 1
                else:
                    delta += 1
        return delta

    # Multi-pass deterministic local search
    # Try multiple random restarts, each running a greedy best-flip local search
    best_count = float('inf')
    best_colors = None

    # Number of restarts: we'll do as many as time allows
    # Each restart: initialize randomly, then run greedy local search
    num_restarts = 0
    while time.time() < deadline - 5.0:  # Leave 5s safety margin
        # Random initialization
        colors = [random.randint(0, 1) for _ in range(N)]
        current_count = count_mono(colors)
        
        # Greedy local search: repeatedly find the best single flip
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            best_delta = 0
            best_idx = -1
            
            # Evaluate all possible single-bit flips
            for idx in range(N):
                delta = get_delta(colors, idx)
                if delta < best_delta:
                    best_delta = delta
                    best_idx = idx
            
            # Apply the best flip if it improves
            if best_delta < 0 and best_idx >= 0:
                colors[best_idx] = 1 - colors[best_idx]
                current_count += best_delta
                improved = True
        
        # Update best
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
        
        num_restarts += 1

    # If no restarts completed (shouldn't happen), return random
    if best_colors is None:
        best_colors = [random.randint(0, 1) for _ in range(N)]
    
    return best_colors
