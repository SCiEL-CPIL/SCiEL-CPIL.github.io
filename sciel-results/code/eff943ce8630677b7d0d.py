import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 145  # Leave 5s buffer for 150s limit
    
    # Precompute all 4-APs: a, a+d, a+2d, a+3d with d >= 1, all in [0, n-1]
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Convert to numpy arrays for fast operations
    ap_array = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)
    
    # Precompute: for each position, which APs contain it
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        for pos in ap_array[idx]:
            pos_aps[pos].append(idx)
    
    # Helper: count monochromatic APs using numpy
    def count_mono(colors):
        # colors is numpy array of 0/1
        # For each AP, check if all 4 positions have same color
        # mono = 1 if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]
        c0 = colors[ap_array[:, 0]]
        c1 = colors[ap_array[:, 1]]
        c2 = colors[ap_array[:, 2]]
        c3 = colors[ap_array[:, 3]]
        mono = ((c0 == c1) & (c1 == c2) & (c2 == c3)).astype(np.int32)
        return int(np.sum(mono))
    
    # Compute deltas: delta[i] = gain from flipping position i
    # delta[i] = (# APs containing i that would become mono if i flips) 
    #            - (# APs containing i that are currently mono)
    def compute_deltas(colors):
        deltas = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if not pos_aps[i]:
                continue
            # Get all APs containing position i
            ap_indices = pos_aps[i]
            if not ap_indices:
                continue
            # Get colors of all positions in these APs
            # For each AP, check if it's currently mono
            # And if flipping i would make it mono
            current_mono = 0
            would_be_mono = 0
            for ap_idx in ap_indices:
                ap = ap_array[ap_idx]
                # Find which position in ap is i
                # Check if currently mono
                c0 = colors[ap[0]]
                c1 = colors[ap[1]]
                c2 = colors[ap[2]]
                c3 = colors[ap[3]]
                if c0 == c1 == c2 == c3:
                    current_mono += 1
                # Check if flipping i would make it mono
                # The flipped color is 1 - colors[i]
                new_ci = 1 - colors[i]
                # Check if all other positions match new_ci
                others_match = True
                for j, pos in enumerate(ap):
                    if pos == i:
                        continue
                    if colors[pos] != new_ci:
                        others_match = False
                        break
                if others_match:
                    would_be_mono += 1
            deltas[i] = would_be_mono - current_mono
        return deltas
    
    def local_search(colors):
        # colors is numpy array of 0/1
        deltas = compute_deltas(colors)
        
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find position with most negative delta (best flip)
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta >= 0:
                break
            
            # Flip the position
            colors[min_idx] = 1 - colors[min_idx]
            
            # Update deltas efficiently
            # Only need to update deltas for positions in APs that contain min_idx
            # and min_idx itself
            affected = set([min_idx])
            for ap_idx in pos_aps[min_idx]:
                for pos in ap_array[ap_idx]:
                    affected.add(pos)
            
            # Recompute deltas for affected positions
            for pos in affected:
                if not pos_aps[pos]:
                    continue
                current_mono = 0
                would_be_mono = 0
                for ap_idx in pos_aps[pos]:
                    ap = ap_array[ap_idx]
                    c0 = colors[ap[0]]
                    c1 = colors[ap[1]]
                    c2 = colors[ap[2]]
                    c3 = colors[ap[3]]
                    if c0 == c1 == c2 == c3:
                        current_mono += 1
                    new_cp = 1 - colors[pos]
                    others_match = True
                    for j, p in enumerate(ap):
                        if p == pos:
                            continue
                        if colors[p] != new_cp:
                            others_match = False
                            break
                    if others_match:
                        would_be_mono += 1
                deltas[pos] = would_be_mono - current_mono
        
        return colors
    
    best_score = float('inf')
    best_colors = None
    
    # Initialization strategies
    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int32)
    
    def alternating(n):
        return np.array([i % 2 for i in range(n)], dtype=np.int32)
    
    def blocks(n, size):
        return np.array([(i // size) % 2 for i in range(n)], dtype=np.int32)
    
    init_strategies = [
        lambda: np.array([random.randint(0, 1) for _ in range(n)], dtype=np.int32),
        lambda: thue_morse(n),
        lambda: alternating(n),
        lambda: blocks(n, 2),
        lambda: blocks(n, 4),
        lambda: blocks(n, 8),
        lambda: blocks(n, 16),
        lambda: blocks(n, 32),
    ]
    
    # Run multiple restarts
    while time.time() < deadline:
        init_fn = random.choice(init_strategies)
        colors = init_fn()
        
        colors = local_search(colors)
        
        score = count_mono(colors)
        if score < best_score:
            best_score = score
            best_colors = colors.copy()
            
        if best_score == 0:
            break
    
    if best_colors is None:
        return [0] * n
    
    return best_colors.tolist()
