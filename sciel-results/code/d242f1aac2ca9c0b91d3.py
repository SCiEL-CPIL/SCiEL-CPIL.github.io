import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 128
    
    # Precompute popcount for masks up to 2^10
    popcount = [bin(i).count('1') for i in range(1024)]
    
    # Precompute all 4-term APs
    aps = []
    ap_indices = [[] for _ in range(n)]
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d + 1):
            idx = len(aps)
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            for pos in (a, a + d, a + 2 * d, a + 3 * d):
                ap_indices[pos].append(idx)
                
    num_aps = len(aps)
    
    def local_search(K, time_limit):
        colors = np.random.randint(0, K, size=n)
        
        # Initialize histograms and masks
        hist = np.zeros((num_aps, K), dtype=np.int32)
        masks = np.zeros(num_aps, dtype=np.int32)
        
        for i, ap in enumerate(aps):
            for pos in ap:
                hist[i, colors[pos]] += 1
            mask = 0
            for c in range(K):
                if hist[i, c] > 0:
                    mask |= (1 << c)
            masks[i] = mask
            
        cost = 0
        for i in range(num_aps):
            if popcount[masks[i]] < 3:
                cost += 1
                
        if cost == 0:
            return colors.tolist()
            
        while time.time() < time_limit:
            p = random.randint(0, n - 1)
            current_color = colors[p]
            
            best_color = current_color
            best_cost = cost
            
            for c in range(K):
                if c == current_color:
                    continue
                    
                delta = 0
                for ap_idx in ap_indices[p]:
                    old_mask = masks[ap_idx]
                    old_bad = 1 if popcount[old_mask] < 3 else 0
                    
                    h_old = hist[ap_idx, current_color]
                    h_new = hist[ap_idx, c]
                    
                    new_mask = old_mask
                    if h_old == 1:
                        new_mask &= ~(1 << current_color)
                    if h_new == 0:
                        new_mask |= (1 << c)
                        
                    new_bad = 1 if popcount[new_mask] < 3 else 0
                    
                    delta += new_bad - old_bad
                    
                new_cost = cost + delta
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_color = c
                elif new_cost == best_cost and random.random() < 0.05:
                    best_color = c
                    best_cost = new_cost

            if best_color != current_color:
                colors[p] = best_color
                for ap_idx in ap_indices[p]:
                    hist[ap_idx, current_color] -= 1
                    hist[ap_idx, best_color] += 1
                    if hist[ap_idx, current_color] == 0:
                        masks[ap_idx] &= ~(1 << current_color)
                    if hist[ap_idx, best_color] == 1:
                        masks[ap_idx] |= (1 << best_color)
                cost = best_cost
                
                if cost == 0:
                    return colors.tolist()
                    
        return None

    for K in range(3, 11):
        time_limit = time.time() + 8.0
        if time.time() > deadline:
            break
        result = local_search(K, time_limit)
        if result is not None:
            return result
            
    return [0] * n
