import time
import random
import numpy as np

def construct():
    N = 128
    deadline = time.time() + 55.0
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    M = len(aps)
    
    # Map position to list of (ap_index, role)
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for j, pos in enumerate(ap):
            pos_to_aps[pos].append((idx, j))

    def is_violation(c0, c1, c2, c3):
        # Violation if distinct colors < 3
        return len({c0, c1, c2, c3}) < 3

    def simulated_annealing(K, seed, max_steps=30000):
        random.seed(seed)
        # Initialize with K colors - use structured initialization
        colors = [random.randint(0, K - 1) for _ in range(N)]
        
        # Initialize violation state
        ap_violation = [False] * M
        total_violations = 0
        for idx, ap in enumerate(aps):
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if is_violation(c0, c1, c2, c3):
                ap_violation[idx] = True
                total_violations += 1
        
        color_counts = [0] * K
        for c in colors:
            color_counts[c] += 1
        distinct_colors = sum(1 for c in color_counts if c > 0)
        
        # Energy: prioritize violations heavily, then minimize colors
        T = 5.0
        cooling = 0.995
        best_E = total_violations * 10000 + distinct_colors
        best_colors = colors[:]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick random position
            pos = random.randint(0, N - 1)
            old_color = colors[pos]
            new_color = random.randint(0, K - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate Delta E
            delta_v = 0
            # Check affected APs
            for ap_idx, role in pos_to_aps[pos]:
                ap = aps[ap_idx]
                # Get current colors
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                old_v = ap_violation[ap_idx]
                
                # Update color in temp tuple
                if role == 0:
                    nc = (new_color, c1, c2, c3)
                elif role == 1:
                    nc = (c0, new_color, c2, c3)
                elif role == 2:
                    nc = (c0, c1, new_color, c3)
                else:
                    nc = (c0, c1, c2, new_color)
                
                new_v = is_violation(*nc)
                
                if new_v and not old_v:
                    delta_v += 1
                elif not new_v and old_v:
                    delta_v -= 1
            
            # Update distinct colors
            delta_d = 0
            if color_counts[old_color] == 1:
                # Removing last of old_color
                delta_d -= 1
            if color_counts[new_color] == 0:
                # Adding first of new_color
                delta_d += 1
            
            delta_E = delta_v * 10000 + delta_d
            
            # Accept?
            if delta_E < 0:
                accept = True
            else:
                if T > 1e-6:
                    accept = random.random() < np.exp(-delta_E / T)
                else:
                    accept = False
            
            if accept:
                # Apply move
                colors[pos] = new_color
                total_violations += delta_v
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
                distinct_colors += delta_d
                
                # Update ap_violation
                for ap_idx, role in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    ap_violation[ap_idx] = is_violation(c0, c1, c2, c3)
                
                current_E = total_violations * 10000 + distinct_colors
                if current_E < best_E:
                    best_E = current_E
                    best_colors = colors[:]
                    if total_violations == 0 and distinct_colors == K:
                        # Found a valid coloring using exactly K colors
                        return best_colors, best_E
            
            T *= cooling
        
        # Check if best is valid
        if best_E < 10000: # i.e. total_violations == 0
            return best_colors, best_E
        return None, best_E

    # Try K from 3 upward, with multiple restarts
    best_solution = None
    best_K = 128
    
    for K in range(3, 128):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple seeds for this K
        for seed in range(15):
            if time.time() > deadline:
                break
            sol, E = simulated_annealing(K, seed, max_steps=25000)
            if sol is not None:
                best_solution = sol
                best_K = K
                found = True
                break
        
        if found:
            break
    
    if best_solution is None:
        # Fallback: All distinct
        best_solution = list(range(N))
        
    return best_solution
