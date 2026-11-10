import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin
    
    # Try n from 34 down to 1
    for n in range(34, 0, -1):
        if time.time() - start_time > deadline - 5:
            break
        result = find_coloring(n, time_budget=min(10, deadline - (time.time() - start_time)))
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback: return a trivial valid coloring for small n
    # For n=1,2,3, no K4 or K5, so any coloring is valid.
    # We need to return a valid one.
    # Let's just return n=1 with empty colors? No, colors for triples.
    # For n=1, 0 triples.
    # For n=3, 1 triple.
    # Let's construct a valid one for n=10 just in case.
    n_fb = 10
    colors = []
    for a in range(n_fb):
        for b in range(a+1, n_fb):
            for c in range(b+1, n_fb):
                # Simple heuristic: avoid all red K4 and all blue K5
                # Random is usually fine for small n
                colors.append(random.randint(0, 1))
    # Verify and fix if needed (should be valid for small n with high prob)
    # Actually, for n=10, K4 count is 210, K5 count is 252.
    # Random has prob (1/2)^4 = 1/16 for red K4. Expected ~13 violations.
    # We should just run a quick local search for the fallback.
    
    # Better fallback: just return n=1
    return {'n': 1, 'colors': []}

def find_coloring(n, time_budget):
    if n < 4:
        # No K4, so no red violation possible.
        # If n < 5, no K5, so no blue violation possible.
        # So any coloring is valid.
        M = n * (n - 1) * (n - 2) // 6
        return [random.randint(0, 1) for _ in range(M)]
        
    # Precompute edges
    edges = list(combinations(range(n), 3))
    M = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    # Precompute K4s and K5s
    k4_list = []
    for quad in combinations(range(n), 4):
        k4_list.append([edge_index[t] for t in combinations(quad, 3)])
        
    k5_list = []
    if n >= 5:
        for quint in combinations(range(n), 5):
            k5_list.append([edge_index[t] for t in combinations(quint, 3)])
            
    num_k4 = len(k4_list)
    num_k5 = len(k5_list)
    
    # Map edge to k4 indices and k5 indices
    edge_to_k4s = [[] for _ in range(M)]
    for i, k4_edges in enumerate(k4_list):
        for e in k4_edges:
            edge_to_k4s[e].append(i)
            
    edge_to_k5s = [[] for _ in range(M)]
    for i, k5_edges in enumerate(k5_list):
        for e in k5_edges:
            edge_to_k5s[e].append(i)
            
    # Initialize random coloring
    colors = np.random.randint(0, 2, M, dtype=np.int8)
    
    # Initialize counts
    # k4_counts[i] = number of red edges (0) in k4_list[i]
    k4_counts = np.zeros(num_k4, dtype=np.int32)
    for i, k4_edges in enumerate(k4_list):
        k4_counts[i] = np.sum(colors[k4_edges] == 0)
        
    # k5_counts[i] = number of blue edges (1) in k5_list[i]
    k5_counts = np.zeros(num_k5, dtype=np.int32)
    for i, k5_edges in enumerate(k5_list):
        k5_counts[i] = np.sum(colors[k5_edges] == 1)
        
    red_viol = np.sum(k4_counts == 4)
    blue_viol = np.sum(k5_counts == 10)
    
    if red_viol == 0 and blue_viol == 0:
        return colors.tolist()
        
    # Simulated Annealing
    T = 1.0
    T_end = 0.01
    cooling = 0.999
    
    best_viol = red_viol + blue_viol
    best_colors = colors.copy()
    
    steps = 0
    no_improve = 0
    
    while T > T_end:
        if time.time() - time.time() > time_budget: # Placeholder, need global time
            break
            
        e = np.random.randint(0, M)
        old_val = colors[e]
        new_val = 1 - old_val
        
        delta_red = 0
        delta_blue = 0
        
        if old_val == 0: # Red -> Blue
            # K4: Red count decreases
            for k4_idx in edge_to_k4s[e]:
                if k4_counts[k4_idx] == 4:
                    delta_red -= 1
                k4_counts[k4_idx] -= 1
            # K5: Blue count increases
            for k5_idx in edge_to_k5s[e]:
                if k5_counts[k5_idx] == 9:
                    delta_blue += 1
                k5_counts[k5_idx] += 1
        else: # Blue -> Red
            # K4: Red count increases
            for k4_idx in edge_to_k4s[e]:
                if k4_counts[k4_idx] == 3:
                    delta_red += 1
                k4_counts[k4_idx] += 1
            # K5: Blue count decreases
            for k5_idx in edge_to_k5s[e]:
                if k5_counts[k5_idx] == 10:
                    delta_blue -= 1
                k5_counts[k5_idx] -= 1
                
        colors[e] = new_val
        red_viol += delta_red
        blue_viol += delta_blue
        total_viol = red_viol + blue_viol
        
        if total_viol < best_viol:
            best_viol = total_viol
            best_colors = colors.copy()
            no_improve = 0
            if total_viol == 0:
                return colors.tolist()
        else:
            # Accept with prob exp(-delta/T)
            delta = total_viol - (red_viol - delta_red + blue_viol - delta_blue) # This is wrong, delta is total_viol - old_total
            # old_total was best_viol? No, current total before flip.
            # Let's store old_total.
            
            # Re-calc old_total properly
            # We updated counts. To revert, we'd need to undo.
            # Easier: calculate delta from old state.
            
            # Let's refactor the loop to calculate delta first, then apply.
            
        # Cooling
        T *= cooling
        steps += 1
        
        # Restart if stuck
        if no_improve > 5000:
            if time.time() > start_time + time_budget:
                break
            colors = np.random.randint(0, 2, M, dtype=np.int8)
            # Re-init counts
            for i in range(num_k4):
                k4_counts[i] = np.sum(colors[k4_list[i]] == 0)
            for i in range(num_k5):
                k5_counts[i] = np.sum(colors[k5_list[i]] == 1)
            red_viol = np.sum(k4_counts == 4)
            blue_viol = np.sum(k5_counts == 10)
            T = 1.0
            no_improve = 0
            if red_viol + blue_viol == 0:
                return colors.tolist()
            
    return None

# The logic above is a bit messy with the delta calculation and revert.
# I will rewrite the inner loop cleanly.

def construct():
    start_time = time.time()
    deadline = 115
    
    for n in range(34, 0, -1):
        if time.time() - start_time > deadline - 2:
            break
        # Allocate time based on n. Larger n needs more time? 
        # Actually, larger n is harder. 
        # Let's give 34 most of the time.
        if n == 34:
            budget = deadline - (time.time() - start_time) - 5
        elif n == 33:
            budget = 5
        else:
            budget = 1
            
        if budget < 1:
            continue
            
        result = find_coloring(n, budget, start_time)
        if result is not None:
            return {'n': n, 'colors': result}
            
    return {'n': 1, 'colors': []}

def find_coloring(n, time_budget, global_start):
    if n < 4:
        M = n * (n - 1) * (n - 2) // 6
        return [random.randint(0, 1) for _ in range(M)]
        
    edges = list(combinations(range(n), 3))
    M = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    k4_list = []
    for quad in combinations(range(n), 4):
        k4_list.append([edge_index[t] for t in combinations(quad, 3)])
        
    k5_list = []
    if n >= 5:
        for quint in combinations(range(n), 5):
            k5_list.append([edge_index[t] for t in combinations(quint, 3)])
            
    num_k4 = len(k4_list)
    num_k5 = len(k5_list)
    
    edge_to_k4s = [[] for _ in range(M)]
    for i, k4_edges in enumerate(k4_list):
        for e in k4_edges:
            edge_to_k4s[e].append(i)
            
    edge_to_k5s = [[] for _ in range(M)]
    for i, k5_edges in enumerate(k5_list):
        for e in k5_edges:
            edge_to_k5s[e].append(i)
            
    colors = np.random.randint(0, 2, M, dtype=np.int8)
    
    k4_counts = np.zeros(num_k4, dtype=np.int32)
    for i, k4_edges in enumerate(k4_list):
        k4_counts[i] = np.sum(colors[k4_edges] == 0)
        
    k5_counts = np.zeros(num_k5, dtype=np.int32)
    for i, k5_edges in enumerate(k5_list):
        k5_counts[i] = np.sum(colors[k5_edges] == 1)
        
    red_viol = np.sum(k4_counts == 4)
    blue_viol = np.sum(k5_counts == 10)
    
    if red_viol == 0 and blue_viol == 0:
        return colors.tolist()
        
    T = 1.0
    T_end = 0.01
    cooling = 0.999
    
    best_viol = red_viol + blue_viol
    best_colors = colors.copy()
    
    no_improve = 0
    
    while T > T_end and (time.time() - global_start) < (global_start + time_budget):
        e = np.random.randint(0, M)
        old_val = colors[e]
        
        # Calculate delta
        delta_red = 0
        delta_blue = 0
        
        if old_val == 0: # Red -> Blue
            for k4_idx in edge_to_k4s[e]:
                if k4_counts[k4_idx] == 4:
                    delta_red -= 1
            for k5_idx in edge_to_k5s[e]:
                if k5_counts[k5_idx] == 9:
                    delta_blue += 1
        else: # Blue -> Red
            for k4_idx in edge_to_k4s[e]:
                if k4_counts[k4_idx] == 3:
                    delta_red += 1
            for k5_idx in edge_to_k5s[e]:
                if k5_counts[k5_idx] == 10:
                    delta_blue -= 1
                    
        delta_total = delta_red + delta_blue
        old_total = red_viol + blue_viol
        new_total = old_total + delta_total
        
        # Acceptance
        if delta_total < 0 or random.random() < np.exp(-delta_total / T):
            # Apply flip
            colors[e] = 1 - old_val
            if old_val == 0:
                for k4_idx in edge_to_k4s[e]:
                    k4_counts[k4_idx] -= 1
                for k5_idx in edge_to_k5s[e]:
                    k5_counts[k5_idx] += 1
            else:
                for k4_idx in edge_to_k4s[e]:
                    k4_counts[k4_idx] += 1
                for k5_idx in edge_to_k5s[e]:
                    k5_counts[k5_idx] -= 1
                    
            red_viol += delta_red
            blue_viol += delta_blue
            
            if new_total < best_viol:
                best_viol = new_total
                best_colors = colors.copy()
                no_improve = 0
                if new_total == 0:
                    return colors.tolist()
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        T *= cooling
        
        if no_improve > 2000:
            if (time.time() - global_start) > (global_start + time_budget - 2):
                break
            # Restart
            colors = np.random.randint(0, 2, M, dtype=np.int8)
            for i in range(num_k4):
                k4_counts[i] = np.sum(colors[k4_list[i]] == 0)
            for i in range(num_k5):
                k5_counts[i] = np.sum(colors[k5_list[i]] == 1)
            red_viol = np.sum(k4_counts == 4)
            blue_viol = np.sum(k5_counts == 10)
            T = 1.0
            no_improve = 0
            if red_viol + blue_viol == 0:
                return colors.tolist()
                
    return None
