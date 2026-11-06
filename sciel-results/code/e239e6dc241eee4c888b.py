import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    edges = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_index[(i, j)] = idx
            idx += 1
    num_edges = 120
    
    k4s = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for a, b, c, d in combinations(range(n), 4):
        k4_edges = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4_idx = len(k4s)
        k4s.append(k4_edges)
        for ei in k4_edges:
            edge_to_k4s[ei].append(k4_idx)
            
    num_k4s = len(k4s)
    
    # Initial coloring
    colors = np.array([(i + j) % 17 for i, j in edges], dtype=int)
    current_colors = colors.copy()
    
    # Precompute K4 color counts
    # Max color value is 16, so size 17 is enough for initial.
    # But colors might change? We restrict to used colors, so max color stays 16.
    # Unless we introduce new colors. We don't.
    # So 17 is safe.
    
    k4_color_counts = np.zeros((num_k4s, 17), dtype=np.int32)
    k4_distinct_counts = np.zeros(num_k4s, dtype=np.int32)
    
    # Initialize
    for k4_idx, k4_e in enumerate(k4s):
        for e in k4_e:
            c = current_colors[e]
            k4_color_counts[k4_idx, c] += 1
        k4_distinct_counts[k4_idx] = np.sum(k4_color_counts[k4_idx] > 0)
        
    # Initial violations
    current_violations = np.sum(k4_distinct_counts < 5)
    current_num_colors = len(set(current_colors))
    
    # Best valid
    best_valid_colors = current_colors.copy()
    best_valid_num_colors = current_num_colors
    
    # Global color counts
    color_counts = np.zeros(17, dtype=np.int32)
    for c in current_colors:
        color_counts[c] += 1
        
    random.seed(42)
    T = 50.0
    cooling_rate = 0.9999
    min_T = 0.01
    
    # List of used colors for fast sampling
    # We will update this list or just filter color_counts
    
    while T > min_T and time.time() < deadline:
        e = random.randint(0, num_edges - 1)
        
        # Get used colors
        # Find indices where color_counts > 0
        used_indices = np.nonzero(color_counts > 0)[0]
        
        if len(used_indices) == 0:
            break
            
        new_color = used_indices[random.randint(0, len(used_indices) - 1)]
        old_color = current_colors[e]
        
        if new_color == old_color:
            continue
            
        # Calculate Delta Violations
        delta_viol = 0
        for k4_idx in edge_to_k4s[e]:
            # Update counts for this K4
            # Old
            old_cnt = k4_color_counts[k4_idx, old_color]
            new_cnt = k4_color_counts[k4_idx, new_color]
            
            old_distinct = k4_distinct_counts[k4_idx]
            
            # Apply change to counts temporarily for logic
            # old_color count decreases by 1
            # new_color count increases by 1
            
            new_old_cnt = old_cnt - 1
            new_new_cnt = new_cnt + 1
            
            new_distinct = old_distinct
            
            if new_old_cnt == 0:
                new_distinct -= 1
            if new_new_cnt == 1:
                new_distinct += 1
                
            old_v = 1 if old_distinct < 5 else 0
            new_v = 1 if new_distinct < 5 else 0
            
            delta_viol += new_v - old_v
            
            # We don't update k4_color_counts yet, only if accepted
        
        # Calculate Delta Num Colors
        delta_nc = 0
        if color_counts[old_color] == 1:
            # Old color removed from global
            # New color is from used_indices, so count > 0
            delta_nc = -1
            
        delta_cost = delta_viol * 100 + delta_nc
        
        # Acceptance
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            if delta_cost < 1000:
                prob = np.exp(-delta_cost / T)
                if random.random() < prob:
                    accept = True
        
        if accept:
            # Update state
            current_colors[e] = new_color
            
            # Update global color counts
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            
            # Update K4 counts
            for k4_idx in edge_to_k4s[e]:
                # Update counts
                k4_color_counts[k4_idx, old_color] -= 1
                k4_color_counts[k4_idx, new_color] += 1
                
                # Update distinct count
                # We already calculated new_distinct in the loop above?
                # No, we only calculated delta_viol.
                # We need to update k4_distinct_counts[k4_idx]
                
                # Recalculate distinct for this K4?
                # Or use the logic from above.
                # Let's just recalculate sum for this K4, it's small (17)
                # Or update incrementally:
                if k4_color_counts[k4_idx, old_color] == 0:
                    k4_distinct_counts[k4_idx] -= 1
                if k4_color_counts[k4_idx, new_color] == 1:
                    k4_distinct_counts[k4_idx] += 1
                    
            current_violations += delta_viol
            current_num_colors += delta_nc
            
            # Update best valid
            if current_violations == 0:
                if current_num_colors < best_valid_num_colors:
                    best_valid_num_colors = current_num_colors
                    best_valid_colors = current_colors.copy()
                    
        T *= cooling_rate
        
    # Final check
    # Verify best_valid_colors
    # Recount violations to be safe
    temp_counts = np.zeros((num_k4s, 17), dtype=np.int32)
    for k4_idx, k4_e in enumerate(k4s):
        for e in k4_e:
            c = best_valid_colors[e]
            temp_counts[k4_idx, c] += 1
    final_viol = np.sum(np.sum(temp_counts > 0, axis=1) < 5)
    
    if final_viol != 0:
        return list([(i + j) % 17 for i, j in edges])
        
    return list(best_valid_colors)

