import time
import random

def construct():
        start_time = time.time()
        deadline = start_time + 140.0
        
        # Initial 17-coloring
        colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
        def edge_idx(i, j):
            if i > j: i, j = j, i
            return 15*i - i*(i-1)//2 + (j - i - 1)
            
        # Generate all K4s and their edge indices
        k4_edges = []
        for a in range(16):
            for b in range(a+1, 16):
                for c in range(b+1, 16):
                    for d in range(c+1, 16):
                        edges = [
                            edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                            edge_idx(b, c), edge_idx(b, d),
                            edge_idx(c, d)
                        ]
                        k4_edges.append(tuple(edges))
                        
        # Map color to set of K4 indices using that color
        # Initialize based on initial coloring
        max_color = 17
        k4s_using = [set() for _ in range(max_color)]
        
        def rebuild_k4s_using(col_list):
            # Rebuild from scratch if colors change drastically or for initialization
            new_k4s_using = [set() for _ in range(max_color)]
            for k4_idx, edges in enumerate(k4_edges):
                distinct_colors_in_k4 = set()
                for e in edges:
                    c = col_list[e]
                    distinct_colors_in_k4.add(c)
                for c in distinct_colors_in_k4:
                    if c < max_color:
                        new_k4s_using[c].add(k4_idx)
            return new_k4s_using

        k4s_using = rebuild_k4s_using(colors)
        
        def is_valid_subset(k4_indices, col_list):
            for k4_idx in k4_indices:
                edges = k4_edges[k4_idx]
                c0 = col_list[edges[0]]
                c1 = col_list[edges[1]]
                c2 = col_list[edges[2]]
                c3 = col_list[edges[3]]
                c4 = col_list[edges[4]]
                c5 = col_list[edges[5]]
                # Check if at least 5 distinct
                # Optimization: if we see 5 distinct, we can stop checking this K4?
                # No, we need to ensure NO K4 has < 5.
                # So if we find a K4 with < 5, return False.
                # Count distinct
                # Using a small set is fast enough for 6 items
                if len({c0, c1, c2, c3, c4, c5}) < 5:
                    return False
            return True

        active_colors = set(colors)
        best_colors = colors
        best_count = len(active_colors)
        
        # Greedy Merging with Incremental Check
        # We will try to merge colors.
        # To avoid getting stuck, we can use a randomized order or restarts.
        
        for restart in range(50):
            if time.time() > deadline:
                break
                
            current_colors = best_colors.copy()
            current_active = set(current_colors)
            current_k4s_using = rebuild_k4s_using(current_colors)
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                cols_list = list(current_active)
                random.shuffle(cols_list)
                
                for i in range(len(cols_list)):
                    for j in range(i+1, len(cols_list)):
                        c1 = cols_list[i]
                        c2 = cols_list[j]
                        if c1 not in current_active or c2 not in current_active:
                            continue
                            
                        # Determine affected K4s
                        affected = current_k4s_using[c1] | current_k4s_using[c2]
                        
                        # Simulate merge
                        # We need to check if the merge is valid.
                        # The merge changes edges with c2 to c1.
                        # We can't easily "undo" without copying.
                        # Copying 120 ints is fast.
                        new_colors = current_colors.copy()
                        for k in range(120):
                            if new_colors[k] == c2:
                                new_colors[k] = c1
                            
                        if is_valid_subset(affected, new_colors):
                            # Accept merge
                            current_colors = new_colors
                            current_active.remove(c2)
                            # Update k4s_using
                            # c2 is removed. c1 gains the K4s from c2?
                            # Actually, any K4 that used c2 now uses c1 (and possibly other colors).
                            # So k4s_using[c1] = k4s_using[c1] | k4s_using[c2]
                            # k4s_using[c2] becomes empty.
                            current_k4s_using[c1] = current_k4s_using[c1] | current_k4s_using[c2]
                            current_k4s_using[c2] = set()
                            
                            improved = True
                            if len(current_active) < best_count:
                                best_count = len(current_active)
                                best_colors = current_colors.copy()
                            break # Restart inner loop with new state
                    if improved:
                        break
                        
        return best_colors
