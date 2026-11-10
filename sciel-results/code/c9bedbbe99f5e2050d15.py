import time
import random
import itertools
import math
import numpy as np

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    n = 16
    vertices = list(range(n))
    
    # Precompute edges
    edge_list = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
    num_edges = len(edge_list) # 120
    
    # Precompute K4s
    k4s = []
    k4_edges = []
    for combo in itertools.combinations(vertices, 4):
        a, b, c, d = combo
        edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        k4s.append(combo)
        k4_edges.append([edge_index[e] for e in edges])
    num_k4s = len(k4_edges) # 1820
    
    # Precompute edge -> k4s mapping
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[e].append(k4_idx)
            
    # Max colors to consider for search
    MAX_COLORS = 20
    
    def init_state(colors):
        """Initialize k4_counts, k4_distinct, total_violations, global_counts, colors_used_count."""
        k4_counts = [[0] * MAX_COLORS for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        global_counts = [0] * MAX_COLORS
        colors_used_count = 0
        
        for e in range(num_edges):
            c = colors[e]
            if c >= MAX_COLORS:
                c = c % MAX_COLORS
            global_counts[c] += 1
            
        for c in range(MAX_COLORS):
            if global_counts[c] > 0:
                colors_used_count += 1
                
        for k4_idx in range(num_k4s):
            for e in k4_edges[k4_idx]:
                c = colors[e]
                if c >= MAX_COLORS:
                    c = c % MAX_COLORS
                k4_counts[k4_idx][c] += 1
                
            distinct = 0
            for c in range(MAX_COLORS):
                if k4_counts[k4_idx][c] > 0:
                    distinct += 1
            k4_distinct[k4_idx] = distinct
            
        total_violations = 0
        for k4_idx in range(num_k4s):
            if k4_distinct[k4_idx] < 5:
                total_violations += 1
                
        return k4_counts, k4_distinct, total_violations, global_counts, colors_used_count

    def get_cost(total_violations, colors_used_count):
        return total_violations * 1000 + colors_used_count

    def random_coloring():
        return [random.randint(0, MAX_COLORS - 1) for _ in range(num_edges)]

    def seed_coloring():
        # (i+j) % 17
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((i + j) % 17)
        # Ensure within MAX_COLORS
        return [c % MAX_COLORS for c in colors]
    
    def greedy_merge(colors):
        """Try to merge colors to reduce count while maintaining validity."""
        # Map colors to 0..k-1
        unique_colors = sorted(set(colors))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        remapped = [color_map[c] for c in colors]
        
        current_colors = remapped[:]
        current_set = set(current_colors)
        
        # If already valid, try to merge
        k4_counts, k4_distinct, total_violations, global_counts, colors_used_count = init_state(current_colors)
        
        if total_violations > 0:
            return current_colors
        
        # Try merging
        for _ in range(50):
            if len(current_set) <= 13:
                break
            
            merged = False
            # Sort colors by count descending? Or just iterate
            sorted_colors = sorted(current_set)
            
            for i in range(len(sorted_colors)):
                for j in range(i+1, len(sorted_colors)):
                    c1 = sorted_colors[i]
                    c2 = sorted_colors[j]
                    
                    # Try merging c1 into c2
                    new_colors = [c2 if x == c1 else x for x in current_colors]
                    
                    # Check validity
                    # Fast check: only need to check K4s affected by edges with color c1
                    # But simple full check is okay for 120 edges
                    # Let's use the state updater for efficiency
                    # Actually, just re-initializing is fine for 120 edges, it's fast
                    # But let's be smarter.
                    
                    # Apply change
                    delta_viol = 0
                    affected_edges = [e for e in range(num_edges) if current_colors[e] == c1]
                    affected_k4s_set = set()
                    for e in affected_edges:
                        for k4_idx in edge_k4s[e]:
                            affected_k4s_set.add(k4_idx)
                            
                    # This is getting complicated. Let's just re-init if the merge is cheap.
                    # Re-init cost is ~ 1820 * 6 ops. Very fast.
                    
                    # However, we need to be careful with the state variables if we want to continue.
                    # For a simple greedy pass, we can just re-validate.
                    
                    # Optimization: Check if merge is valid using the existing state
                    # We need to update the state for the affected K4s.
                    
                    # Let's just do a quick validation function that uses the precomputed structures
                    # but operates on the new colors list directly for the affected K4s.
                    
                    # Actually, the simplest robust way for a "polish" step is to just check validity.
                    # Since we are doing this after SA, we can afford a few full checks.
                    
                    # Let's implement a fast validator
                    # Build a 16x16 matrix
                    mat = [[0]*16 for _ in range(16)]
                    idx = 0
                    for i in range(16):
                        for j in range(i+1, 16):
                            mat[i][j] = new_colors[idx]
                            mat[j][i] = new_colors[idx]
                            idx += 1
                            
                    valid = True
                    for combo in k4s:
                        a, b, c, d = combo
                        e1 = mat[a][b]
                        e2 = mat[a][c]
                        e3 = mat[a][d]
                        e4 = mat[b][c]
                        e5 = mat[b][d]
                        e6 = mat[c][d]
                        if len({e1, e2, e3, e4, e5, e6}) < 5:
                            valid = False
                            break
                            
                    if valid:
                        current_colors = new_colors
                        current_set.remove(c1)
                        merged = True
                        break
                        
            if not merged:
                break
                
        return current_colors

    best_score = float('inf')
    best_colors = None
    best_valid_colors = None
    
    # We will run multiple restarts
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        if restart == 0:
            colors = seed_coloring()
        else:
            colors = random_coloring()
            
        k4_counts, k4_distinct, total_violations, global_counts, colors_used_count = init_state(colors)
        current_cost = get_cost(total_violations, colors_used_count)
        
        if total_violations == 0 and colors_used_count < best_score:
            best_score = colors_used_count
            best_colors = colors[:]
            best_valid_colors = colors[:]
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.9999
        max_iters = 200000
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick random edge
            e = random.randint(0, num_edges - 1)
            c_old = colors[e]
            
            # Pick new color
            if random.random() < 0.8:
                used = [c for c in range(MAX_COLORS) if global_counts[c] > 0]
                if used:
                    c_new = random.choice(used)
                else:
                    c_new = random.randint(0, MAX_COLORS - 1)
            else:
                c_new = random.randint(0, MAX_COLORS - 1)
                
            if c_new == c_old:
                continue
                
            # Calculate delta
            delta_viol = 0
            affected_k4s = edge_k4s[e]
            
            old_distincts = [0] * len(affected_k4s)
            
            for i, k4_idx in enumerate(affected_k4s):
                old_distincts[i] = k4_distinct[k4_idx]
                
                cnt = k4_counts[k4_idx]
                d = k4_distinct[k4_idx]
                
                cnt[c_old] -= 1
                if cnt[c_old] == 0:
                    d -= 1
                    
                cnt[c_new] += 1
                if cnt[c_new] == 1:
                    d += 1
                    
                k4_distinct[k4_idx] = d
                
                old_viol = 1 if old_distincts[i] < 5 else 0
                new_viol = 1 if d < 5 else 0
                delta_viol += new_viol - old_viol
                
            gc_old = global_counts[c_old]
            gc_new = global_counts[c_new]
            
            global_counts[c_old] -= 1
            global_counts[c_new] += 1
            
            delta_color_count = 0
            if gc_old == 1 and gc_new == 0:
                delta_color_count -= 1
            if gc_new == 1:
                delta_color_count += 1
                
            total_violations += delta_viol
            colors_used_count += delta_color_count
            
            colors[e] = c_new
            
            new_cost = get_cost(total_violations, colors_used_count)
            delta_cost = new_cost - current_cost
            
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
                if total_violations == 0 and colors_used_count < best_score:
                    best_score = colors_used_count
                    best_colors = colors[:]
                    best_valid_colors = colors[:]
            else:
                colors[e] = c_old
                
                for i, k4_idx in enumerate(affected_k4s):
                    cnt = k4_counts[k4_idx]
                    d = k4_distinct[k4_idx]
                    
                    cnt[c_new] -= 1
                    if cnt[c_new] == 0:
                        d -= 1
                        
                    cnt[c_old] += 1
                    if cnt[c_old] == 1:
                        d += 1
                        
                    k4_distinct[k4_idx] = d
                    
                global_counts[c_old] += 1
                global_counts[c_new] -= 1
                total_violations -= delta_viol
                colors_used_count -= delta_color_count
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        # End SA loop for this restart
        
        # Polish: Try greedy merge on the best result of this restart
        if best_valid_colors is not None:
            polished = greedy_merge(best_valid_colors)
            # Verify polished
            k4_counts, k4_distinct, total_violations, global_counts, colors_used_count = init_state(polished)
            if total_violations == 0 and colors_used_count < best_score:
                best_score = colors_used_count
                best_valid_colors = polished[:]

    # Final check
    if best_valid_colors is not None:
        k4_counts, k4_distinct, total_violations, global_counts, colors_used_count = init_state(best_valid_colors)
        if total_violations == 0:
            return best_valid_colors
        else:
            return seed_coloring()
    else:
        return seed_coloring()
