import time
import random
import sys
import itertools

def construct():
    n = 20
    num_colors = 20
    deadline = time.time() + 110.0
    
    adj = [[-1] * n for _ in range(n)]
    
    # 1. 1-Factorization
    for r in range(19):
        adj[19][r] = r
        adj[r][19] = r
        for i in range(1, 10):
            u = (r + i) % 19
            v = (r - i) % 19
            adj[u][v] = r
            adj[v][u] = r
            
    # 2. Pre-compute cycles and edge-cycle mapping
    all_4cycles = []
    edge_to_cycles = {}
    
    verts = range(n)
    for combo in itertools.combinations(verts, 4):
        a, b, c, d = combo
        cycles = [
            [(a,b), (b,c), (c,d), (d,a)],
            [(a,b), (b,d), (d,c), (c,a)],
            [(a,c), (c,b), (b,d), (d,a)]
        ]
        for cycle in cycles:
            idx = len(all_4cycles)
            all_4cycles.append(cycle)
            for u, v in cycle:
                key = (min(u,v), max(u,v))
                if key not in edge_to_cycles:
                    edge_to_cycles[key] = []
                edge_to_cycles[key].append(idx)
                
    def is_rainbow(cycle):
        colors = [adj[u][v] for u, v in cycle]
        return len(set(colors)) == 4

    def count_bad_cycles():
        bad = 0
        for cycle in all_4cycles:
            if not is_rainbow(cycle):
                bad += 1
        return bad

    current_bad = count_bad_cycles()
    best_bad = current_bad
    best_adj = [row[:] for row in adj]
    
    edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() < deadline:
        if best_bad == 0:
            break
            
        # Find a bad cycle
        bad_cycle_idx = -1
        for i, cycle in enumerate(all_4cycles):
            if not is_rainbow(cycle):
                bad_cycle_idx = i
                break
                
        if bad_cycle_idx == -1:
            best_bad = 0
            best_adj = [row[:] for row in adj]
            break
            
        cycle = all_4cycles[bad_cycle_idx]
        improved = False
        
        for u, v in cycle:
            key = (min(u,v), max(u,v))
            affected_cycles = edge_to_cycles[key]
            
            old_color = adj[u][v]
            
            # Count bad among affected
            bad_before = 0
            for idx in affected_cycles:
                if not is_rainbow(all_4cycles[idx]):
                    bad_before += 1
            
            for new_color in range(num_colors):
                if new_color == old_color:
                    continue
                adj[u][v] = new_color
                adj[v][u] = new_color
                
                bad_after = 0
                for idx in affected_cycles:
                    if not is_rainbow(all_4cycles[idx]):
                        bad_after += 1
                        
                new_global_bad = current_bad - bad_before + bad_after
                
                if new_global_bad < best_bad:
                    best_bad = new_global_bad
                    best_adj = [row[:] for row in adj]
                    current_bad = new_global_bad
                    improved = True
                    break
                else:
                    adj[u][v] = old_color
                    adj[v][u] = old_color
                    
            if improved:
                break
                
        if not improved:
            u, v = random.choice(edges)
            new_color = random.randint(0, num_colors - 1)
            adj[u][v] = new_color
            adj[v][u] = new_color
            current_bad = count_bad_cycles()
            if current_bad < best_bad:
                best_bad = current_bad
                best_adj = [row[:] for row in adj]

    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v] != -1:
                result_edges.append([u, v, best_adj[u][v]])
                
    return result_edges
