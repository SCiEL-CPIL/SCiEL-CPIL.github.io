import random
import time

def construct():
    target_ns = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    start_time = time.time()
    time_limit = 115.0
    
    for n in target_ns:
        if time.time() - start_time > time_limit:
            break
            
        # Allocate time based on priority
        # n=26 gets 80% of remaining time if it's the first target
        # Simpler: Just run SA until time limit for current n, then move on?
        # No, we need to return the best.
        
        # Precompute edge list
        edges = []
        edge_idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                edge_idx_map[(i, j)] = idx
                idx += 1
        m = len(edges)
        
        best_colors = None
        best_energy = float('inf')
        
        # Multiple restarts
        max_restarts = 5 if n == 26 else 3
        time_per_restart = (time_limit - (time.time() - start_time)) / max_restarts
        # For n=26, we want to spend most time here.
        # If n < 26, we only get what's left.
        
        for restart in range(max_restarts):
            restart_start = time.time()
            restart_limit = restart_start + time_per_restart
            # For n=26, give it more time
            if n == 26 and restart == 0:
                restart_limit = start_time + time_limit * 0.8 # Use 80% of total time for first restart of 26
            
            # Init
            colors = [random.randint(0, 4) for _ in range(m)]
            adj = [ [set() for _ in range(n)] for _ in range(5) ]
            
            for k, (i, j) in enumerate(edges):
                c = colors[k]
                adj[c][i].add(j)
                adj[c][j].add(i)
                
            def count_c4s_in_color(c):
                total = 0
                for u in range(n):
                    for v in adj[c][u]:
                        if u < v:
                            count = 0
                            for x in adj[c][v]:
                                if x == u: continue
                                for y in adj[c][u]:
                                    if y == v: continue
                                    if x == y: continue
                                    if y in adj[c][x]:
                                        count += 1
                            total += count
                return total // 4

            total_energy = sum(count_c4s_in_color(c) for c in range(5))
            
            T = 10.0
            cooling = 0.9995
            min_T = 0.01
            
            steps = 0
            max_steps = 100000 # Increased steps
            
            while T > min_T:
                if time.time() > restart_limit:
                    break
                    
                # Pick random edge
                k = random.randint(0, m - 1)
                i, j = edges[k]
                
                c_old = colors[k]
                c_new = random.randint(0, 4)
                if c_old == c_new: continue
                
                # Delta calc
                # Remove (i,j) from c_old
                # Add (i,j) to c_new
                
                # Count C4s involving (i,j) in c_old
                cnt_old = 0
                for x in adj[c_old][j]:
                    if x == i: continue
                    for y in adj[c_old][i]:
                        if y == j: continue
                        if x == y: continue
                        if y in adj[c_old][x]:
                            cnt_old += 1
                            
                # Count C4s involving (i,j) in c_new (before adding)
                cnt_new = 0
                for x in adj[c_new][j]:
                    if x == i: continue
                    for y in adj[c_new][i]:
                        if y == j: continue
                        if x == y: continue
                        if y in adj[c_new][x]:
                            cnt_new += 1
                            
                delta_E = cnt_new - cnt_old
                
                if delta_E < 0 or random.random() < pow(2.718, -delta_E / T):
                    colors[k] = c_new
                    adj[c_old][i].remove(j)
                    adj[c_old][j].remove(i)
                    adj[c_new][i].add(j)
                    adj[c_new][j].add(i)
                    
                    total_energy += delta_E
                    
                    if total_energy < best_energy:
                        best_energy = total_energy
                        best_colors = colors.copy()
                        if best_energy == 0:
                            return {'n': n, 'colors': best_colors}
                            
                T *= cooling
                steps += 1
                
            if best_energy == 0:
                return {'n': n, 'colors': best_colors}
                
    # Fallback
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
