"""Fast Simulated Annealing for C4-free 5-coloring of K_n."""
import time
import random
import math

def _is_valid_coloring(n, colors):
    """Check that no color class contains a C4."""
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            idx += 1
            adj[c].add((i, j))
    
    for c in range(5):
        neigh = [set() for _ in range(n)]
        for (u, v) in adj[c]:
            neigh[u].add(v)
            neigh[v].add(u)
        for i in range(n):
            for j in range(i + 1, n):
                if len(neigh[i] & neigh[j]) >= 2:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    best_n = 0
    best_colors = []
    
    # Try to find a valid coloring for n=26, 25, 24
    for target_n in [26, 25, 24]:
        if time.time() > deadline - 10:
            break
            
        found = False
        # Multiple restarts
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            random.seed(attempt * 1000 + target_n)
            
            num_edges = target_n * (target_n - 1) // 2
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            def count_conflicts(colors):
                conflicts = 0
                for c in range(5):
                    neigh = [set() for _ in range(target_n)]
                    idx = 0
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if colors[idx] == c:
                                neigh[i].add(j)
                                neigh[j].add(i)
                            idx += 1
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if len(neigh[i] & neigh[j]) >= 2:
                                conflicts += 1
                return conflicts
            
            current_conflicts = count_conflicts(colors)
            if current_conflicts == 0:
                best_n = target_n
                best_colors = colors
                found = True
                break
            
            # Simulated Annealing
            T = 1.0
            cooling = 0.99
            min_T = 0.01
            max_steps = 10000
            step = 0
            
            while T > min_T and step < max_steps:
                if time.time() > deadline - 10:
                    break
                
                step += 1
                idx = random.randint(0, num_edges - 1)
                old_color = colors[idx]
                new_color = random.randint(0, 4)
                if old_color == new_color:
                    continue
                
                colors[idx] = new_color
                new_conflicts = count_conflicts(colors)
                
                delta = new_conflicts - current_conflicts
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_conflicts = new_conflicts
                    if current_conflicts == 0:
                        best_n = target_n
                        best_colors = colors
                        found = True
                        break
                else:
                    colors[idx] = old_color
                
                T *= cooling
            
            if found:
                break
        
        if found:
            break
    
    if best_n == 0:
        # Fallback to a known small valid coloring, e.g., n=5
        n = 5
        colors = [0] * (n*(n-1)//2)
        # Simple greedy
        cn = [[[0]*n for _ in range(n)] for _ in range(5)]
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                idx = 0
                for k in range(i):
                    idx += n - 1 - k
                idx += j - i - 1
                best_c = 0
                best_score = 100
                for c in range(5):
                    score = 0
                    for u in range(n):
                        if u == i or u == j: continue
                        idx_iu = 0
                        a, b = min(i, u), max(i, u)
                        for k in range(a):
                            idx_iu += n - 1 - k
                        idx_iu += b - a - 1
                        if colors[idx_iu] == c:
                            if cn[c][u][j] == 1: score += 1
                    for v in range(n):
                        if v == i or v == j: continue
                        idx_jv = 0
                        a, b = min(j, v), max(j, v)
                        for k in range(a):
                            idx_jv += n - 1 - k
                        idx_jv += b - a - 1
                        if colors[idx_jv] == c:
                            if cn[c][i][v] == 1: score += 1
                    if score < best_score:
                        best_score = score
                        best_c = c
                colors[idx] = best_c
                c = best_c
                for u in range(n):
                    if u == i or u == j: continue
                    idx_iu = 0
                    a, b = min(i, u), max(i, u)
                    for k in range(a):
                        idx_iu += n - 1 - k
                    idx_iu += b - a - 1
                    if colors[idx_iu] == c:
                        cn[c][u][j] += 1
                        cn[c][j][u] += 1
                for v in range(n):
                    if v == i or v == j: continue
                    idx_jv = 0
                    a, b = min(j, v), max(j, v)
                    for k in range(a):
                        idx_jv += n - 1 - k
                    idx_jv += b - a - 1
                    if colors[idx_jv] == c:
                        cn[c][i][v] += 1
                        cn[c][v][i] += 1
        
        if _is_valid_coloring(n, colors):
            best_n = n
            best_colors = colors
        else:
            best_n = 1
            best_colors = []

    return {'n': best_n, 'colors': best_colors}
