import random
import time
import math

N = 20
C = 20

def _is_rainbow(c1, c2, c3, c4):
    return len({c1, c2, c3, c4}) == 4

def _get_cycles_for_edge(u, v):
    """
    Yield all 4-cycles containing edge (u, v).
    Each cycle is represented as a tuple of 4 vertices (a, b, c, d) 
    such that the cycle is a-b-c-d-a and (u,v) is one of the edges.
    For consistency with update logic, we yield the specific 
    (u, v, x, y) tuple where the cycle is u-v-x-y-u.
    """
    others = [i for i in range(N) if i != u and i != v]
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            x = others[i]
            y = others[j]
            # Cycle u-v-x-y-u
            yield (u, v, x, y)
            # Cycle u-v-y-x-u
            yield (u, v, y, x)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initialize K20 with random colors
    colors = [[-1] * N for _ in range(N)]
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            c = random.randint(0, C - 1)
            colors[u][v] = c
            colors[v][u] = c
            edges.append((u, v))
            
    # 2. Count initial bad cycles
    def count_bad_cycles(col):
        bad = 0
        for a in range(N):
            for b in range(a+1, N):
                for c in range(b+1, N):
                    for d in range(c+1, N):
                        if not _is_rainbow(col[a][b], col[b][c], col[c][d], col[d][a]):
                            bad += 1
                        if not _is_rainbow(col[a][b], col[b][d], col[d][c], col[c][a]):
                            bad += 1
                        if not _is_rainbow(col[a][c], col[c][b], col[b][d], col[d][a]):
                            bad += 1
        return bad

    current_bad = count_bad_cycles(colors)
    best_bad = current_bad
    best_colors = [row[:] for row in colors]
    
    # 3. Simulated Annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.995
    
    while time.time() < deadline - 10.0 and T > T_min:
        u, v = random.choice(edges)
        old_c = colors[u][v]
        new_c = random.randint(0, C - 1)
        if old_c == new_c:
            continue
            
        delta = 0
        # Update delta based on affected cycles
        for (cu, cv, cx, cy) in _get_cycles_for_edge(u, v):
            # Cycle: cu-cv-cx-cy-cu
            # Edges: (cu,cv), (cv,cx), (cx,cy), (cy,cu)
            # Only (cu,cv) changes color.
            c_cv = colors[cv][cx]
            c_cx = colors[cx][cy]
            c_cy = colors[cy][cu]
            
            was_bad = not _is_rainbow(old_c, c_cv, c_cx, c_cy)
            is_bad = not _is_rainbow(new_c, c_cv, c_cx, c_cy)
            
            if was_bad and not is_bad:
                delta -= 1
            elif not was_bad and is_bad:
                delta += 1
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            colors[u][v] = new_c
            colors[v][u] = new_c
            current_bad += delta
            
            if current_bad < best_bad:
                best_bad = current_bad
                best_colors = [row[:] for row in colors]
                if best_bad == 0:
                    break
                    
        T *= alpha
        
    # 4. Pruning if not perfect
    if best_bad > 0:
        cur_colors = [row[:] for row in best_colors]
        present = [[True] * N for _ in range(N)]
        
        # Precompute all 4-cycles as tuples of vertices
        all_4cycles = []
        for a in range(N):
            for b in range(a+1, N):
                for c in range(b+1, N):
                    for d in range(c+1, N):
                        all_4cycles.append((a, b, c, d))
                        all_4cycles.append((a, b, d, c))
                        all_4cycles.append((a, c, b, d))
                        
        # Greedy removal
        for _ in range(200): # Max 200 removals
            if time.time() > deadline - 2.0:
                break
            found_bad = False
            for (a, b, c, d) in all_4cycles:
                if not (present[a][b] and present[b][c] and present[c][d] and present[d][a]):
                    continue
                if not _is_rainbow(cur_colors[a][b], cur_colors[b][c], cur_colors[c][d], cur_colors[d][a]):
                    found_bad = True
                    # Remove edge (a,b)
                    present[a][b] = False
                    present[b][a] = False
                    break
            if not found_bad:
                break
                
        result = []
        for u in range(N):
            for v in range(u + 1, N):
                if present[u][v]:
                    result.append([u, v, cur_colors[u][v]])
    else:
        result = []
        for u in range(N):
            for v in range(u + 1, N):
                result.append([u, v, best_colors[u][v]])
                
    return result
