import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find n=26
    # If fails, try 25, 24, etc.
    for n in [26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline:
            break
        result = local_search(n, time_limit=15)
        if result is not None:
            return result
            
    # Fallback
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def local_search(n, time_limit=15):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    def get_violations(colors):
        neighbors = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    current_violations = get_violations(colors)
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    
    # Greedy repair + Random walk
    while time.time() < start_time + time_limit:
        if current_violations == 0:
            return {'n': n, 'colors': colors}
            
        # Find a violation
        # Re-compute neighbors to find a specific violation
        neighbors = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violation_found = False
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    if common.bit_count() >= 2:
                        # Found a C4: u-v-w1-u and u-v-w2-u? No, u-w1-v-w2-u
                        # Edges: (u,w1), (w1,v), (v,w2), (w2,u)
                        # We need to change one of these 4 edges.
                        # Pick one randomly
                        bits = [b for b in range(n) if (common >> b) & 1]
                        w1 = bits[0]
                        w2 = bits[1]
                        
                        # Candidate edges to change:
                        # (u, w1), (w1, v), (v, w2), (w2, u)
                        candidates = [
                            edge_idx.get((u, w1)) if u < w1 else edge_idx.get((w1, u)),
                            edge_idx.get((w1, v)) if w1 < v else edge_idx.get((v, w1)),
                            edge_idx.get((v, w2)) if v < w2 else edge_idx.get((w2, v)),
                            edge_idx.get((w2, u)) if w2 < u else edge_idx.get((u, w2))
                        ]
                        
                        # Try changing each candidate to a random new color
                        best_delta = float('inf')
                        best_change = None
                        for e_idx in candidates:
                            if e_idx is None: continue
                            old_c = colors[e_idx]
                            for new_c in range(5):
                                if new_c == old_c: continue
                                colors[e_idx] = new_c
                                new_v = get_violations(colors)
                                delta = new_v - current_violations
                                if delta < best_delta:
                                    best_delta = delta
                                    best_change = (e_idx, new_c)
                                colors[e_idx] = old_c
                            
                        if best_change:
                            e_idx, new_c = best_change
                            colors[e_idx] = new_c
                            current_violations = get_violations(colors)
                            if current_violations < best_violations:
                                best_violations = current_violations
                                best_colors = colors[:]
                            violation_found = True
                            break
            if violation_found:
                break
        
        if not violation_found:
            # No violations found? Should not happen if current_violations > 0
            # Or maybe my violation detection is inconsistent
            break
            
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
        
    return None
