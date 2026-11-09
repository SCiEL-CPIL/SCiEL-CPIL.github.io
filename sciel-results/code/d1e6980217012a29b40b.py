import numpy as np
import time

def _greedy_score(prizes, dist, max_len, heuristic_matrix, start_node=0):
    n = len(prizes)
    current = start_node
    visited = np.zeros(n, dtype=bool)
    visited[start_node] = True
    length = 0.0
    prize_sum = prizes[start_node]
    
    while True:
        remaining = max_len - length
        cost_vec = dist[current, :] + dist[:, 0]
        feasible_mask = cost_vec <= remaining
        unvisited_mask = ~visited
        candidate_mask = unvisited_mask & feasible_mask
        
        if not np.any(candidate_mask):
            break
            
        h_vals = heuristic_matrix[current, :].copy()
        h_vals[~candidate_mask] = -np.inf
        
        next_node = np.argmax(h_vals)
        
        length += dist[current, next_node]
        visited[next_node] = True
        prize_sum += prizes[next_node]
        current = next_node
        
    return prize_sum

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    start_time = time.time()
    timeout = 290
    
    dist = np.array(distance_matrix, dtype=float)
    p = np.array(prizes, dtype=float)
    
    return_cost = dist[0, :] + dist[:, 0]
    reachable_mask = return_cost <= max_len
    
    def normalize(h):
        # Scale so that max is 1, min is 1e-9
        max_val = np.max(h)
        if max_val > 0:
            h = h / max_val
        h = np.where(h > 0, h, 1e-9)
        return h

    def gen_1(p, d, ml, reach):
        h = (p[None, :] + 1e-9) / (d + 1e-9)
        h[:, ~reach] = 1e-9
        return normalize(h)
        
    def gen_2(p, d, ml, reach):
        d_0j = d[0, :]
        h = (p[None, :] + 1e-9) / (d + d_0j[None, :] + 1e-9)
        h[:, ~reach] = 1e-9
        return normalize(h)
        
    def gen_3(p, d, ml, reach):
        d_0j = d[0, :]
        h = (p[None, :] + 1e-9) / (d * d_0j[None, :] + 1e-9)
        h[:, ~reach] = 1e-9
        return normalize(h)
        
    def gen_4(p, d, ml, reach):
        h = (p[None, :] + 1e-9) * np.exp(-(d + 1e-9))
        h[:, ~reach] = 1e-9
        return normalize(h)
        
    def gen_5(p, d, ml, reach):
        d_0 = d[0, :]
        h = (p[None, :] + 1e-9) / (d + d_0[None, :] + d_0[:, None] + 1e-9)
        h[:, ~reach] = 1e-9
        return normalize(h)
        
    def gen_6(p, d, ml, reach):
        scores = p / (d[0, :] + 1e-9)
        order = np.argsort(scores)[::-1]
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n+1)
        h = (1.0 / ranks)[None, :]
        h = h / (d + 1e-9)
        h[:, ~reach] = 1e-9
        return normalize(h)

    def gen_7(p, d, ml, reach):
        d_0j = d[0, :]
        h = (p[None, :] + 1e-9) / (d + 0.5 * d_0j[None, :] + 1e-9)
        h[:, ~reach] = 1e-9
        return normalize(h)

    def gen_8(p, d, ml, reach):
        d_0j = d[0, :]
        h = (p[None, :] + 1e-9) / (d + 1.5 * d_0j[None, :] + 1e-9)
        h[:, ~reach] = 1e-9
        return normalize(h)
        
    def gen_9(p, d, ml, reach):
        # Prize density in local neighborhood?
        # Simple: p_j / (d_ij + 0.1)
        h = (p[None, :] + 1e-9) / (d + 0.1)
        h[:, ~reach] = 1e-9
        return normalize(h)

    candidates = [gen_1, gen_2, gen_3, gen_4, gen_5, gen_6, gen_7, gen_8, gen_9]
    
    best_score = -np.inf
    best_matrix = None
    
    for gen in candidates:
        if time.time() - start_time > timeout:
            break
            
        try:
            h_mat = gen(p, dist, max_len, reachable_mask)
            h_mat = np.where(np.isfinite(h_mat) & (h_mat > 0), h_mat, 1e-9)
            
            score = _greedy_score(p, dist, max_len, h_mat)
            
            if score > best_score:
                best_score = score
                best_matrix = h_mat
        except Exception:
            continue
            
    if best_matrix is None:
        best_matrix = np.ones((n, n))
        
    return best_matrix
