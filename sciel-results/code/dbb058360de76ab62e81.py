import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 5.0:
            break
        
        if n >= 17:
            time_limit = 35.0
        elif n >= 15:
            time_limit = 15.0
        else:
            time_limit = 8.0
        
        target_end = min(deadline - 5.0, start_time + time_limit)
        result = _local_search(n, target_end)
        if result is not None:
            return result
    
    return _small_valid_solution(9)

def _small_valid_solution(n):
    for _ in range(5000):
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        if _is_valid_coloring(n, colors):
            return {'n': n, 'colors': colors}
    return {'n': 1, 'colors': []}

def _get_edge_vertices(idx, n):
    i = 0
    remaining = idx
    while i < n:
        edges_from_i = n - 1 - i
        if remaining < edges_from_i:
            j = i + 1 + remaining
            return (i, j)
        remaining -= edges_from_i
        i += 1
    return (0, 1)

def _build_adj(n, colors, edge_verts=None):
    if edge_verts is None:
        edge_verts = [_get_edge_vertices(idx, n) for idx in range(n * (n - 1) // 2)]
    adj_bits = [[0] * n for _ in range(4)]
    for idx, c in enumerate(colors):
        i, j = edge_verts[idx]
        adj_bits[c][i] |= (1 << j)
        adj_bits[c][j] |= (1 << i)
    return adj_bits

def _has_c6(adj, n):
    for verts in combinations(range(n), 6):
        a, b, c, d, e, f = verts
        if (adj[a] & (1 << b)) and (adj[b] & (1 << c)) and (adj[c] & (1 << d)) and \
           (adj[d] & (1 << e)) and (adj[e] & (1 << f)) and (adj[f] & (1 << a)):
            return True
        if (adj[a] & (1 << b)) and (adj[b] & (1 << c)) and (adj[c] & (1 << d)) and \
           (adj[d] & (1 << f)) and (adj[f] & (1 << e)) and (adj[e] & (1 << a)):
            return True
        if (adj[a] & (1 << b)) and (adj[b] & (1 << c)) and (adj[c] & (1 << e)) and \
           (adj[e] & (1 << d)) and (adj[d] & (1 << f)) and (adj[f] & (1 << a)):
            return True
        if (adj[a] & (1 << b)) and (adj[b] & (1 << d)) and (adj[d] & (1 << c)) and \
           (adj[c] & (1 << e)) and (adj[e] & (1 << f)) and (adj[f] & (1 << a)):
            return True
        if (adj[a] & (1 << b)) and (adj[b] & (1 << d)) and (adj[d] & (1 << e)) and \
           (adj[e] & (1 << c)) and (adj[c] & (1 << f)) and (adj[f] & (1 << a)):
            return True
        if (adj[a] & (1 << b)) and (adj[b] & (1 << e)) and (adj[e] & (1 << c)) and \
           (adj[c] & (1 << d)) and (adj[d] & (1 << f)) and (adj[f] & (1 << a)):
            return True
        if (adj[a] & (1 << b)) and (adj[b] & (1 << e)) and (adj[e] & (1 << d)) and \
           (adj[d] & (1 << c)) and (adj[c] & (1 << f)) and (adj[f] & (1 << a)):
            return True
        if (adj[a] & (1 << b)) and (adj[b] & (1 << f)) and (adj[f] & (1 << c)) and \
           (adj[c] & (1 << d)) and (adj[d] & (1 << e)) and (adj[e] & (1 << a)):
            return True
        if (adj[a] & (1 << b)) and (adj[b] & (1 << f)) and (adj[f] & (1 << c)) and \
           (adj[c] & (1 << e)) and (adj[e] & (1 << d)) and (adj[d] & (1 << a)):
            return True
        if (adj[a] & (1 << b)) and (adj[b] & (1 << f)) and (adj[f] & (1 << d)) and \
           (adj[d] & (1 << c)) and (adj[c] & (1 << e)) and (adj[e] & (1 << a)):
            return True
        if (adj[a] & (1 << b)) and (adj[b] & (1 << f)) and (adj[f] & (1 << d)) and \
           (adj[d] & (1 << e)) and (adj[e] & (1 << c)) and (adj[c] & (1 << a)):
            return True
    return False

def _count_total_c6(n, colors, edge_verts=None):
    adj_bits = _build_adj(n, colors, edge_verts)
    return sum(1 for c in range(4) if _has_c6(adj_bits[c], n))

def _is_valid_coloring(n, colors, edge_verts=None):
    adj_bits = _build_adj(n, colors, edge_verts)
    for c in range(4):
        if _has_c6(adj_bits[c], n):
            return False
    return True

def _local_search(n, deadline):
    num_edges = n * (n - 1) // 2
    edge_verts = [_get_edge_vertices(idx, n) for idx in range(num_edges)]
    
    max_restarts = 20
    restart_budget = (deadline - time.time()) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
        
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        adj_bits = _build_adj(n, colors, edge_verts)
        current_c6 = sum(1 for c in range(4) if _has_c6(adj_bits[c], n))
        
        if current_c6 == 0:
            return {'n': n, 'colors': colors}
        
        temp = 2.0
        cool_rate = 0.999
        min_temp = 0.01
        steps = int(restart_budget * 50)
        
        for step in range(steps):
            if time.time() > deadline - 2.0:
                break
            
            idx = random.randint(0, num_edges - 1)
            i, j = edge_verts[idx]
            old_c = colors[idx]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            adj_bits[old_c][i] &= ~(1 << j)
            adj_bits[old_c][j] &= ~(1 << i)
            adj_bits[new_c][i] |= (1 << j)
            adj_bits[new_c][j] |= (1 << i)
            
            new_c6 = sum(1 for c in range(4) if _has_c6(adj_bits[c], n))
            delta = new_c6 - current_c6
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 0.001)):
                colors[idx] = new_c
                current_c6 = new_c6
            else:
                adj_bits[new_c][i] &= ~(1 << j)
                adj_bits[new_c][j] &= ~(1 << i)
                adj_bits[old_c][i] |= (1 << j)
                adj_bits[old_c][j] |= (1 << i)
            
            temp *= cool_rate
            if temp < min_temp:
                temp = min_temp
        
        if current_c6 == 0:
            return {'n': n, 'colors': colors}
    
    return None
