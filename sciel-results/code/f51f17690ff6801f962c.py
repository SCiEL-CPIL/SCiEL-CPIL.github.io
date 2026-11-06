import time
import itertools
import random
import math

def _generate_k4s(n=16):
    """Generate all K4s as lists of edge indices."""
    k4s = []
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - i - 1) // 2 + (j - i - 1)
    
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [edge_idx(a, b), edge_idx(a, c), edge_idx(a, d), 
                 edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)]
        k4s.append(edges)
    return k4s

def _init_penalty(colors, k4s, n=16):
    """Calculate initial penalty for a coloring."""
    penalty = 0
    for edges in k4s:
        cset = set(colors[e] for e in edges)
        if len(cset) < 5:
            penalty += 5 - len(cset)
    return penalty

def _simulated_annealing(k, n=16, time_limit=60):
    """Try to find a valid coloring with k colors using simulated annealing."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    k4s = _generate_k4s(n)
    
    edge_to_k4s = [[] for _ in range(n*(n-1)//2)]
    for i, edges in enumerate(k4s):
        for e in edges:
            edge_to_k4s[e].append(i)
    
    colors = [random.randint(0, k-1) for _ in range(n*(n-1)//2)]
    
    penalty = _init_penalty(colors, k4s, n)
    
    if penalty == 0:
        return colors
    
    T = 1.0
    cooling = 0.9995
    best_colors = colors[:]
    best_penalty = penalty
    
    while time.time() < deadline:
        if penalty == 0:
            break
            
        e = random.randint(0, len(colors)-1)
        old_c = colors[e]
        new_c = random.randint(0, k-1)
        if new_c == old_c:
            continue
            
        delta = 0
        for k4_idx in edge_to_k4s[e]:
            edges = k4s[k4_idx]
            cset_before = set(colors[ed] for ed in edges)
            pen_before = max(0, 5 - len(cset_before))
            
            colors[e] = new_c
            cset_after = set(colors[ed] for ed in edges)
            pen_after = max(0, 5 - len(cset_after))
            
            delta += pen_after - pen_before
            
        if delta < 0 or random.random() < math.exp(-delta / T):
            colors[e] = new_c
            penalty += delta
            if penalty < best_penalty:
                best_penalty = penalty
                best_colors = colors[:]
                if penalty == 0:
                    return colors
        else:
            colors[e] = old_c
            
        T *= cooling
        if T < 0.001:
            T = 0.001
    
    return best_colors if best_penalty == 0 else None

def _proper_edge_coloring(n=16):
    """Generate a proper edge coloring with n-1 colors."""
    if n % 2 != 0:
        return None
        
    num_edges = n * (n - 1) // 2
    colors = [0] * num_edges
    
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - i - 1) // 2 + (j - i - 1)
        
    for k in range(n-1):
        color = k
        idx = edge_idx(k, n-1)
        colors[idx] = color
        
        for i in range(1, n//2):
            u = (k + i) % (n-1)
            v = (k - i) % (n-1)
            idx = edge_idx(u, v)
            colors[idx] = color
            
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    k4s = _generate_k4s(n)
    
    best_result = None
    best_k = 15
    
    # Try 14 colors
    if time.time() < deadline - 50:
        res = _simulated_annealing(14, n, 50)
        if res is not None:
            if _init_penalty(res, k4s, n) == 0:
                best_result = res
                best_k = 14
                
    # Try 13 colors
    if best_k > 13 and time.time() < deadline - 40:
        res = _simulated_annealing(13, n, 40)
        if res is not None:
            if _init_penalty(res, k4s, n) == 0:
                best_result = res
                best_k = 13

    if best_result is not None:
        return best_result
    else:
        return _proper_edge_coloring(n)
