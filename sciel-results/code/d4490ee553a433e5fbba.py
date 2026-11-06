import time
import itertools
import random
import numpy as np

def _check_validity_fast(colors, n=16):
    """Check if edge coloring satisfies: every K4 spans >= 5 distinct colors."""
    col = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            col[i][j] = col[j][i] = colors[idx]
            idx += 1
    
    for a,b,c,d in itertools.combinations(range(n), 4):
        cset = {col[a][b], col[a][c], col[a][d], col[b][c], col[b][d], col[c][d]}
        if len(cset) < 5:
            return False
    return True

def _build_k4_edges(n=16):
    """Precompute edge indices for all K4 subgraphs."""
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
    
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
    
    k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for quad in k4s:
        a, b, c, d = quad
        e = [edge_index[(a,b)], edge_index[(a,c)], edge_index[(a,d)],
             edge_index[(b,c)], edge_index[(b,d)], edge_index[(c,d)]]
        k4_edges.append(e)
    
    return k4_edges

def _count_violations(colors, k4_edges):
    """Count K4s with fewer than 5 distinct colors."""
    viol = 0
    for e in k4_edges:
        cset = {colors[e[0]], colors[e[1]], colors[e[2]],
                colors[e[3]], colors[e[4]], colors[e[5]]}
        if len(cset) < 5:
            viol += 1
    return viol

def _simulated_annealing(C, deadline, n=16, seed=42):
    """Simulated annealing to find valid coloring with C colors."""
    rng = random.Random(seed)
    k4_edges = _build_k4_edges(n)
    num_k4 = len(k4_edges)
    
    # Precompute for each edge, which K4s contain it
    edge_k4s = [[] for _ in range(120)]
    for k4_idx, e in enumerate(k4_edges):
        for pos in range(6):
            edge_k4s[e[pos]].append(k4_idx)
    
    # Initial random coloring
    cur_colors = [rng.randint(0, C-1) for _ in range(120)]
    cur_energy = _count_violations(cur_colors, k4_edges)
    
    if cur_energy == 0:
        return cur_colors
    
    best_colors = cur_colors[:]
    best_energy = cur_energy
    
    T = 2.0
    cooling = 0.9997
    min_T = 0.01
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        if T < min_T:
            break
        
        # Pick a random edge
        edge_idx = rng.randint(0, 119)
        old_c = cur_colors[edge_idx]
        
        # Try a new color
        new_c = rng.randint(0, C-1)
        if new_c == old_c:
            new_c = (new_c + 1) % C
        
        cur_colors[edge_idx] = new_c
        
        # Compute energy delta efficiently
        # Only K4s containing this edge can change status
        delta = 0
        for k4_idx in edge_k4s[edge_idx]:
            e = k4_edges[k4_idx]
            cset = {cur_colors[e[0]], cur_colors[e[1]], cur_colors[e[2]],
                    cur_colors[e[3]], cur_colors[e[4]], cur_colors[e[5]]}
            new_count = len(cset)
            
            # Restore old color temporarily to check old state
            cur_colors[edge_idx] = old_c
            cset_old = {cur_colors[e[0]], cur_colors[e[1]], cur_colors[e[2]],
                        cur_colors[e[3]], cur_colors[e[4]], cur_colors[e[5]]}
            old_count = len(cset_old)
            cur_colors[edge_idx] = new_c
            
            # Energy contribution: 1 if count < 5, else 0
            old_contrib = 1 if old_count < 5 else 0
            new_contrib = 1 if new_count < 5 else 0
            delta += new_contrib - old_contrib
        
        new_energy = cur_energy + delta
        
        # Accept or reject
        if delta <= 0:
            cur_energy = new_energy
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_colors = cur_colors[:]
                if best_energy == 0:
                    return best_colors
        else:
            # Boltzmann acceptance
            if rng.random() < np.exp(-delta / T):
                cur_energy = new_energy
        
        T *= cooling
    
    return best_colors

def _try_linear_form(m, a_coeff, b_coeff, n=16):
    """Try coloring (i,j) -> (a_coeff*i + b_coeff*j) % m"""
    colors = [(a_coeff*i + b_coeff*j) % m for i in range(n) for j in range(i+1, n)]
    if _check_validity_fast(colors):
        return colors
    return None

def _try_quadratic_form(m, a_coeff, b_coeff, c_coeff, n=16):
    """Try coloring (i,j) -> (a_coeff*i + b_coeff*j + c_coeff*i*j) % m"""
    colors = [(a_coeff*i + b_coeff*j + c_coeff*i*j) % m 
              for i in range(n) for j in range(i+1, n)]
    if _check_validity_fast(colors):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n = 16
    
    # Baseline: 17-color solution
    baseline = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    if _check_validity_fast(baseline):
        best_colors = baseline
        best_num = 17
    else:
        best_colors = None
        best_num = 18
    
    # Try linear forms with m < 17
    for m in range(13, 17):
        if time.time() > deadline - 30:
            break
        for a_coeff in range(1, m):
            for b_coeff in range(1, m):
                if time.time() > deadline - 30:
                    break
                colors = _try_linear_form(m, a_coeff, b_coeff, n)
                if colors is not None:
                    num_used = len(set(colors))
                    if num_used < best_num:
                        best_num = num_used
                        best_colors = colors
                        if num_used <= 13:
                            return best_colors
    
    # Try quadratic forms
    if time.time() < deadline - 40 and best_num > 13:
        for m in range(13, 17):
            if time.time() > deadline - 40:
                break
            for a_coeff in range(1, m):
                for b_coeff in range(1, m):
                    for c_coeff in range(1, m):
                        if time.time() > deadline - 40:
                            break
                        colors = _try_quadratic_form(m, a_coeff, b_coeff, c_coeff, n)
                        if colors is not None:
                            num_used = len(set(colors))
                            if num_used < best_num:
                                best_num = num_used
                                best_colors = colors
                                if num_used <= 13:
                                    return best_colors
    
    # Simulated annealing for fewer colors
    if time.time() < deadline - 10 and best_num > 13:
        for C in range(13, best_num):
            if time.time() > deadline - 10:
                break
            # Multiple restarts with different seeds
            for seed in range(20):
                if time.time() > deadline - 10:
                    break
                result = _simulated_annealing(C, deadline, n, seed)
                if result is not None:
                    if _check_validity_fast(result):
                        best_colors = result
                        best_num = C
                        break
            if best_num <= 13:
                break
    
    # Fallback
    if best_colors is None:
        best_colors = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    
    return best_colors
