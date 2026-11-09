"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _triple_indices(n):
    """Return list of (i,j,k) for all triples a<b<c of vertices 0..n-1."""
    return [(a, b, c) for a in range(n) for b in range(a+1, n) for c in range(b+1, n)]

def _all_4sets(n):
    """Return list of all 4-sets of vertices."""
    return [tuple(combinations(range(n), 4))] if False else [c for c in combinations(range(n), 4)]

def _all_5sets(n):
    """Return list of all 5-sets of vertices."""
    return [c for c in combinations(range(n), 5)]

def _triple_idx_map(n):
    """Map (a,b,c) -> index in the color list."""
    m = {}
    idx = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                m[(a,b,c)] = idx
                idx += 1
    return m

def _violation_count(colors, n, tidx_map, red_4sets, blue_5sets):
    """Count red K4 and blue K5 violations."""
    red_viol = 0
    for quad in red_4sets:
        a, b, c, d = quad
        if (colors[tidx_map[(a,b,c)]] == 0 and
            colors[tidx_map[(a,b,d)]] == 0 and
            colors[tidx_map[(a,c,d)]] == 0 and
            colors[tidx_map[(b,c,d)]] == 0):
            red_viol += 1
    blue_viol = 0
    for quint in blue_5sets:
        a, b, c, d, e = quint
        cnt = 0
        for triple in combinations(quint, 3):
            if colors[tidx_map[triple]] == 1:
                cnt += 1
        if cnt == 10:
            blue_viol += 1
    return red_viol + blue_viol

def _mod_coloring(n, mod):
    """Color triple (a,b,c) red if (a+b+c) % mod == 0, else blue."""
    colors = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                colors.append(0 if (a + b + c) % mod == 0 else 1)
    return np.array(colors, dtype=np.uint8)

def _random_coloring(n):
    ntriples = n * (n-1) * (n-2) // 6
    return np.array([random.randint(0, 1) for _ in range(ntriples)], dtype=np.uint8)

def _simulated_annealing(colors, n, tidx_map, red_4sets, blue_5sets, 
                          start_temp, end_temp, steps, deadline):
    """Run simulated annealing to minimize violations."""
    current_colors = colors.copy()
    current_viol = _violation_count(current_colors, n, tidx_map, red_4sets, blue_5sets)
    best_colors = current_colors.copy()
    best_viol = current_viol
    
    ntriples = len(current_colors)
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        # Choose a random triple to flip
        idx = random.randint(0, ntriples - 1)
        old_color = current_colors[idx]
        current_colors[idx] = 1 - old_color
        
        new_viol = _violation_count(current_colors, n, tidx_map, red_4sets, blue_5sets)
        delta = new_viol - current_viol
        
        temp = start_temp * (end_temp / start_temp) ** (step / max(steps, 1))
        
        if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_viol = new_viol
            if new_viol < best_viol:
                best_viol = new_viol
                best_colors = current_colors.copy()
        else:
            current_colors[idx] = old_color
        
        if best_viol == 0:
            break
    
    return best_colors, best_viol

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Try multiple approaches with increasing n
    best_result = None
    best_n = 0
    
    # Start with a known valid small construction and try to grow
    # Use modular coloring as warm start
    
    for n in range(9, 35):
        if time.time() > deadline:
            break
        
        # Check if we have enough time for this n
        # Estimate time based on n
        ntriples = n * (n-1) * (n-2) // 6
        n4sets = n * (n-1) * (n-2) * (n-3) // 24
        n5sets = n * (n-1) * (n-2) * (n-3) * (n-4) // 120
        
        # For large n, violation checking is expensive
        # Use a faster check for larger n
        if n > 20:
            # Use a more efficient approach
            pass
        
        tidx_map = _triple_idx_map(n)
        red_4sets = _all_4sets(n)
        blue_5sets = _all_5sets(n)
        
        # Try multiple initializations
        inits = []
        
        # Modular coloring mod 3
        inits.append(_mod_coloring(n, 3))
        # Modular coloring mod 2
        inits.append(_mod_coloring(n, 2))
        # Random
        inits.append(_random_coloring(n))
        
        best_viol = float('inf')
        best_colors = None
        
        for init_colors in inits:
            if time.time() > deadline:
                break
            
            # Quick violation check
            viol = _violation_count(init_colors, n, tidx_map, red_4sets, blue_5sets)
            if viol == 0:
                return {'n': n, 'colors': init_colors.tolist()}
            
            if viol < best_viol:
                best_viol = viol
                best_colors = init_colors.copy()
        
        if best_colors is None:
            continue
        
        # Run simulated annealing
        start_temp = 1.0
        end_temp = 0.01
        steps = 5000
        
        # Adjust steps based on remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            steps = min(steps, 500)
        
        best_colors, best_viol = _simulated_annealing(
            best_colors, n, tidx_map, red_4sets, blue_5sets,
            start_temp, end_temp, steps, deadline
        )
        
        if best_viol == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        
        if n > best_n:
            best_n = n
            best_result = {'n': n, 'colors': best_colors.tolist()}
    
    # If we didn't find a valid coloring, return the best we have
    if best_result is not None:
        return best_result
    
    # Fallback: return a small valid construction
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
