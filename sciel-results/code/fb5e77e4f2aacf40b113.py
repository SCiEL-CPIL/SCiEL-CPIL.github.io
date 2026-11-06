import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit
    
    # 1. Try to find a valid n=17 solution quickly
    # We attempt a structured construction (Paley) first, then fallback to SA.
    best_n = 0
    best_colors = []
    
    # Try Paley(17)
    n_try = 17
    colors_try = _generate_paley_17()
    if _verify_no_c6(colors_try, n_try):
        best_n = n_try
        best_colors = colors_try
    else:
        # Fallback to SA for n=17
        result_17 = _solve_sa(17, deadline, time_budget=5.0)
        if result_17:
            best_n = 17
            best_colors = result_17['colors']
        else:
            # If even n=17 fails, try smaller? Unlikely, but safe.
            # For now, return empty or seed if fail
            best_n = 9
            best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

    # 2. If we have time, try n=18
    if time.time() < deadline - 10.0 and best_n < 18:
        # We have plenty of time. Try n=18.
        # Use a more aggressive SA or multiple restarts.
        result_18 = _solve_sa(18, deadline, time_budget=deadline - time.time() - 5.0)
        if result_18:
            return {'n': 18, 'colors': result_18['colors']}
            
    return {'n': best_n, 'colors': best_colors}

def _generate_paley_17():
    n = 17
    qr = {1, 2, 4, 8, 9, 13, 15, 16}
    colors = [0] * (n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in qr:
                colors[idx] = 0
            else:
                colors[idx] = 1
            idx += 1
    return colors

def _verify_no_c6(colors, n):
    # Build adjacency bitmasks for each color
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    # Check for monochromatic C6
    # Iterate over all combinations of 6 vertices
    # For each combination, check all 60 cycles
    for vertices in itertools.combinations(range(n), 6):
        seen_canonical = set()
        for perm in itertools.permutations(vertices):
            # Canonicalize
            min_idx = perm.index(min(perm))
            rot = perm[min_idx:] + perm[:min_idx]
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            if rot in seen_canonical:
                continue
            seen_canonical.add(rot)
            
            # Check if this cycle is monochromatic
            c_val = -1
            is_mono = True
            for k in range(6):
                u, v = rot[k], rot[(k+1)%6]
                if u > v: u, v = v, u
                # Find color
                # We can precompute a lookup table for speed, but n=17 is small
                # Linear search in adj? No, adj is bitmask.
                # We need to know which color has the edge.
                # We can check adj[0], adj[1], adj[2], adj[3]
                found = False
                for c in range(4):
                    if adj[c][u] & (1 << v):
                        if c_val == -1:
                            c_val = c
                        elif c_val != c:
                            is_mono = False
                            break
                        found = True
                        break
                if not found:
                    # Edge must exist in one of the colors
                    # This shouldn't happen if partition is correct
                    is_mono = False
                    break
                if not is_mono:
                    break
                    
            if is_mono and c_val != -1:
                return False
                
    return True

def _solve_sa(n, deadline, time_budget):
    # Precompute cycles
    if time.time() > deadline - 5.0:
        return None
        
    vertices = list(range(n))
    cycles = []
    edge_to_cycles = {}
    
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_map[(i, j)] = idx
            idx += 1
            
    num_edges = idx
    
    for e in range(num_edges):
        edge_to_cycles[e] = []
        
    # Generate cycles
    # Optimization: Only generate cycles if n >= 6
    if n < 6:
        return {'n': n, 'colors': [0]*num_edges}
        
    for subset in itertools.combinations(vertices, 6):
        seen_canonical = set()
        for perm in itertools.permutations(subset):
            min_idx = perm.index(min(perm))
            rot = perm[min_idx:] + perm[:min_idx]
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            if rot in seen_canonical:
                continue
            seen_canonical.add(rot)
            
            edges = []
            for k in range(6):
                u = rot[k]
                v = rot[(k+1)%6]
                if u > v:
                    u, v = v, u
                edges.append(edge_map[(u, v)])
            
            cycle_id = len(cycles)
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(cycle_id)
                
    num_cycles = len(cycles)
    
    # Initialize
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    cycle_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
    total_violations = 0
    
    for c_id in range(num_cycles):
        edges = cycles[c_id]
        for e in edges:
            c = colors[e]
            cycle_counts[c_id][c] += 1
        for c in range(4):
            if cycle_counts[c_id][c] == 6:
                total_violations += 1
                break
                
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Run SA
    while time.time() < deadline - 2.0:
        if total_violations == 0:
            return {'n': n, 'colors': colors}
            
        e = random.randint(0, num_edges - 1)
        old_color = colors[e]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
            
        delta = 0
        affected_cycles = edge_to_cycles[e]
        
        for c_id in affected_cycles:
            cc = cycle_counts[c_id]
            old_count = cc[old_color]
            
            old_violation = 1 if old_count == 6 else 0
            
            cc[old_color] -= 1
            cc[new_color] += 1
            new_count = cc[new_color]
            
            new_violation = 1 if new_count == 6 else 0
            
            delta += new_violation - old_violation
            
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            colors[e] = new_color
            total_violations += delta
            T *= cooling
            if T < min_T:
                T = min_T
        else:
            for c_id in affected_cycles:
                cc = cycle_counts[c_id]
                cc[old_color] += 1
                cc[new_color] -= 1
            T *= cooling
            if T < min_T:
                T = min_T

    if total_violations == 0:
        return {'n': n, 'colors': colors}
        
    return None
# EVOLVE-BLOCK-END
