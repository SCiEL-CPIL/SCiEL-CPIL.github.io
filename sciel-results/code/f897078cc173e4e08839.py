import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128

    # Precompute conflict graph using bitmasks
    # conflict_mask[i] has bit j set if i and j are in a bad triple
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if c % g_ab != 0:
                    continue
                g_bc = gcd(b, c)
                if g_bc != g_ab:
                    continue
                g_ac = gcd(a, c)
                if g_ac == g_ab:
                    conflict_mask[a] |= (1 << b)
                    conflict_mask[a] |= (1 << c)
                    conflict_mask[b] |= (1 << a)
                    conflict_mask[b] |= (1 << c)
                    conflict_mask[c] |= (1 << a)
                    conflict_mask[c] |= (1 << b)

    # Build complement graph masks
    # comp_mask[i] has bits for j where i and j are NOT in conflict (and i != j)
    full_mask = (1 << (N + 1)) - 2 # Bits 1 to N set
    
    comp_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        others = full_mask ^ (1 << i)
        comp_mask[i] = others & ~conflict_mask[i]

    # Initial Solution: Greedy Independent Set on G
    def is_valid_add(S_mask, x):
        # Check if x has no conflict with any node in S_mask
        if (conflict_mask[x] & S_mask) == 0:
            return True
        return False

    best_mask = 0
    best_size = 0
    
    # Greedy
    for _ in range(20):
        S_mask = 0
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(S_mask, x):
                S_mask |= (1 << x)
        if S_mask.bit_count() > best_size:
            best_size = S_mask.bit_count()
            best_mask = S_mask

    # Tomita Max Clique on Comp Graph
    # We want to find a clique in comp_mask, which is an independent set in conflict_mask.
    
    def color_sort_with_colors(vertices):
        verts = []
        temp = vertices
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        v_colors = {}
        max_c = 0
        for v in verts:
            neighbors_mask = comp_mask[v] & vertices
            used_colors = set()
            temp_n = neighbors_mask
            while temp_n:
                lsb = temp_n & -temp_n
                n_idx = lsb.bit_length() - 1
                if n_idx in v_colors:
                    used_colors.add(v_colors[n_idx])
                temp_n ^= lsb
                
            c = 1
            while c in used_colors:
                c += 1
            v_colors[v] = c
            if c > max_c:
                max_c = c
                
        # Sort by color, then by index
        sorted_verts = sorted(verts, key=lambda x: (v_colors[x], x))
        return sorted_verts, v_colors, max_c

    def max_clique(vertices, current_mask, depth):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        if not vertices:
            if depth > best_size:
                best_size = depth
                best_mask = current_mask
            return
            
        sorted_verts, v_colors, max_color = color_sort_with_colors(vertices)
        
        if depth + max_color <= best_size:
            return
        
        for v in reversed(sorted_verts):
            if time.time() > deadline:
                return
                
            if depth + v_colors[v] <= best_size:
                break
                
            new_vertices = vertices & comp_mask[v]
            
            max_clique(new_vertices, current_mask | (1 << v), depth + 1)
            
            vertices ^= (1 << v)

    # Start the search
    max_clique(full_mask, 0, 0)

    # Extract best clique
    result = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(idx)
        temp ^= lsb
        
    return result
