import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    
    # List of all edges (i, j)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # List of all K4s (sets of 4 vertices)
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s) # 1820
    
    # For each K4, find the indices of its 6 edges
    k4_edge_indices = []
    for k4 in k4s:
        vs = list(k4)
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(vs[i], vs[j])])
        k4_edge_indices.append(e_indices)
        
    # For each edge, find which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)

    def is_valid(colors):
        for e_indices in k4_edge_indices:
            if len(set(colors[idx] for idx in e_indices)) < 5:
                return False
        return True

    # GF(4) Initialization
    # F4 = {0, 1, 2, 3}
    # 0=0, 1=1, 2=w, 3=w+1
    # Add table
    add4 = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ]
    
    # Map vertex i to (x, y) in F4^2
    # i = 4*x + y
    def get_coords(i):
        return (i // 4, i % 4)
    
    def f4_sub(a, b):
        return add4[a][b]
        
    # Initialize colors based on difference
    # Color = index of non-zero difference vector
    # There are 15 non-zero vectors in F4^2
    # We map each non-zero vector to a color 0..14
    diff_to_color = {}
    color_id = 0
    for x in range(4):
        for y in range(4):
            if x == 0 and y == 0:
                continue
            diff_to_color[(x, y)] = color_id
            color_id += 1
            
    init_colors = [0] * num_edges
    for idx, (u, v) in enumerate(edges):
        ux, uy = get_coords(u)
        vx, vy = get_coords(v)
        dx = f4_sub(ux, vx)
        dy = f4_sub(uy, vy)
        # Note: difference is symmetric in char 2, so (dx, dy) is same as (vx-ux, vy-uy)
        init_colors[idx] = diff_to_color[(dx, dy)]

    # This initialization uses 15 colors.
    # It is known to be invalid for parallelograms (3 colors).
    # But it's a good start.
    
    def local_search(max_colors, time_limit, init=None):
        if init is None:
            colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        else:
            # Ensure colors are within 0..max_colors-1
            # If max_colors < 15, we need to remap or just let it handle high values?
            # The problem asks for colors in 0..119, but we want to minimize count.
            # If we use 15 colors, they are 0..14.
            # If we want to try 13 colors, we should probably start with a valid 15-coloring
            # and then try to merge colors? Or just run SA with max_colors=13?
            # Starting from a 15-coloring and restricting to 13 colors immediately might cause huge violations.
            # Better: Start with 15 colors, find a valid 15-coloring, then try to reduce.
            # Or: Run SA with max_colors=15, then check if we can compress.
            
            # For now, let's assume we run SA with the given max_colors.
            # If init has values >= max_colors, we mod them? No, that breaks structure.
            # If max_colors < len(set(init)), we should probably randomize those?
            # Let's just use the init as is, assuming max_colors >= 15 for the first pass.
            colors = init[:]
            
        # Precompute current state
        k4_counts = []
        k4_distinct = []
        total_violations = 0
        
        # We need to handle the case where colors might exceed max_colors if init is used
        # But we will call local_search(15, ...) first.
        
        for e_indices in k4_edge_indices:
            counts = [0] * max_colors
            for idx in e_indices:
                c = colors[idx]
                if c >= max_colors:
                    # This shouldn't happen if we pass correct max_colors
                    c = c % max_colors
                counts[c] += 1
            distinct = sum(1 for c in counts if c > 0)
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            if distinct < 5:
                total_violations += 1
                
        best_violations = total_violations
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 5.0
        min_temp = 0.01
        cool_rate = 0.9999
        
        # Main Loop
        while time.time() < start_time + time_limit:
            if total_violations == 0:
                return colors, 0
            
            # Pick a random edge
            # Bias towards edges in violating K4s?
            # Simple random is ok, but let's try to pick from a random violating K4's edges
            if total_violations > 0:
                # Pick a random violating K4
                viol_k4s = [k for k in range(num_k4s) if k4_distinct[k] < 5]
                if viol_k4s:
                    k4_idx = random.choice(viol_k4s)
                    e_idx = random.choice(k4_edge_indices[k4_idx])
                else:
                    e_idx = random.randint(0, num_edges - 1)
            else:
                e_idx = random.randint(0, num_edges - 1)
                
            old_color = colors[e_idx]
            
            # Pick a new color
            # Try to pick a color that helps the violating K4s
            new_color = random.randint(0, max_colors - 1)
            if new_color == old_color:
                continue
            
            # Calculate Delta Violations
            delta_viol = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Store old distinct values for revert
            old_distincts = []
            
            for k4_idx in affected_k4s:
                counts = k4_counts[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_distincts.append(old_d)
                
                # Update counts
                counts[old_color] -= 1
                counts[new_color] += 1
                
                # Update distinct
                new_d = old_d
                if counts[old_color] == 0:
                    new_d -= 1
                if counts[new_color] == 1: # It was 0 before
                    new_d += 1
                
                k4_distinct[k4_idx] = new_d
                
                # Update violation status
                was_viol = 1 if old_d < 5 else 0
                is_viol = 1 if new_d < 5 else 0
                delta_viol += is_viol - was_viol
            
            # Acceptance Criteria
            accept = False
            if delta_viol <= 0:
                accept = True
            else:
                # Avoid overflow in exp
                if delta_viol < 100 * temp:
                    if random.random() < math.exp(-delta_viol / temp):
                        accept = True
            
            if accept:
                colors[e_idx] = new_color
                total_violations += delta_viol
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
            else:
                # Revert counts
                for k4_idx, old_d in zip(affected_k4s, old_distincts):
                    counts = k4_counts[k4_idx]
                    counts[old_color] += 1
                    counts[new_color] -= 1
                    k4_distinct[k4_idx] = old_d
            
            # Cool down
            temp *= cool_rate
            if temp < min_temp:
                pass
                
        return best_colors, best_violations

    # Strategy:
    # 1. Find a valid 15-coloring using the GF(4) init.
    # 2. Try to compress to 14 colors.
    # 3. Try to compress to 13 colors.
    
    # Phase 1: 15 colors
    sol, viol = local_search(15, 40.0, init=init_colors)
    if viol == 0:
        best_sol = sol
        best_colors_used = 15
    else:
        # Fallback to random if GF(4) init fails to converge (unlikely)
        sol, viol = local_search(15, 20.0)
        if viol == 0:
            best_sol = sol
            best_colors_used = 15
        else:
            # Return seed if all else fails
            seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
            return seed_colors

    # Phase 2: Try to reduce to 14 colors
    # We can try to run SA with max_colors=14, starting from the 15-color solution.
    # However, if the 15-color solution uses color 14, we need to map it.
    # A better way: Run SA with max_colors=14, but initialize with the 15-color solution
    # mapped to 14 colors (e.g., merge color 14 into 0)?
    # Or just run SA from scratch with 14 colors?
    # Given time, let's try a short run with 14 colors, initializing with the 15-color solution
    # but clamping colors.
    
    if time.time() < deadline - 30.0:
        # Map 15-color solution to 14 colors
        # Simple mapping: if c == 14, c = 0
        init_14 = [c % 14 for c in best_sol]
        sol14, viol14 = local_search(14, 30.0, init=init_14)
        if viol14 == 0:
            best_sol = sol14
            best_colors_used = 14

    # Phase 3: Try to reduce to 13 colors
    remaining = deadline - time.time()
    if remaining > 5.0:
        init_13 = [c % 13 for c in best_sol]
        sol13, viol13 = local_search(13, remaining - 2.0, init=init_13)
        if viol13 == 0:
            best_sol = sol13
            best_colors_used = 13

    return best_sol
