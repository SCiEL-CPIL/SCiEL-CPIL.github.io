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
        """Check if a coloring is valid (all K4s have >= 5 colors)."""
        for e_indices in k4_edge_indices:
            if len(set(colors[idx] for idx in e_indices)) < 5:
                return False
        return True

    def gf4_init(max_colors):
        """
        Initialize using GF(4) structure.
        Represent 16 vertices as 4-bit vectors.
        Color edge (u,v) based on some function of u XOR v.
        We try several different linear forms over GF(4) mapped to colors.
        """
        # Each vertex is a 4-bit number 0..15
        # For edge (u,v), let d = u ^ v (nonzero 4-bit value)
        # We can map d to a color. There are 15 possible nonzero values.
        # We want to assign colors such that any 4 vertices span >= 5 colors.
        
        # Strategy: Use a random bijection from {1..15} to {0..max_colors-1}
        # but we only have max_colors colors, and 15 values.
        # If max_colors >= 15, we can use distinct colors for each d.
        # But we want to minimize. Let's try a few structured approaches.
        
        # Approach 1: Random assignment of the 15 XOR values to colors
        # This ensures that for any K4, the 6 edges have 6 XOR values.
        # Are they distinct? Not necessarily.
        # Example: vertices 0, 1, 2, 3 (00, 01, 10, 11)
        # Edges: 0-1 (01), 0-2 (10), 0-3 (11), 1-2 (11), 1-3 (10), 2-3 (01)
        # XOR values: {01, 10, 11}. Only 3 distinct XOR values!
        # So this approach fails for K4s that are affine subspaces of dimension 2.
        
        # We need a coloring that depends on more than just XOR.
        # Let's try: color(u,v) = (a*u + b*v) mod p for some prime p?
        # Or use a hash of (u,v) with structure.
        
        # Better: Use the fact that we need 5 colors per K4.
        # Let's just do random init but with a bias toward spreading colors.
        
        colors = [0] * num_edges
        # Assign random colors
        for idx in range(num_edges):
            colors[idx] = random.randint(0, max_colors - 1)
        
        # Do a quick greedy repair: for each K4 in random order, if deficit > 0, 
        # try to fix by changing one edge's color.
        k4_order = list(range(num_k4s))
        random.shuffle(k4_order)
        
        for _ in range(3):  # 3 passes
            for k4_idx in k4_order:
                e_indices = k4_edge_indices[k4_idx]
                cur_colors = [colors[idx] for idx in e_indices]
                distinct = set(cur_colors)
                if len(distinct) >= 5:
                    continue
                # Try to change one edge to increase distinct count
                for pos, e_idx in enumerate(e_indices):
                    old_c = colors[e_idx]
                    # Try a new color
                    for new_c in range(max_colors):
                        if new_c == old_c:
                            continue
                        if new_c in distinct:
                            continue
                        # Check if this helps (it will, since new_c is new)
                        # But we need to check it doesn't hurt other K4s.
                        # For simplicity in init, just accept if it increases local distinct.
                        colors[e_idx] = new_c
                        distinct.add(new_c)
                        break
                    if len(distinct) >= 5:
                        break
        
        return colors

    def local_search_smooth(max_colors, time_limit, init_colors=None):
        """
        Local search using a smooth objective:
        Minimize total_deficit = sum(max(0, 5 - distinct_colors) for all K4s)
        """
        if init_colors is not None:
            colors = init_colors[:]
        else:
            colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        # Precompute current state
        k4_counts = []
        k4_distinct = []
        k4_deficit = []
        total_deficit = 0
        
        for e_indices in k4_edge_indices:
            counts = np.zeros(max_colors, dtype=np.int8)
            for idx in e_indices:
                counts[colors[idx]] += 1
            distinct = np.count_nonzero(counts)
            deficit = 5 - distinct if distinct < 5 else 0
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            k4_deficit.append(deficit)
            total_deficit += deficit
            
        best_deficit = total_deficit
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 5.0
        min_temp = 0.05
        cool_rate = 0.9995
        
        # Track stagnation
        stagnation_count = 0
        max_stagnation = 5000
        
        # Main Loop
        while time.time() < start_time + time_limit:
            if total_deficit == 0:
                return colors, 0
            
            # Pick an edge from a K4 with high deficit (weighted selection)
            # Build weights based on deficit
            if total_deficit > 0:
                # Weight each edge by sum of deficits of K4s it belongs to
                edge_weights = np.zeros(num_edges)
                for k4_idx in range(num_k4s):
                    if k4_deficit[k4_idx] > 0:
                        for e_idx in k4_edge_indices[k4_idx]:
                            edge_weights[e_idx] += k4_deficit[k4_idx]
                
                # Normalize to probabilities
                total_w = edge_weights.sum()
                if total_w > 0:
                    probs = edge_weights / total_w
                    # Add small epsilon to avoid zero probs
                    probs += 1e-6
                    probs /= probs.sum()
                    e_idx = np.random.choice(num_edges, p=probs)
                else:
                    e_idx = random.randint(0, num_edges - 1)
            else:
                e_idx = random.randint(0, num_edges - 1)
            
            old_color = colors[e_idx]
            
            # Pick a new color different from old
            new_color = random.randint(0, max_colors - 1)
            if new_color == old_color:
                continue
            
            # Calculate Delta Deficit
            delta_deficit = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Store old states for revert if needed
            old_states = []
            
            for k4_idx in affected_k4s:
                counts = k4_counts[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_def = k4_deficit[k4_idx]
                old_states.append((k4_idx, old_d, old_def))
                
                # Update counts
                counts[old_color] -= 1
                counts[new_color] += 1
                
                # Update distinct
                new_d = old_d
                if counts[old_color] == 0:
                    new_d -= 1
                if counts[new_color] == 1: # It was 0 before
                    new_d += 1
                
                # Update deficit
                new_def = 5 - new_d if new_d < 5 else 0
                
                k4_distinct[k4_idx] = new_d
                k4_deficit[k4_idx] = new_def
                
                delta_deficit += new_def - old_def
            
            # Acceptance Criteria
            accept = False
            if delta_deficit <= 0:
                accept = True
                if delta_deficit < 0:
                    stagnation_count = 0
            else:
                # Avoid overflow in exp
                if delta_deficit < 100 * temp:
                    if random.random() < math.exp(-delta_deficit / temp):
                        accept = True
            
            if accept:
                colors[e_idx] = new_color
                total_deficit += delta_deficit
                if total_deficit < best_deficit:
                    best_deficit = total_deficit
                    best_colors = colors[:]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                # Revert counts
                for k4_idx, old_d, old_def in old_states:
                    counts = k4_counts[k4_idx]
                    counts[old_color] += 1
                    counts[new_color] -= 1
                    k4_distinct[k4_idx] = old_d
                    k4_deficit[k4_idx] = old_def
                stagnation_count += 1
            
            # Cool down
            temp *= cool_rate
            if temp < min_temp:
                temp = min_temp
            
            # Restart if stuck
            if stagnation_count > max_stagnation:
                # Perturb: change a random subset of edges
                num_perturb = random.randint(5, 15)
                for _ in range(num_perturb):
                    e_idx = random.randint(0, num_edges - 1)
                    new_c = random.randint(0, max_colors - 1)
                    old_c = colors[e_idx]
                    if new_c == old_c:
                        continue
                    # Update state
                    for k4_idx in edge_to_k4s[e_idx]:
                        counts = k4_counts[k4_idx]
                        old_d = k4_distinct[k4_idx]
                        old_def = k4_deficit[k4_idx]
                        counts[old_c] -= 1
                        counts[new_c] += 1
                        new_d = old_d
                        if counts[old_c] == 0:
                            new_d -= 1
                        if counts[new_c] == 1:
                            new_d += 1
                        new_def = 5 - new_d if new_d < 5 else 0
                        k4_distinct[k4_idx] = new_d
                        k4_deficit[k4_idx] = new_def
                        total_deficit += new_def - old_def
                    colors[e_idx] = new_c
                temp = max(1.0, temp * 10)
                stagnation_count = 0
                
        return best_colors, best_deficit

    # Try C=13
    # We have ~110s.
    # Try 13 for 45s with multiple restarts
    found = False
    for attempt in range(3):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        t_alloc = min(15.0, remaining - 5.0)
        if t_alloc < 3:
            break
        
        # Use GF4-inspired init for first attempt, random for others
        if attempt == 0:
            init = gf4_init(13)
        else:
            init = None
        
        sol, deficit = local_search_smooth(13, t_alloc, init)
        if deficit == 0:
            return sol
    
    # Try C=14 for remaining time
    remaining = deadline - time.time()
    if remaining > 10:
        for attempt in range(2):
            t_alloc = min(20.0, remaining - 5.0)
            if t_alloc < 3:
                break
            sol, deficit = local_search_smooth(14, t_alloc, None)
            if deficit == 0:
                return sol
    
    # Try C=15 for remaining time
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol, deficit = local_search_smooth(15, remaining - 2.0, None)
        if deficit == 0:
            return sol

    # Fallback: Return a valid solution with more colors
    # The seed (i+j)%17 is valid. 
    seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    return seed_colors
