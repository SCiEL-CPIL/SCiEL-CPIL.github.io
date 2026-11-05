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
    num_k4s = len(k4s)  # 1820
    
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
    
    def compute_deficit(colors, max_colors):
        """Compute total deficit for a coloring."""
        deficit = 0
        for e_indices in k4_edge_indices:
            distinct = len(set(colors[idx] for idx in e_indices))
            if distinct < 5:
                deficit += 5 - distinct
        return deficit

    def structured_init(max_colors, trial_seed=0):
        """
        Use vertex labels to derive edge colors via multiple non-linear functions.
        Try several structured approaches including GF(4) and XOR-based schemes.
        """
        rng = random.Random(trial_seed)
        best_init = None
        best_deficit = float('inf')
        
        # Strategy 1: Quadratic form over Z_p
        for trial in range(30):
            labels = list(range(n))
            rng.shuffle(labels)
            a = rng.randint(1, max_colors - 1)
            b = rng.randint(0, max_colors - 1)
            c = rng.randint(0, max_colors - 1)
            
            colors = [0] * num_edges
            for idx, (i, j) in enumerate(edges):
                u, v = labels[i], labels[j]
                colors[idx] = (a * u * v + b * (u + v) + c) % max_colors
            
            deficit = compute_deficit(colors, max_colors)
            if deficit < best_deficit:
                best_deficit = deficit
                best_init = colors[:]
            if deficit == 0:
                break
        
        # Strategy 2: XOR-based (works well for power-of-2 color counts)
        if max_colors >= 16:
            for trial in range(20):
                # Assign each vertex a 4-bit label
                labels = [rng.randint(0, 15) for _ in range(n)]
                # Ensure all labels are distinct
                while len(set(labels)) < n:
                    labels = [rng.randint(0, 15) for _ in range(n)]
                
                colors = [0] * num_edges
                for idx, (i, j) in enumerate(edges):
                    colors[idx] = labels[i] ^ labels[j]
                
                deficit = compute_deficit(colors, max_colors)
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_init = colors[:]
                if deficit == 0:
                    break
        
        # Strategy 3: Multiplication mod prime (for prime max_colors)
        if max_colors == 13:
            for trial in range(30):
                labels = list(range(n))
                rng.shuffle(labels)
                a = rng.randint(1, 12)
                b = rng.randint(0, 12)
                
                colors = [0] * num_edges
                for idx, (i, j) in enumerate(edges):
                    u, v = labels[i], labels[j]
                    # Use (u-v)^2 mod 13 to get symmetric colors
                    colors[idx] = ((u - v) ** 2) % 13
                
                deficit = compute_deficit(colors, max_colors)
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_init = colors[:]
                if deficit == 0:
                    break
        
        # Strategy 4: Pairwise distance-based
        for trial in range(20):
            # Random permutation
            perm = list(range(n))
            rng.shuffle(perm)
            # Edge color based on circular distance
            colors = [0] * num_edges
            for idx, (i, j) in enumerate(edges):
                d = abs(perm[i] - perm[j])
                colors[idx] = d % max_colors
            
            deficit = compute_deficit(colors, max_colors)
            if deficit < best_deficit:
                best_deficit = deficit
                best_init = colors[:]
            if deficit == 0:
                break
        
        # Strategy 5: Random with greedy repair
        if best_deficit > 0:
            colors = [rng.randint(0, max_colors - 1) for _ in range(num_edges)]
            
            # Greedy repair: for each K4 with deficit, try to fix
            k4_order = list(range(num_k4s))
            rng.shuffle(k4_order)
            for k4_idx in k4_order:
                e_indices = k4_edge_indices[k4_idx]
                cur_colors = [colors[idx] for idx in e_indices]
                distinct = set(cur_colors)
                if len(distinct) >= 5:
                    continue
                for pos, e_idx in enumerate(e_indices):
                    old_c = colors[e_idx]
                    for new_c in range(max_colors):
                        if new_c == old_c or new_c in distinct:
                            continue
                        colors[e_idx] = new_c
                        distinct.add(new_c)
                        break
                    if len(distinct) >= 5:
                        break
            
            deficit = compute_deficit(colors, max_colors)
            if deficit < best_deficit:
                best_deficit = deficit
                best_init = colors[:]
        
        return best_init

    def local_search_smooth(max_colors, time_limit, init_colors=None):
        """
        Local search minimizing total_deficit = sum(max(0, 5 - distinct_colors) for all K4s)
        Optimized with precomputed structures.
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
            counts = [0] * max_colors
            for idx in e_indices:
                counts[colors[idx]] += 1
            distinct = sum(1 for c in counts if c > 0)
            deficit = 5 - distinct if distinct < 5 else 0
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            k4_deficit.append(deficit)
            total_deficit += deficit
            
        best_deficit = total_deficit
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 3.0
        min_temp = 0.01
        cool_rate = 0.9998
        
        # Track stagnation
        stagnation_count = 0
        max_stagnation = 2000
        
        # Precompute edge weights for weighted selection
        edge_k4_count = [len(edge_to_k4s[e]) for e in range(num_edges)]
        
        # Main Loop
        iterations = 0
        while time.time() < start_time + time_limit:
            iterations += 1
            if total_deficit == 0:
                return colors, 0
            
            # Pick an edge: biased toward edges in K4s with high deficit
            if total_deficit > 0 and iterations % 5 == 0:
                # Recompute edge weights every 5 iterations
                edge_weights = [0.0] * num_edges
                for k4_idx in range(num_k4s):
                    d = k4_deficit[k4_idx]
                    if d > 0:
                        for e_idx in k4_edge_indices[k4_idx]:
                            edge_weights[e_idx] += d
                total_w = sum(edge_weights)
                if total_w > 0:
                    # Weighted random selection
                    r = random.random() * total_w
                    cumulative = 0
                    e_idx = 0
                    for i in range(num_edges):
                        cumulative += edge_weights[i]
                        if r <= cumulative:
                            e_idx = i
                            break
                else:
                    e_idx = random.randint(0, num_edges - 1)
            else:
                e_idx = random.randint(0, num_edges - 1)
            
            old_color = colors[e_idx]
            
            # Pick a new color
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
                if counts[new_color] == 1:  # It was 0 before
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
                num_perturb = random.randint(8, 20)
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
                temp = max(2.0, temp * 5)
                stagnation_count = 0
                
        return best_colors, best_deficit

    # Try C=13 with multiple restarts using structured initialization
    found = False
    for attempt in range(8):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        t_alloc = min(12.0, remaining - 5.0)
        if t_alloc < 3:
            break
        
        # Use structured init with different seeds
        init = structured_init(13, trial_seed=attempt * 1000)
        
        sol, deficit = local_search_smooth(13, t_alloc, init)
        if deficit == 0:
            return sol
    
    # Try C=14 for remaining time
    remaining = deadline - time.time()
    if remaining > 10:
        for attempt in range(4):
            t_alloc = min(12.0, remaining - 5.0)
            if t_alloc < 3:
                break
            init = structured_init(14, trial_seed=attempt * 1000 + 500)
            sol, deficit = local_search_smooth(14, t_alloc, init)
            if deficit == 0:
                return sol
    
    # Try C=15 for remaining time
    remaining = deadline - time.time()
    if remaining > 5.0:
        init = structured_init(15, trial_seed=9999)
        sol, deficit = local_search_smooth(15, remaining - 2.0, init)
        if deficit == 0:
            return sol

    # Fallback: Return a valid solution with more colors
    # The seed (i+j)%17 is valid. 
    seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    return seed_colors
