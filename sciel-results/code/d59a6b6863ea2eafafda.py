import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    n = 16
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
    
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        e1 = edge_idx[(a, b)]
        e2 = edge_idx[(a, c)]
        e3 = edge_idx[(a, d)]
        e4 = edge_idx[(b, c)]
        e5 = edge_idx[(b, d)]
        e6 = edge_idx[(c, d)]
        k4s.append((e1, e2, e3, e4, e5, e6))
        
    k4_edges_np = np.array(k4s, dtype=np.int32)  # shape (1820, 6)
    num_k4s = len(k4s)
    
    # Precompute for each edge, which K4s it belongs to (for targeted search)
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, k4 in enumerate(k4s):
        for e in k4:
            edge_to_k4s[e].append(k4_idx)
    
    def count_violations(colours):
        """Return number of K4s with fewer than 5 distinct colors using NumPy."""
        c_np = np.asarray(colours, dtype=np.int32)
        k4_colors = c_np[k4_edges_np]  # shape (1820, 6)
        sorted_colors = np.sort(k4_colors, axis=1)
        unique_mask = np.ones((num_k4s, 6), dtype=bool)
        unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
        counts = np.sum(unique_mask, axis=1)
        return int(np.sum(counts < 5))
    
    def get_violated_k4_indices(colours):
        """Return indices of violated K4s."""
        c_np = np.asarray(colours, dtype=np.int32)
        k4_colors = c_np[k4_edges_np]
        sorted_colors = np.sort(k4_colors, axis=1)
        unique_mask = np.ones((num_k4s, 6), dtype=bool)
        unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
        counts = np.sum(unique_mask, axis=1)
        return np.where(counts < 5)[0]
    
    def is_valid(colours):
        return count_violations(colours) == 0
    
    # Start with seed (i+j) % 17
    current = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    if not is_valid(current):
        # Fallback: try random
        for _ in range(50):
            if time.time() > deadline:
                break
            rand = [random.randint(0, 15) for _ in range(120)]
            if is_valid(rand):
                current = rand
                break
    
    best_colours = list(current)
    best_num_colors = len(set(current))
    
    # Try to find valid colorings with fewer colors
    # Strategy: for each target K from 13 to best_num_colors-1, try random init + local search
    for K in range(13, best_num_colors):
        if time.time() > deadline:
            break
        
        # Multiple restarts for this K
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Initialize: try structured random
            # Mix of random and semi-structured
            if restart % 3 == 0:
                # Semi-structured: use (i+j) % K with some noise
                colours = [((i + j) % K + random.randint(0, 1)) % K 
                          for i in range(n) for j in range(i + 1, n)]
            elif restart % 3 == 1:
                # Random
                colours = [random.randint(0, K - 1) for _ in range(120)]
            else:
                # Another structured: (i*3 + j*5) % K
                colours = [((i * 3 + j * 5) % K) 
                          for i in range(n) for j in range(i + 1, n)]
            
            violations = count_violations(colours)
            
            if violations == 0:
                best_colours = list(colours)
                best_num_colors = K
                break  # Found valid with K colors, try lower K
            
            # Local search with simulated annealing
            temperature = 1.0
            cooling = 0.995
            min_temp = 0.01
            max_steps = 3000
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                if violations == 0:
                    best_colours = list(colours)
                    best_num_colors = K
                    break
                
                # Pick an edge to change
                # Bias towards edges in violated K4s
                if violations > 0 and random.random() < 0.7:
                    # Pick a random violated K4, then a random edge in it
                    violated_idx = get_violated_k4_indices(colours)
                    if len(violated_idx) > 0:
                        k4_idx = violated_idx[random.randint(0, len(violated_idx) - 1)]
                        k4 = k4s[k4_idx]
                        eidx = k4[random.randint(0, 5)]
                    else:
                        eidx = random.randint(0, 119)
                else:
                    eidx = random.randint(0, 119)
                
                old_color = colours[eidx]
                new_color = random.randint(0, K - 1)
                
                if new_color == old_color:
                    continue
                
                colours[eidx] = new_color
                new_violations = count_violations(colours)
                
                # Accept or reject based on simulated annealing
                delta = new_violations - violations
                if delta <= 0:
                    violations = new_violations
                else:
                    # Accept with probability exp(-delta / temperature)
                    if random.random() < np.exp(-delta / temperature):
                        violations = new_violations
                    else:
                        colours[eidx] = old_color
                
                temperature *= cooling
                if temperature < min_temp:
                    temperature = min_temp
            
            # If we found a valid coloring, break
            if violations == 0:
                best_colours = list(colours)
                best_num_colors = K
                break
    
    # Final greedy merge pass on the best solution found
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            current_set = set(best_colours)
            if len(current_set) <= 13:
                break
            
            sorted_colors = sorted(current_set)
            merged = False
            
            for i in range(len(sorted_colors)):
                for j in range(i + 1, len(sorted_colors)):
                    if time.time() > deadline - 5:
                        break
                    
                    c1 = sorted_colors[i]
                    c2 = sorted_colors[j]
                    
                    new_colours = [c2 if c == c1 else c for c in best_colours]
                    
                    if is_valid(new_colours):
                        best_colours = new_colours
                        merged = True
                        break
                if merged:
                    break
            
            if not merged:
                break
    
    # Ensure validity
    if not is_valid(best_colours):
        # Fallback to seed
        seed = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        if is_valid(seed):
            best_colours = seed
    
    return best_colours
