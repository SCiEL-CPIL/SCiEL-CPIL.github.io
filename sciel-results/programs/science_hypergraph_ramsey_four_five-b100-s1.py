import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Try n from 34 downwards
    for n in range(34, 1, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Precomputation for this n
        edges = list(combinations(range(n), 3))
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        quads = list(combinations(range(n), 4))
        pents = list(combinations(range(n), 5))
        
        # For each edge, which quads contain it?
        edge_to_quads = [[] for _ in range(m)]
        for q_idx, q in enumerate(quads):
            q_edges = list(combinations(q, 3))
            for e in q_edges:
                e_idx = edge_to_idx[e]
                edge_to_quads[e_idx].append(q_idx)
                
        # For each edge, which pents contain it?
        edge_to_pents = [[] for _ in range(m)]
        for p_idx, p in enumerate(pents):
            p_edges = list(combinations(p, 3))
            for e in p_edges:
                e_idx = edge_to_idx[e]
                edge_to_pents[e_idx].append(p_idx)

        best_violations = float('inf')
        best_colors = None
        
        # Run local search for a few seconds
        search_deadline = min(deadline, time.time() + 10.0)
        
        # We will try multiple restarts. 
        # To incorporate "spectral" ideas without heavy computation that might timeout,
        # we use a structured initialization based on vertex positions.
        # For a complete 3-uniform hypergraph, a "spectral" or geometric initialization
        # often involves distributing vertices on a sphere or using a low-discrepancy sequence.
        # However, the color is per-edge, not per-vertex.
        # A good heuristic for edge coloring to avoid monochromatic cliques is to
        # make the coloring depend on the "distance" or sum of vertices in a way that
        # breaks symmetry. 
        # Let's try a few different initialization strategies:
        # 1. Random
        # 2. Sum mod 2
        # 3. Sum mod 3 (0->red, 1->blue, 2->random)
        # 4. Geometric: project vertices to 1D (just index), color based on (a+b+c) % k
        
        # Let's stick to random restarts with a slightly smarter local search that
        # prioritizes fixing the most violated constraints.
        
        num_restarts = 0
        while time.time() < search_deadline:
            num_restarts += 1
            
            # Initialization
            # Try a mix of random and structured
            if num_restarts % 3 == 0:
                # Structured: color based on (a+b+c) % 2
                colors = [0] * m
                for i, (a, b, c) in enumerate(edges):
                    colors[i] = (a + b + c) % 2
            elif num_restarts % 3 == 1:
                # Structured: color based on (a*b + b*c + c*a) % 2
                colors = [0] * m
                for i, (a, b, c) in enumerate(edges):
                    colors[i] = (a * b + b * c + c * a) % 2
            else:
                # Random
                colors = [random.randint(0, 1) for _ in range(m)]
            
            # Calculate initial violations
            quad_red_counts = [0] * len(quads)
            pent_blue_counts = [0] * len(pents)
            
            for i, c in enumerate(colors):
                if c == 0: # Red
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] += 1
                else: # Blue
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] += 1
                        
            violations = sum(1 for c in quad_red_counts if c == 4) + sum(1 for c in pent_blue_counts if c == 10)
            
            if violations == 0:
                return {'n': n, 'colors': colors}
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
                
            # Local search loop with simulated annealing / greedy
            # We use a simple hill climbing with random restarts within the local search
            # to escape local minima.
            
            # Temperature for simulated annealing
            temp = 1.0
            cooling = 0.995
            
            local_deadline = min(search_deadline, time.time() + 2.0)
            stuck_count = 0
            
            while time.time() < local_deadline:
                # Pick a random edge to flip
                i = random.randint(0, m - 1)
                old_c = colors[i]
                new_c = 1 - old_c
                
                # Calculate delta violations
                delta_v = 0
                
                if old_c == 0: # Red -> Blue
                    # Quads: red count decreases
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 4:
                            delta_v -= 1 # Violation removed
                    # Pents: blue count increases
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 9:
                            delta_v += 1 # Violation added
                else: # Blue -> Red
                    # Quads: red count increases
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 3:
                            delta_v += 1 # Violation added
                    # Pents: blue count decreases
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 10:
                            delta_v -= 1 # Violation removed
                
                # Acceptance criterion
                # If delta_v < 0, always accept
                # If delta_v == 0, accept with some probability
                # If delta_v > 0, accept with probability exp(-delta_v / temp)
                
                accept = False
                if delta_v < 0:
                    accept = True
                elif delta_v == 0:
                    accept = random.random() < 0.5
                else:
                    # Simulated annealing step
                    if temp > 0.01:
                        prob = np.exp(-delta_v / temp)
                        accept = random.random() < prob
                    else:
                        accept = False
                
                if accept:
                    # Apply move
                    colors[i] = new_c
                    if new_c == 0: # Now Red
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] += 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] -= 1
                    else: # Now Blue
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] -= 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] += 1
                            
                    violations += delta_v
                    
                    if violations == 0:
                        return {'n': n, 'colors': colors}
                        
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors[:]
                        stuck_count = 0
                    else:
                        stuck_count += 1
                        # If stuck for too long, increase temperature slightly or restart local search
                        if stuck_count > 1000:
                            temp = max(temp * 1.1, 0.1)
                            stuck_count = 0
                            
                    temp *= cooling
                else:
                    # If rejected, do nothing
                    pass

        # If we didn't find a solution for this n in the allotted time, try smaller n
        # We don't return best_colors because it's not valid.
        
    # If no n >= 2 found, return n=1 (trivially valid)
    return {'n': 1, 'colors': []}
