import time
import random
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
        
        edge_to_quads = [[] for _ in range(m)]
        for q_idx, q in enumerate(quads):
            q_edges = list(combinations(q, 3))
            for e in q_edges:
                e_idx = edge_to_idx[e]
                edge_to_quads[e_idx].append(q_idx)
                
        edge_to_pents = [[] for _ in range(m)]
        for p_idx, p in enumerate(pents):
            p_edges = list(combinations(p, 3))
            for e in p_edges:
                e_idx = edge_to_idx[e]
                edge_to_pents[e_idx].append(p_idx)

        # Heuristic initialization: color based on sum of vertices mod 3
        # This gives a structured starting point with fewer violations
        def heuristic_init(n):
            colors = []
            for a, b, c in combinations(range(n), 3):
                s = (a + b + c) % 3
                # Map to 0 or 1: 0->red, 1->blue, 2->red (or some pattern)
                # Try: 0->0, 1->1, 2->0
                colors.append(0 if s != 1 else 1)
            return colors
        
        best_violations = float('inf')
        best_colors = None
        
        # Simulated annealing / local search with multiple strategies
        # Phase 1: Start with heuristic, do greedy local search
        # Phase 2: Random restarts with simulated annealing
        
        search_deadline = min(deadline - 5.0, time.time() + 100.0)
        
        # Try heuristic init first
        colors = heuristic_init(n)
        
        # Calculate initial violations
        quad_red_counts = [0] * len(quads)
        pent_blue_counts = [0] * len(pents)
        
        for i, c in enumerate(colors):
            if c == 0:  # Red
                for q_idx in edge_to_quads[i]:
                    quad_red_counts[q_idx] += 1
            else:  # Blue
                for p_idx in edge_to_pents[i]:
                    pent_blue_counts[p_idx] += 1
                    
        violations = sum(1 for c in quad_red_counts if c == 4) + sum(1 for c in pent_blue_counts if c == 10)
        
        if violations == 0:
            return {'n': n, 'colors': colors}
            
        if violations < best_violations:
            best_violations = violations
            best_colors = colors[:]
            
        # Greedy local search from heuristic start
        local_deadline = min(search_deadline, time.time() + 10.0)
        while time.time() < local_deadline:
            i = random.randint(0, m - 1)
            old_c = colors[i]
            new_c = 1 - old_c
            
            delta_v = 0
            
            if old_c == 0:  # Red -> Blue
                for q_idx in edge_to_quads[i]:
                    if quad_red_counts[q_idx] == 4:
                        delta_v -= 1
                for p_idx in edge_to_pents[i]:
                    if pent_blue_counts[p_idx] == 9:
                        delta_v += 1
            else:  # Blue -> Red
                for q_idx in edge_to_quads[i]:
                    if quad_red_counts[q_idx] == 3:
                        delta_v += 1
                for p_idx in edge_to_pents[i]:
                    if pent_blue_counts[p_idx] == 10:
                        delta_v -= 1
            
            if delta_v <= 0:
                colors[i] = new_c
                if new_c == 0:
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] += 1
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] -= 1
                else:
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
        
        # Phase 2: Random restarts with simulated annealing
        # Temperature starts high, cools down
        while time.time() < search_deadline:
            # Random restart
            colors = [random.randint(0, 1) for _ in range(m)]
            
            quad_red_counts = [0] * len(quads)
            pent_blue_counts = [0] * len(pents)
            
            for i, c in enumerate(colors):
                if c == 0:
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] += 1
                else:
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] += 1
                        
            violations = sum(1 for c in quad_red_counts if c == 4) + sum(1 for c in pent_blue_counts if c == 10)
            
            if violations == 0:
                return {'n': n, 'colors': colors}
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
                
            # Simulated annealing with cooling
            temperature = 5.0
            cooling_rate = 0.95
            local_deadline = min(search_deadline, time.time() + 5.0)
            
            while time.time() < local_deadline:
                i = random.randint(0, m - 1)
                old_c = colors[i]
                new_c = 1 - old_c
                
                delta_v = 0
                
                if old_c == 0:
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 4:
                            delta_v -= 1
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 9:
                            delta_v += 1
                else:
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 3:
                            delta_v += 1
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 10:
                            delta_v -= 1
                
                # Accept if improving or with probability exp(-delta/temperature)
                if delta_v <= 0:
                    accept = True
                else:
                    # Use min(1, exp(-delta_v / temperature)) to avoid overflow
                    if temperature < 0.01:
                        accept = False
                    else:
                        accept_prob = min(1.0, 2.718281828 ** (-delta_v / temperature))
                        accept = random.random() < accept_prob
                
                if accept:
                    colors[i] = new_c
                    if new_c == 0:
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] += 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] -= 1
                    else:
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
                
                # Cool down
                temperature *= cooling_rate
        
        # If we didn't find a solution for this n, try smaller n
        
    # If no n >= 2 found, return n=1 (trivially valid)
    return {'n': 1, 'colors': []}
