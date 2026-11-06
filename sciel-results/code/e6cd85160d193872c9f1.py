import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 150s kill

    def search_for_n(n, time_limit, seed=0):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}

        # Precompute incidence
        quad_triples = []
        for quad in combinations(range(n), 4):
            idxs = [tri_to_idx[tuple(x for j, x in enumerate(quad) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
        quint_triples = []
        for quint in combinations(range(n), 5):
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)

        n_quads = len(quad_triples)
        n_quints = len(quint_triples)

        # Inverse incidence: triple -> list of quads/quints it belongs to
        triple_to_quads = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                triple_to_quads[t].append(qidx)
        
        triple_to_quints = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                triple_to_quints[t].append(qidx)

        # Convert to numpy arrays for faster iteration if needed, or keep lists
        # Lists are fine for small n, but n=34 -> M=5984, n_quads=46376, n_quints=237336
        # Iterating over lists of indices is fast in Python if we don't do heavy work inside.
        
        rng = np.random.RandomState(seed)

        # Initialize colors using spectral method
        # Build the line graph adjacency matrix
        # Two triples are adjacent if they share exactly 2 vertices
        # This is expensive for large n, so we'll use a simpler geometric initialization
        # based on vertex positions in a 3D space
        
        # Place vertices on a 3D sphere
        # Use Fibonacci sphere for even distribution
        def fibonacci_sphere(n_points):
            points = []
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(n_points):
                y = 1 - (i / (n_points - 1)) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y
                theta = golden_angle * i
                x = np.cos(theta) * radius
                z = np.sin(theta) * radius
                points.append((x, y, z))
            return np.array(points)
        
        if n <= 1:
            return [0] * M
        
        # Use geometric initialization: color based on dot product of vertex positions
        # with a random direction
        positions = fibonacci_sphere(n)
        
        # Try multiple random directions
        best_colors = None
        best_violations = float('inf')
        
        for attempt in range(10):
            # Random direction in 3D
            direction = rng.randn(3)
            direction /= np.linalg.norm(direction)
            
            # Project vertices onto direction
            projections = positions @ direction
            
            # Color triples based on whether the sum of projections is positive or negative
            colors = np.zeros(M, dtype=np.int8)
            for i, (a, b, c) in enumerate(triples):
                s = projections[a] + projections[b] + projections[c]
                colors[i] = 0 if s >= 0 else 1  # 0 = red, 1 = blue
            
            # Count violations
            red_counts = np.zeros(n_quads, dtype=np.int32)
            blue_counts = np.zeros(n_quints, dtype=np.int32)
            
            for t in range(M):
                if colors[t] == 0:  # Red
                    for q in triple_to_quads[t]:
                        red_counts[q] += 1
                else:  # Blue
                    for k in triple_to_quints[t]:
                        blue_counts[k] += 1
            
            total_violations = np.sum(red_counts == 4) + np.sum(blue_counts == 10)
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                best_red_counts = red_counts.copy()
                best_blue_counts = blue_counts.copy()
        
        # Use the best initialization
        colors = best_colors
        red_counts = best_red_counts
        blue_counts = best_blue_counts
        total_violations = best_violations
        
        # If violations are still high, try modular initialization as backup
        if total_violations > M * 0.1:
            colors_mod = np.zeros(M, dtype=np.int8)
            for i, (a, b, c) in enumerate(triples):
                colors_mod[i] = 0 if (a + b + c) % 3 == 0 else 1
            
            red_counts_mod = np.zeros(n_quads, dtype=np.int32)
            blue_counts_mod = np.zeros(n_quints, dtype=np.int32)
            
            for t in range(M):
                if colors_mod[t] == 0:  # Red
                    for q in triple_to_quads[t]:
                        red_counts_mod[q] += 1
                else:  # Blue
                    for k in triple_to_quints[t]:
                        blue_counts_mod[k] += 1
            
            total_violations_mod = np.sum(red_counts_mod == 4) + np.sum(blue_counts_mod == 10)
            
            if total_violations_mod < total_violations:
                colors = colors_mod
                red_counts = red_counts_mod
                blue_counts = blue_counts_mod
                total_violations = total_violations_mod
        
        def flip(t):
            nonlocal total_violations
            old_color = colors[t]
            
            if old_color == 0: # Red -> Blue
                # Update Red counts (decrease)
                for q in triple_to_quads[t]:
                    old_c = red_counts[q]
                    if old_c == 4:
                        total_violations -= 1
                    red_counts[q] = old_c - 1
                
                # Update Blue counts (increase)
                for k in triple_to_quints[t]:
                    old_c = blue_counts[k]
                    blue_counts[k] = old_c + 1
                    if old_c + 1 == 10:
                        total_violations += 1
                
                colors[t] = 1

            else: # Blue -> Red
                # Update Blue counts (decrease)
                for k in triple_to_quints[t]:
                    old_c = blue_counts[k]
                    if old_c == 10:
                        total_violations -= 1
                    blue_counts[k] = old_c - 1
                
                # Update Red counts (increase)
                for q in triple_to_quads[t]:
                    old_c = red_counts[q]
                    red_counts[q] = old_c + 1
                    if old_c + 1 == 4:
                        total_violations += 1
                
                colors[t] = 0

        # Local Search with multiple phases
        # Phase 1: Greedy violation elimination
        max_iterations = 5000
        for it in range(max_iterations):
            if time.time() > start_time + time_limit:
                break
            if total_violations == 0:
                return colors.tolist()
            
            # Find a violating quad or quint and pick a random triple from it to flip
            # Try all triples in violating structures
            best_delta = 0
            best_t = -1
            
            # Sample candidates from violating structures
            candidates = set()
            
            # Add triples from violating quads
            for q in range(n_quads):
                if red_counts[q] == 4:
                    for t in quad_triples[q]:
                        candidates.add(t)
            
            # Add triples from violating quints
            for k in range(n_quints):
                if blue_counts[k] == 10:
                    for t in quint_triples[k]:
                        candidates.add(t)
            
            # If no violations, add random triples
            if not candidates:
                for _ in range(100):
                    candidates.add(rng.randint(M))
            
            # Evaluate candidates
            for t in candidates:
                old_color = colors[t]
                delta = 0
                if old_color == 0:
                    # Red -> Blue
                    for q in triple_to_quads[t]:
                        if red_counts[q] == 4: delta -= 1
                    for k in triple_to_quints[t]:
                        if blue_counts[k] == 9: delta += 1
                else:
                    # Blue -> Red
                    for k in triple_to_quints[t]:
                        if blue_counts[k] == 10: delta -= 1
                    for q in triple_to_quads[t]:
                        if red_counts[q] == 3: delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_t = t
            
            if best_t != -1:
                flip(best_t)
            else:
                # No improving move, do a random flip
                t = rng.randint(M)
                flip(t)
        
        # Phase 2: Simulated Annealing
        if time.time() < start_time + time_limit and total_violations > 0:
            T = 3.0
            cooling = 0.9995
            min_T = 0.05
            while time.time() < start_time + time_limit:
                if total_violations == 0:
                    return colors.tolist()
                
                t = rng.randint(M)
                
                # Calculate delta
                old_color = colors[t]
                delta = 0
                if old_color == 0:
                    for q in triple_to_quads[t]:
                        if red_counts[q] == 4: delta -= 1
                    for k in triple_to_quints[t]:
                        if blue_counts[k] == 9: delta += 1
                else:
                    for k in triple_to_quints[t]:
                        if blue_counts[k] == 10: delta -= 1
                    for q in triple_to_quads[t]:
                        if red_counts[q] == 3: delta += 1
                
                flip(t)
                
                if delta <= 0 or (T > min_T and rng.random() < np.exp(-delta / T)):
                    pass # Accept
                else:
                    flip(t) # Reject (flip back)
                
                T *= cooling
        
        return None

    best_n = 1
    best_colors = [0]

    # Try n values in order, allocating more time to larger n
    # Focus on finding the maximum n that can be solved
    
    # First, quickly verify small n work
    for n in range(1, 11):
        if time.time() > deadline - 30:
            break
        t_limit = min(2.0, deadline - 30 - (time.time() - start_time))
        res = search_for_n(n, t_limit, seed=n)
        if res is not None:
            best_n = n
            best_colors = res
        else:
            # Should not happen for small n
            break
    
    # Now try larger n with more time
    # Use a strategy: try n=34 first with most time, then work down if needed
    large_ns = list(range(34, 10, -1))
    
    for n in large_ns:
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - 15 - (time.time() - start_time)
        if remaining < 10:
            break
        
        # Allocate time based on how much is left
        t_limit = min(remaining - 5, 40.0)
        
        # Try multiple seeds for larger n
        seeds_to_try = 1
        if remaining > 30:
            seeds_to_try = 2
        if remaining > 50:
            seeds_to_try = 3
        
        res = None
        for s in range(seeds_to_try):
            if time.time() > deadline - 10:
                break
            t_per_seed = t_limit / seeds_to_try
            res = search_for_n(n, t_per_seed, seed=n * 100 + s)
            if res is not None:
                break
        
        if res is not None:
            best_n = n
            best_colors = res
            # Found a valid coloring, try larger n
            break
    
    # If we didn't find n=34, try intermediate values
    if best_n < 34:
        mid_ns = list(range(25, 10, -1))
        for n in mid_ns:
            if n <= best_n:
                break
            if time.time() > deadline - 10:
                break
            
            remaining = deadline - 10 - (time.time() - start_time)
            if remaining < 5:
                break
            
            t_limit = min(remaining - 3, 20.0)
            res = search_for_n(n, t_limit, seed=n * 50)
            if res is not None:
                best_n = n
                best_colors = res
    
    return {'n': best_n, 'colors': best_colors}
