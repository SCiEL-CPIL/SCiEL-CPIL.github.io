"""Rewritten with bit-parallel local search and multi-start for R(4,5;3)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Try to find valid coloring for n=34, then fall back to smaller n
    for n_target in [34, 33, 32, 31, 30, 29, 28, 27, 26, 25]:
        if time.time() > deadline - 10:
            break
        result = _solve_n(n_target, deadline - time.time() - 5)
        if result is not None:
            return {'n': n_target, 'colors': result}
    
    # Final fallback: return a small valid coloring
    n = 9
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}

def _solve_n(n, time_limit):
    """Solve for given n using bit-parallel local search with multiple restarts."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Precompute all triples in lexicographic order
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute quads and their constituent triples
    quads = list(combinations(range(n), 4))
    num_quads = len(quads)
    quad_triples = []
    for q in quads:
        a, b, c, d = q
        quad_triples.append([triple_idx[(a,b,c)], triple_idx[(a,b,d)], 
                            triple_idx[(a,c,d)], triple_idx[(b,c,d)]])
    quad_triples = np.array(quad_triples, dtype=np.int32)
    
    # Precompute quints and their constituent triples
    quints = list(combinations(range(n), 5))
    num_quints = len(quints)
    quint_triples = []
    for q in quints:
        tri_list = [triple_idx[tri] for tri in combinations(q, 3)]
        quint_triples.append(tri_list)
    quint_triples = np.array(quint_triples, dtype=np.int32)
    
    # For efficient flip evaluation: for each triple, which quads and quints contain it
    tri_quad_membership = [[] for _ in range(num_triples)]
    for qi in range(num_quads):
        for ti in quad_triples[qi]:
            tri_quad_membership[ti].append(qi)
    
    tri_quint_membership = [[] for _ in range(num_triples)]
    for qi in range(num_quints):
        for ti in quint_triples[qi]:
            tri_quint_membership[ti].append(qi)
    
    # Convert to numpy arrays for faster access
    tri_quad_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) 
                    for lst in tri_quad_membership]
    tri_quint_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) 
                     for lst in tri_quint_membership]
    
    best_result = None
    best_score = float('inf')
    
    # Multiple restarts with different strategies
    max_restarts = 8
    rng_master = random.Random(12345)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Random with slight blue bias
            seed_val = 42
            colors = np.random.default_rng(seed_val).integers(0, 2, size=num_triples, dtype=np.int8)
        elif restart == 1:
            # Modular: color based on sum mod 3
            colors = np.array([0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples], dtype=np.int8)
        elif restart == 2:
            # Modular: color based on (a+b) mod 2
            colors = np.array([(a+b) % 2 for a,b,c in triples], dtype=np.int8)
        elif restart == 3:
            # Random with red bias
            seed_val = 7
            colors = np.random.default_rng(seed_val).integers(0, 2, size=num_triples, dtype=np.int8)
            # Bias: make 40% red
            mask = np.random.default_rng(seed_val+1).random(num_triples) < 0.4
            colors[mask] = 0
        elif restart == 4:
            # Product-based
            colors = np.array([(a*b*c) % 2 for a,b,c in triples], dtype=np.int8)
        elif restart == 5:
            # Difference-based
            colors = np.array([(a-b+c) % 2 for a,b,c in triples], dtype=np.int8)
        else:
            # Random
            seed_val = restart * 1000 + 7
            colors = np.random.default_rng(seed_val).integers(0, 2, size=num_triples, dtype=np.int8)
        
        # Run local search
        result = _local_search_bitparallel(
            colors.copy(), n, num_triples, num_quads, num_quints,
            quad_triples, quint_triples, tri_quad_arr, tri_quint_arr,
            deadline - time.time() - 1, rng_master
        )
        
        if result is not None:
            return result
        
        # Track best partial result
        if result is None:
            # Count violations to track progress
            violations = _count_violations_fast(colors, quad_triples, quint_triples)
            if violations < best_score:
                best_score = violations
                best_result = colors.copy()
    
    return best_result if best_score == 0 else None

def _count_violations_fast(colors, quad_triples, quint_triples):
    """Count total violations (red K4 + blue K5) using vectorized operations."""
    # Red K4: quad where all 4 triples are red (color 0)
    quad_colors = colors[quad_triples]  # shape (num_quads, 4)
    red_viol = np.sum(np.all(quad_colors == 0, axis=1))
    
    # Blue K5: quint where all 10 triples are blue (color 1)
    quint_colors = colors[quint_triples]  # shape (num_quints, 10)
    blue_viol = np.sum(np.all(quint_colors == 1, axis=1))
    
    return int(red_viol + blue_viol)

def _local_search_bitparallel(colors, n, num_triples, num_quads, num_quints,
                               quad_triples, quint_triples, tri_quad_arr, tri_quint_arr,
                               time_limit, rng_master):
    """Local search with efficient delta computation."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initialize violation counts
    quad_red = np.zeros(num_quads, dtype=np.int8)
    for qi in range(num_quads):
        quad_red[qi] = np.sum(colors[quad_triples[qi]] == 0)
    
    quint_blue = np.zeros(num_quints, dtype=np.int8)
    for qi in range(num_quints):
        quint_blue[qi] = np.sum(colors[quint_triples[qi]] == 1)
    
    current_viol = int(np.sum(quad_red == 4) + np.sum(quint_blue == 10))
    
    if current_viol == 0:
        return list(colors)
    
    best_viol = current_viol
    best_colors = colors.copy()
    
    rng = random.Random(rng_master.randrange(1000000))
    
    # Local search with simulated annealing
    temperature = 0.1
    cooling = 0.99999
    
    steps = 0
    max_steps = 200000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Pick random triple
        ti = rng.randrange(num_triples)
        
        old_color = colors[ti]
        new_color = 1 - old_color
        
        # Compute delta in violations
        delta_viol = 0
        
        # Red K4 impact
        if new_color == 0:  # becoming red
            for qi in tri_quad_arr[ti]:
                if quad_red[qi] == 3:
                    delta_viol += 1
        else:  # becoming blue (was red)
            for qi in tri_quad_arr[ti]:
                if quad_red[qi] == 4:
                    delta_viol -= 1
        
        # Blue K5 impact
        if new_color == 1:  # becoming blue
            for qi in tri_quint_arr[ti]:
                if quint_blue[qi] == 9:
                    delta_viol += 1
        else:  # becoming red (was blue)
            for qi in tri_quint_arr[ti]:
                if quint_blue[qi] == 10:
                    delta_viol -= 1
        
        # Acceptance criterion
        if delta_viol < 0:
            accept = True
        elif delta_viol == 0:
            accept = rng.random() < 0.15
        else:
            # Simulated annealing: accept worse moves with decreasing probability
            accept = rng.random() < np.exp(-delta_viol / max(temperature, 0.01))
        
        if accept:
            # Apply flip
            colors[ti] = new_color
            for qi in tri_quad_arr[ti]:
                if new_color == 0:
                    quad_red[qi] += 1
                else:
                    quad_red[qi] -= 1
            for qi in tri_quint_arr[ti]:
                if new_color == 1:
                    quint_blue[qi] += 1
                else:
                    quint_blue[qi] -= 1
            
            current_viol += delta_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_colors = colors.copy()
                if best_viol == 0:
                    return list(best_colors)
        
        # Cool down
        temperature *= cooling
        
        # Periodic check
        if steps % 50000 == 0 and best_viol == 0:
            return list(best_colors)
    
    if best_viol == 0:
        return list(best_colors)
    return None
# EVOLVE-BLOCK-END
