"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import comb

def construct():
    """Construct a valid 2-coloring of triples maximizing n.
    
    Returns dict with 'n' and 'colors' (list of 0/1 for each triple in lex order).
    """
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    def greedy_violation_elimination(n, deadline, max_iter=50000):
        """
        Greedy violation-elimination: start from modular heuristic, 
        iteratively flip triples to eliminate violations.
        """
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple tuple to index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute quads and quints
        quads = list(combinations(range(n), 4))
        quints = list(combinations(range(n), 5))
        
        # For each triple, find which quads and quints it belongs to
        triple_to_quads = [[] for _ in range(num_triples)]
        triple_to_quints = [[] for _ in range(num_triples)]
        
        # For each quad, find its triples
        quad_triples = [[] for _ in range(len(quads))]
        # For each quint, find its triples
        quint_triples = [[] for _ in range(len(quints))]
        
        # Build mappings
        for qi, quad in enumerate(quads):
            a, b, c, d = quad
            t1 = (a, b, c); t2 = (a, b, d); t3 = (a, c, d); t4 = (b, c, d)
            idxs = [triple_to_idx[t1], triple_to_idx[t2], triple_to_idx[t3], triple_to_idx[t4]]
            quad_triples[qi] = idxs
            for idx in idxs:
                triple_to_quads[idx].append(qi)
                
        for qi, quint in enumerate(quints):
            trs = list(combinations(quint, 3))
            idxs = [triple_to_idx[t] for t in trs]
            quint_triples[qi] = idxs
            for idx in idxs:
                triple_to_quints[idx].append(qi)

        # Start with modular heuristic: color based on (a+b+c) mod 3
        colors = np.zeros(num_triples, dtype=int)
        for i, (a, b, c) in enumerate(triples):
            s = (a + b + c) % 3
            colors[i] = 0 if s == 0 else 1  # Red if sum % 3 == 0, else Blue
            
        # Initialize counts
        quad_red = np.zeros(len(quads), dtype=int)
        quint_blue = np.zeros(len(quints), dtype=int)
        
        for t_idx in range(num_triples):
            if colors[t_idx] == 0:
                for qi in triple_to_quads[t_idx]:
                    quad_red[qi] += 1
            else:
                for qi in triple_to_quints[t_idx]:
                    quint_blue[qi] += 1
        
        red_violations = set(np.where(quad_red == 4)[0].tolist())
        blue_violations = set(np.where(quint_blue == 10)[0].tolist())
        
        if not red_violations and not blue_violations:
            return colors.tolist(), True
        
        # Greedy elimination loop
        stagnant = 0
        for _ in range(max_iter):
            if time.time() > deadline - 1:
                break
                
            if not red_violations and not blue_violations:
                return colors.tolist(), True
            
            # Pick a violation (prioritize red as they're smaller)
            if red_violations:
                v_idx = random.choice(list(red_violations))
                is_red = True
            else:
                v_idx = random.choice(list(blue_violations))
                is_red = False
            
            # Get triples in this violation
            if is_red:
                trs = quad_triples[v_idx]
                candidates = [t for t in trs if colors[t] == 0]
            else:
                trs = quint_triples[v_idx]
                candidates = [t for t in trs if colors[t] == 1]
            
            if not candidates:
                stagnant += 1
                if stagnant > 200:
                    break
                continue
            
            # Find flip with best delta
            best_delta = 0
            best_candidate = None
            
            for t in candidates:
                delta = 0
                if colors[t] == 0:  # Red -> Blue
                    for qi in triple_to_quads[t]:
                        if quad_red[qi] == 4:
                            delta -= 1
                    for qi in triple_to_quints[t]:
                        if quint_blue[qi] == 9:
                            delta += 1
                else:  # Blue -> Red
                    for qi in triple_to_quads[t]:
                        if quad_red[qi] == 3:
                            delta += 1
                    for qi in triple_to_quints[t]:
                        if quint_blue[qi] == 10:
                            delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_candidate = t
            
            if best_candidate is not None:
                t = best_candidate
                old_color = colors[t]
                colors[t] = 1 - old_color
                
                if old_color == 0:
                    for qi in triple_to_quads[t]:
                        quad_red[qi] -= 1
                        if quad_red[qi] == 3 and qi in red_violations:
                            red_violations.remove(qi)
                    for qi in triple_to_quints[t]:
                        quint_blue[qi] += 1
                        if quint_blue[qi] == 10 and qi not in blue_violations:
                            blue_violations.add(qi)
                else:
                    for qi in triple_to_quads[t]:
                        quad_red[qi] += 1
                        if quad_red[qi] == 4 and qi not in red_violations:
                            red_violations.add(qi)
                    for qi in triple_to_quints[t]:
                        quint_blue[qi] -= 1
                        if quint_blue[qi] == 9 and qi in blue_violations:
                            blue_violations.remove(qi)
                
                stagnant = 0
            else:
                stagnant += 1
                if stagnant > 200:
                    break
        
        current_violations = len(red_violations) + len(blue_violations)
        return colors.tolist(), (current_violations == 0)
    
    # Fallback: small valid coloring
    def make_fallback(n):
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return n, colors
    
    # Try n=34 first with greedy approach
    best_n, best_colors = make_fallback(9)
    
    for n in [34, 33, 32, 30]:
        if time.time() > deadline - 5:
            break
        
        colors, valid = greedy_violation_elimination(n, deadline)
        if valid:
            best_n = n
            best_colors = colors
            break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
