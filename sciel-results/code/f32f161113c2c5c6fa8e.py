"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best_signs = None
    best_radius = float('inf')
    
    # Pre-compute edge types for structured generation
    # Edges are: top cycle (m), bottom cycle (m), matching (m)
    top_edges = list(range(m))
    bottom_edges = list(range(m, 2*m))
    match_edges = list(range(2*m, 3*m))
    
    def evaluate(signs):
        nonlocal best_signs, best_radius
        try:
            radius, details = check_lift(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
            return radius
        except Exception:
            return None
    
    def make_structured_signs(pattern_top, pattern_bottom, pattern_match):
        signs = np.ones(num_edges, dtype=int)
        for i in range(m):
            signs[top_edges[i]] = pattern_top[i % len(pattern_top)]
        for i in range(m):
            signs[bottom_edges[i]] = pattern_bottom[i % len(pattern_bottom)]
        for i in range(m):
            signs[match_edges[i]] = pattern_match[i % len(pattern_match)]
        return signs
    
    def local_search(current, max_iters=500):
        nonlocal best_signs, best_radius
        cur = current.copy()
        cur_radius = evaluate(cur)
        if cur_radius is None:
            return
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Try single flips
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 2:
                    break
                trial = cur.copy()
                trial[i] = -trial[i]
                trial_radius = evaluate(trial)
                if trial_radius is not None and trial_radius < cur_radius - 1e-12:
                    cur = trial
                    cur_radius = trial_radius
                    improved = True
                    break
            
            # Try double flips
            if not improved:
                for i in range(num_edges):
                    if time.time() > deadline - 2:
                        break
                    for j in range(i+1, num_edges):
                        trial = cur.copy()
                        trial[i] = -trial[i]
                        trial[j] = -trial[j]
                        trial_radius = evaluate(trial)
                        if trial_radius is not None and trial_radius < cur_radius - 1e-12:
                            cur = trial
                            cur_radius = trial_radius
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        # Update global best
        if cur_radius < best_radius:
            best_radius = cur_radius
            best_signs = cur.copy()
    
    def local_search_3flip(current, max_iters=100):
        nonlocal best_signs, best_radius
        cur = current.copy()
        cur_radius = evaluate(cur)
        if cur_radius is None:
            return
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 2:
                    break
                for j in range(i+1, num_edges):
                    if time.time() > deadline - 2:
                        break
                    for k in range(j+1, num_edges):
                        trial = cur.copy()
                        trial[i] = -trial[i]
                        trial[j] = -trial[j]
                        trial[k] = -trial[k]
                        trial_radius = evaluate(trial)
                        if trial_radius is not None and trial_radius < cur_radius - 1e-12:
                            cur = trial
                            cur_radius = trial_radius
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if cur_radius < best_radius:
            best_radius = cur_radius
            best_signs = cur.copy()
    
    # Gauge-reduced exhaustive search
    # Key insight: The spectral radius of the signed adjacency matrix is invariant under gauge transformations.
    # A gauge transformation corresponds to flipping signs on all edges incident to a vertex.
    # We can fix a gauge by setting the first edge of the top cycle to +1, and fixing the first 
    # matching edge to +1. This reduces the search space from 2^(3m) to 2^(3m-2).
    # For m=4,5,7,9 this is 2^10, 2^13, 2^19, 2^25 which is feasible for small m.
    
    def gauge_fix(signs):
        """Apply gauge transformation to fix gauge: 
        - Flip all edges incident to vertex 0 if signs[top_edges[0]] == -1
        - Flip all edges incident to vertex m if signs[match_edges[0]] == -1
        """
        signs = signs.copy()
        # If first top edge is -1, flip all edges incident to vertex 0
        if signs[top_edges[0]] == -1:
            # Edges incident to vertex 0: top_edges[0], top_edges[m-1] (cycle wrap), match_edges[0]
            signs[top_edges[0]] *= -1
            signs[top_edges[m-1]] *= -1
            signs[match_edges[0]] *= -1
        
        # If first match edge is -1, flip all edges incident to vertex m
        if signs[match_edges[0]] == -1:
            # Edges incident to vertex m: match_edges[0], bottom_edges[0], bottom_edges[m-1]
            signs[match_edges[0]] *= -1
            signs[bottom_edges[0]] *= -1
            signs[bottom_edges[m-1]] *= -1
        
        return signs
    
    def exhaustive_search_gauge_reduced():
        """Exhaustive search over gauge-reduced space."""
        nonlocal best_signs, best_radius
        
        # We'll iterate over all 2^(3m-2) combinations
        # The free variables are:
        # - top_edges[1..m-1] (m-1 variables)
        # - bottom_edges[0..m-1] (m variables, but bottom_edges[0] and bottom_edges[m-1] are coupled)
        # - match_edges[1..m-1] (m-1 variables)
        
        # Actually, let's just iterate over all 2^(3m) and use gauge fixing to deduplicate
        # For m=4: 2^12 = 4096
        # For m=5: 2^15 = 32768
        # For m=7: 2^21 = 2097152 (might be too slow)
        # For m=9: 2^27 = 134217728 (too slow)
        
        # Better approach: iterate over reduced space
        # After gauge fixing, we have 3m-2 free variables
        
        if m <= 5:
            # Full exhaustive search with gauge fixing
            for mask in range(2 ** (3 * m)):
                if time.time() > deadline - 3:
                    return
                
                # Decode mask into signs
                signs = np.ones(num_edges, dtype=int)
                for i in range(num_edges):
                    if (mask >> i) & 1:
                        signs[i] = -1
                
                # Apply gauge fixing
                signs = gauge_fix(signs)
                
                # Check if this is the canonical representative
                # (first top edge should be +1, first match edge should be +1)
                if signs[top_edges[0]] != 1 or signs[match_edges[0]] != 1:
                    continue
                
                radius = evaluate(signs)
        
        elif m == 7:
            # For m=7, do a more targeted search
            # Try all combinations for the first few edges, then use local search
            num_tries = 0
            for mask in range(2 ** (3 * m)):
                if time.time() > deadline - 3:
                    return
                if num_tries > 500000:  # Cap to avoid timeout
                    break
                
                signs = np.ones(num_edges, dtype=int)
                for i in range(num_edges):
                    if (mask >> i) & 1:
                        signs[i] = -1
                
                signs = gauge_fix(signs)
                
                if signs[top_edges[0]] != 1 or signs[match_edges[0]] != 1:
                    continue
                
                radius = evaluate(signs)
                num_tries += 1
        
        # For m=9, we'll rely on the other methods below
    
    # Generate structured initial guesses
    patterns = [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
        [1, 1, 1, -1],
        [1, -1, 1, 1],
        [1, -1, -1, -1],
        [-1, 1, 1, 1],
    ]
    
    # Try all combinations of patterns
    for pt in patterns:
        for pb in patterns:
            for pm in patterns:
                if time.time() > deadline - 10:
                    break
                signs = make_structured_signs(pt, pb, pm)
                evaluate(signs)
    
    # Exhaustive search for small m
    exhaustive_search_gauge_reduced()
    
    # Random sampling
    rng = np.random.default_rng(42)
    
    num_random = 3000 if m <= 5 else 1000
    for _ in range(num_random):
        if time.time() > deadline - 15:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)
    
    # Local search from best
    if best_signs is not None:
        local_search(best_signs, max_iters=300)
    
    # More random restarts with local search
    num_restarts = 50 if m <= 5 else 20
    for _ in range(num_restarts):
        if time.time() > deadline - 5:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)
        if best_signs is not None:
            local_search(best_signs, max_iters=100)
    
    # Final polish: try 3-flip moves from the best solution
    if best_signs is not None and time.time() < deadline - 3:
        local_search_3flip(best_signs, max_iters=50)
    
    if best_signs is None:
        return np.ones(num_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
