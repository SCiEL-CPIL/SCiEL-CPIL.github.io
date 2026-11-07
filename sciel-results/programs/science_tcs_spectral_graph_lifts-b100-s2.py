"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    deadline = time.time() + 55
    
    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(42)
    
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
    
    def local_search(start_signs, max_iters=200):
        nonlocal best_signs, best_radius
        current = start_signs.copy()
        cur_radius = evaluate(current)
        if cur_radius is None:
            return
        
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 3:
                    break
                trial = current.copy()
                trial[i] = -trial[i]
                trial_radius = evaluate(trial)
                if trial_radius is not None and trial_radius < cur_radius:
                    current = trial
                    cur_radius = trial_radius
                    improved = True
                    break
            if not improved:
                break
    
    # Phase 1: Random sampling
    for _ in range(300):
        if time.time() > deadline - 15:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)
    
    # Phase 2: Local search from best
    if best_signs is not None:
        local_search(best_signs, max_iters=300)
    
    # Phase 3: Random restarts with local search
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        evaluate(signs)
        if best_signs is not None:
            local_search(best_signs, max_iters=100)
    
    if best_signs is None:
        return np.ones(num_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
