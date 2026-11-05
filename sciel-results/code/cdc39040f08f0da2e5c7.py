import random
import time
import itertools
import numpy as np

def construct():
    # We want to maximize n.
    # Known lower bound is 34.
    # We will attempt to find a valid coloring for n=34 using local search.
    # If that fails within time, we try smaller n.
    # Fallback is n=9.
    
    target_ns = [34, 33, 32, 30, 25, 20, 15, 10, 9]
    
    def try_solve(n, time_limit=110):
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        quads = list(itertools.combinations(range(n), 4))
        num_quads = len(quads)
        
        quints = list(itertools.combinations(range(n), 5))
        num_quints = len(quints)
        
        quad_to_triples = []
        for q in quads:
            ts = list(itertools.combinations(q, 3))
            idxs = [triple_to_idx[t] for t in ts]
            quad_to_triples.append(idxs)
            
        quint_to_triples = []
        for q in quints:
            ts = list(itertools.combinations(q, 3))
            idxs = [triple_to_idx[t] for t in ts]
            quint_to_triples.append(idxs)
            
        triple_to_quads = [[] for _ in range(num_triples)]
        for i, ts in enumerate(quad_to_triples):
            for t_idx in ts:
                triple_to_quads[t_idx].append(i)
                
        triple_to_quints = [[] for _ in range(num_triples)]
        for i, ts in enumerate(quint_to_triples):
            for t_idx in ts:
                triple_to_quints[t_idx].append(i)
                
        # Initialize colors randomly
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Initialize counts
        red_count_4 = np.zeros(num_quads, dtype=np.int8)
        blue_count_5 = np.zeros(num_quints, dtype=np.int8)
        
        for t_idx, c in enumerate(colors):
            if c == 0: # Red
                for q_idx in triple_to_quads[t_idx]:
                    red_count_4[q_idx] += 1
            else: # Blue
                for q_idx in triple_to_quints[t_idx]:
                    blue_count_5[q_idx] += 1
                    
        violated_quads = set(np.where(red_count_4 == 4)[0])
        violated_quints = set(np.where(blue_count_5 == 10)[0])
        
        start_time = time.time()
        stuck_counter = 0
        
        while violated_quads or violated_quints:
            if time.time() - start_time > time_limit:
                return None
            
            # Pick a violation to fix
            # Prioritize the type with fewer violations to balance load? Or random?
            # Let's pick randomly from the union of violations, weighted by type if needed.
            # Actually, just pick one.
            
            if violated_quads and (not violated_quints or random.random() < 0.5):
                if not violated_quads:
                    break
                q_idx = random.choice(list(violated_quads))
                ts = quad_to_triples[q_idx]
                # Find red triples in this quad
                red_ts = [t for t in ts if colors[t] == 0]
                if not red_ts:
                    # Should not happen if logic is correct, but safety check
                    violated_quads.discard(q_idx)
                    continue
                
                # Evaluate cost for each candidate
                # Cost of flipping t to Blue: number of quints containing t with blue_count_5 == 9
                best_t = None
                best_cost = float('inf')
                candidates = []
                
                for t_idx in red_ts:
                    cost = 0
                    for q5_idx in triple_to_quints[t_idx]:
                        if blue_count_5[q5_idx] == 9:
                            cost += 1
                    if cost < best_cost:
                        best_cost = cost
                        best_t = t_idx
                        candidates = [t_idx]
                    elif cost == best_cost:
                        candidates.append(t_idx)
                        
                t_idx = random.choice(candidates)
                
                # Flip to Blue
                colors[t_idx] = 1
                
                # Update counts
                for q in triple_to_quads[t_idx]:
                    red_count_4[q] -= 1
                    if red_count_4[q] == 4:
                        violated_quads.add(q)
                    elif red_count_4[q] == 3:
                        violated_quads.discard(q)
                        
                for q in triple_to_quints[t_idx]:
                    blue_count_5[q] += 1
                    if blue_count_5[q] == 10:
                        violated_quints.add(q)
                    elif blue_count_5[q] == 9:
                        violated_quints.discard(q)
                        
                if red_count_4[q_idx] < 4:
                    violated_quads.discard(q_idx)
                    
                # Check if we made progress?
                # If we created more violations than we fixed, we might be stuck.
                # Simple heuristic: if total violations increased significantly, restart or randomize?
                # Let's just count steps.
                
            elif violated_quints:
                if not violated_quints:
                    break
                q_idx = random.choice(list(violated_quints))
                ts = quint_to_triples[q_idx]
                # Find blue triples in this quint
                blue_ts = [t for t in ts if colors[t] == 1]
                if not blue_ts:
                    violated_quints.discard(q_idx)
                    continue
                
                # Evaluate cost for each candidate
                # Cost of flipping t to Red: number of quads containing t with red_count_4 == 3
                best_t = None
                best_cost = float('inf')
                candidates = []
                
                for t_idx in blue_ts:
                    cost = 0
                    for q4_idx in triple_to_quads[t_idx]:
                        if red_count_4[q4_idx] == 3:
                            cost += 1
                    if cost < best_cost:
                        best_cost = cost
                        best_t = t_idx
                        candidates = [t_idx]
                    elif cost == best_cost:
                        candidates.append(t_idx)
                        
                t_idx = random.choice(candidates)
                
                # Flip to Red
                colors[t_idx] = 0
                
                # Update counts
                for q in triple_to_quads[t_idx]:
                    red_count_4[q] += 1
                    if red_count_4[q] == 4:
                        violated_quads.add(q)
                    elif red_count_4[q] == 3:
                        violated_quads.discard(q)
                        
                for q in triple_to_quints[t_idx]:
                    blue_count_5[q] -= 1
                    if blue_count_5[q] == 10:
                        violated_quints.add(q)
                    elif blue_count_5[q] == 9:
                        violated_quints.discard(q)
                        
                if blue_count_5[q_idx] < 10:
                    violated_quints.discard(q_idx)
                    
            else:
                break
                
            # Simple restart logic: if we've done many steps without reducing total violations, re-randomize?
            # For now, let's just let it run. The min-cost heuristic is strong.
            
        if not violated_quads and not violated_quints:
            return {'n': n, 'colors': colors.tolist()}
            
        return None

    # Try to solve for larger n first
    # Give more time to 34
    time_alloc = {34: 100, 33: 10, 32: 10, 30: 5, 25: 5, 20: 2, 15: 1, 10: 0.5, 9: 0.1}
    
    for n in target_ns:
        if n not in time_alloc:
            continue
        res = try_solve(n, time_limit=time_alloc[n])
        if res:
            return res
            
    # Fallback to seed solution
    from itertools import combinations
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
