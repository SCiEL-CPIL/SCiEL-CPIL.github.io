import numpy as np
import itertools
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try n=34 first, then 33, etc.
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 10.0:
            break
        result = solve_sa(n, deadline)
        if result is not None:
            return result
            
    # Fallback to a small valid n if all else fails
    # n=1 is trivially valid
    return {'n': 1, 'colors': []}

def solve_sa(n, deadline):
    triples = list(itertools.combinations(range(n), 3))
    m = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute quads and pents
    quads = list(itertools.combinations(range(n), 4))
    quads_triples = []
    for q in quads:
        trs = [triple_to_idx[t] for t in itertools.combinations(q, 3)]
        quads_triples.append(trs)
        
    pents = list(itertools.combinations(range(n), 5))
    pents_triples = []
    for p in pents:
        trs = [triple_to_idx[t] for t in itertools.combinations(p, 3)]
        pents_triples.append(trs)
        
    # Map triple to quads and pents containing it
    triple_to_quads = [[] for _ in range(m)]
    triple_to_pents = [[] for _ in range(m)]
    
    for i, trs in enumerate(quads_triples):
        for t_idx in trs:
            triple_to_quads[t_idx].append(i)
            
    for j, trs in enumerate(pents_triples):
        for t_idx in trs:
            triple_to_pents[t_idx].append(j)
            
    # Simulated Annealing
    best_result = None
    
    restart_deadline = deadline - 5.0
    restart_num = 0
    
    while time.time() < restart_deadline:
        restart_num += 1
        
        # Initialize
        # Strategy: Random
        colors = np.random.randint(0, 2, m, dtype=np.int8)
        
        # Initialize counts
        red_count_quads = np.zeros(len(quads), dtype=np.int32)
        blue_count_pents = np.zeros(len(pents), dtype=np.int32)
        
        # Efficient initialization:
        for t_idx in range(m):
            if colors[t_idx] == 0: # Red
                for q_idx in triple_to_quads[t_idx]:
                    red_count_quads[q_idx] += 1
            else: # Blue
                for p_idx in triple_to_pents[t_idx]:
                    blue_count_pents[p_idx] += 1
                    
        # Identify violations
        red_violations = set(np.where(red_count_quads == 4)[0])
        blue_violations = set(np.where(blue_count_pents == 10)[0])
        
        cost = len(red_violations) + len(blue_violations)
        
        if cost == 0:
            return {'n': n, 'colors': colors.tolist()}
        
        # SA Parameters
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        # Local loop
        local_deadline = min(restart_deadline, time.time() + 2.0)
        stuck_counter = 0
        best_cost_in_run = cost
        
        while time.time() < local_deadline:
            if cost == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            if cost < best_cost_in_run:
                best_cost_in_run = cost
                stuck_counter = 0
            else:
                stuck_counter += 1
                
            # Pick a move
            # Prefer moves that reduce violations
            if red_violations or blue_violations:
                # Choose which violation type to target
                # Weight by count or just 50/50
                if red_violations and (not blue_violations or random.random() < 0.5):
                    v_idx = random.choice(list(red_violations))
                    trs = quads_triples[v_idx]
                    # Flip a red triple in this quad to blue
                    t_idx = random.choice(trs)
                    new_color = 1
                else:
                    v_idx = random.choice(list(blue_violations))
                    trs = pents_triples[v_idx]
                    # Flip a blue triple in this pent to red
                    t_idx = random.choice(trs)
                    new_color = 0
            else:
                # No violations? Should not happen if cost>0
                t_idx = random.randint(0, m - 1)
                new_color = 1 - colors[t_idx]
                
            old_color = colors[t_idx]
            if old_color == new_color:
                continue
                
            # Calculate delta cost
            delta_cost = 0
            
            # Update red counts for quads (simulation)
            for q_idx in triple_to_quads[t_idx]:
                old_rc = red_count_quads[q_idx]
                if old_color == 0: # Red -> Blue
                    new_rc = old_rc - 1
                else: # Blue -> Red
                    new_rc = old_rc + 1
                
                was_viol = (old_rc == 4)
                is_viol = (new_rc == 4)
                
                if was_viol and not is_viol:
                    delta_cost -= 1
                elif not was_viol and is_viol:
                    delta_cost += 1
                
            # Update blue counts for pents (simulation)
            for p_idx in triple_to_pents[t_idx]:
                old_bc = blue_count_pents[p_idx]
                if old_color == 1: # Blue -> Red
                    new_bc = old_bc - 1
                else: # Red -> Blue
                    new_bc = old_bc + 1
                
                was_viol = (old_bc == 10)
                is_viol = (new_bc == 10)
                
                if was_viol and not is_viol:
                    delta_cost -= 1
                elif not was_viol and is_viol:
                    delta_cost += 1
                
            # Acceptance
            if delta_cost < 0:
                accept = True
            else:
                if T > min_T:
                    prob = math.exp(-delta_cost / T)
                    accept = random.random() < prob
                else:
                    accept = False
                    
            if accept:
                # Apply changes
                colors[t_idx] = new_color
                cost += delta_cost
                
                for q_idx in triple_to_quads[t_idx]:
                    old_rc = red_count_quads[q_idx]
                    if old_color == 0:
                        new_rc = old_rc - 1
                    else:
                        new_rc = old_rc + 1
                    
                    if old_rc == 4 and new_rc != 4:
                        red_violations.discard(q_idx)
                    elif old_rc != 4 and new_rc == 4:
                        red_violations.add(q_idx)
                        
                    red_count_quads[q_idx] = new_rc
                
                for p_idx in triple_to_pents[t_idx]:
                    old_bc = blue_count_pents[p_idx]
                    if old_color == 1:
                        new_bc = old_bc - 1
                    else:
                        new_bc = old_bc + 1
                    
                    if old_bc == 10 and new_bc != 10:
                        blue_violations.discard(p_idx)
                    elif old_bc != 10 and new_bc == 10:
                        blue_violations.add(p_idx)
                        
                    blue_count_pents[p_idx] = new_bc
                
                T *= cooling
                
            if stuck_counter > 2000:
                break
                
    return None
