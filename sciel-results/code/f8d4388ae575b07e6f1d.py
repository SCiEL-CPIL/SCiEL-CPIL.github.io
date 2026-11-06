import numpy as np
import itertools
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    for n in [34, 33, 32, 31, 30, 28, 25, 20, 15, 10, 5, 1]:
        if time.time() > deadline - 3.0:
            break
        result = solve_conflict(n, deadline)
        if result is not None:
            return result
            
    return {'n': 1, 'colors': []}

def solve_conflict(n, deadline):
    triples = list(itertools.combinations(range(n), 3))
    m = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    four_sets = list(itertools.combinations(range(n), 4))
    four_set_triples = []
    for s in four_sets:
        trs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
        four_set_triples.append(trs)
        
    five_sets = list(itertools.combinations(range(n), 5))
    five_set_triples = []
    for s in five_sets:
        trs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
        five_set_triples.append(trs)
        
    triple_to_four_sets = [[] for _ in range(m)]
    triple_to_five_sets = [[] for _ in range(m)]
    
    for i, trs in enumerate(four_set_triples):
        for t_idx in trs:
            triple_to_four_sets[t_idx].append(i)
            
    for j, trs in enumerate(five_set_triples):
        for t_idx in trs:
            triple_to_five_sets[t_idx].append(j)
            
    # Multiple restart strategies
    best_result = None
    best_cost = float('inf')
    
    restart_deadline = deadline - 5.0
    restart_num = 0
    
    while time.time() < restart_deadline:
        restart_num += 1
        
        if restart_num == 1:
            # Strategy 1: Balanced initialization - aim for ~50% red/blue
            colors = np.random.randint(0, 2, m)
        elif restart_num == 2:
            # Strategy 2: Structured - alternate by vertex parity
            colors = np.zeros(m, dtype=np.int8)
            for i, (a, b, c) in enumerate(triples):
                colors[i] = (a + b + c) % 2
        elif restart_num == 3:
            # Strategy 3: Mostly blue with some red
            colors = np.ones(m, dtype=np.int8)
            for i in range(m):
                if random.random() < 0.3:
                    colors[i] = 0
        elif restart_num == 4:
            # Strategy 4: Mostly red with some blue
            colors = np.zeros(m, dtype=np.int8)
            for i in range(m):
                if random.random() < 0.3:
                    colors[i] = 1
        else:
            # Random
            colors = np.random.randint(0, 2, m)
        
        # Initialize counts
        red_count_4 = np.zeros(len(four_sets), dtype=np.int32)
        blue_count_5 = np.zeros(len(five_sets), dtype=np.int32)
        
        for i, trs in enumerate(four_set_triples):
            red_count_4[i] = np.sum(colors[trs] == 0)
        for j, trs in enumerate(five_set_triples):
            blue_count_5[j] = np.sum(colors[trs] == 1)
        
        cost = int(np.sum(red_count_4 == 4)) + int(np.sum(blue_count_5 == 10))
        
        if cost == 0:
            return {'n': n, 'colors': colors.tolist()}
        
        # Conflict-directed local search
        local_deadline = min(restart_deadline, time.time() + 2.0)
        stuck_counter = 0
        max_stuck = 5000
        
        while time.time() < local_deadline:
            if cost == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            if cost < best_cost:
                best_cost = cost
                best_result = {'n': n, 'colors': colors.tolist()}
            
            # Find a random violation to target
            # Either a red K4 or a blue K5
            red_violations = np.where(red_count_4 == 4)[0]
            blue_violations = np.where(blue_count_5 == 10)[0]
            
            if len(red_violations) > 0 and (len(blue_violations) == 0 or random.random() < 0.5):
                # Target a red K4 violation
                v_idx = red_violations[random.randint(0, len(red_violations) - 1)]
                # Pick a random triple from this 4-set to flip to blue
                trs = four_set_triples[v_idx]
                # Only flip triples that are currently red
                red_trs = [t for t in trs if colors[t] == 0]
                if len(red_trs) == 0:
                    continue
                t_idx = red_trs[random.randint(0, len(red_trs) - 1)]
                new_color = 1
            else:
                # Target a blue K5 violation
                v_idx = blue_violations[random.randint(0, len(blue_violations) - 1)]
                trs = five_set_triples[v_idx]
                # Only flip triples that are currently blue
                blue_trs = [t for t in trs if colors[t] == 1]
                if len(blue_trs) == 0:
                    continue
                t_idx = blue_trs[random.randint(0, len(blue_trs) - 1)]
                new_color = 0
            
            old_color = colors[t_idx]
            if old_color == new_color:
                continue
            
            # Calculate delta cost
            delta_cost = 0
            
            for i in triple_to_four_sets[t_idx]:
                if old_color == 0:
                    old_rc = red_count_4[i]
                    new_rc = old_rc - 1
                    if old_rc == 4:
                        delta_cost -= 1
                    if new_rc == 4:
                        delta_cost += 1
                    red_count_4[i] = new_rc
                else:
                    old_rc = red_count_4[i]
                    new_rc = old_rc + 1
                    if old_rc == 4:
                        delta_cost -= 1
                    if new_rc == 4:
                        delta_cost += 1
                    red_count_4[i] = new_rc
                    
            for j in triple_to_five_sets[t_idx]:
                if old_color == 1:
                    old_bc = blue_count_5[j]
                    new_bc = old_bc - 1
                    if old_bc == 10:
                        delta_cost -= 1
                    if new_bc == 10:
                        delta_cost += 1
                    blue_count_5[j] = new_bc
                else:
                    old_bc = blue_count_5[j]
                    new_bc = old_bc + 1
                    if old_bc == 10:
                        delta_cost -= 1
                    if new_bc == 10:
                        delta_cost += 1
                    blue_count_5[j] = new_bc
                    
            cost += delta_cost
            colors[t_idx] = new_color
            
            if delta_cost > 0:
                stuck_counter += 1
            else:
                stuck_counter = 0
            
            # If stuck too long, do a random perturbation
            if stuck_counter > max_stuck:
                stuck_counter = 0
                # Flip a random batch of triples
                num_flips = random.randint(5, 20)
                for _ in range(num_flips):
                    t_idx = random.randint(0, m - 1)
                    old_c = colors[t_idx]
                    new_c = 1 - old_c
                    
                    d = 0
                    for i in triple_to_four_sets[t_idx]:
                        if old_c == 0:
                            old_rc = red_count_4[i]
                            new_rc = old_rc - 1
                            if old_rc == 4: d -= 1
                            if new_rc == 4: d += 1
                            red_count_4[i] = new_rc
                        else:
                            old_rc = red_count_4[i]
                            new_rc = old_rc + 1
                            if old_rc == 4: d -= 1
                            if new_rc == 4: d += 1
                            red_count_4[i] = new_rc
                    for j in triple_to_five_sets[t_idx]:
                        if old_c == 1:
                            old_bc = blue_count_5[j]
                            new_bc = old_bc - 1
                            if old_bc == 10: d -= 1
                            if new_bc == 10: d += 1
                            blue_count_5[j] = new_bc
                        else:
                            old_bc = blue_count_5[j]
                            new_bc = old_bc + 1
                            if old_bc == 10: d -= 1
                            if new_bc == 10: d += 1
                            blue_count_5[j] = new_bc
                    cost += d
                    colors[t_idx] = new_c
        
        # Check if we improved
        if best_result is None or best_cost > cost:
            if cost == 0:
                return {'n': n, 'colors': colors.tolist()}
    
    if best_result is not None and best_cost == 0:
        return best_result
        
    return None
