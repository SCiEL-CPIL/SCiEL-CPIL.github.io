import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    # Try to find a valid coloring for n=34 down to 10
    time_budgets = {
        34: 100.0,
        33: 20.0,
        32: 8.0,
        31: 4.0,
        30: 2.0,
        29: 1.0,
        28: 0.5,
    }
    
    for n in range(34, 9, -1):
        if time.time() > deadline:
            break
        budget = time_budgets.get(n, 0.5)
        budget = min(budget, deadline - time.time())
        if budget < 0.1:
            continue
            
        try:
            result = solve_n(n, budget)
            if result is not None:
                return result
        except Exception:
            continue
            
    # Fallback to seed solution
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

def solve_n(n, time_limit):
    start = time.time()
    
    # Precompute triples
    triples = list(combinations(range(n), 3))
    M = len(triples)
    idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and 5-sets
    sets4 = list(combinations(range(n), 4))
    sets5 = list(combinations(range(n), 5))
    N4 = len(sets4)
    N5 = len(sets5)
    
    # Map triple index to list of 4-set indices containing it
    triple_to_4 = [[] for _ in range(M)]
    for i4, s4 in enumerate(sets4):
        for j in range(4):
            t = tuple(sorted(s4[:j] + s4[j+1:]))
            triple_to_4[idx[t]].append(i4)
            
    # Map triple index to list of 5-set indices containing it
    triple_to_5 = [[] for _ in range(M)]
    for i5, s5 in enumerate(sets5):
        for j in range(5):
            t = tuple(sorted(s5[:j] + s5[j+1:]))
            triple_to_5[idx[t]].append(i5)
            
    # Precompute triples for each 4-set and 5-set
    s4_triples = []
    for s4 in sets4:
        ts = [idx[tuple(sorted(s4[:j] + s4[j+1:]))] for j in range(4)]
        s4_triples.append(ts)
        
    s5_triples = []
    for s5 in sets5:
        ts = [idx[t] for t in combinations(s5, 3)]
        s5_triples.append(ts)
        
    # Convert to numpy arrays for faster access
    s4_triples_arr = [np.array(ts) for ts in s4_triples]
    s5_triples_arr = [np.array(ts) for ts in s5_triples]
    triple_to_4_arr = [np.array(lst) for lst in triple_to_4]
    triple_to_5_arr = [np.array(lst) for lst in triple_to_5]
    
    def init_counts(colors):
        red_count_4 = np.zeros(N4, dtype=np.int8)
        blue_count_5 = np.zeros(N5, dtype=np.int8)
        for i in range(M):
            c = colors[i]
            if c == 0:
                for i4 in triple_to_4_arr[i]:
                    red_count_4[i4] += 1
            else:
                for i5 in triple_to_5_arr[i]:
                    blue_count_5[i5] += 1
        return red_count_4, blue_count_5
    
    def get_violations(red_count_4, blue_count_5):
        v4 = set(np.where(red_count_4 == 4)[0])
        v5 = set(np.where(blue_count_5 == 10)[0])
        return v4, v5
    
    # Multiple initialization strategies
    init_strategies = [
        lambda: np.random.randint(0, 2, M),
        lambda: np.zeros(M, dtype=np.int8),  # all red
        lambda: np.ones(M, dtype=np.int8),   # all blue
        lambda: np.array([0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples], dtype=np.int8),
        lambda: np.array([0 if (a*b+c) % 2 == 0 else 1 for a,b,c in triples], dtype=np.int8),
        lambda: np.array([0 if (a^b^c) % 2 == 0 else 1 for a,b,c in triples], dtype=np.int8),
        lambda: np.array([0 if (a+b) % 2 == (c % 2) else 1 for a,b,c in triples], dtype=np.int8),
        lambda: np.array([0 if (a*b*c) % 3 == 0 else 1 for a,b,c in triples], dtype=np.int8),
    ]
    
    max_restarts = 12
    best_result = None
    
    for restart in range(max_restarts):
        if time.time() - start > time_limit:
            break
            
        # Choose initialization strategy
        strat_idx = restart % len(init_strategies)
        colors = init_strategies[strat_idx]().copy()
        
        red_count_4, blue_count_5 = init_counts(colors)
        violating_4, violating_5 = get_violations(red_count_4, blue_count_5)
        
        if not violating_4 and not violating_5:
            return {'n': n, 'colors': colors.tolist()}
        
        # Local search with simulated annealing-like temperature
        temperature = 1.0
        no_improve = 0
        max_iters = 30000
        
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
                
            if not violating_4 and not violating_5:
                return {'n': n, 'colors': colors.tolist()}
            
            # Pick a violation - prioritize the larger set
            if violating_4 and violating_5:
                if len(violating_4) > len(violating_5) * 1.5:
                    i4 = random.choice(list(violating_4))
                    t_idx = random.choice(s4_triples_arr[i4])
                elif len(violating_5) > len(violating_4) * 1.5:
                    i5 = random.choice(list(violating_5))
                    t_idx = random.choice(s5_triples_arr[i5])
                else:
                    if random.random() < 0.5:
                        i4 = random.choice(list(violating_4))
                        t_idx = random.choice(s4_triples_arr[i4])
                    else:
                        i5 = random.choice(list(violating_5))
                        t_idx = random.choice(s5_triples_arr[i5])
            elif violating_4:
                i4 = random.choice(list(violating_4))
                t_idx = random.choice(s4_triples_arr[i4])
            else:
                i5 = random.choice(list(violating_5))
                t_idx = random.choice(s5_triples_arr[i5])
                
            # Flip t_idx
            old_color = colors[t_idx]
            new_color = 1 - old_color
            colors[t_idx] = new_color
            
            # Update counts
            for i4 in triple_to_4_arr[t_idx]:
                if old_color == 0:
                    red_count_4[i4] -= 1
                    if red_count_4[i4] == 4:
                        violating_4.add(i4)
                    else:
                        violating_4.discard(i4)
                else:
                    red_count_4[i4] += 1
                    if red_count_4[i4] == 4:
                        violating_4.add(i4)
                    else:
                        violating_4.discard(i4)
                        
            for i5 in triple_to_5_arr[t_idx]:
                if old_color == 1:
                    blue_count_5[i5] -= 1
                    if blue_count_5[i5] == 10:
                        violating_5.add(i5)
                    else:
                        violating_5.discard(i5)
                else:
                    blue_count_5[i5] += 1
                    if blue_count_5[i5] == 10:
                        violating_5.add(i5)
                    else:
                        violating_5.discard(i5)
            
            # Track improvement
            total_viol = len(violating_4) + len(violating_5)
            if no_improve > 5000:
                # Reset with perturbation
                for _ in range(10):
                    t_idx = random.randint(0, M-1)
                    old_color = colors[t_idx]
                    new_color = 1 - old_color
                    colors[t_idx] = new_color
                    for i4 in triple_to_4_arr[t_idx]:
                        if old_color == 0:
                            red_count_4[i4] -= 1
                            if red_count_4[i4] == 4:
                                violating_4.add(i4)
                            else:
                                violating_4.discard(i4)
                        else:
                            red_count_4[i4] += 1
                            if red_count_4[i4] == 4:
                                violating_4.add(i4)
                            else:
                                violating_4.discard(i4)
                    for i5 in triple_to_5_arr[t_idx]:
                        if old_color == 1:
                            blue_count_5[i5] -= 1
                            if blue_count_5[i5] == 10:
                                violating_5.add(i5)
                            else:
                                violating_5.discard(i5)
                        else:
                            blue_count_5[i5] += 1
                            if blue_count_5[i5] == 10:
                                violating_5.add(i5)
                            else:
                                violating_5.discard(i5)
                no_improve = 0
                temperature = 1.0
            
            if total_viol < getattr(solve_n, '_last_total', float('inf')):
                no_improve = 0
                solve_n._last_total = total_viol
            else:
                no_improve += 1
                
    return None
