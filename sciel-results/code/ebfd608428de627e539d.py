import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # Target n values to try, descending
    target_ns = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    def is_valid(n, colors):
        # Quick check for validity
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def solve_for_n(n):
        if time.time() > deadline:
            return None, False

        num_triples = len(list(combinations(range(n), 3)))
        triple_list = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        # Precompute structures
        # For each triple, list of quads (indices of the other 3 triples in the quad)
        triple_in_quads = [[] for _ in range(num_triples)]
        all_quads = list(combinations(range(n), 4))
        quad_data = [] # List of (quad_indices, [t1, t2, t3, t4])
        for a, b, c, d in all_quads:
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            quad_data.append((t1, t2, t3, t4))
            for t in [t1, t2, t3, t4]:
                triple_in_quads[t].append(len(quad_data) - 1)

        # For each triple, list of quints (indices of the other 9 triples in the quint)
        triple_in_quints = [[] for _ in range(num_triples)]
        all_quints = list(combinations(range(n), 5))
        quint_data = [] # List of 10 triple indices
        for q5 in all_quints:
            ts = [triple_to_idx[t] for t in combinations(q5, 3)]
            quint_data.append(ts)
            for t in ts:
                triple_in_quints[t].append(len(quint_data) - 1)

        num_quads = len(quad_data)
        num_quints = len(quint_data)

        # State: 0=Red, 1=Blue
        # We want to minimize cost.
        # Cost = sum over quads of penalty(red_count) + sum over quints of penalty(blue_count)
        
        def calc_cost(colors):
            cost = 0.0
            # Quad penalties
            for t1, t2, t3, t4 in quad_data:
                reds = 0
                if colors[t1] == 0: reds += 1
                if colors[t2] == 0: reds += 1
                if colors[t3] == 0: reds += 1
                if colors[t4] == 0: reds += 1
                if reds == 4:
                    cost += 10.0
                elif reds == 3:
                    cost += 1.0
            # Quint penalties
            for ts in quint_data:
                blues = 0
                for t in ts:
                    if colors[t] == 1: blues += 1
                if blues == 10:
                    cost += 10.0
                elif blues == 9:
                    cost += 1.0
            return cost

        def delta_cost(colors, tidx):
            """Calculate change in cost if we flip tidx."""
            old_color = colors[tidx]
            new_color = 1 - old_color
            
            d_cost = 0.0
            
            # Update quad costs
            for qi in triple_in_quads[tidx]:
                t1, t2, t3, t4 = quad_data[qi]
                old_reds = 0
                new_reds = 0
                for t in [t1, t2, t3, t4]:
                    if t == tidx:
                        if old_color == 0: old_reds += 1
                        if new_color == 0: new_reds += 1
                    else:
                        if colors[t] == 0:
                            old_reds += 1
                            new_reds += 1
                
                old_pen = 10.0 if old_reds == 4 else (1.0 if old_reds == 3 else 0.0)
                new_pen = 10.0 if new_reds == 4 else (1.0 if new_reds == 3 else 0.0)
                d_cost += new_pen - old_pen

            # Update quint costs
            for qi in triple_in_quints[tidx]:
                ts = quint_data[qi]
                old_blues = 0
                new_blues = 0
                for t in ts:
                    if t == tidx:
                        if old_color == 1: old_blues += 1
                        if new_color == 1: new_blues += 1
                    else:
                        if colors[t] == 1:
                            old_blues += 1
                            new_blues += 1
                
                old_pen = 10.0 if old_blues == 10 else (1.0 if old_blues == 9 else 0.0)
                new_pen = 10.0 if new_blues == 10 else (1.0 if new_blues == 9 else 0.0)
                d_cost += new_pen - old_pen

            return d_cost

        # Simulated Annealing
        # Try multiple restarts with different initial probabilities
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initial probability of Red
            # Based on heuristic, p_red around 0.4-0.6 is good
            p_red = 0.3 + 0.02 * restart
            colors = [0 if random.random() < p_red else 1 for _ in range(num_triples)]
            
            current_cost = calc_cost(colors)
            best_cost = current_cost
            best_colors = colors[:]
            
            T_start = 5.0
            T_end = 0.01
            max_iters = 50000
            # Adjust iterations based on time
            time_per_iter = 0.0001 # Rough estimate
            allowed_time = deadline - time.time() - 2
            if allowed_time <= 0: break
            max_iters = min(max_iters, int(allowed_time / time_per_iter))
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                
                T = T_start * (T_end / T_start) ** (it / max_iters)
                
                # Pick random triple
                tidx = random.randint(0, num_triples - 1)
                
                d_cost = delta_cost(colors, tidx)
                
                if d_cost <= 0 or random.random() < np.exp(-d_cost / T):
                    colors[tidx] = 1 - colors[tidx]
                    current_cost += d_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors[:]
                        
                        if best_cost == 0:
                            # Found valid solution
                            return best_colors, True
                            
            if best_cost == 0:
                return best_colors, True

        # If no valid solution found, return best found (might be invalid)
        # But we need a valid one. If we can't find one, we should return None
        # However, the task requires a valid coloring.
        # If SA fails, we might have to return a smaller n.
        return None, False

    for n in target_ns:
        if time.time() > deadline:
            break
            
        colors, ok = solve_for_n(n)
        if ok and colors is not None:
            # Verify
            if is_valid(n, colors):
                return {'n': n, 'colors': colors}
        
        # If not ok, continue to smaller n
        # We could also try to use the best partial solution from n as init for n-1?
        # That's complex. Just restart.

    # Fallback to small n
    return {'n': best_n, 'colors': best_colors}
