import time
import random
import itertools
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def try_find_coloring(n, time_limit):
        if n < 4:
            return [0] * len(list(itertools.combinations(range(n), 3)))
            
        edges = list(itertools.combinations(range(n), 3))
        num_edges = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        
        sets4 = list(itertools.combinations(range(n), 4))
        sets5 = list(itertools.combinations(range(n), 5))
        num_sets4 = len(sets4)
        num_sets5 = len(sets5)
        
        # Precompute adjacency
        edge_to_4sets = [[] for _ in range(num_edges)]
        for s4_idx, s4 in enumerate(sets4):
            for i in range(4):
                triple = s4[:i] + s4[i+1:]
                e_idx = edge_index[triple]
                edge_to_4sets[e_idx].append(s4_idx)
                
        edge_to_5sets = [[] for _ in range(num_edges)]
        for s5_idx, s5 in enumerate(sets5):
            for triple in itertools.combinations(s5, 3):
                e_idx = edge_index[triple]
                edge_to_5sets[e_idx].append(s5_idx)
                
        # Precompute penalty deltas
        # P4(c) = (c-2)**2 if c > 2 else 0
        P4 = [0]*5
        for c in range(5):
            if c > 2:
                P4[c] = (c-2)**2
        d4_dec = [P4[c-1] - P4[c] for c in range(5)] # c -> c-1
        d4_inc = [P4[c+1] - P4[c] for c in range(4)] # c -> c+1, c in 0..3
        
        # P5(c) = (c-8)**2 if c > 8 else 0
        P5 = [0]*11
        for c in range(11):
            if c > 8:
                P5[c] = (c-8)**2
        d5_inc = [P5[c+1] - P5[c] for c in range(10)] # c -> c+1, c in 0..9
        d5_dec = [P5[c-1] - P5[c] for c in range(11)] # c -> c-1, c in 1..10
        
        # State
        colors = [random.randint(0, 1) for _ in range(num_edges)]
        red_count_4 = [0] * num_sets4
        blue_count_5 = [0] * num_sets5
        
        for e_idx, color in enumerate(colors):
            if color == 0:
                for s4_idx in edge_to_4sets[e_idx]:
                    red_count_4[s4_idx] += 1
            else:
                for s5_idx in edge_to_5sets[e_idx]:
                    blue_count_5[s5_idx] += 1
                    
        def calc_cost():
            cost = 0
            for c in red_count_4:
                cost += P4[c]
            for c in blue_count_5:
                cost += P5[c]
            return cost
            
        current_cost = calc_cost()
        
        if current_cost == 0:
            return colors
            
        T = 10.0
        cooling = 0.99
        min_T = 0.01
        end_time = start_time + time_limit
        
        while time.time() < end_time:
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            delta_cost = 0
            
            if old_color == 0: # Red -> Blue
                # 4-sets: red count decreases (c -> c-1)
                for s4_idx in edge_to_4sets[e_idx]:
                    c = red_count_4[s4_idx]
                    delta_cost += d4_dec[c]
                    red_count_4[s4_idx] = c - 1
                    
                # 5-sets: blue count increases (c -> c+1)
                for s5_idx in edge_to_5sets[e_idx]:
                    c = blue_count_5[s5_idx]
                    delta_cost += d5_inc[c]
                    blue_count_5[s5_idx] = c + 1
                    
            else: # Blue -> Red
                # 5-sets: blue count decreases (c -> c-1)
                for s5_idx in edge_to_5sets[e_idx]:
                    c = blue_count_5[s5_idx]
                    delta_cost += d5_dec[c]
                    blue_count_5[s5_idx] = c - 1
                    
                # 4-sets: red count increases (c -> c+1)
                for s4_idx in edge_to_4sets[e_idx]:
                    c = red_count_4[s4_idx]
                    delta_cost += d4_inc[c]
                    red_count_4[s4_idx] = c + 1
                    
            new_cost = current_cost + delta_cost
            
            if new_cost < current_cost or random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
                colors[e_idx] = 1 - old_color
                
                if current_cost == 0:
                    return colors
            else:
                # Rollback
                if old_color == 0:
                    for s4_idx in edge_to_4sets[e_idx]:
                        red_count_4[s4_idx] += 1
                    for s5_idx in edge_to_5sets[e_idx]:
                        blue_count_5[s5_idx] -= 1
                else:
                    for s5_idx in edge_to_5sets[e_idx]:
                        blue_count_5[s5_idx] += 1
                    for s4_idx in edge_to_4sets[e_idx]:
                        red_count_4[s4_idx] -= 1
                
            T *= cooling
            if T < min_T:
                T = min_T
                
        return None

    time_alloc = {
        34: 60.0,
        33: 30.0,
        32: 20.0,
        31: 10.0,
        30: 5.0,
        29: 2.0,
        28: 1.0,
        27: 0.5
    }
    
    for n in sorted(time_alloc.keys(), reverse=True):
        if time.time() > deadline - 5.0:
            break
            
        t_limit = time_alloc[n]
        if time.time() + t_limit > deadline:
            t_limit = deadline - time.time() - 1.0
            if t_limit < 0.5:
                continue
                
        # print(f"Trying n={n} with time limit {t_limit}", file=sys.stderr)
        
        result = try_find_coloring(n, t_limit)
        
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
