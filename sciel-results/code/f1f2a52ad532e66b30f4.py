"""Direct constructive approach for small sums and products along selected graph edges."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 14
    m = 35
    max_val = 10000
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        ns = len(sums)
        np_ = len(prods)
        return ns, np_, max(ns, np_)
    
    def greedy_select_edges(values):
        """Select m edges to minimize max(distinct sums, distinct products) using greedy."""
        pair_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            pair_data.append((u, v, s, p))
        
        selected = []
        selected_set = set()
        current_sums = set()
        current_prods = set()
        
        for _ in range(m):
            best_edge = None
            best_score_edge = float('inf')
            best_s = None
            best_p = None
            
            for u, v, s, p in pair_data:
                if (u, v) in selected_set or (v, u) in selected_set:
                    continue
                
                add_s = 1 if s not in current_sums else 0
                add_p = 1 if p not in current_prods else 0
                
                new_num_sums = len(current_sums) + add_s
                new_num_prods = len(current_prods) + add_p
                score = max(new_num_sums, new_num_prods)
                
                # Tie-break: prefer edges that add to both (more "efficient")
                tie_break = add_s + add_p
                
                if (score, tie_break) < (best_score_edge, float('inf')):
                    best_score_edge = score
                    best_edge = (u, v)
                    best_s = s
                    best_p = p
            
            if best_edge:
                selected.append(best_edge)
                selected_set.add(best_edge)
                current_sums.add(best_s)
                current_prods.add(best_p)
        
        return selected
    
    def local_improve_edges(values, edges, time_limit):
        """Try swapping one edge for another to reduce distinct counts."""
        selected = list(edges)
        selected_set = set(selected)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            current_sums = set()
            current_prods = set()
            for u, v in selected:
                current_sums.add(values[u] + values[v])
                current_prods.add(values[u] * values[v])
            current_score = max(len(current_sums), len(current_prods))
            
            for i in range(len(selected)):
                u1, v1 = selected[i]
                
                # Remove edge i temporarily
                temp_sums = set()
                temp_prods = set()
                for j, (u, v) in enumerate(selected):
                    if j != i:
                        temp_sums.add(values[u] + values[v])
                        temp_prods.add(values[u] * values[v])
                
                # Try adding each unselected edge
                for u2, v2 in all_edges:
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                    if (u2, v2) in selected_set or (v2, u2) in selected_set:
                        continue
                    
                    s2 = values[u2] + values[u2] if u2 == v2 else values[u2] + values[v2]
                    p2 = values[u2] * values[v2]
                    
                    new_sums = temp_sums | {s2}
                    new_prods = temp_prods | {p2}
                    new_score = max(len(new_sums), len(new_prods))
                    
                    if new_score < current_score:
                        selected[i] = (u2, v2)
                        selected_set.discard((u1, v1))
                        selected_set.discard((v1, u1))
                        selected_set.add((u2, v2))
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        return selected
    
    def simulated_annealing(values, edges, time_budget):
        """Simulated annealing that co-optimizes values and edges."""
        rng = random.Random(42)
        deadline = time.time() + time_budget
        
        current_values = list(values)
        current_edges = list(edges)
        _, _, current_score = evaluate(current_values, current_edges)
        
        best_values = list(current_values)
        best_edges = list(current_edges)
        best_score = current_score
        
        T = 5.0  # Initial temperature
        cooling = 0.995
        
        while time.time() < deadline:
            # Choose perturbation type
            r = rng.random()
            
            if r < 0.4:
                # Perturb a value
                idx = rng.randint(0, n - 1)
                delta = rng.choice([-5, -3, -2, -1, 1, 2, 3, 5, -10, 10, -20, 20])
                new_val = current_values[idx] + delta
                if new_val < 1 or new_val > max_val or new_val in current_values:
                    continue
                new_values = current_values.copy()
                new_values[idx] = new_val
                new_edges = current_edges
            elif r < 0.7:
                # Swap two values
                i, j = rng.sample(range(n), 2)
                new_values = current_values.copy()
                new_values[i], new_values[j] = new_values[j], new_values[i]
                new_edges = current_edges
            else:
                # Perturb edges: swap one edge
                i = rng.randint(0, len(current_edges) - 1)
                old_u, old_v = current_edges[i]
                # Pick a random unselected edge
                candidates = [e for e in all_edges if e != (old_u, old_v) and e != (old_v, old_u) and e not in current_edges and tuple(reversed(e)) not in current_edges]
                if not candidates:
                    continue
                new_u, new_v = rng.choice(candidates)
                new_values = current_values
                new_edges = list(current_edges)
                new_edges[i] = (new_u, new_v)
            
            _, _, new_score = evaluate(new_values, new_edges)
            
            # Accept if better or with probability exp(-(new_score - current_score)/T)
            if new_score <= current_score:
                current_values = new_values
                current_edges = new_edges
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(new_values)
                    best_edges = list(new_edges)
            else:
                delta = new_score - current_score
                if rng.random() < math.exp(-delta / T):
                    current_values = new_values
                    current_edges = new_edges
                    current_score = new_score
            
            T *= cooling
        
        return best_values, best_edges, best_score
    
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(12345)
    
    value_strategies = [
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],
        [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168],
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 30],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
        [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [2, 6, 18, 54, 162, 486, 1458, 4374, 3, 9, 27, 81, 243, 729],
    ]
    
    # Phase 1: Try each value strategy with greedy edge selection
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline - 10.0:
            break
        
        edges = greedy_select_edges(init_values)
        _, _, score = evaluate(init_values, edges)
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(init_values), 'edges': [list(e) for e in edges]}
        
        # Local search on edges
        if best_result is not None and (time.time() - start_time) < deadline - 8.0:
            edges_ls = [list(e) for e in best_result['edges']]
            edges_ls = local_improve_edges(init_values, edges_ls, deadline - 8.0)
            _, _, score_e = evaluate(init_values, edges_ls)
            if score_e < best_overall_score:
                best_overall_score = score_e
                best_result = {'values': list(init_values), 'edges': [list(e) for e in edges_ls]}
        
        # Simulated annealing co-optimization
        if best_result is not None and (time.time() - start_time) < deadline - 5.0:
            values_ls = list(best_result['values'])
            edges_ls = [list(e) for e in best_result['edges']]
            values_ls, edges_ls, score_sa = simulated_annealing(values_ls, edges_ls, deadline - 5.0)
            if score_sa < best_overall_score:
                best_overall_score = score_sa
                best_result = {'values': list(values_ls), 'edges': [list(e) for e in edges_ls]}
    
    # Phase 2: More aggressive refinement on best result
    if best_result is not None and (time.time() - start_time) < deadline - 3.0:
        values = list(best_result['values'])
        edges = [list(e) for e in best_result['edges']]
        
        # Multiple rounds of simulated annealing with different seeds
        for seed in [42, 123, 456, 789, 101]:
            if (time.time() - start_time) > deadline - 2.0:
                break
            
            rng_sa = random.Random(seed)
            # Reset SA with different seed
            current_values = list(values)
            current_edges = list(edges)
            _, _, current_score = evaluate(current_values, current_edges)
            
            best_values = list(current_values)
            best_edges = list(current_edges)
            best_score = current_score
            
            T = 3.0
            cooling = 0.99
            
            deadline_sa = time.time() + 2.0
            while time.time() < deadline_sa:
                r = rng_sa.random()
                
                if r < 0.4:
                    idx = rng_sa.randint(0, n - 1)
                    delta = rng_sa.choice([-5, -3, -2, -1, 1, 2, 3, 5, -10, 10, -20, 20])
                    new_val = current_values[idx] + delta
                    if new_val < 1 or new_val > max_val or new_val in current_values:
                        continue
                    new_values = current_values.copy()
                    new_values[idx] = new_val
                    new_edges = current_edges
                elif r < 0.7:
                    i, j = rng_sa.sample(range(n), 2)
                    new_values = current_values.copy()
                    new_values[i], new_values[j] = new_values[j], new_values[i]
                    new_edges = current_edges
                else:
                    i = rng_sa.randint(0, len(current_edges) - 1)
                    old_u, old_v = current_edges[i]
                    candidates = [e for e in all_edges if e != (old_u, old_v) and e != (old_v, old_u) and e not in current_edges and tuple(reversed(e)) not in current_edges]
                    if not candidates:
                        continue
                    new_u, new_v = rng_sa.choice(candidates)
                    new_values = current_values
                    new_edges = list(current_edges)
                    new_edges[i] = (new_u, new_v)
                
                _, _, new_score = evaluate(new_values, new_edges)
                
                if new_score <= current_score:
                    current_values = new_values
                    current_edges = new_edges
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(new_values)
                        best_edges = list(new_edges)
                else:
                    delta = new_score - current_score
                    if rng_sa.random() < math.exp(-delta / T):
                        current_values = new_values
                        current_edges = new_edges
                        current_score = new_score
                
                T *= cooling
            
            if best_score < best_overall_score:
                best_overall_score = best_score
                best_result = {'values': list(best_values), 'edges': [list(e) for e in best_edges]}
    
    # Phase 3: Final polish with tighter search
    if best_result is not None and (time.time() - start_time) < deadline - 1.0:
        values = list(best_result['values'])
        edges = [list(e) for e in best_result['edges']]
        
        # Try more diverse value perturbations
        for _ in range(20):
            if (time.time() - start_time) > deadline - 1.0:
                break
            idx = rng.randint(0, n-1)
            old_val = values[idx]
            candidates = set()
            for delta in range(-20, 21):
                v = old_val + delta
                if 1 <= v <= max_val:
                    candidates.add(v)
            for _ in range(30):
                candidates.add(rng.randint(1, max_val))
            
            best_new_val = None
            best_new_score = evaluate(values, edges)[2]
            for new_val in sorted(candidates):
                if new_val < 1 or new_val > max_val or new_val in values:
                    continue
                values[idx] = new_val
                ns, np_, new_score = evaluate(values, edges)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
                else:
                    values[idx] = old_val
                    break
                values[idx] = old_val
            
            if best_new_val is not None:
                values[idx] = best_new_val
                if best_new_score < best_overall_score:
                    best_overall_score = best_new_score
                    best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    if best_result is None:
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in list(all_edges)[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
