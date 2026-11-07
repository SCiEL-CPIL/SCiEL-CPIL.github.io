"""Direct constructive approach for small sums and products along selected graph edges."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    max_val = 10000
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        ns = len(sums)
        np_ = len(prods)
        return ns, np_, max(ns, np_)
    
    def fast_evaluate_delta(values, edges, edge_changes):
        """Evaluate with only changed edges (for speed in SA)."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        ns = len(sums)
        np_ = len(prods)
        return ns, np_, max(ns, np_)
    
    def sa_co_optimize(rng, time_limit):
        """Simulated annealing co-optimizing values and edges simultaneously."""
        # Initialize with structured values
        base_values = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 30]
        values = list(base_values)
        rng.shuffle(values)
        
        # Initial edges: random selection
        edges = list(rng.sample(all_edges, m))
        edges_set = set(tuple(e) for e in edges)
        
        ns, np_, current_score = evaluate(values, edges)
        best_score = current_score
        best_values = list(values)
        best_edges = [list(e) for e in edges]
        
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        steps_per_iter = 1
        
        while time.time() < time_limit and T > T_min:
            for _ in range(steps_per_iter):
                if time.time() > time_limit:
                    break
                
                move_type = rng.random()
                
                if move_type < 0.4:
                    # Value move: perturb one value
                    idx = rng.randint(0, n - 1)
                    old_val = values[idx]
                    # Try structured candidates
                    r = rng.random()
                    if r < 0.3:
                        new_val = max(1, min(max_val, old_val + rng.randint(-10, 10)))
                    elif r < 0.5:
                        base = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 48, 60])
                        new_val = base * rng.randint(1, max(1, max_val // base))
                    elif r < 0.7:
                        # Power of small base
                        base = rng.choice([2, 3, 4, 5])
                        new_val = base ** rng.randint(1, 12)
                        if new_val > max_val:
                            new_val = rng.randint(1, max_val)
                    else:
                        new_val = rng.randint(1, max_val)
                    
                    if new_val == old_val or new_val in values:
                        continue
                    
                    values[idx] = new_val
                    ns, np_, new_score = evaluate(values, edges)
                    delta = new_score - current_score
                    
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_values = list(values)
                            best_edges = [list(e) for e in edges]
                    else:
                        values[idx] = old_val
                    
                elif move_type < 0.7:
                    # Edge move: swap one edge
                    i = rng.randint(0, m - 1)
                    new_edge = rng.choice(all_edges)
                    if tuple(new_edge) in edges_set:
                        continue
                    
                    old_edge = edges[i]
                    edges[i] = new_edge
                    edges_set.discard(tuple(old_edge))
                    edges_set.add(tuple(new_edge))
                    
                    ns, np_, new_score = evaluate(values, edges)
                    delta = new_score - current_score
                    
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_values = list(values)
                            best_edges = [list(e) for e in edges]
                    else:
                        edges[i] = old_edge
                        edges_set.discard(tuple(new_edge))
                        edges_set.add(tuple(old_edge))
                    
                else:
                    # Pair move: change two values simultaneously
                    idx1, idx2 = rng.sample(range(n), 2)
                    old_v1, old_v2 = values[idx1], values[idx2]
                    
                    r = rng.random()
                    if r < 0.5:
                        # Make them share a factor
                        common = rng.choice([2, 3, 4, 5, 6, 8, 10, 12])
                        values[idx1] = common * rng.randint(1, max(1, max_val // common))
                        values[idx2] = common * rng.randint(1, max(1, max_val // common))
                    else:
                        # Arithmetic progression pair
                        base = rng.randint(1, 200)
                        step = rng.randint(1, 50)
                        values[idx1] = base
                        values[idx2] = base + step
                    
                    if values[idx1] < 1 or values[idx1] > max_val:
                        values[idx1] = old_v1
                    if values[idx2] < 1 or values[idx2] > max_val:
                        values[idx2] = old_v2
                    if values[idx1] == values[idx2]:
                        values[idx1] = old_v1
                        values[idx2] = old_v2
                    
                    ns, np_, new_score = evaluate(values, edges)
                    delta = new_score - current_score
                    
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_values = list(values)
                            best_edges = [list(e) for e in edges]
                    else:
                        values[idx1] = old_v1
                        values[idx2] = old_v2
                
                T *= cooling
        
        return best_values, best_edges, best_score
    
    def local_polish(values, edges, rng, time_limit):
        """Fast local search to polish the SA result."""
        _, _, current_score = evaluate(values, edges)
        edges_set = set(tuple(e) for e in edges)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Value moves
            for _ in range(50):
                if time.time() > time_limit:
                    break
                idx = rng.randint(0, n - 1)
                old_val = values[idx]
                
                candidates = set()
                for d in range(-15, 16):
                    v = old_val + d
                    if 1 <= v <= max_val:
                        candidates.add(v)
                for base in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]:
                    for mult in range(1, min(50, max_val // base + 1)):
                        candidates.add(base * mult)
                
                best_new_val = None
                best_new_score = current_score
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
                    current_score = best_new_score
                    improved = True
            
            # Edge moves
            for _ in range(200):
                if time.time() > time_limit:
                    break
                i = rng.randint(0, m - 1)
                new_edge = rng.choice(all_edges)
                if tuple(new_edge) in edges_set:
                    continue
                
                old_edge = edges[i]
                edges[i] = new_edge
                edges_set.discard(tuple(old_edge))
                edges_set.add(tuple(new_edge))
                
                ns, np_, new_score = evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(tuple(new_edge))
                    edges_set.add(tuple(old_edge))
        
        return values, edges, current_score
    
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Multiple SA restarts with different initializations
    init_strategies = [
        lambda r: [r.choice([2,3,4,5,6,8,9,10,12,15,16,20,24,30]) for _ in range(n)],
        lambda r: sorted(r.sample(range(1, 200), n)),
        lambda r: [r.choice([2,3,4,5,6,8,9,10,12,15,16,20,24,30,36,48,60]) for _ in range(n)],
        lambda r: [2 ** r.randint(1, 13) if 2 ** r.randint(1,13) <= max_val else r.randint(1, max_val) for _ in range(n)],
        lambda r: [3 ** r.randint(1, 8) if 3 ** r.randint(1,8) <= max_val else r.randint(1, max_val) for _ in range(n)],
    ]
    
    num_restarts = 8
    time_per_restart = (deadline - start_time - 5.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Ensure distinct values
        init_fn = init_strategies[restart % len(init_strategies)]
        values = init_fn(rng)
        # Make distinct
        seen = set()
        for i in range(n):
            while values[i] in seen or values[i] < 1 or values[i] > max_val:
                values[i] = rng.randint(1, max_val)
            seen.add(values[i])
        
        edges = list(rng.sample(all_edges, m))
        
        # SA phase
        sa_time_limit = min(time.time() + time_per_restart * 0.7, deadline - 8.0)
        values, edges, score = sa_co_optimize(rng, sa_time_limit)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
        
        # Polish phase
        if time.time() < deadline - 3.0:
            polish_limit = min(time.time() + time_per_restart * 0.3, deadline - 3.0)
            values, edges, score = local_polish(values, edges, rng, polish_limit)
            
            if score < best_overall_score:
                best_overall_score = score
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    # Final global polish on best result
    if best_result is not None and time.time() < deadline - 2.0:
        values = list(best_result['values'])
        edges = [list(e) for e in best_result['edges']]
        values, edges, score = local_polish(values, edges, rng, deadline - 2.0)
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    if best_result is None:
        best_result = {
            'values': list(range(1, n + 1)),
            'edges': [list(e) for e in all_edges[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
