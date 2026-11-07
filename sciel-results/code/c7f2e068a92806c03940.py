"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val = 10000
    
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)  # 91
    
    def fast_evaluate(values, edges):
        """Optimized evaluation."""
        vals = values
        s_set = set()
        p_set = set()
        for u, v in edges:
            s_set.add(vals[u] + vals[v])
            p_set.add(vals[u] * vals[v])
        return max(len(s_set), len(p_set))
    
    def evaluate_detailed(values, edges):
        """Returns (score, num_sums, num_prods)."""
        vals = values
        s_set = set()
        p_set = set()
        for u, v in edges:
            s_set.add(vals[u] + vals[v])
            p_set.add(vals[u] * vals[v])
        return max(len(s_set), len(p_set)), len(s_set), len(p_set)
    
    def greedy_edge_selection(values, rng, time_limit):
        """Pick edges greedily to minimize new distinct sums/products."""
        sums_seen = set()
        prods_seen = set()
        selected = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m):
            if (time.time() - start_time) > time_limit:
                break
            best_edge = None
            best_score = float('inf')
            for edge in remaining:
                u, v = edge
                s = values[u] + values[v]
                p = values[u] * values[v]
                score = 0
                if s not in sums_seen:
                    score += 1
                if p not in prods_seen:
                    score += 1
                if score < best_score:
                    best_score = score
                    best_edge = edge
                    if best_score == 0:
                        break
            if best_edge is None:
                break
            selected.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
            prods_seen.add(values[u] * values[v])
        
        return selected
    
    def local_search(values, edges, rng, time_limit):
        """
        Joint local search on values and edges.
        Alternates between perturbing values and swapping edges.
        """
        current_score = fast_evaluate(values, edges)
        val_set = set(values)
        edges_set = set(tuple(e) for e in edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Try value perturbations
            for _ in range(200):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n-1)
                old_val = values[idx]
                
                # Generate candidate values
                candidates = set()
                # Small perturbations
                for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10]:
                    candidates.add(old_val + delta)
                # Multiplicative perturbations
                for factor in [0.5, 0.75, 1.5, 2.0, 0.33, 3.0]:
                    candidates.add(int(old_val * factor))
                # Structured candidates
                for base in [2, 3, 4, 6, 8, 12, 16, 24, 32, 48]:
                    candidates.add(base * rng.randint(1, 200))
                # Random
                candidates.add(rng.randint(1, max_val))
                
                for new_val in candidates:
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in val_set:
                        continue
                    
                    values[idx] = new_val
                    val_set.discard(old_val)
                    val_set.add(new_val)
                    new_score = fast_evaluate(values, edges)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        values[idx] = old_val
                        val_set.discard(new_val)
                        val_set.add(old_val)
            
            if (time.time() - start_time) > time_limit:
                break
            
            # Try edge swaps
            for _ in range(300):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges)-1)
                new_edge = rng.choice(all_edges)
                new_edge_tuple = tuple(new_edge)
                if new_edge_tuple in edges_set:
                    continue
                
                old_edge = edges[i]
                old_edge_tuple = tuple(old_edge)
                
                edges[i] = new_edge
                edges_set.discard(old_edge_tuple)
                edges_set.add(new_edge_tuple)
                
                new_score = fast_evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(new_edge_tuple)
                    edges_set.add(old_edge_tuple)
        
        return values, edges, current_score
    
    def simulated_annealing(values, edges, rng, time_limit):
        """
        Simulated annealing that co-optimizes values and edges.
        """
        current_score = fast_evaluate(values, edges)
        val_set = set(values)
        edges_set = set(tuple(e) for e in edges)
        
        best_score = current_score
        best_values = list(values)
        best_edges = list(edges)
        
        T = 50.0
        T_min = 0.1
        cooling = 0.98
        
        while T > T_min and (time.time() - start_time) < time_limit:
            T *= cooling
            acceptance_threshold = T / 10.0
            
            # Generate a random move
            move_type = rng.random()
            
            if move_type < 0.4:
                # Perturb a value
                idx = rng.randint(0, n-1)
                old_val = values[idx]
                if rng.random() < 0.3:
                    new_val = old_val + rng.choice([-5, -3, -2, -1, 1, 2, 3, 5])
                elif rng.random() < 0.5:
                    new_val = int(old_val * rng.choice([0.5, 0.75, 1.5, 2.0, 0.33, 3.0]))
                elif rng.random() < 0.7:
                    base = rng.choice([2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
                    new_val = base * rng.randint(1, 200)
                else:
                    new_val = rng.randint(1, max_val)
                
                if new_val < 1 or new_val > max_val or new_val in val_set:
                    continue
                
                values[idx] = new_val
                val_set.discard(old_val)
                val_set.add(new_val)
                new_score = fast_evaluate(values, edges)
                
                delta = new_score - current_score
                if delta <= 0 or rng.random() < 0.5 ** (delta / (acceptance_threshold + 1e-10)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(values)
                        best_edges = list(edges)
                else:
                    values[idx] = old_val
                    val_set.discard(new_val)
                    val_set.add(old_val)
                    
            elif move_type < 0.7:
                # Swap an edge
                i = rng.randint(0, len(edges)-1)
                new_edge = rng.choice(all_edges)
                new_edge_tuple = tuple(new_edge)
                if new_edge_tuple in edges_set:
                    continue
                
                old_edge = edges[i]
                old_edge_tuple = tuple(old_edge)
                
                edges[i] = new_edge
                edges_set.discard(old_edge_tuple)
                edges_set.add(new_edge_tuple)
                
                new_score = fast_evaluate(values, edges)
                
                delta = new_score - current_score
                if delta <= 0 or rng.random() < 0.5 ** (delta / (acceptance_threshold + 1e-10)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(values)
                        best_edges = list(edges)
                else:
                    edges[i] = old_edge
                    edges_set.discard(new_edge_tuple)
                    edges_set.add(old_edge_tuple)
                    
            else:
                # Simultaneous value + edge change
                idx = rng.randint(0, n-1)
                old_val = values[idx]
                new_val = rng.randint(1, max_val)
                
                i = rng.randint(0, len(edges)-1)
                new_edge = rng.choice(all_edges)
                new_edge_tuple = tuple(new_edge)
                
                if new_val < 1 or new_val > max_val or new_val in val_set:
                    continue
                if new_edge_tuple in edges_set:
                    continue
                
                old_edge = edges[i]
                old_edge_tuple = tuple(old_edge)
                
                values[idx] = new_val
                val_set.discard(old_val)
                val_set.add(new_val)
                edges[i] = new_edge
                edges_set.discard(old_edge_tuple)
                edges_set.add(new_edge_tuple)
                
                new_score = fast_evaluate(values, edges)
                
                delta = new_score - current_score
                if delta <= 0 or rng.random() < 0.5 ** (delta / (acceptance_threshold + 1e-10)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(values)
                        best_edges = list(edges)
                else:
                    values[idx] = old_val
                    val_set.discard(new_val)
                    val_set.add(old_val)
                    edges[i] = old_edge
                    edges_set.discard(new_edge_tuple)
                    edges_set.add(old_edge_tuple)
            
            # Occasional restart from best
            if rng.random() < 0.01:
                values = list(best_values)
                edges = list(best_edges)
                current_score = best_score
                val_set = set(values)
                edges_set = set(tuple(e) for e in edges)
        
        return best_values, best_edges, best_score
    
    def find_edges_for_values(values, time_limit):
        """Find the best 35 edges for given values using greedy + local refinement."""
        # First pass: greedy
        rng = random.Random(hash(tuple(values)) % (2**32))
        edges = greedy_edge_selection(values, rng, time_limit)
        
        if len(edges) < m:
            # Fill remaining with random edges
            existing = set(tuple(e) for e in edges)
            for e in all_edges:
                if len(edges) >= m:
                    break
                if tuple(e) not in existing:
                    edges.append(e)
                    existing.add(tuple(e))
        
        # Second pass: local search on edges only
        edges_set = set(tuple(e) for e in edges)
        current_score = fast_evaluate(values, edges)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(500):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges)-1)
                new_edge = rng.choice(all_edges)
                new_edge_tuple = tuple(new_edge)
                if new_edge_tuple in edges_set:
                    continue
                
                old_edge = edges[i]
                old_edge_tuple = tuple(old_edge)
                
                edges[i] = new_edge
                edges_set.discard(old_edge_tuple)
                edges_set.add(new_edge_tuple)
                
                new_score = fast_evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(new_edge_tuple)
                    edges_set.add(old_edge_tuple)
        
        return edges, current_score
    
    # Curated initial value sets designed for high collision
    value_strategies = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58],
        [1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144],
        [2, 3, 4, 6, 12, 18, 24, 36, 48, 72, 96, 144, 192, 288],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
        [2, 4, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120],
        [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
        [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384],
        [2, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 8192, 9216],
        [3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 48, 60],
        [1, 2, 4, 6, 9, 12, 18, 24, 36, 48, 72, 96, 144, 192],
        [2, 3, 6, 9, 12, 18, 27, 36, 54, 72, 108, 144, 216, 288],
        [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512],
        [6, 12, 18, 24, 36, 48, 72, 96, 144, 192, 288, 384, 576, 768],
    ]
    
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Process curated strategies
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline - 3.0:
            break
        
        # Find optimal edges for these values
        edges, score = find_edges_for_values(init_values, deadline - 3.0)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(init_values), 'edges': edges}
        
        # Joint local search
        if (time.time() - start_time) < deadline - 3.0:
            values, edges2, score2 = local_search(list(init_values), edges, rng, deadline - 3.0)
            if score2 < best_overall_score:
                best_overall_score = score2
                best_result = {'values': list(values), 'edges': edges2}
        
        # Simulated annealing
        if (time.time() - start_time) < deadline - 5.0:
            values3, edges3, score3 = simulated_annealing(list(init_values), edges, rng, deadline - 5.0)
            if score3 < best_overall_score:
                best_overall_score = score3
                best_result = {'values': list(values3), 'edges': edges3}
    
    # Try additional random initializations
    num_random = 10
    for gen_idx in range(num_random):
        if (time.time() - start_time) > deadline - 5.0:
            break
        
        gen_rng = random.Random(1000 + gen_idx * 7919)
        
        # Generate values with structure
        strategy = gen_rng.randint(0, 3)
        if strategy == 0:
            # Arithmetic progression
            a = gen_rng.randint(1, 30)
            d = gen_rng.randint(1, 15)
            init_values = [a + i * d for i in range(n)]
        elif strategy == 1:
            # Geometric-like
            base = gen_rng.choice([2, 3, 4])
            init_values = []
            v = base
            for _ in range(n):
                if v > max_val:
                    v = gen_rng.randint(1, max_val)
                if v not in init_values:
                    init_values.append(v)
                v = v * gen_rng.choice([2, 3]) + gen_rng.randint(1, 10)
            while len(init_values) < n:
                v = gen_rng.randint(1, max_val)
                if v not in init_values:
                    init_values.append(v)
            init_values = init_values[:n]
        elif strategy == 2:
            # Multiples of a base
            base = gen_rng.choice([2, 3, 4, 6, 8, 12])
            init_values = []
            for _ in range(50):
                v = base * gen_rng.randint(1, max_val // base)
                if 1 <= v <= max_val and v not in init_values:
                    init_values.append(v)
                if len(init_values) >= n:
                    break
            while len(init_values) < n:
                v = gen_rng.randint(1, max_val)
                if v not in init_values:
                    init_values.append(v)
            init_values = init_values[:n]
        else:
            # Random small numbers
            init_values = []
            for _ in range(200):
                v = gen_rng.randint(1, 200)
                if v not in init_values:
                    init_values.append(v)
                if len(init_values) >= n:
                    break
            while len(init_values) < n:
                v = gen_rng.randint(1, max_val)
                if v not in init_values:
                    init_values.append(v)
            init_values = init_values[:n]
        
        # Ensure all distinct
        seen = set()
        final_values = []
        for v in init_values:
            if v not in seen and 1 <= v <= max_val:
                final_values.append(v)
                seen.add(v)
        while len(final_values) < n:
            v = gen_rng.randint(1, max_val)
            if v not in seen:
                final_values.append(v)
                seen.add(v)
        init_values = final_values[:n]
        
        # Find optimal edges
        edges, score = find_edges_for_values(init_values, deadline - 5.0)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(init_values), 'edges': edges}
        
        # Joint local search
        if (time.time() - start_time) < deadline - 5.0:
            values, edges2, score2 = local_search(list(init_values), edges, rng, deadline - 5.0)
            if score2 < best_overall_score:
                best_overall_score = score2
                best_result = {'values': list(values), 'edges': edges2}
        
        # Simulated annealing
        if (time.time() - start_time) < deadline - 8.0:
            values3, edges3, score3 = simulated_annealing(list(init_values), edges, rng, deadline - 8.0)
            if score3 < best_overall_score:
                best_overall_score = score3
                best_result = {'values': list(values3), 'edges': edges3}
    
    # Final polish on best result
    if best_result is not None and (time.time() - start_time) < deadline - 2.0:
        values = list(best_result['values'])
        edges = list(best_result['edges'])
        
        for _ in range(10):
            if (time.time() - start_time) > deadline - 2.0:
                break
            values, edges, s1 = local_search(values, edges, rng, deadline - 2.0)
            if s1 < best_overall_score:
                best_overall_score = s1
                best_result = {'values': list(values), 'edges': edges}
        
        # One more SA pass
        if (time.time() - start_time) < deadline - 2.0:
            values, edges, s2 = simulated_annealing(values, edges, rng, deadline - 2.0)
            if s2 < best_overall_score:
                best_overall_score = s2
                best_result = {'values': list(values), 'edges': edges}
    
    if best_result is None:
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in all_edges[:m]]
        }
    
    # Ensure types are correct
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
