"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def construct():
    deadline = time.time() + 55  # safety margin before 60s
    n = P['n']
    m = P['m']
    all_edges = list(combinations(range(n), 2))
    
    def fast_evaluate(values, edge_set):
        """Fast evaluation using sets for sums and products."""
        sums = set()
        prods = set()
        for i, j in edge_set:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    def greedy_initial(values):
        """Greedy edge selection to minimize distinct sums/products."""
        edges = []
        edge_set = set()
        sums = set()
        prods = set()
        # Sort all edges by product (smaller products tend to collide more)
        sorted_edges = sorted(all_edges, key=lambda e: values[e[0]] * values[e[1]])
        for e in sorted_edges:
            i, j = e
            s = values[i] + values[j]
            p = values[i] * values[j]
            # Prefer edges where both sum and product are already seen
            new_s = 1 if s not in sums else 0
            new_p = 1 if p not in prods else 0
            if new_s + new_p <= 1:  # At most one new value
                edges.append([i, j])
                edge_set.add(e)
                sums.add(s)
                prods.add(p)
                if len(edges) == m:
                    break
        # Fill remaining with any edges
        if len(edges) < m:
            for e in all_edges:
                if e not in edge_set:
                    edges.append(list(e))
                    edge_set.add(e)
                    if len(edges) == m:
                        break
        return edges, edge_set
    
    def simulated_annealing(values, edges, edge_set, max_iters=100000):
        """SA over both values and edges with fast delta evaluation."""
        current_values = list(values)
        current_edges = [list(e) for e in edges]
        current_edge_set = set(edge_set)
        
        # Precompute current sums and products
        cur_sums = set()
        cur_prods = set()
        for i, j in current_edge_set:
            cur_sums.add(current_values[i] + current_values[j])
            cur_prods.add(current_values[i] * current_values[j])
        cur_score = max(len(cur_sums), len(cur_prods))
        
        best_score = cur_score
        best_values = list(current_values)
        best_edges = [list(e) for e in current_edges]
        best_edge_set = set(current_edge_set)
        
        T = 10.0
        cooling = 0.99995
        min_T = 0.01
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if T < min_T:
                break
            
            # Choose move type: 60% edge swap, 40% value change
            if random.random() < 0.6:
                # Edge swap: remove one edge, add another
                idx = random.randint(0, m - 1)
                old_i, old_j = current_edges[idx]
                old_edge = (old_i, old_j)
                
                # Pick a random edge not in current set
                new_edge = random.choice(all_edges)
                if new_edge in current_edge_set:
                    continue
                new_i, new_j = new_edge
                
                # Delta evaluation
                old_s = current_values[old_i] + current_values[old_j]
                old_p = current_values[old_i] * current_values[old_j]
                new_s = current_values[new_i] + current_values[new_j]
                new_p = current_values[new_i] * current_values[new_j]
                
                # Count how many edges have this sum/product
                s_count = {}
                p_count = {}
                for ei, ej in current_edge_set:
                    si = current_values[ei] + current_values[ej]
                    pi = current_values[ei] * current_values[ej]
                    s_count[si] = s_count.get(si, 0) + 1
                    p_count[pi] = p_count.get(pi, 0) + 1
                
                # Remove old edge
                s_count[old_s] -= 1
                p_count[old_p] -= 1
                if s_count[old_s] == 0:
                    s_count.pop(old_s)
                if p_count[old_p] == 0:
                    p_count.pop(old_p)
                
                # Add new edge
                s_count[new_s] = s_count.get(new_s, 0) + 1
                p_count[new_p] = p_count.get(new_p, 0) + 1
                
                new_score = max(len(s_count), len(p_count))
                
                # Accept or reject
                if new_score < cur_score or random.random() < min(1.0, ((cur_score - new_score) / T) if new_score > cur_score else 1.0):
                    # Accept
                    current_edges[idx] = [new_i, new_j]
                    current_edge_set.remove(old_edge)
                    current_edge_set.add(new_edge)
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_values = list(current_values)
                        best_edges = [list(e) for e in current_edges]
                        best_edge_set = set(current_edge_set)
                else:
                    # Reject
                    pass
            else:
                # Value change: pick a vertex and change its value
                v_idx = random.randint(0, n - 1)
                old_val = current_values[v_idx]
                # Generate new value: nearby or random
                if random.random() < 0.5:
                    new_val = max(1, min(P['maximum'], old_val + random.randint(-5, 5)))
                else:
                    new_val = random.randint(1, P['maximum'])
                if new_val == old_val:
                    continue
                # Check distinctness
                if new_val in current_values:
                    continue
                current_values[v_idx] = new_val
                
                # Delta evaluation for value change
                # Find all edges incident to v_idx
                incident = []
                for i, j in current_edge_set:
                    if i == v_idx or j == v_idx:
                        incident.append((i, j))
                
                # Compute old and new sums/products for incident edges
                old_sums_prods = []
                new_sums_prods = []
                for i, j in incident:
                    other = j if i == v_idx else i
                    old_s = old_val + current_values[other]
                    old_p = old_val * current_values[other]
                    new_s = new_val + current_values[other]
                    new_p = new_val * current_values[other]
                    old_sums_prods.append((old_s, old_p))
                    new_sums_prods.append((new_s, new_p))
                
                # Build count maps excluding incident edges
                s_count = {}
                p_count = {}
                for ei, ej in current_edge_set:
                    if ei == v_idx or ej == v_idx:
                        continue
                    si = current_values[ei] + current_values[ej]
                    pi = current_values[ei] * current_values[ej]
                    s_count[si] = s_count.get(si, 0) + 1
                    p_count[pi] = p_count.get(pi, 0) + 1
                
                # Add back old incident contributions
                for os, op in old_sums_prods:
                    s_count[os] = s_count.get(os, 0) + 1
                    p_count[op] = p_count.get(op, 0) + 1
                
                # Remove old incident contributions
                for os, op in old_sums_prods:
                    s_count[os] -= 1
                    p_count[op] -= 1
                    if s_count[os] == 0:
                        s_count.pop(os)
                    if p_count[op] == 0:
                        p_count.pop(op)
                
                # Add new incident contributions
                for ns, np in new_sums_prods:
                    s_count[ns] = s_count.get(ns, 0) + 1
                    p_count[np] = p_count.get(np, 0) + 1
                
                new_score = max(len(s_count), len(p_count))
                
                # Accept or reject
                if new_score < cur_score or random.random() < min(1.0, ((cur_score - new_score) / T) if new_score > cur_score else 1.0):
                    # Accept
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_values = list(current_values)
                        best_edges = [list(e) for e in current_edges]
                        best_edge_set = set(current_edge_set)
                else:
                    # Reject
                    current_values[v_idx] = old_val
            
            T *= cooling
        
        return best_values, best_edges, best_score
    
    # Initialize with greedy values and edges
    # Try a few value initializations
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    value_initializations = [
        list(range(1, 15)),  # 1..14
        [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610],  # Fibonacci-like
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],  # Powers of 2 (too large for products)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],  # Primes
    ]
    
    for init_vals in value_initializations:
        if time.time() > deadline - 5:
            break
        # Ensure distinct and <= 10000
        if len(set(init_vals)) != n or any(v < 1 or v > P['maximum'] for v in init_vals):
            continue
        
        edges, edge_set = greedy_initial(init_vals)
        vals, edges_sa, score_sa = simulated_annealing(init_vals, edges, edge_set, max_iters=50000)
        
        if score_sa < best_overall_score:
            best_overall_score = score_sa
            best_overall_values = vals
            best_overall_edges = edges_sa
    
    # Final local refinement on best
    if best_overall_edges is not None:
        # Quick local search on edges only
        best_vals = best_overall_values
        best_edges = [list(e) for e in best_overall_edges]
        best_edge_set = set(tuple(e) for e in best_edges)
        
        for _ in range(20000):
            if time.time() > deadline - 2:
                break
            idx = random.randint(0, m - 1)
            old_i, old_j = best_edges[idx]
            old_edge = (old_i, old_j)
            new_edge = random.choice(all_edges)
            if new_edge in best_edge_set:
                continue
            new_i, new_j = new_edge
            
            # Evaluate
            cur_sums = set()
            cur_prods = set()
            for ei, ej in best_edge_set:
                cur_sums.add(best_vals[ei] + best_vals[ej])
                cur_prods.add(best_vals[ei] * best_vals[ej])
            cur_score = max(len(cur_sums), len(cur_prods))
            
            # Try swap
            best_edge_set.remove(old_edge)
            best_edge_set.add(new_edge)
            new_sums = set()
            new_prods = set()
            for ei, ej in best_edge_set:
                new_sums.add(best_vals[ei] + best_vals[ej])
                new_prods.add(best_vals[ei] * best_vals[ej])
            new_score = max(len(new_sums), len(new_prods))
            
            if new_score < cur_score:
                best_edges[idx] = [new_i, new_j]
                best_overall_score = new_score
            else:
                best_edge_set.remove(new_edge)
                best_edge_set.add(old_edge)
    
    # Ensure valid output
    final_values = [int(v) for v in best_overall_values]
    final_edges = [[int(i), int(j)] for i, j in best_overall_edges]
    
    # Verify constraints
    assert len(final_values) == n
    assert len(set(final_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= P['maximum'] for v in final_values)
    assert len(final_edges) == m
    edge_tuples = set()
    for e in final_edges:
        assert len(e) == 2
        i, j = e
        assert 0 <= i < n and 0 <= j < n
        assert i != j
        t = tuple(sorted((i, j)))
        assert t not in edge_tuples
        edge_tuples.add(t)
    
    return {'values': final_values, 'edges': final_edges}

# EVOLVE-BLOCK-END
