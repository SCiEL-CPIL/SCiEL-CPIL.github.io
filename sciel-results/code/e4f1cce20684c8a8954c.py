"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Conservative safety margin
    
    n = P['n']
    m = P['m']
    max_val = P['maximum']
    
    all_pairs = list(itertools.combinations(range(n), 2))
    
    def get_score(vals, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))
    
    def find_best_edges(vals):
        """Greedy edge selection: pick edges that reuse existing sums/products first."""
        # Precompute all pair data
        pair_sums = {}
        pair_prods = {}
        for i, j in all_pairs:
            pair_sums[(i, j)] = vals[i] + vals[j]
            pair_prods[(i, j)] = vals[i] * vals[j]
        
        # Sort pairs by (sum_freq, prod_freq) descending, then by sum, prod for determinism
        sum_freq = Counter(pair_sums.values())
        prod_freq = Counter(pair_prods.values())
        
        scored_pairs = []
        for i, j in all_pairs:
            s = pair_sums[(i, j)]
            p = pair_prods[(i, j)]
            # Higher frequency = more likely to be reused
            score = (sum_freq[s], prod_freq[p], -s, -p)
            scored_pairs.append((score, i, j))
        
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        
        selected = []
        selected_set = set()
        seen_sums = set()
        seen_prods = set()
        
        for score, i, j in scored_pairs:
            if len(selected) >= m:
                break
            key = (min(i, j), max(i, j))
            if key in selected_set:
                continue
            selected.append([i, j])
            selected_set.add(key)
            seen_sums.add(pair_sums[(i, j)])
            seen_prods.add(pair_prods[(i, j)])
        
        return selected
    
    def local_search_edges(vals, edges, max_iters=50):
        """Improve edges by swapping with non-edges."""
        edges = [e[:] for e in edges]
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        non_edges = [(i, j) for i, j in all_pairs if (min(i, j), max(i, j)) not in edge_set]
        
        current_score = get_score(vals, edges)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try random swaps
            for _ in range(10):
                e_idx = random.randint(0, m - 1)
                new_edge = random.choice(non_edges)
                
                # Check if new edge is already in edges
                if (min(new_edge[0], new_edge[1]), max(new_edge[0], new_edge[1])) in edge_set:
                    continue
                
                temp_edges = edges[:]
                temp_edges[e_idx] = list(new_edge)
                temp_score = get_score(vals, temp_edges)
                
                if temp_score < current_score:
                    # Remove old edge from edge_set, add new
                    old_e = edges[e_idx]
                    edge_set.discard((min(old_e[0], old_e[1]), max(old_e[0], old_e[1])))
                    edge_set.add((min(new_edge[0], new_edge[1]), max(new_edge[0], new_edge[1])))
                    # Update non_edges
                    non_edges.remove(new_edge)
                    non_edges.append(tuple(old_e))
                    
                    edges = temp_edges
                    current_score = temp_score
                    improved = True
                    break
            
            if not improved:
                break
        
        return edges, current_score
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Generate diverse initial value sets
    def gen_initial_values(restart_idx):
        """Generate structured value sets with high collision potential."""
        strategies = [
            # Arithmetic progression
            lambda: sorted(random.sample(range(1, 50), n)),
            # Small numbers with many divisors
            lambda: sorted(random.sample([1,2,3,4,5,6,7,8,9,10,12,15,16,18,20,24,25,27,28,30,32,35,36,40,42,45,48,49,50,54,56,60,63,64,70,72,75,80,81,84,90,96,98,99,100], n)),
            # Mix of very small and small
            lambda: sorted(random.sample(range(1, 30), min(n, 12)) + random.sample(range(30, 80), n - min(n, 12))),
            # Geometric-like
            lambda: sorted(set([int(2**i * random.uniform(0.8, 1.2)) for i in range(n)]))[:n] if len(set([int(2**i * random.uniform(0.8, 1.2)) for i in range(n)])) >= n else sorted(random.sample(range(1, 100), n)),
            # Random small
            lambda: sorted(random.sample(range(1, 60), n)),
        ]
        strat = strategies[restart_idx % len(strategies)]
        vals = strat()
        # Ensure distinct
        if len(set(vals)) < n:
            vals = sorted(random.sample(range(1, 100), n))
        return vals
    
    # Multiple restarts
    for restart in range(30):
        if time.time() > deadline:
            break
        
        current_values = gen_initial_values(restart)
        
        # Find optimal edges
        current_edges = find_best_edges(current_values)
        current_score = get_score(current_values, current_edges)
        
        # Refine edges
        current_edges, current_score = local_search_edges(current_values, current_edges, max_iters=30)
        
        if current_score < best_score:
            best_score = current_score
            best_values = current_values[:]
            best_edges = [e[:] for e in current_edges]
        
        # Local search on values
        max_val_iters = 200
        for _ in range(max_val_iters):
            if time.time() > deadline:
                break
            
            # Try replacing one value with a nearby value
            idx = random.randint(0, n - 1)
            old_val = current_values[idx]
            
            found_improvement = False
            # Try several candidate replacements
            candidates_to_try = set()
            # Neighbors
            for delta in range(-10, 11):
                nv = old_val + delta
                if 1 <= nv <= max_val and nv != old_val:
                    candidates_to_try.add(nv)
            # Random small values
            for _ in range(5):
                nv = random.randint(1, 50)
                if nv != old_val:
                    candidates_to_try.add(nv)
            
            for new_val in list(candidates_to_try)[:10]:
                if new_val in current_values:
                    continue
                
                temp_vals = current_values[:]
                temp_vals[idx] = new_val
                temp_vals.sort()
                
                # Re-find best edges
                temp_edges = find_best_edges(temp_vals)
                temp_score = get_score(temp_vals, temp_edges)
                
                if temp_score < current_score:
                    current_values = temp_vals
                    current_edges = temp_edges
                    current_score = temp_score
                    found_improvement = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_values = current_values[:]
                        best_edges = [e[:] for e in current_edges]
                    break
            
            if not found_improvement:
                # Try edge local search
                current_edges, current_score = local_search_edges(current_values, current_edges, max_iters=20)
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = current_values[:]
                    best_edges = [e[:] for e in current_edges]
    
    # Final type safety
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = find_best_edges(best_values)
    
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
