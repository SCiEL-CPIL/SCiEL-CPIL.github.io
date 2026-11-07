"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val = 10000
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def greedy_edges(values, all_edges, rng):
        """Pick edges greedily to maximize reuse of existing sums/products."""
        sums_seen = set()
        prods_seen = set()
        selected = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            for edge in remaining:
                u, v = edge
                s = values[u] + values[v]
                p = values[u] * values[v]
                # Score: prefer edges that reuse existing sums/products
                score = (0 if s in sums_seen else 1) + (0 if p in prods_seen else 1)
                if score < best_score:
                    best_score = score
                    best_edge = edge
                elif score == best_score and best_edge is not None:
                    # Tie-break: prefer smaller new sum/product
                    s1, p1 = values[best_edge[0]] + values[best_edge[1]], values[best_edge[0]] * values[best_edge[1]]
                    if (s not in sums_seen and p not in prods_seen):
                        if (s, p) < (s1, p1):
                            best_edge = edge
            if best_edge is None:
                break
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
            prods_seen.add(values[u] * values[v])
            selected.append(best_edge)
            remaining.remove(best_edge)
        
        return selected
    
    def generate_structured_values(rng):
        """Generate values with shared structure to create collisions."""
        strategies = [
            lambda: sorted(set([2**i for i in range(4, 12)] + [3**i for i in range(1, 8)] + [rng.randint(1, max_val) for _ in range(14)]))[:14],
            lambda: sorted(set([i * 7 for i in range(1, 15)] + [i * 11 for i in range(1, 10)] + [rng.randint(1, max_val) for _ in range(14)]))[:14],
            lambda: sorted(set([2**i * 3**j for i in range(5) for j in range(4)] + [rng.randint(1, max_val) for _ in range(14)]))[:14],
            lambda: sorted(set([i * 6 for i in range(1, 15)] + [i * 10 for i in range(1, 11)] + [rng.randint(1, max_val) for _ in range(14)]))[:14],
        ]
        strat = rng.choice(strategies)
        vals = strat()
        # Ensure distinct and within bounds
        vals = sorted(set(v for v in vals if 1 <= v <= max_val))
        while len(vals) < n:
            vals.append(rng.randint(1, max_val))
        vals = list(dict.fromkeys(vals))[:n]
        return sorted(vals)
    
    def local_search(values, edges, rng, max_iters=500):
        """Improve via value swaps and edge adjustments."""
        best_score = evaluate(values, edges)
        all_edges = list(combinations(range(n), 2))
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try swapping two values
            i, j = rng.sample(range(n), 2)
            new_values = values[:]
            new_values[i], new_values[j] = new_values[j], new_values[i]
            if len(set(new_values)) == n and all(1 <= v <= max_val for v in new_values):
                score = evaluate(new_values, edges)
                if score < best_score:
                    values = new_values
                    best_score = score
                    improved = True
            
            # Try replacing an edge
            if not improved and edges:
                old_edge = edges[rng.randrange(len(edges))]
                candidates = [e for e in all_edges if e not in edges]
                if candidates:
                    new_edge = rng.choice(candidates)
                    new_edges = edges[:]
                    new_edges.remove(old_edge)
                    new_edges.append(new_edge)
                    score = evaluate(values, new_edges)
                    if score < best_score:
                        edges = new_edges
                        best_score = score
                        improved = True
            
            if not improved:
                # Try greedy re-selection of some edges
                if rng.random() < 0.3:
                    new_edges = greedy_edges(values, all_edges, rng)
                    score = evaluate(values, new_edges)
                    if score < best_score:
                        edges = new_edges
                        best_score = score
        
        return values, edges, best_score
    
    # Main search
    rng = random.Random(42)
    all_edges = list(combinations(range(n), 2))
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Try multiple initial value configurations
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < 20:
        attempts += 1
        values = generate_structured_values(rng)
        edges = greedy_edges(values, all_edges, rng)
        score = evaluate(values, edges)
        
        if score < best_score:
            best_values = values[:]
            best_edges = edges[:]
            best_score = score
        
        # Local search from this starting point
        if time.time() < deadline - 15.0:
            vals, eds, sc = local_search(values, edges, rng, max_iters=300)
            if sc < best_score:
                best_values = vals
                best_edges = eds
                best_score = sc
    
    # Final local search on best solution
    if time.time() < deadline - 5.0 and best_values is not None:
        vals, eds, sc = local_search(best_values, best_edges, rng, max_iters=500)
        if sc < best_score:
            best_values = vals
            best_edges = eds
            best_score = sc
    
    # Ensure valid output
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in all_edges[:m]]
    
    # Convert to proper types
    best_values = [int(v) for v in best_values]
    best_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
