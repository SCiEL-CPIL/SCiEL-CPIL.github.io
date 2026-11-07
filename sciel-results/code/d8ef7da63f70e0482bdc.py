"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 14
    m = 35
    max_val = 10000
    all_pairs = list(combinations(range(n), 2))  # 91 total pairs

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def clique_search(values, target_k, rng, max_nodes_to_try=91):
        """
        Find a clique of size m in the compatibility graph where two edges
        are compatible if their sums and products can coexist within target_k
        distinct sums and target_k distinct products.
        
        Returns (edges, actual_score) or (None, None) if no clique found.
        """
        # Precompute sum and product for each pair
        pair_sums = {}
        pair_prods = {}
        for idx, (u, v) in enumerate(all_pairs):
            pair_sums[idx] = values[u] + values[v]
            pair_prods[idx] = values[u] * values[v]
        
        # For a given target_k, we need to find m edges such that
        # the number of distinct sums <= target_k AND distinct products <= target_k.
        # 
        # Strategy: Greedy construction with backtracking.
        # We build the edge set incrementally, tracking which sums and prods are used.
        # An edge can be added if it doesn't push us over target_k in either dimension.
        
        # First, try a greedy approach: sort edges by "cost" (how many new sums/prods they introduce)
        # and pick the cheapest m edges that are mutually compatible.
        
        # For each edge, compute its (sum, prod)
        # Two edges are compatible if we can have both without exceeding target_k distinct sums and prods.
        
        # Simpler approach: greedy build with feasibility check
        def greedy_build(values, target_k, order):
            """Build edge set greedily following given order."""
            selected = []
            sums = set()
            prods = set()
            
            for idx in order:
                if len(selected) >= m:
                    break
                s = pair_sums[idx]
                p = pair_prods[idx]
                
                # Check if adding this edge keeps us within target_k
                new_sum_count = len(sums) + (1 if s not in sums else 0)
                new_prod_count = len(prods) + (1 if p not in prods else 0)
                
                if new_sum_count <= target_k and new_prod_count <= target_k:
                    selected.append(idx)
                    sums.add(s)
                    prods.add(p)
            
            if len(selected) == m:
                edges = [list(all_pairs[idx]) for idx in selected]
                return edges, max(len(sums), len(prods))
            return None, None
        
        # Try multiple orderings
        # Ordering 1: by frequency of sum (most common sums first)
        from collections import Counter
        sum_counts = Counter(pair_sums.values())
        prod_counts = Counter(pair_prods.values())
        
        # Sort by: prefer edges whose sum appears many times, then whose product appears many times
        order1 = sorted(range(len(all_pairs)), key=lambda i: (-sum_counts[pair_sums[i]], -prod_counts[pair_prods[i]]))
        result = greedy_build(values, target_k, order1)
        if result[0] is not None:
            return result
        
        # Ordering 2: random shuffles
        for attempt in range(20):
            order = list(range(len(all_pairs)))
            rng.shuffle(order)
            result = greedy_build(values, target_k, order)
            if result[0] is not None:
                return result
        
        # Ordering 3: sorted by sum value (cluster similar sums)
        order3 = sorted(range(len(all_pairs)), key=lambda i: (pair_sums[i], pair_prods[i]))
        result = greedy_build(values, target_k, order3)
        if result[0] is not None:
            return result
        
        return None, None

    def local_search_edges(values, edges, rng, max_iters=500):
        """Local search: try swapping one edge for another to reduce distinct sums/products."""
        edges_set = set(tuple(sorted(e)) for e in edges)
        current_edges = [list(e) for e in edges]
        
        def compute_score(edgelist):
            sums = set()
            prods = set()
            for u, v in edgelist:
                sums.add(values[u] + values[v])
                prods.add(values[u] * values[v])
            return max(len(sums), len(prods))
        
        best_score = compute_score(current_edges)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            idx = rng.randint(0, m - 1)
            old_edge = tuple(sorted(current_edges[idx]))
            
            for _ in range(30):
                new_u, new_v = rng.choice(all_pairs)
                new_edge = tuple(sorted((new_u, new_v)))
                if new_edge in edges_set:
                    continue
                
                # Temporarily replace
                trial_edges = [list(e) for e in current_edges]
                trial_edges[idx] = [new_u, new_v]
                new_score = compute_score(trial_edges)
                
                if new_score < best_score:
                    current_edges[idx] = [new_u, new_v]
                    edges_set.discard(old_edge)
                    edges_set.add(new_edge)
                    best_score = new_score
                    improved = True
                    break
            
            if not improved:
                continue
        
        return current_edges, best_score

    def generate_structured_values(rng):
        """Generate values with shared structure to reduce product diversity."""
        strategy = rng.randint(0, 5)
        
        if strategy == 0:
            # Powers of 2
            base = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
            values = base[:13] + [rng.randint(1, 100)]
            values = sorted(set(values))
            while len(values) < 14:
                values.append(rng.randint(1, max_val))
                values = sorted(set(values))
            return values[:14]
        
        elif strategy == 1:
            # Multiples of a small number
            base_mult = rng.choice([2, 3, 4, 5, 6])
            values = []
            for i in range(1, 25):
                v = base_mult * i
                if v <= max_val and v not in values:
                    values.append(v)
                if len(values) == 14:
                    break
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]
        
        elif strategy == 2:
            # Values of form 2^a * 3^b
            values = set()
            for a in range(0, 8):
                for b in range(0, 5):
                    v = (2 ** a) * (3 ** b)
                    if 1 <= v <= max_val:
                        values.add(v)
            values = list(values)
            rng.shuffle(values)
            values = values[:14]
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]
        
        elif strategy == 3:
            # Small consecutive values
            start = rng.randint(1, 5)
            step = rng.randint(1, 3)
            values = []
            v = start
            while len(values) < 14 and v <= max_val:
                values.append(v)
                v += step
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]
        
        elif strategy == 4:
            # Geometric progression with small ratio
            base = rng.randint(2, 5)
            ratio = rng.randint(2, 4)
            values = []
            v = base
            while len(values) < 14 and v <= max_val:
                values.append(v)
                v *= ratio
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]
        
        else:
            # Random but biased toward small numbers
            values = []
            for _ in range(14):
                v = rng.randint(1, 150)
                if v not in values:
                    values.append(v)
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]

    def optimize_values(values, edges, rng, max_iters=150):
        """Try swapping pairs of values to improve score."""
        best_values = list(values)
        best_score = evaluate(best_values, edges)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            i, j = rng.sample(range(n), 2)
            new_values = list(best_values)
            new_values[i], new_values[j] = new_values[j], new_values[i]
            
            if len(set(new_values)) != 14:
                continue
            if any(v < 1 or v > max_val for v in new_values):
                continue
            
            new_score = evaluate(new_values, edges)
            if new_score < best_score:
                best_values = new_values
                best_score = new_score
        
        return best_values, best_score

    # Main search loop
    rng = random.Random(42)
    best_result = None
    best_score = float('inf')
    
    for restart in range(300):
        if time.time() > deadline:
            break
        
        values = generate_structured_values(rng)
        values = list(set(values))
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        values = values[:14]
        
        # Find lower bound for target_k
        # With 35 edges, minimum distinct sums is at least 1, but practically higher
        # Try target_k from a reasonable lower bound upward
        # Theoretical min: if all sums are the same, k=1 for sums. Same for products.
        # But with 14 distinct values, sums range from v[0]+v[1] to v[12]+v[13]
        
        # First, compute what target_k we need
        # Greedy: try increasing target_k until we find a valid edge set
        found = False
        for target_k in range(3, 40):
            if time.time() > deadline:
                break
            edges, score = clique_search(values, target_k, rng)
            if edges is not None:
                found = True
                # Local search to potentially improve
                edges, score = local_search_edges(values, edges, rng, max_iters=200)
                
                # Optimize values
                values, score = optimize_values(values, edges, rng, max_iters=80)
                
                # Re-run clique search with optimized values
                for target_k2 in range(3, 40):
                    if time.time() > deadline:
                        break
                    edges2, score2 = clique_search(values, target_k2, rng)
                    if edges2 is not None:
                        edges2, score2 = local_search_edges(values, edges2, rng, max_iters=150)
                        if score2 < score:
                            edges, score = edges2, score2
                        break
                
                if score < best_score:
                    best_score = score
                    best_result = (list(values), [list(e) for e in edges])
                    if best_score <= 10:
                        break
                break
        
        if best_score <= 10:
            break
    
    if best_result is None:
        # Fallback: use simple values and greedy edges
        values = list(range(1, 15))
        sums = set()
        prods = set()
        edges = []
        for u, v in all_pairs:
            if len(edges) >= m:
                break
            edges.append([u, v])
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
    else:
        values, edges = best_result
    
    # Final validation
    assert len(values) == 14
    assert len(set(values)) == 14
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in values)
    assert len(edges) == m
    edge_set = set()
    for e in edges:
        assert len(e) == 2
        assert e[0] != e[1]
        assert 0 <= e[0] < n and 0 <= e[1] < n
        key = tuple(sorted(e))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': list(values), 'edges': edges}
# EVOLVE-BLOCK-END
