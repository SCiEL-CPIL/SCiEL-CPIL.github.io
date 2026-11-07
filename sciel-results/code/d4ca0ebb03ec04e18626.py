"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

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

    def generate_multiplicative_structure(rng):
        """
        Generate values with strong multiplicative structure.
        Key idea: use values of form a * 2^i * 3^j so that products
        cluster into few distinct values.
        """
        strategy = rng.randint(0, 3)
        
        if strategy == 0:
            # Small base times powers of 2 and 3
            bases = [1, 2, 3, 5, 7, 11, 13]
            candidates = set()
            for base in bases:
                for a in range(0, 6):
                    for b in range(0, 4):
                        v = base * (2 ** a) * (3 ** b)
                        if 1 <= v <= max_val:
                            candidates.add(v)
            candidates = list(candidates)
            rng.shuffle(candidates)
            values = candidates[:14]
        
        elif strategy == 1:
            # All multiples of a small number, spread out
            base_mult = rng.choice([2, 3, 4, 5, 6, 8, 10, 12])
            values = []
            for i in range(1, 50):
                v = base_mult * i
                if v <= max_val and v not in values:
                    values.append(v)
                if len(values) == 14:
                    break
            # Shuffle to avoid consecutive patterns
            rng.shuffle(values)
        
        elif strategy == 2:
            # Mix of small numbers with shared factors
            # Use numbers that are products of small primes
            small_primes = [2, 3, 5, 7]
            candidates = set()
            for p1 in small_primes:
                for p2 in small_primes:
                    for p3 in small_primes:
                        v = p1 * p2 * p3
                        if 1 <= v <= max_val:
                            candidates.add(v)
            # Add some single primes and prime powers
            for p in small_primes:
                for exp in range(1, 8):
                    v = p ** exp
                    if 1 <= v <= max_val:
                        candidates.add(v)
            candidates = list(candidates)
            rng.shuffle(candidates)
            values = candidates[:14]
        
        else:
            # Geometric progression with small ratio
            ratio = rng.choice([2, 3])
            start_val = rng.randint(1, 5)
            values = []
            v = start_val
            while len(values) < 14 and v <= max_val:
                values.append(v)
                v *= ratio
            # If not enough, fill with multiples
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
        
        # Ensure distinct and valid
        values = list(set(values))
        values = [v for v in values if 1 <= v <= max_val]
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        values = values[:14]
        
        return values

    def systematic_edge_selection(values):
        """
        Systematic edge selection: group all possible edges by their (sum, product) pair.
        Then select edges to maximize coverage of few distinct (sum, product) groups.
        This is a set cover / greedy approach.
        """
        all_pairs = list(combinations(range(n), 2))
        
        # Compute sum and product for each pair
        pair_props = {}
        for u, v in all_pairs:
            s = values[u] + values[v]
            p = values[u] * values[v]
            pair_props[(u, v)] = (s, p)
        
        # Group edges by their (sum, product) signature
        # We want to select 35 edges that use as few distinct signatures as possible
        # Greedy: repeatedly pick the signature that covers the most new edges
        # while keeping the number of distinct signatures low
        
        # First, let's see how many distinct (sum, product) pairs exist
        sig_to_edges = {}
        for (u, v) in all_pairs:
            sig = pair_props[(u, v)]
            if sig not in sig_to_edges:
                sig_to_edges[sig] = []
            sig_to_edges[sig].append((u, v))
        
        # Sort signatures by number of edges they cover (descending)
        # This helps us pick signatures that cover many edges with few distinct values
        sig_list = sorted(sig_to_edges.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Greedy selection: pick signatures in order of coverage, but we need exactly 35 edges
        # We can pick from the top signatures until we have 35 edges
        selected_edges = []
        used_sigs = set()
        
        # First pass: take all edges from the top signatures
        for sig, edges in sig_list:
            if len(selected_edges) >= m:
                break
            for e in edges:
                if len(selected_edges) >= m:
                    break
                if e not in selected_edges:
                    selected_edges.append(e)
                    used_sigs.add(sig)
        
        # If we don't have enough, add more from remaining
        if len(selected_edges) < m:
            selected_set = set(selected_edges)
            for sig, edges in sig_list:
                if len(selected_edges) >= m:
                    break
                for e in edges:
                    if len(selected_edges) >= m:
                        break
                    if e not in selected_set:
                        selected_edges.append(e)
                        selected_set.add(e)
                        used_sigs.add(sig)
        
        # Now local search to improve
        edges = selected_edges[:m]
        
        return edges

    def local_search_edges_improved(values, edges, rng, max_iters=2000):
        """
        Improved local search: try swapping edges to reduce distinct sums and products.
        Uses efficient delta computation.
        """
        edges_set = set(tuple(sorted(e)) for e in edges)
        current_edges = [list(e) for e in edges]
        
        # Precompute sum and product for each edge
        edge_props = {}
        for u, v in current_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_props[(u, v)] = (s, p)
        
        # Count occurrences of each sum and product
        sum_counts = {}
        prod_counts = {}
        for u, v in current_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        distinct_sums = len(sum_counts)
        distinct_prods = len(prod_counts)
        best_score = max(distinct_sums, distinct_prods)
        
        all_pairs = list(combinations(range(n), 2))
        non_edges = [p for p in all_pairs if tuple(sorted(p)) not in edges_set]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try removing a random edge and adding a random non-edge
            idx = rng.randint(0, m - 1)
            old_u, old_v = current_edges[idx]
            old_s = values[old_u] + values[old_v]
            old_p = values[old_u] * values[old_v]
            
            # Try several candidate replacements
            candidates = rng.sample(non_edges, min(30, len(non_edges)))
            
            best_new_score = best_score
            best_new_edge = None
            best_new_sums = None
            best_new_prods = None
            best_new_sum_counts = None
            best_new_prod_counts = None
            
            for new_u, new_v in candidates:
                if (new_u, new_v) in edges_set or (new_v, new_u) in edges_set:
                    continue
                
                new_s = values[new_u] + values[new_v]
                new_p = values[new_u] * values[new_v]
                
                # Simulate removal and addition
                new_sum_counts = dict(sum_counts)
                new_prod_counts = dict(prod_counts)
                
                # Remove old edge
                new_sum_counts[old_s] -= 1
                if new_sum_counts[old_s] == 0:
                    del new_sum_counts[old_s]
                new_prod_counts[old_p] -= 1
                if new_prod_counts[old_p] == 0:
                    del new_prod_counts[old_p]
                
                # Add new edge
                new_sum_counts[new_s] = new_sum_counts.get(new_s, 0) + 1
                new_prod_counts[new_p] = new_prod_counts.get(new_p, 0) + 1
                
                new_distinct_sums = len(new_sum_counts)
                new_distinct_prods = len(new_prod_counts)
                new_score = max(new_distinct_sums, new_distinct_prods)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_edge = (new_u, new_v)
                    best_new_sums = set(new_sum_counts.keys())
                    best_new_prods = set(new_prod_counts.keys())
                    best_new_sum_counts = new_sum_counts
                    best_new_prod_counts = new_prod_counts
            
            if best_new_edge is not None:
                # Apply the improvement
                current_edges[idx] = list(best_new_edge)
                edges_set.discard(tuple(sorted((old_u, old_v))))
                edges_set.add(tuple(sorted(best_new_edge)))
                
                # Update non_edges list
                non_edges.remove(tuple(sorted(best_new_edge)))
                non_edges.append(tuple(sorted((old_u, old_v))))
                
                sum_counts = best_new_sum_counts
                prod_counts = best_new_prod_counts
                distinct_sums = len(sum_counts)
                distinct_prods = len(prod_counts)
                best_score = best_new_score
                improved = True
        
        return current_edges, best_score

    def optimize_values_via_replacement(values, edges, rng, max_iters=500):
        """
        Try replacing individual values with other values to improve score.
        """
        best_values = list(values)
        best_score = evaluate(best_values, edges)
        
        # Generate candidate replacement values
        candidates = set()
        for _ in range(200):
            v = rng.randint(1, max_val)
            candidates.add(v)
        # Also add values near existing ones
        for v in values:
            for delta in [-3, -2, -1, 1, 2, 3]:
                nv = v + delta
                if 1 <= nv <= max_val:
                    candidates.add(nv)
        
        candidates = list(candidates)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            i = rng.randint(0, n - 1)
            old_val = best_values[i]
            
            # Try replacing with a candidate
            for new_val in rng.sample(candidates, min(20, len(candidates))):
                if new_val == old_val or new_val in best_values:
                    continue
                
                new_values = list(best_values)
                new_values[i] = new_val
                
                new_score = evaluate(new_values, edges)
                if new_score < best_score:
                    best_values = new_values
                    best_score = new_score
                    break
        
        return best_values, best_score

    # Main search loop
    rng = random.Random(42)
    best_result = None
    best_score = float('inf')
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Generate structured values
        values = generate_multiplicative_structure(rng)
        
        # Systematic edge selection
        edges = systematic_edge_selection(values)
        
        # Local search on edges
        edges, score = local_search_edges_improved(values, edges, rng, max_iters=1000)
        
        # Optimize values
        values, score = optimize_values_via_replacement(values, edges, rng, max_iters=300)
        
        # Re-optimize edges after value changes
        edges = systematic_edge_selection(values)
        edges, score = local_search_edges_improved(values, edges, rng, max_iters=500)
        
        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])
            if best_score <= 10:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
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
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
