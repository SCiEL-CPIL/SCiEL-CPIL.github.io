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

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def clique_search(adj_matrix, target_size, rng, max_nodes=91):
        """
        Maximum clique via branch-and-bound.
        adj_matrix: list of bitmasks (one per node)
        target_size: we want a clique of at least this size
        Returns a list of node indices forming a clique, or None.
        """
        num_nodes = len(adj_matrix)
        if num_nodes < target_size:
            return None

        # Convert to list of int bitmasks for speed
        adj = [0] * num_nodes
        for i in range(num_nodes):
            mask = 0
            row = adj_matrix[i]
            for j in range(num_nodes):
                if row[j]:
                    mask |= (1 << j)
            adj[i] = mask

        best_clique = []
        best_size = 0

        def popcount(x):
            return bin(x).count('1')

        def expand(clique, candidates):
            nonlocal best_clique, best_size
            # Prune: even if we add all candidates, can't beat best
            if len(clique) + popcount(candidates) <= best_size:
                return
            if len(clique) >= target_size:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = list(clique)
                return

            # Try nodes in candidates, ordered by degree (heuristic)
            while candidates:
                # Pick node with highest degree within candidates
                best_node = -1
                best_deg = -1
                temp = candidates
                while temp:
                    lb = temp & (-temp)
                    node = lb.bit_length() - 1
                    deg = popcount(adj[node] & candidates)
                    if deg > best_deg:
                        best_deg = deg
                        best_node = node
                    temp ^= lb

                if best_node == -1:
                    break

                # Remove best_node from candidates
                candidates &= ~(1 << best_node)

                # Try adding best_node
                new_clique = clique + [best_node]
                new_candidates = candidates & adj[best_node]

                if len(new_clique) + popcount(new_candidates) > best_size:
                    expand(new_clique, new_candidates)

                # Prune: if even with all remaining candidates we can't beat best
                if len(clique) + popcount(candidates) <= best_size:
                    return

        # Initial call with all nodes as candidates
        all_candidates = (1 << num_nodes) - 1
        expand([], all_candidates)

        return best_clique if best_size >= target_size else None

    def max_clique_size(adj_matrix, max_nodes=91):
        """Find maximum clique size using branch-and-bound."""
        num_nodes = len(adj_matrix)
        if num_nodes == 0:
            return 0

        adj = [0] * num_nodes
        for i in range(num_nodes):
            mask = 0
            row = adj_matrix[i]
            for j in range(num_nodes):
                if row[j]:
                    mask |= (1 << j)
            adj[i] = mask

        best_size = 0

        def popcount(x):
            return bin(x).count('1')

        def expand(clique_size, candidates):
            nonlocal best_size
            if clique_size + popcount(candidates) <= best_size:
                return
            if clique_size > best_size:
                best_size = clique_size
            if clique_size + popcount(candidates) <= best_size:
                return

            while candidates:
                lb = candidates & (-candidates)
                node = lb.bit_length() - 1
                candidates ^= lb

                new_candidates = candidates & adj[node]
                if clique_size + 1 + popcount(new_candidates) > best_size:
                    expand(clique_size + 1, new_candidates)

        all_candidates = (1 << num_nodes) - 1
        expand(0, all_candidates)
        return best_size

    def generate_structured_values(rng):
        """Generate values with shared structure to reduce product diversity."""
        strategy = rng.randint(0, 5)

        if strategy == 0:
            # Multiples of a small number - good for product collisions
            base_mult = rng.choice([2, 3, 4, 5, 6, 7, 8])
            values = []
            for i in range(1, 30):
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

        elif strategy == 1:
            # Values of form 2^a * 3^b - rich multiplicative structure
            values = set()
            for a in range(0, 10):
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

        elif strategy == 2:
            # Small values with common factors
            small_primes = [2, 3, 5, 7, 11, 13]
            values = set()
            for p in small_primes:
                for mult in range(1, 10):
                    v = p * mult
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
            # Arithmetic progression - good for sum collisions
            start = rng.randint(1, 10)
            step = rng.choice([1, 2, 3, 4, 5])
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
            # Powers of 2 - extreme product structure
            values = set()
            for e in range(0, 14):
                v = 2 ** e
                if v <= max_val:
                    values.add(v)
            values = list(values)
            rng.shuffle(values)
            values = values[:14]
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]

        else:
            # Geometric-like sequence with small ratio
            base = rng.randint(2, 4)
            values = set()
            v = 1
            for _ in range(20):
                if v <= max_val:
                    values.add(v)
                v *= base
                if len(values) >= 14:
                    break
            values = list(values)
            rng.shuffle(values)
            values = values[:14]
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]

    def build_conflict_graph(values, K):
        """
        Build conflict graph for target K.
        Two edge-pairs are compatible if adding both doesn't exceed K distinct sums and products.
        Actually, we build a graph where two potential edges are adjacent if they can coexist
        in a solution with at most K distinct sums and K distinct products.
        
        Simplified: two edges (u1,v1) and (u2,v2) are compatible if:
        - The set of sums {s1, s2} has size <= 2 (always true)
        - The set of products {p1, p2} has size <= 2 (always true)
        This is too loose. Instead, we use a different approach:
        
        We'll use a greedy + local search hybrid with the clique idea as guidance.
        """
        pass

    def greedy_with_clique_guidance(values, rng, K_target):
        """
        Greedy edge selection guided by clique structure.
        Try to find edges that form a "clique" in the sense of sharing sums/products.
        """
        all_pairs = list(combinations(range(n), 2))
        rng.shuffle(all_pairs)

        # Precompute sum and product for all pairs
        pair_sum = {}
        pair_prod = {}
        for u, v in all_pairs:
            s = values[u] + values[v]
            p = values[u] * values[v]
            pair_sum[(u, v)] = s
            pair_prod[(u, v)] = p

        # Group edges by their sum
        sum_groups = {}
        for pair in all_pairs:
            s = pair_sum[pair]
            if s not in sum_groups:
                sum_groups[s] = []
            sum_groups[s].append(pair)

        # Group edges by their product
        prod_groups = {}
        for pair in all_pairs:
            p = pair_prod[pair]
            if p not in prod_groups:
                prod_groups[p] = []
            prod_groups[p].append(pair)

        # Strategy: pick edges that share sums or products
        # First, find the most common sums and products
        sorted_sum_groups = sorted(sum_groups.items(), key=lambda x: -len(x[1]))
        sorted_prod_groups = sorted(prod_groups.items(), key=lambda x: -len(x[1]))

        selected = []
        selected_set = set()
        sums_used = set()
        prods_used = set()

        # Phase 1: Greedily pick edges from top sum groups
        for s, edges_in_group in sorted_sum_groups:
            if len(selected) >= m:
                break
            for pair in edges_in_group:
                if len(selected) >= m:
                    break
                if pair in selected_set:
                    continue
                u, v = pair
                p = pair_prod[pair]
                # Prefer if product is also shared
                selected.append(pair)
                selected_set.add(pair)
                sums_used.add(s)
                prods_used.add(p)

        # Phase 2: Fill remaining with edges that share products
        if len(selected) < m:
            for p, edges_in_group in sorted_prod_groups:
                if len(selected) >= m:
                    break
                for pair in edges_in_group:
                    if len(selected) >= m:
                        break
                    if pair in selected_set:
                        continue
                    u, v = pair
                    s = pair_sum[pair]
                    selected.append(pair)
                    selected_set.add(pair)
                    sums_used.add(s)
                    prods_used.add(p)

        # Phase 3: Fill remaining randomly
        if len(selected) < m:
            for pair in all_pairs:
                if len(selected) >= m:
                    break
                if pair not in selected_set:
                    u, v = pair
                    selected.append(pair)
                    selected_set.add(pair)
                    sums_used.add(pair_sum[pair])
                    prods_used.add(pair_prod[pair])

        return selected[:m]

    def local_search_edges(values, edges, rng, max_iters=500):
        """Local search: try swapping one edge for another to reduce distinct sums/products."""
        edges_set = set(tuple(e) for e in edges)
        current_edges = list(edges)
        sums = set()
        prods = set()
        for u, v in current_edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        best_score = max(len(sums), len(prods))

        all_pairs = list(combinations(range(n), 2))

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try removing a random edge and adding a different one
            idx = rng.randint(0, m - 1)
            old_edge = tuple(current_edges[idx])

            # Try replacing with a random non-edge
            for _ in range(30):
                new_u, new_v = rng.choice(all_pairs)
                if (new_u, new_v) in edges_set or (new_v, new_u) in edges_set:
                    continue

                # Compute new sums and prods efficiently
                os = values[old_edge[0]] + values[old_edge[1]]
                op = values[old_edge[0]] * values[old_edge[1]]
                ns = values[new_u] + values[new_v]
                np_ = values[new_u] * values[new_v]

                # Count occurrences to know if removing old edge frees a sum/product
                sum_counts = {}
                prod_counts = {}
                for u, v in current_edges:
                    s = values[u] + values[v]
                    p = values[u] * values[v]
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                    prod_counts[p] = prod_counts.get(p, 0) + 1

                sum_counts[os] -= 1
                prod_counts[op] -= 1

                new_sum_count = len(sums) - (1 if sum_counts[os] == 0 else 0) + (1 if ns not in sums and sum_counts.get(ns, 0) == 0 else 0)
                # More careful:
                new_sums = set(sums)
                if sum_counts[os] == 0:
                    new_sums.discard(os)
                if ns not in new_sums:
                    new_sums.add(ns)

                new_prods = set(prods)
                if prod_counts[op] == 0:
                    new_prods.discard(op)
                if np_ not in new_prods:
                    new_prods.add(np_)

                new_score = max(len(new_sums), len(new_prods))
                if new_score < best_score:
                    # Accept
                    current_edges[idx] = [new_u, new_v]
                    edges_set.discard(old_edge)
                    edges_set.discard(tuple(reversed(old_edge)))
                    edges_set.add((new_u, new_v))
                    sums = new_sums
                    prods = new_prods
                    best_score = new_score
                    improved = True
                    break

            if not improved:
                continue

        return current_edges, best_score

    def optimize_values(values, edges, rng, max_iters=200):
        """Try swapping pairs of values to improve score."""
        best_values = list(values)
        best_score = evaluate(best_values, edges)

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            i, j = rng.sample(range(n), 2)
            new_values = list(best_values)
            new_values[i], new_values[j] = new_values[j], new_values[i]

            # Check constraints
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

    # Initial structured values
    for restart in range(150):
        if time.time() > deadline:
            break

        values = generate_structured_values(rng)

        # Ensure all distinct and positive
        values = list(set(values))
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        values = values[:14]

        # Greedy edge selection with clique guidance
        edges = greedy_with_clique_guidance(values, rng, K_target=15)

        # Local search on edges
        edges, score = local_search_edges(values, edges, rng, max_iters=200)

        # Local search on values
        values, score = optimize_values(values, edges, rng, max_iters=80)

        # Re-optimize edges after value changes
        edges = greedy_with_clique_guidance(values, rng, K_target=15)
        edges, score = local_search_edges(values, edges, rng, max_iters=150)

        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])
            # Early exit if we hit a very good score
            if best_score <= 12:
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
