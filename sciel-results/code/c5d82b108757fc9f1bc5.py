import random
import time
import itertools
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def is_triangle_free(n, adj):
        for u in range(n):
            for v in range(n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        return False
        return True

    def greedy_alpha_lower_bound(n, adj):
        """Return size of a greedy independent set (lower bound on alpha)."""
        remaining = (1 << n) - 1
        size = 0
        while remaining:
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = bin(remaining & adj[v]).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            size += 1
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return size

    def has_independent_set_10(n, adj):
        """Exact branch and bound for independent set of size 10."""
        # Quick greedy check
        if greedy_alpha_lower_bound(n, adj) >= 10:
            return True

        def backtrack(candidates, size):
            if size == 10:
                return True
            if size + bin(candidates).count('1') < 10:
                return False

            # Pick vertex with max degree in candidate subgraph
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = bin(candidates & adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

            if best_v == -1:
                return False

            candidates &= ~(1 << best_v)

            # Option 1: Include best_v
            new_candidates = candidates & ~adj[best_v]
            if backtrack(new_candidates, size + 1):
                return True

            # Option 2: Exclude best_v
            if backtrack(candidates, size):
                return True

            return False

        return backtrack((1 << n) - 1, 0)

    def validate(n, adj):
        if not is_triangle_free(n, adj):
            return False
        if has_independent_set_10(n, adj):
            return False
        return True

    def edges_from_adj(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Fallback: known valid graph on 26 vertices
    best_n = 26
    best_adj = [0] * 26
    for i in range(26):
        for j in range(i + 1, 26):
            if (j - i) % 26 % 3 == 1:
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
    best_edges = edges_from_adj(26, best_adj)

    def spectral_alpha_proxy(n, adj_list):
        """Compute a spectral proxy for independence number.
        Uses the second-largest eigenvalue of the adjacency matrix.
        Lower lambda_2 tends to correlate with lower independence number
        for regular/near-regular graphs."""
        A = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            for v in range(u + 1, n):
                if adj_list[u] & (1 << v):
                    A[u, v] = 1.0
                    A[v, u] = 1.0
        try:
            eigvals = np.linalg.eigvalsh(A)
            # For triangle-free graphs, we want to minimize the spread
            # that allows large independent sets. Use a combination.
            lambda_max = eigvals[-1]
            lambda_2 = eigvals[-2] if n > 1 else 0.0
            lambda_min = eigvals[0]
            # A graph with small |lambda_min| and controlled lambda_2
            # tends to have smaller independence number
            return lambda_2 + 0.5 * abs(lambda_min)
        except:
            return float('inf')

    def spectral_local_search(n, adj_list, max_time=30.0):
        """Iteratively swap edges to reduce spectral proxy while maintaining triangle-freeness."""
        start_t = time.time()
        current_adj = list(adj_list)
        current_score = spectral_alpha_proxy(n, current_adj)
        best_adj = list(current_adj)
        best_score = current_score

        # Precompute for fast triangle checks
        def can_add_edge(u, v, adj):
            """Check if adding edge (u,v) maintains triangle-freeness."""
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            return common == 0

        def can_remove_edge(u, v, adj):
            """Always safe to remove an edge for triangle-freeness."""
            return True

        # Build set of existing edges for quick lookup
        edge_set = set()
        for u in range(n):
            for v in range(u + 1, n):
                if current_adj[u] & (1 << v):
                    edge_set.add((u, v))

        # Build set of non-edges
        all_pairs = set((i, j) for i in range(n) for j in range(i + 1, n))
        non_edges = all_pairs - edge_set

        for iteration in range(20000):
            if time.time() - start_t > max_time:
                break

            # Try random edge swaps
            for _ in range(20):
                if time.time() - start_t > max_time:
                    break

                if not non_edges:
                    break

                # Pick a random non-edge to add
                add_edge = random.choice(list(non_edges))
                u, v = add_edge

                if not can_add_edge(u, v, current_adj):
                    continue

                # Pick a random existing edge to remove (not incident to u or v to keep degree similar)
                removable = [e for e in edge_set if u not in e and v not in e]
                if not removable:
                    continue

                rem_edge = random.choice(removable)
                ru, rv = rem_edge

                # Apply swap
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_adj[ru] &= ~(1 << rv)
                current_adj[rv] &= ~(1 << ru)

                new_score = spectral_alpha_proxy(n, current_adj)

                if new_score < current_score:
                    # Accept
                    current_score = new_score
                    edge_set.discard(add_edge)
                    edge_set.add(rem_edge)
                    non_edges.discard(rem_edge)
                    non_edges.add(add_edge)

                    if new_score < best_score:
                        best_score = new_score
                        best_adj = list(current_adj)
                else:
                    # Reject, undo
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_adj[ru] |= (1 << rv)
                    current_adj[rv] |= (1 << ru)

        return best_adj

    # Structural change: Systematic circulant graph search with sum-free difference sets
    # This replaces weak random edge-adding with algebraic construction
    # A circulant graph C_n(S) has edge (i,j) iff (j-i) mod n in S
    # Triangle-free condition: S is sum-free mod n (no a,b,c in S with a+b+c ≡ 0 mod n)
    # Actually for triangle-free: no a,b in S with a-b in S (directed) or a+b in S (undirected)
    # For undirected circulant: S is symmetric (if s in S then n-s in S)
    # Triangle-free: no a,b,c in S with a+b ≡ c mod n (since a,b,c edges means a-b, b-c, a-c in S)
    # Actually: vertices 0,a,b form triangle iff a in S, b in S, (b-a) mod n in S
    # So S must be such that for no a,b in S is (a-b) mod n in S (and a != b)
    
    def is_sum_free_circulant(n, S):
        """Check if circulant graph C_n(S) is triangle-free.
        S is a set of differences. Edge (i,j) iff (j-i) mod n in S.
        Triangle 0,a,b exists iff a in S, b in S, (b-a) mod n in S.
        """
        S_set = set(S)
        for a in S:
            for b in S:
                if a != b:
                    if (b - a) % n in S_set:
                        return False
        return True

    def circulant_alpha_lower_bound(n, S):
        """Quick greedy alpha lower bound for circulant graph."""
        adj = [0] * n
        for s in S:
            for i in range(n):
                adj[i] |= (1 << ((i + s) % n))
                adj[i] |= (1 << ((i - s) % n))
        # Remove self-loops
        for i in range(n):
            adj[i] &= ~(1 << i)
        return greedy_alpha_lower_bound(n, adj)

    def build_circulant(n, S):
        """Build adjacency list for circulant graph C_n(S)."""
        adj = [0] * n
        S_set = set(S)
        for i in range(n):
            for s in S:
                adj[i] |= (1 << ((i + s) % n))
                adj[i] |= (1 << ((i - s) % n))
            adj[i] &= ~(1 << i)
        return adj

    # Phase 1: Spectral local search starting from good initial graphs
    # Start with known good constructions and refine via edge swaps

    def get_initial_graphs(n):
        """Generate several initial triangle-free graphs for spectral refinement."""
        graphs = []

        # 1. Random greedy triangle-free
        for trial in range(5):
            adj = [0] * n
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            graphs.append(adj)

        # 2. Circulant-based
        for size in range(3, n // 3):
            for _ in range(3):
                S = set()
                for s in range(1, n // 2 + 1):
                    if random.random() < 0.25:
                        S.add(s)
                        S.add(n - s)
                if len(S) < 3:
                    continue
                if not is_sum_free_circulant(n, S):
                    continue
                adj = build_circulant(n, S)
                graphs.append(adj)

        return graphs

    for n in range(40, 26, -1):
        if time.time() > deadline - 10:
            break

        # Try multiple initial graphs with spectral local search
        initial_graphs = get_initial_graphs(n)

        for init_adj in initial_graphs:
            if time.time() > deadline - 5:
                break

            # Quick check: if already has independent set of 10, skip
            if greedy_alpha_lower_bound(n, init_adj) >= 10:
                continue

            # Apply spectral local search
            remaining = deadline - time.time() - 3
            if remaining < 5:
                # Just validate directly
                if validate(n, init_adj):
                    edges = edges_from_adj(n, init_adj)
                    return {'n': n, 'edges': edges}
                continue

            refined_adj = spectral_local_search(n, init_adj, max_time=min(remaining, 20.0))

            # Check if refined graph is valid
            if greedy_alpha_lower_bound(n, refined_adj) >= 10:
                continue

            if validate(n, refined_adj):
                edges = edges_from_adj(n, refined_adj)
                return {'n': n, 'edges': edges}

    # Phase 2: Fallback to random maximal triangle-free search
    for n in range(40, 26, -1):
        if time.time() > deadline - 5:
            break

        remaining_time = deadline - time.time()
        max_attempts = min(500, max(50, int(remaining_time * 20)))

        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break

            adj = [0] * n
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)

            for u, v in all_edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
            if edge_count < n * n // 10:
                continue

            if greedy_alpha_lower_bound(n, adj) >= 10:
                continue

            edges = edges_from_adj(n, adj)

            if validate(n, adj):
                return {'n': n, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}
