import random
import time
import itertools

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

    # Try to find larger n, starting from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline - 5:
            break

        found = False
        # Adaptive number of attempts based on remaining time
        remaining_time = deadline - time.time()
        max_attempts = min(3000, max(200, int(remaining_time * 50)))

        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break

            # Generate random maximal triangle-free graph
            adj = [0] * n
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)

            for u, v in all_edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            # Quick density check
            edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
            if edge_count < n * n // 10:
                continue

            # Fast greedy pre-filter: if greedy finds alpha >= 10, skip
            if greedy_alpha_lower_bound(n, adj) >= 10:
                continue

            # Build edge list for validation
            edges = edges_from_adj(n, adj)

            # Run exact validation
            if validate(n, adj):
                return {'n': n, 'edges': edges}

    # Local search: try to grow smaller valid graphs by adding edges
    # Try to find a valid graph on n vertices by starting from n-1 and adding a vertex
    for target_n in range(39, 26, -1):
        if time.time() > deadline - 3:
            break

        # Try random construction for target_n
        for attempt in range(500):
            if time.time() > deadline - 2:
                break

            adj = [0] * target_n
            all_edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
            random.shuffle(all_edges)

            for u, v in all_edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            edge_count = sum(bin(adj[i]).count('1') for i in range(target_n)) // 2
            if edge_count < target_n * target_n // 10:
                continue

            if greedy_alpha_lower_bound(target_n, adj) >= 10:
                continue

            edges = edges_from_adj(target_n, adj)

            if validate(target_n, adj):
                return {'n': target_n, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}
