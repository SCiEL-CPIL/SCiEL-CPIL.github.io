import random
import time

def popcount(x):
    return bin(x).count('1')

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_pairs_set = set(all_pairs)

    def is_valid_addition(adj, u, v):
        # Check if adding edge (u, v) keeps the graph K2,3-free
        # 1. Common neighbors of u and v must be < 3
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False

        # 2. For any w in N(v)\{u}, v becomes a common neighbor of u and w.
        #    So common neighbors of (u, w) must currently be < 2.
        neighbors_v = adj[v] & ~(1 << u)
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = (lsb.bit_length() - 1)
            common_uw = adj[u] & adj[w]
            if popcount(common_uw) >= 2:
                return False
            neighbors_v ^= lsb

        # 3. Similarly for w in N(u)\{v}
        neighbors_u = adj[u] & ~(1 << v)
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = (lsb.bit_length() - 1)
            common_vw = adj[v] & adj[w]
            if popcount(common_vw) >= 2:
                return False
            neighbors_u ^= lsb

        return True

    def try_add_edge(adj, u, v):
        if is_valid_addition(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False

    def greedy_build(adj, order):
        for u, v in order:
            try_add_edge(adj, u, v)

    def local_search_add(adj, max_rounds=5):
        """Iteratively add any missing edges until no more can be added."""
        for _ in range(max_rounds):
            improved = False
            for u, v in all_pairs:
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(adj, u, v):
                        improved = True
            if not improved:
                break

    def two_swap_local_search(adj, max_iters=50):
        """Try removing 2 edges and adding up to 3 to gain net +1 edge."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Pick 2 random existing edges to remove
            existing = [(i, j) for i in range(n) for j in range(i+1, n)
                        if (adj[i] >> j) & 1]
            if len(existing) < 2:
                break
            e1, e2 = random.sample(existing, 2)
            u1, v1 = e1
            u2, v2 = e2

            # Save state
            adj_save = list(adj)

            # Remove both edges
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] &= ~(1 << v2)
            adj[v2] &= ~(1 << u2)

            # Try to add as many missing edges as possible
            missing = [(i, j) for i in range(n) for j in range(i+1, n)
                       if not ((adj[i] >> j) & 1)]
            random.shuffle(missing)
            added = 0
            for a, b in missing:
                if try_add_edge(adj, a, b):
                    added += 1

            # Check if we gained net edges
            old_count = sum(popcount(adj_save[i]) for i in range(n)) // 2
            new_count = sum(popcount(adj[i]) for i in range(n)) // 2
            if new_count <= old_count:
                # Revert
                adj[:] = adj_save

    def one_swap_local_search(adj, max_iters=50):
        """Try removing 1 edge and adding up to 2 to gain net +1 edge."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Pick a random existing edge to remove
            existing = [(i, j) for i in range(n) for j in range(i+1, n)
                        if (adj[i] >> j) & 1]
            if not existing:
                break
            u, v = random.choice(existing)

            # Save state
            adj_save = list(adj)

            # Remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

            # Try to add as many missing edges as possible
            missing = [(i, j) for i in range(n) for j in range(i+1, n)
                       if not ((adj[i] >> j) & 1)]
            random.shuffle(missing)
            added = 0
            for a, b in missing:
                if try_add_edge(adj, a, b):
                    added += 1

            # Check if we gained net edges
            old_count = sum(popcount(adj_save[i]) for i in range(n)) // 2
            new_count = sum(popcount(adj[i]) for i in range(n)) // 2
            if new_count <= old_count:
                # Revert
                adj[:] = adj_save

    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2

    def extract_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = (lsb.bit_length() - 1)
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    best_edges = []
    best_count = 0

    for restart in range(500):
        if time.time() > deadline:
            break

        adj = [0] * n

        # Greedy construction with random edge order
        edges = list(all_pairs)
        random.shuffle(edges)
        greedy_build(adj, edges)

        # Local search: add any missing edges
        local_search_add(adj)

        # One-swap local search
        one_swap_local_search(adj, max_iters=20)

        # Two-swap local search
        two_swap_local_search(adj, max_iters=15)

        # Final local search
        local_search_add(adj)

        # Count edges
        edge_count = count_edges(adj)
        if edge_count > best_count:
            best_count = edge_count
            best_edges = extract_edges(adj)

    return best_edges
