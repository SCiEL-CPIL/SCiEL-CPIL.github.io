"""Improved K2,3-free graph construction for n=20 vertices using bitmasks."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    full_mask = (1 << n) - 1

    def popcount(x):
        return x.bit_count()

    def is_valid_addition(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free using bitmasks."""
        # u and v must not have 3+ common neighbors
        if popcount(adj[u] & adj[v]) >= 3:
            return False
        # For each neighbor w of v (excluding u), common(u,w) must be < 2
        neighbors_v = adj[v] & ~(1 << u) & ~(1 << v)
        # For each neighbor w of u (excluding v), common(v,w) must be < 2
        neighbors_u = adj[u] & ~(1 << u) & ~(1 << v)
        
        # Check all neighbors of v
        w = neighbors_v
        while w:
            lowbit = w & (-w)
            wi = lowbit.bit_length() - 1
            if popcount(adj[u] & adj[wi]) >= 2:
                return False
            w ^= lowbit
        
        # Check all neighbors of u
        w = neighbors_u
        while w:
            lowbit = w & (-w)
            wi = lowbit.bit_length() - 1
            if popcount(adj[v] & adj[wi]) >= 2:
                return False
            w ^= lowbit
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def get_missing_edges(adj):
        missing = []
        for u, v in all_pairs:
            if not (adj[u] & (1 << v)):
                missing.append((u, v))
        return missing

    def greedy_build(seed):
        """Build graph using greedy edge addition."""
        rng = random.Random(seed)
        adj = [0] * n
        pairs = list(all_pairs)
        rng.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                add_edge(adj, u, v)
        return adj

    def local_search(adj, max_iterations=2000):
        """
        Local search: try removing one edge, then greedily add as many as possible.
        """
        if time.time() > deadline:
            return adj

        adj = adj[:]  # copy
        current_count = sum(popcount(adj[i]) for i in range(n)) // 2
        best_adj = adj[:]
        best_count = current_count

        rng = random.Random(42)

        for iteration in range(max_iterations):
            if time.time() > deadline:
                break

            # Pick a random existing edge to remove
            edges_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges_list.append((i, j))
            if not edges_list:
                break

            rem_u, rem_v = rng.choice(edges_list)
            remove_edge(adj, rem_u, rem_v)
            current_count -= 1

            # Greedily add missing edges
            added = []
            for u, v in all_pairs:
                if not (adj[u] & (1 << v)):
                    if is_valid_addition(adj, u, v):
                        add_edge(adj, u, v)
                        added.append((u, v))
                        current_count += 1

            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
            else:
                # Revert
                for u, v in added:
                    remove_edge(adj, u, v)
                add_edge(adj, rem_u, rem_v)

        return best_adj

    def multi_start(adj, num_rounds=5):
        """Run multiple rounds of local search with perturbations."""
        adj = adj[:]
        best_count = sum(popcount(adj[i]) for i in range(n)) // 2
        best_adj = adj[:]

        for rnd in range(num_rounds):
            if time.time() > deadline:
                break

            adj_pert = best_adj[:]
            rng = random.Random(200 + rnd)
            num_remove = rng.randint(2, 4)
            for _ in range(num_remove):
                edges_list = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj_pert[i] & (1 << j):
                            edges_list.append((i, j))
                if not edges_list:
                    break
                u, v = rng.choice(edges_list)
                remove_edge(adj_pert, u, v)

            adj_ls = local_search(adj_pert, max_iterations=500)
            count_ls = sum(popcount(adj_ls[i]) for i in range(n)) // 2
            if count_ls > best_count:
                best_count = count_ls
                best_adj = adj_ls[:]

        return best_adj

    # Main search
    best_adj = None
    best_count = 0

    # Strategy 1: Multiple greedy restarts with local search
    num_restarts = 60
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        adj = greedy_build(seed)
        count = sum(popcount(adj[i]) for i in range(n)) // 2

        adj = local_search(adj, max_iterations=800)
        count = sum(popcount(adj[i]) for i in range(n)) // 2

        if count > best_count:
            best_count = count
            best_adj = adj[:]

    # Strategy 2: Multi-start with perturbations
    if best_adj is not None and time.time() < deadline:
        best_adj = multi_start(best_adj, num_rounds=10)
        best_count = sum(popcount(best_adj[i]) for i in range(n)) // 2

    # Strategy 3: More greedy restarts if time permits
    if time.time() < deadline:
        for seed in range(num_restarts, num_restarts + 60):
            if time.time() > deadline:
                break
            adj = greedy_build(seed)
            count = sum(popcount(adj[i]) for i in range(n)) // 2
            if count + 3 > best_count:
                adj = local_search(adj, max_iterations=400)
                count = sum(popcount(adj[i]) for i in range(n)) // 2
            if count > best_count:
                best_count = count
                best_adj = adj[:]

    if best_adj is not None:
        return get_edge_list(best_adj)

    # Fallback: disjoint K4s
    edges = []
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                edges.append([4 * k + a, 4 * k + b])
    return edges
