import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    n = 20

    def is_valid_addition(adj, u, v):
        """
        Checks if adding edge (u, v) violates the K3,3-free condition.
        Optimized: precompute neighbor lists once.
        """
        if adj[u] & (1 << v):
            return False

        # Precompute neighbor lists
        adj_u = adj[u]
        adj_v = adj[v]

        # Check triples containing u: {u, a, b}
        for a in range(n):
            if a == u or a == v:
                continue
            adj_a = adj[a]
            if not (adj_a & adj_u):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                adj_b = adj[b]
                if not (adj_b & adj_u):
                    continue

                # Check if v is a common neighbor of a and b
                if (adj_a & (1 << v)) and (adj_b & (1 << v)):
                    count = (adj_u & adj_a & adj_b).bit_count()
                    if count > 1:
                        return False

                # Check if u is a common neighbor of a and b
                if (adj_a & (1 << u)) and (adj_b & (1 << u)):
                    count = (adj_v & adj_a & adj_b).bit_count()
                    if count > 1:
                        return False

        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n

        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)

        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

        return adj

    def bipartite_spectral_initialize(seed):
        """
        Bipartite spectral construction: partition 20 vertices into two sets of 10,
        generate edges based on eigenvectors of a random 10x10 matrix to naturally
        limit common neighbors across the partition, then add intra-partition edges.
        """
        random.seed(seed)
        np.random.seed(seed)

        left = list(range(10))
        right = list(range(10, 20))

        A = np.random.randn(10, 10)
        A = (A + A.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(A)

        num_top = 3
        adj = [0] * n

        bipartite_weights = []
        for i in range(10):
            for j in range(10):
                w = 0.0
                for k in range(num_top):
                    w += eigenvectors[i, k] * eigenvectors[j, k]
                bipartite_weights.append((w, i, j + 10))

        bipartite_weights.sort(key=lambda x: -x[0])

        for w, u, v in bipartite_weights:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

        intra_pairs = []
        for i in range(10):
            for j in range(i + 1, 10):
                intra_pairs.append((i, j))
                intra_pairs.append((i + 10, j + 10))

        random.shuffle(intra_pairs)
        for u, v in intra_pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

        return adj

    def greedy_fill(adj):
        """Add all possible edges greedily with multiple passes."""
        current_adj = adj[:]

        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if time.time() > deadline:
                        return current_adj
                    if not (current_adj[i] & (1 << j)):
                        if is_valid_addition(current_adj, i, j):
                            current_adj[i] |= (1 << j)
                            current_adj[j] |= (1 << i)
                            improved = True

        return current_adj

    def kick(adj, num_removals=2):
        """Randomly remove some edges to escape local optima."""
        current_adj = adj[:]

        existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
        random.shuffle(existing_edges)

        for u, v in existing_edges[:num_removals]:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)

        return current_adj

    def local_search(adj, max_kick_iterations=30):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)

        # Phase 1: Greedy fill
        current_adj = greedy_fill(current_adj)
        current_edges = count_edges(current_adj)

        best_adj = current_adj[:]
        best_edges = current_edges

        # Phase 2: Kick + greedy fill cycles
        for _ in range(max_kick_iterations):
            if time.time() > deadline:
                break

            kicked_adj = kick(current_adj, num_removals=2)
            filled_adj = greedy_fill(kicked_adj)
            filled_edges = count_edges(filled_adj)

            if filled_edges > best_edges:
                best_adj = filled_adj[:]
                best_edges = filled_edges
                current_adj = best_adj[:]
            else:
                kicked_adj = kick(current_adj, num_removals=3)
                filled_adj = greedy_fill(kicked_adj)
                filled_edges = count_edges(filled_adj)
                if filled_edges > best_edges:
                    best_adj = filled_adj[:]
                    best_edges = filled_edges
                    current_adj = best_adj[:]
                else:
                    kicked_adj = kick(current_adj, num_removals=5)
                    filled_adj = greedy_fill(kicked_adj)
                    filled_edges = count_edges(filled_adj)
                    if filled_edges > best_edges:
                        best_adj = filled_adj[:]
                        best_edges = filled_edges
                        current_adj = best_adj[:]
                    else:
                        break

        current_adj = best_adj
        current_edges = best_edges

        # Phase 3: 2-opt edge swap local search
        improved = True
        swap_iterations = 0
        max_swap_iterations = 50
        while improved and swap_iterations < max_swap_iterations:
            if time.time() > deadline:
                break
            improved = False
            swap_iterations += 1

            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            random.shuffle(existing_edges)
            random.shuffle(missing_edges)

            for u, v in existing_edges:
                if time.time() > deadline:
                    break
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)

                best_add = 0
                best_pair = None
                for a, b in missing_edges:
                    if time.time() > deadline:
                        break
                    if (a == u or a == v or b == u or b == v):
                        continue
                    if is_valid_addition(current_adj, a, b):
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        if best_add < 2:
                            best_add += 1
                            best_pair = (a, b)
                        if best_add >= 2:
                            break

                if best_add >= 1:
                    current_edges += best_add - 1
                    improved = True
                    missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)

        # Phase 4: Final greedy fill
        current_adj = greedy_fill(current_adj)
        current_edges = count_edges(current_adj)

        return current_adj, current_edges

    best_adj = [0] * n
    best_edges = 0

    num_restarts = 3000
    for i in range(num_restarts):
        if time.time() > deadline:
            break

        strategy = i % 4
        if strategy == 0:
            adj = greedy_construct(i)
        elif strategy == 1:
            adj = bipartite_spectral_initialize(i)
        elif strategy == 2:
            adj = greedy_construct(i + 10000)
        else:
            adj = bipartite_spectral_initialize(i + 5000)

        adj, edges = local_search(adj)

        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]

    # Final polish: multiple local search passes on best solution
    polish_count = 0
    while time.time() < deadline - 10 and polish_count < 5:
        adj, edges = local_search(best_adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
            polish_count = 0
        else:
            polish_count += 1

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])

    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
