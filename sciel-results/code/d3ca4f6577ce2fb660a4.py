import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    n = 20

    # Precompute all triples of vertices
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))

    def is_valid(adj):
        """
        adj: list of integers representing adjacency bitmasks.
        Returns True if graph is K3,3-free.
        """
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def get_violating_triple(adj):
        """
        Returns a triple (i, j, k) such that common neighbors >= 3, or None.
        """
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return (i, j, k, common)
        return None

    def repair(adj):
        """
        Removes edges until the graph is valid.
        Returns the number of edges removed.
        """
        removed = 0
        while True:
            v = get_violating_triple(adj)
            if v is None:
                break
            i, j, k, common = v
            # Get list of nodes in common
            candidates = []
            for m in range(n):
                if common & (1 << m):
                    candidates.append(m)

            # Pick a random candidate
            m = random.choice(candidates)
            # Remove edge (i, m)
            adj[i] &= ~(1 << m)
            adj[m] &= ~(1 << i)
            removed += 1

        return removed

    def spectral_init(rng, density_target=0.6):
        """
        Use a spectral heuristic to initialize the graph.
        Construct the adjacency matrix as the top eigenvector of a random symmetric matrix,
        thresholding at the mean to create a dense random-like starting point.
        """
        # Generate a random symmetric matrix
        A = rng.random((n, n))
        A = (A + A.T) / 2.0
        np.fill_diagonal(A, 0)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # Take the top eigenvector (largest eigenvalue)
        top_eigenvector = eigenvectors[:, -1]

        # Threshold at the mean to create a dense random-like starting point
        threshold = np.mean(top_eigenvector)

        # Create adjacency bitmask
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                # Use the eigenvector values to determine edge inclusion
                # If both values are above threshold, include the edge
                if top_eigenvector[i] > threshold and top_eigenvector[j] > threshold:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                elif top_eigenvector[i] < threshold and top_eigenvector[j] < threshold:
                    # Also include edges where both are below threshold (for density)
                    if rng.random() < 0.3:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

        return adj

    def random_init(rng, density=0.7):
        """Initialize a random graph with given edge probability."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def local_search(adj, current_edges, rng, max_iterations=3000):
        """
        Local search: Try to add edges and perform edge swaps.
        Returns the updated edge count.
        """
        no_improve_count = 0
        iter_count = 0

        while iter_count < max_iterations:
            if time.time() > deadline - 3.0:
                break

            iter_count += 1

            # Strategy 1: Try to add a random non-edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue

            # Check if edge exists
            if adj[u] & (1 << v):
                continue

            # Try adding it
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

            if is_valid(adj):
                current_edges += 1
                no_improve_count = 0
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                no_improve_count += 1

                # Strategy 2: If stuck, try an edge swap
                if no_improve_count > 30:
                    # Find a random existing edge to remove
                    removed_edge = None
                    while removed_edge is None:
                        u_rm = rng.randint(0, n - 1)
                        if adj[u_rm] == 0:
                            continue
                        # Pick random set bit
                        neighbors = []
                        for w in range(n):
                            if adj[u_rm] & (1 << w):
                                neighbors.append(w)
                        if neighbors:
                            v_rm = rng.choice(neighbors)
                            if u_rm < v_rm:
                                removed_edge = (u_rm, v_rm)
                            else:
                                removed_edge = (v_rm, u_rm)

                    if removed_edge is not None:
                        u_rm, v_rm = removed_edge
                        adj[u_rm] &= ~(1 << v_rm)
                        adj[v_rm] &= ~(1 << u_rm)
                        current_edges -= 1

                        # Try to add a different edge
                        for _ in range(10):
                            u_add = rng.randint(0, n - 1)
                            v_add = rng.randint(0, n - 1)
                            if u_add == v_add:
                                continue
                            if adj[u_add] & (1 << v_add):
                                continue
                            if (u_add, v_add) == removed_edge or (v_add, u_add) == removed_edge:
                                continue

                            adj[u_add] |= (1 << v_add)
                            adj[v_add] |= (1 << u_add)

                            if is_valid(adj):
                                current_edges += 1
                                no_improve_count = 0
                                break
                            else:
                                adj[u_add] &= ~(1 << v_add)
                                adj[v_add] &= ~(1 << u_add)

                        no_improve_count = 0

        return current_edges

    best_edges = 0
    best_adj = None

    # Try multiple random starts with different strategies
    for seed in range(150):
        if time.time() > deadline - 10.0:
            break

        rng = random.Random(seed)

        # Alternate between spectral and random initialization
        if seed % 3 == 0:
            adj = spectral_init(rng)
        else:
            # Vary density for diversity
            density = 0.5 + 0.3 * rng.random()
            adj = random_init(rng, density)

        # Repair to valid state
        repair(adj)

        current_edges = count_edges(adj)

        # Local search
        current_edges = local_search(adj, current_edges, rng, max_iterations=2000)

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [x for x in adj]

    # Final greedy addition pass on best
    if best_adj is not None:
        # Try to add all missing edges in a systematic order
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 2.0:
                    break
                if not (best_adj[u] & (1 << v)):
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    if is_valid(best_adj):
                        best_edges += 1
                    else:
                        best_adj[u] &= ~(1 << v)
                        best_adj[v] &= ~(1 << u)

    # Construct edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return edges
