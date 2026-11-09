import time
import random

def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin before 90s kill

    n = 21
    edges = [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]

    # Build edge set and adjacency
    edge_set = set()
    adj = [set() for _ in range(n)]
    for e in edges:
        a, b = e
        edge_set.add((min(a, b), max(a, b)))
        adj[a].add(b)
        adj[b].add(a)

    all_vertices = set(range(n))

    def find_hamiltonian_cycle(available_edges, start):
        """Backtracking to find a Hamiltonian cycle starting and ending at 'start'."""
        path = [start]
        used = set()  # set of (u,v) with u<v for edges used in this cycle

        def backtrack():
            if time.time() > deadline:
                return None
            if len(path) == n:
                # Check if last vertex connects back to start
                last = path[-1]
                key = (min(last, start), max(last, start))
                if key in available_edges and key not in used:
                    return list(path)
                return None

            last = path[-1]
            # Get available neighbors
            neighbors = [v for v in adj[last] if v not in set(path)]
            # Also check if we can close the cycle (only when len==n handled above)

            # Sort neighbors by degree in remaining graph (heuristic)
            def neighbor_degree(v):
                return len([u for u in adj[v] if u not in set(path) and (min(u, v), max(u, v)) in available_edges])
            neighbors.sort(key=neighbor_degree)

            for nb in neighbors:
                key = (min(last, nb), max(last, nb))
                if key not in available_edges:
                    continue
                if key in used:
                    continue
                used.add(key)
                path.append(nb)
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                used.discard(key)
            return None

        return backtrack()

    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    cycles = []
    available = set(edge_set)

    # Try different starting vertices and orderings
    start_orderings = [
        list(range(n)),
        list(range(n - 1, -1, -1)),
        [0, 20, 1, 19, 2, 18, 3, 17, 4, 16, 5, 15, 6, 14, 7, 13, 8, 12, 9, 11, 10],
    ]

    for ordering in start_orderings:
        if len(cycles) >= 3:
            break
        if time.time() > deadline:
            break
        for start in ordering:
            if len(cycles) >= 3:
                break
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(available, start)
            if cycle is not None:
                # Record the cycle and remove its edges
                cycles.append(cycle)
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    key = (min(a, b), max(a, b))
                    available.discard(key)

    if len(cycles) == 0:
        # Fallback: return a single Hamiltonian cycle if we can find one
        available = set(edge_set)
        cycle = find_hamiltonian_cycle(available, 0)
        if cycle is not None:
            cycles = [cycle]
        else:
            # Last resort: return empty? No, must be nonempty. Try harder.
            # Try all starts
            for s in range(n):
                cycle = find_hamiltonian_cycle(available, s)
                if cycle is not None:
                    cycles = [cycle]
                    break

    # Ensure we return at most 10 cycles
    cycles = cycles[:10]

    # Convert to list of lists of Python ints
    result = []
    for c in cycles:
        result.append([int(v) for v in c])

    return result
