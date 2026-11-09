import time
import sys
from collections import defaultdict

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill

    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

    n = P['n']
    all_edges = P['edges']

    # Build adjacency
    adj = defaultdict(set)
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in all_edges:
        edge_set.add((min(u,v), max(u,v)))

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def find_hamiltonian_cycle(available_edges, start=0):
        """Find a Hamiltonian cycle using available edges via backtracking."""
        # Build adjacency from available edges
        adj_avail = defaultdict(set)
        for (u, v) in available_edges:
            adj_avail[u].add(v)
            adj_avail[v].add(u)

        # Check all vertices have degree >= 2
        for i in range(n):
            if len(adj_avail[i]) < 2:
                return None

        # Try to find Hamiltonian cycle starting from 'start'
        # State: current vertex, visited set, path, remaining edges
        visited = set([start])
        path = [start]
        remaining = set(available_edges)

        # Order neighbors for pruning: prefer vertices with fewer remaining connections
        def backtrack(current, visited, path, remaining):
            if time.time() > deadline:
                return None

            if len(path) == n:
                # Check if current connects back to start
                ek = edge_key(current, start)
                if ek in remaining:
                    return list(path)
                return None

            # Get unvisited neighbors of current
            neighbors = [v for v in adj_avail[current] if v not in visited]

            # Sort by degree (fewest options first - MRV heuristic)
            neighbors.sort(key=lambda v: len([x for x in adj_avail[v] if x not in visited]))

            for next_v in neighbors:
                ek = edge_key(current, next_v)
                if ek not in remaining:
                    continue

                # Pruning: check if next_v will have enough remaining edges
                # After using this edge, next_v loses one connection
                # It needs at least 1 more edge to connect to rest (unless it's the last)
                # Actually for Hamiltonian cycle, every vertex needs degree 2 in the cycle

                # Check connectivity: removing this edge shouldn't disconnect unvisited vertices
                # from being reachable

                new_remaining = remaining - {ek}
                new_visited = visited | {next_v}
                new_path = path + [next_v]

                # Pruning: ensure all unvisited vertices still have >= 2 available edges
                # (they need 2 edges in the cycle, one to predecessor, one to successor)
                # Actually, the last vertex only needs 1 more edge (back to start)
                # For intermediate vertices, they need at least 2 edges total in cycle
                # Since they'll get one from predecessor, they need at least 1 more available

                # Better pruning: check that no unvisited vertex (except possibly the last one)
                # has fewer than 2 available edges in new_remaining
                # But the "last" vertex is unknown, so check all unvisited vertices have >= 1
                # available edge (they need at least one to connect forward)

                feasible = True
                for v in range(n):
                    if v in new_visited:
                        continue
                    # v is unvisited, it needs at least 1 edge to connect to visited or other unvisited
                    avail_deg = sum(1 for e in new_remaining if v in e)
                    if avail_deg < 1:
                        feasible = False
                        break

                if not feasible:
                    continue

                # Additional pruning: check that the graph induced by unvisited vertices
                # plus the current vertex is connected enough
                # Simple check: number of unvisited vertices with degree 0 in remaining
                # (excluding the edge back to start which is special)

                result = backtrack(next_v, new_visited, new_path, new_remaining)
                if result is not None:
                    return result

            return None

        return backtrack(start, visited, path, remaining)

    # Try to find 3 edge-disjoint Hamiltonian cycles
    best_result = None
    found_cycles = []

    # Strategy: iteratively find cycles, removing used edges each time
    # Try different starting vertices and orderings

    remaining_edges = set(edge_set)

    for attempt in range(3):
        if time.time() > deadline - 5:
            break

        cycle = None
        # Try different starting vertices
        for start in range(n):
            if time.time() > deadline - 5:
                break
            cycle = find_hamiltonian_cycle(remaining_edges, start=start)
            if cycle is not None:
                break

        if cycle is None:
            break

        found_cycles.append(cycle)

        # Remove edges used in this cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = edge_key(u, v)
            if ek in remaining_edges:
                remaining_edges.discard(ek)

    if len(found_cycles) >= 1:
        # Convert to proper format: list of lists of int
        result = []
        for cycle in found_cycles:
            result.append([int(v) for v in cycle])
        return result
    else:
        # Fallback: return a single cycle if we can find one
        # This shouldn't happen but as a safety net
        cycle = find_hamiltonian_cycle(set(edge_set), start=0)
        if cycle is not None:
            return [[int(v) for v in cycle]]
        # Absolute fallback
        return [list(range(n))]
