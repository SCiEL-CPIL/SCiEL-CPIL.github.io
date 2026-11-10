"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

sys.setrecursionlimit(100000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = N
    edges = P['edges']

    # Build adjacency and edge lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        ek = (u, v) if u < v else (v, u)
        edge_set.add(ek)

    def edge_key(u, v):
        return (u, v) if u < v else (v, u)

    def find_ham_cycle(used_edges, start_vertex, rng=None):
        """
        Find a Hamiltonian cycle starting at start_vertex, avoiding used_edges.
        Uses Warnsdorff's heuristic with optional randomization for diversity.
        Returns list of vertices or None.
        """
        if time.time() > deadline:
            return None

        path = [start_vertex]
        visited = [False] * n
        visited[start_vertex] = True

        # Pre-sort neighbors for deterministic base ordering
        neighbor_order = {v: sorted(adj[v]) for v in range(n)}

        def dfs(current, depth):
            if time.time() > deadline:
                return None

            if depth == n - 1:
                # Check closing edge to start_vertex
                ek = edge_key(current, start_vertex)
                if ek in edge_set and ek not in used_edges:
                    return list(path)
                return None

            # Collect candidate neighbors
            candidates = []
            for nb in neighbor_order[current]:
                if visited[nb]:
                    continue
                ek = edge_key(current, nb)
                if ek not in edge_set or ek in used_edges:
                    continue
                # Warnsdorff: count unvisited neighbors of nb (with edge availability)
                deg = 0
                for w in adj[nb]:
                    if not visited[w]:
                        ek2 = edge_key(nb, w)
                        if ek2 in edge_set and ek2 not in used_edges:
                            deg += 1
                # If this is the second-to-last step, nb must connect to start
                if depth == n - 2:
                    ek_close = edge_key(nb, start_vertex)
                    if ek_close not in edge_set or ek_close in used_edges:
                        continue  # Can't close through nb
                candidates.append((deg, nb))

            if not candidates:
                return None

            # Sort by degree (fewest options first)
            candidates.sort(key=lambda x: x[0])

            # Add small random perturbation for diversity if rng provided
            if rng is not None:
                for i in range(len(candidates) - 1, 0, -1):
                    if rng.random() < 0.3:
                        j = rng.randint(0, i)
                        candidates[i], candidates[j] = candidates[j], candidates[i]

            for deg, nb in candidates:
                path.append(nb)
                visited[nb] = True
                result = dfs(nb, depth + 1)
                if result is not None:
                    return result
                path.pop()
                visited[nb] = False

            return None

        return dfs(start_vertex, 0)

    def cycle_edges(cycle):
        """Return set of edge keys for a cycle."""
        ekeys = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ekeys.add(edge_key(u, v))
        return ekeys

    def validate_cycle(cycle, used_edges):
        """Check cycle is valid: all vertices distinct, all edges exist and unused."""
        if len(cycle) != n:
            return False
        if len(set(cycle)) != n:
            return False
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = edge_key(u, v)
            if ek not in edge_set:
                return False
            if ek in used_edges:
                return False
        return True

    best_cycles = []

    # Strategy: Try to find 2 edge-disjoint Hamiltonian cycles.
    # Use multiple restarts with different start vertices and randomizations.

    # First, try deterministic approach with start=0
    used_edges = set()
    cycles = []

    # Try to find first cycle
    for attempt in range(5):
        if time.time() > deadline:
            break
        if len(cycles) >= 2:
            break

        if len(cycles) == 0:
            # Find first cycle
            cyc = find_ham_cycle(used_edges, 0, rng=None)
            if cyc is not None and validate_cycle(cyc, used_edges):
                cycles.append(cyc)
                used_edges.update(cycle_edges(cyc))
        else:
            # Find second cycle
            # Try different start vertices for diversity
            found = False
            for start_v in range(n):
                if time.time() > deadline:
                    break
                cyc = find_ham_cycle(used_edges, start_v, rng=None)
                if cyc is not None and validate_cycle(cyc, used_edges):
                    cycles.append(cyc)
                    used_edges.update(cycle_edges(cyc))
                    found = True
                    break
            if not found:
                # Try with randomization
                for trial in range(10):
                    if time.time() > deadline:
                        break
                    rng = random.Random(trial * 1000 + 42)
                    for start_v in range(n):
                        if time.time() > deadline:
                            break
                        cyc = find_ham_cycle(used_edges, start_v, rng=rng)
                        if cyc is not None and validate_cycle(cyc, used_edges):
                            cycles.append(cyc)
                            used_edges.update(cycle_edges(cyc))
                            found = True
                            break
                    if found:
                        break
            if not found:
                break  # Can't find second cycle, stop

    if len(cycles) > len(best_cycles):
        best_cycles = cycles

    # If we only found 1 cycle, try harder with more randomization for the second
    if len(best_cycles) < 2:
        used_edges = set()
        for cyc in best_cycles:
            used_edges.update(cycle_edges(cyc))

        for trial in range(30):
            if time.time() > deadline:
                break
            rng = random.Random(trial * 7919 + 13)
            for start_v in range(n):
                if time.time() > deadline:
                    break
                cyc = find_ham_cycle(used_edges, start_v, rng=rng)
                if cyc is not None and validate_cycle(cyc, used_edges):
                    if len(best_cycles) < 2:
                        best_cycles.append(cyc)
                        used_edges.update(cycle_edges(cyc))
                    break
            if len(best_cycles) >= 2:
                break

    # Ensure we have at least 1 valid cycle
    if not best_cycles:
        # Last resort: find any single Hamiltonian cycle
        for start_v in range(n):
            if time.time() > deadline:
                break
            cyc = find_ham_cycle(set(), start_v, rng=None)
            if cyc is not None and validate_cycle(cyc, set()):
                best_cycles = [cyc]
                break

    # Final output: convert to list of lists of Python ints
    out = []
    for c in best_cycles:
        out.append([int(v) for v in c])

    return out
# EVOLVE-BLOCK-END
