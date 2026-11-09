import time
import random
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency sets for quick lookup
    orig_adj = [set() for _ in range(n)]
    for u, v in edges:
        orig_adj[u].add(v)
        orig_adj[v].add(u)

    def find_biclique(uncovered, max_size_left=4, max_size_right=4):
        """
        Find a good biclique from uncovered edges.
        Returns (L, R, covered_edges) or None.
        """
        active = [u for u in range(n) if uncovered[u]]
        if not active:
            return None

        # Sort by uncovered degree
        active_sorted = sorted(active, key=lambda x: len(uncovered[x]), reverse=True)

        best_covered = 0
        best_L = None
        best_R = None

        # Strategy 1: Single vertex on left
        for u in active_sorted:
            R = set(uncovered[u])
            if not R:
                continue
            covered = len(R)
            if covered > best_covered:
                best_covered = covered
                best_L = [u]
                best_R = list(R)

        # Strategy 2: Two vertices on left
        K = min(15, len(active_sorted))
        top_active = active_sorted[:K]
        for i in range(len(top_active)):
            u = top_active[i]
            for j in range(i + 1, len(top_active)):
                v = top_active[j]
                R = uncovered[u].intersection(uncovered[v])
                R.discard(u)
                R.discard(v)
                if not R:
                    continue
                covered = 2 * len(R)
                if covered > best_covered:
                    best_covered = covered
                    best_L = [u, v]
                    best_R = list(R)

        # Strategy 3: Three vertices on left
        M = min(10, len(active_sorted))
        top_active_m = active_sorted[:M]
        for i in range(len(top_active_m)):
            u = top_active_m[i]
            for j in range(i + 1, len(top_active_m)):
                v = top_active_m[j]
                common_uv = uncovered[u].intersection(uncovered[v])
                common_uv.discard(u)
                common_uv.discard(v)
                if not common_uv:
                    continue
                # Try adding a third vertex
                w_candidates = list(common_uv)
                if len(w_candidates) > 10:
                    w_candidates = sorted(w_candidates, key=lambda x: len(uncovered[x]), reverse=True)[:10]
                for w in w_candidates:
                    R = common_uv.intersection(uncovered[w])
                    R.discard(u)
                    R.discard(v)
                    R.discard(w)
                    if not R:
                        continue
                    covered = 3 * len(R)
                    if covered > best_covered:
                        best_covered = covered
                        best_L = [u, v, w]
                        best_R = list(R)

        # Strategy 4: Symmetric - single vertex on right
        for v in active_sorted:
            L = set(uncovered[v])
            if not L:
                continue
            covered = len(L)
            if covered > best_covered:
                best_covered = covered
                best_L = list(L)
                best_R = [v]

        if best_L is None:
            # Fallback: pick any uncovered edge
            for u in active:
                if uncovered[u]:
                    v = next(iter(uncovered[u]))
                    return [u], [v], 1
            return None

        return best_L, best_R, best_covered

    def greedy_solve(seed):
        random.seed(seed)

        # Track uncovered edges
        uncovered = [set() for _ in range(n)]
        for u, v in edges:
            uncovered[u].add(v)
            uncovered[v].add(u)

        bicliques = []

        while any(uncovered[u] for u in range(n)):
            if time.time() > deadline:
                return None

            result = find_biclique(uncovered)
            if result is None:
                break

            L, R, covered = result

            # Validate: all cross pairs must be edges
            valid = True
            for u in L:
                for v in R:
                    if v not in orig_adj[u]:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                # Fallback to single edge
                for u in range(n):
                    if uncovered[u]:
                        v = next(iter(uncovered[u]))
                        L = [u]
                        R = [v]
                        break

            bicliques.append([L, R])

            # Update uncovered
            for u in L:
                for v in R:
                    if v in uncovered[u]:
                        uncovered[u].discard(v)
                    if u in uncovered[v]:
                        uncovered[v].discard(u)

        return bicliques

    def local_search(bicliques, max_iters=500):
        """
        Try to reduce the number of bicliques by merging or moving edges.
        This is a heuristic local search.
        """
        if not bicliques:
            return bicliques

        # Rebuild uncovered tracking from current partition
        covered = [set() for _ in range(n)]
        for L, R in bicliques:
            for u in L:
                for v in R:
                    covered[u].add(v)
                    covered[v].add(u)

        # Check if valid
        for u, v in edges:
            if v not in covered[u]:
                return bicliques  # Invalid, return as-is

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1

            # Try to merge two bicliques
            for i in range(len(bicliques)):
                for j in range(i + 1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]

                    # Try merging: new_L = L1 ∪ L2, new_R = R1 ∪ R2
                    # Check if all cross pairs are edges
                    new_L = L1 + L2
                    new_R = R1 + R2
                    valid = True
                    for u in new_L:
                        for v in new_R:
                            if v not in orig_adj[u]:
                                valid = False
                                break
                        if not valid:
                            break

                    if valid:
                        # Check no duplicate edges
                        edge_set = set()
                        for u in new_L:
                            for v in new_R:
                                edge_set.add((u, v))
                                edge_set.add((v, u))
                        # Ensure all are in original edges
                        for e in edge_set:
                            if e not in [(u, v) for u, v in edges] and (e[1], e[0]) not in [(u, v) for u, v in edges]:
                                valid = False
                                break

                    if valid:
                        bicliques.pop(j)
                        bicliques[i] = [new_L, new_R]
                        improved = True
                        break

                if improved:
                    break

            # Try to move an edge from one biclique to another
            if not improved:
                for i in range(len(bicliques)):
                    L, R = bicliques[i]
                    if len(L) <= 1 and len(R) <= 1:
                        continue
                    # Try removing one edge and seeing if we can place it elsewhere
                    # This is complex, skip for now

        return bicliques

    best_bicliques = None
    best_count = float('inf')

    # Run multiple restarts with different seeds
    seed = 42
    num_restarts = 0
    while time.time() < deadline - 5:
        result = greedy_solve(seed)
        if result is not None:
            # Apply local search
            result = local_search(result)
            count = len(result)
            if count < best_count:
                best_count = count
                best_bicliques = result
        seed += 1
        num_restarts += 1

        # Early stopping if we've found a very good solution
        if best_count <= 10:
            break

    if best_bicliques is None:
        # Fallback: trivial partition
        trivial = []
        for u, v in edges:
            trivial.append([[u], [v]])
        best_bicliques = trivial

    # Convert to required format with strict types
    result = []
    for L, R in best_bicliques:
        result.append([[int(x) for x in L], [int(y) for y in R]])

    return result
