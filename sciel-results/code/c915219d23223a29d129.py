import time
import sys
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    def make_edge(u, v):
        return (min(u, v), max(u, v))

    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        e = make_edge(u, v)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add(e)

    total_edges = len(edge_set)

    def time_left():
        return deadline - time.time()

    def biclique_edges(L, R):
        """Return set of edges covered by biclique (L, R)."""
        out = set()
        for l in L:
            for r in R:
                if l != r:
                    out.add(make_edge(l, r))
        return out

    def is_biclique(L, R, remaining):
        """Check if (L, R) is a valid biclique within remaining edges."""
        for l in L:
            for r in R:
                if l == r:
                    continue
                if make_edge(l, r) not in remaining:
                    return False
        return True

    def expand_biclique(L, R, remaining):
        """Expand (L, R) to a maximal biclique within remaining edges."""
        L = set(L)
        R = set(R)
        if not L or not R:
            return L, R
        changed = True
        while changed and time_left() > 0.1:
            changed = False
            # Try adding vertices to L
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for r in R:
                    if make_edge(w, r) not in remaining:
                        ok = False
                        break
                if ok:
                    L.add(w)
                    changed = True
            # Try adding vertices to R
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for l in L:
                    if make_edge(w, l) not in remaining:
                        ok = False
                        break
                if ok:
                    R.add(w)
                    changed = True
        return L, R

    def find_biclique_star(center, remaining):
        """Find a biclique centered at 'center' using its neighbors."""
        neighbors = set()
        for v in range(n):
            if v != center and make_edge(center, v) in remaining:
                neighbors.add(v)
        if not neighbors:
            return None
        R = set(neighbors)
        L = {center}
        L, R = expand_biclique(L, R, remaining)
        if L and R:
            return (L, R)
        return None

    def find_biclique_edge(u, v, remaining):
        """Find a biclique seeded by edge (u, v)."""
        L = {u}
        R = {v}
        L, R = expand_biclique(L, R, remaining)
        if L and R:
            return (L, R)
        return None

    def greedy_partition(remaining, strategy='star'):
        """Greedy partition using the specified strategy."""
        bicliques = []
        rem = set(remaining)

        while rem and time_left() > 1.0:
            best_biclique = None
            best_size = 0

            if strategy == 'star':
                # Try each vertex as center
                for center in range(n):
                    if time_left() < 0.5:
                        break
                    b = find_biclique_star(center, rem)
                    if b:
                        L, R = b
                        sz = len(L) * len(R)
                        if sz > best_size:
                            best_size = sz
                            best_biclique = b
            elif strategy == 'edge':
                # Try top-degree edges
                deg = {v: 0 for v in range(n)}
                for (u, v) in rem:
                    deg[u] += 1
                    deg[v] += 1
                candidates = sorted(rem, key=lambda e: (-(deg[e[0]] + deg[e[1]]), e[0], e[1]))
                for (u, v) in candidates[:30]:  # Limit candidates for speed
                    if time_left() < 0.5:
                        break
                    b = find_biclique_edge(u, v, rem)
                    if b:
                        L, R = b
                        sz = len(L) * len(R)
                        if sz > best_size:
                            best_size = sz
                            best_biclique = b
            else:
                # Mixed: try both
                for center in range(n):
                    if time_left() < 0.5:
                        break
                    b = find_biclique_star(center, rem)
                    if b:
                        L, R = b
                        sz = len(L) * len(R)
                        if sz > best_size:
                            best_size = sz
                            best_biclique = b
                deg = {v: 0 for v in range(n)}
                for (u, v) in rem:
                    deg[u] += 1
                    deg[v] += 1
                candidates = sorted(rem, key=lambda e: (-(deg[e[0]] + deg[e[1]]), e[0], e[1]))
                for (u, v) in candidates[:20]:
                    if time_left() < 0.5:
                        break
                    b = find_biclique_edge(u, v, rem)
                    if b:
                        L, R = b
                        sz = len(L) * len(R)
                        if sz > best_size:
                            best_size = sz
                            best_biclique = b

            if best_biclique is None:
                # Fallback: single edge
                e = next(iter(rem))
                best_biclique = ({e[0]}, {e[1]})

            bicliques.append(best_biclique)
            covered = biclique_edges(best_biclique[0], best_biclique[1])
            for e in covered:
                rem.discard(e)

        # Handle remaining edges
        for e in rem:
            bicliques.append(({e[0]}, {e[1]}))

        return bicliques

    def validate_partition(bicliques, all_edges):
        """Check that the partition is valid."""
        if not bicliques:
            return False
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            Ls, Rs = set(L), set(R)
            if Ls & Rs:
                return False
            for u in L:
                for v in R:
                    e = make_edge(u, v)
                    if e not in all_edges:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == all_edges

    def local_search(bicliques, all_edges):
        """Local search to reduce biclique count via merges and moves."""
        bicliques = [(set(L), set(R)) for L, R in bicliques]
        improved = True
        max_iterations = 100

        while improved and time_left() > 2.0:
            improved = False
            for _ in range(max_iterations):
                if time_left() < 2.0:
                    break

                # Try merging pairs
                merged = False
                for i in range(len(bicliques)):
                    if time_left() < 2.0:
                        break
                    for j in range(i + 1, len(bicliques)):
                        if time_left() < 2.0:
                            break
                        L_i, R_i = bicliques[i]
                        L_j, R_j = bicliques[j]

                        # Try (L_i ∪ L_j, R_i ∪ R_j)
                        new_L = L_i | L_j
                        new_R = R_i | R_j
                        if not (new_L & new_R):
                            valid = True
                            for u in new_L:
                                for v in new_R:
                                    if make_edge(u, v) not in all_edges:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                new_bicliques = []
                                for k in range(len(bicliques)):
                                    if k != i and k != j:
                                        new_bicliques.append(bicliques[k])
                                new_bicliques.append((new_L, new_R))
                                bicliques = new_bicliques
                                improved = True
                                merged = True
                                break

                        # Try (L_i ∪ R_j, R_i ∪ L_j)
                        new_L = L_i | R_j
                        new_R = R_i | L_j
                        if not (new_L & new_R):
                            valid = True
                            for u in new_L:
                                for v in new_R:
                                    if make_edge(u, v) not in all_edges:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                new_bicliques = []
                                for k in range(len(bicliques)):
                                    if k != i and k != j:
                                        new_bicliques.append(bicliques[k])
                                new_bicliques.append((new_L, new_R))
                                bicliques = new_bicliques
                                improved = True
                                merged = True
                                break
                    if merged:
                        break

                if merged:
                    continue

                # Try moving a vertex from one biclique to another
                moved = False
                for i in range(len(bicliques)):
                    if time_left() < 2.0:
                        break
                    L_i, R_i = bicliques[i]
                    if len(L_i) < 2 and len(R_i) < 2:
                        continue

                    # Try moving a vertex from L_i to another biclique's L or R
                    for v in list(L_i):
                        if time_left() < 2.0:
                            break
                        for j in range(len(bicliques)):
                            if j == i:
                                continue
                            L_j, R_j = bicliques[j]

                            # Option 1: Move v to L_j
                            new_L = L_j | {v}
                            new_R = R_j
                            if not (new_L & new_R):
                                valid = True
                                for u in new_L:
                                    for w in new_R:
                                        if make_edge(u, w) not in all_edges:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    # Check if removing v from L_i still leaves a valid biclique
                                    new_L_i = L_i - {v}
                                    if new_L_i:
                                        valid_i = True
                                        for u in new_L_i:
                                            for w in R_i:
                                                if make_edge(u, w) not in all_edges:
                                                    valid_i = False
                                                    break
                                            if not valid_i:
                                                break
                                        if valid_i:
                                            bicliques[i] = (new_L_i, R_i)
                                            bicliques[j] = (new_L, new_R)
                                            improved = True
                                            moved = True
                                            break
                            if moved:
                                break

                            # Option 2: Move v to R_j
                            new_L = L_j
                            new_R = R_j | {v}
                            if not (new_L & new_R):
                                valid = True
                                for u in new_L:
                                    for w in new_R:
                                        if make_edge(u, w) not in all_edges:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    new_L_i = L_i - {v}
                                    if new_L_i:
                                        valid_i = True
                                        for u in new_L_i:
                                            for w in R_i:
                                                if make_edge(u, w) not in all_edges:
                                                    valid_i = False
                                                    break
                                            if not valid_i:
                                                break
                                        if valid_i:
                                            bicliques[i] = (new_L_i, R_i)
                                            bicliques[j] = (new_L, new_R)
                                            improved = True
                                            moved = True
                                            break
                            if moved:
                                break
                        if moved:
                            break
                    if moved:
                        break

                # Try moving a vertex from R_i to another biclique
                if not moved:
                    for i in range(len(bicliques)):
                        if time_left() < 2.0:
                            break
                        L_i, R_i = bicliques[i]
                        if len(L_i) < 2 and len(R_i) < 2:
                            continue

                        for v in list(R_i):
                            if time_left() < 2.0:
                                break
                            for j in range(len(bicliques)):
                                if j == i:
                                    continue
                                L_j, R_j = bicliques[j]

                                # Option 1: Move v to L_j
                                new_L = L_j | {v}
                                new_R = R_j
                                if not (new_L & new_R):
                                    valid = True
                                    for u in new_L:
                                        for w in new_R:
                                            if make_edge(u, w) not in all_edges:
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if valid:
                                        new_R_i = R_i - {v}
                                        if new_R_i:
                                            valid_i = True
                                            for u in L_i:
                                                for w in new_R_i:
                                                    if make_edge(u, w) not in all_edges:
                                                        valid_i = False
                                                        break
                                                if not valid_i:
                                                    break
                                            if valid_i:
                                                bicliques[i] = (L_i, new_R_i)
                                                bicliques[j] = (new_L, new_R)
                                                improved = True
                                                moved = True
                                                break
                                if moved:
                                    break

                                # Option 2: Move v to R_j
                                new_L = L_j
                                new_R = R_j | {v}
                                if not (new_L & new_R):
                                    valid = True
                                    for u in new_L:
                                        for w in new_R:
                                            if make_edge(u, w) not in all_edges:
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if valid:
                                        new_R_i = R_i - {v}
                                        if new_R_i:
                                            valid_i = True
                                            for u in L_i:
                                                for w in new_R_i:
                                                    if make_edge(u, w) not in all_edges:
                                                        valid_i = False
                                                        break
                                                if not valid_i:
                                                    break
                                            if valid_i:
                                                bicliques[i] = (L_i, new_R_i)
                                                bicliques[j] = (new_L, new_R)
                                                improved = True
                                                moved = True
                                                break
                                if moved:
                                    break
                            if moved:
                                break
                        if moved:
                            break

                if not improved:
                    break

        return bicliques

    # Try multiple strategies and keep the best
    best_result = None
    best_count = float('inf')

    strategies = ['star', 'edge', 'mixed']
    for strat in strategies:
        if time_left() < 5.0:
            break
        initial = greedy_partition(edge_set, strategy=strat)
        if validate_partition(initial, edge_set):
            refined = local_search(initial, edge_set)
            if validate_partition(refined, edge_set):
                count = len(refined)
                if count < best_count:
                    best_count = count
                    best_result = refined

    # If no valid result, fallback to single-edge bicliques
    if best_result is None:
        best_result = [({u}, {v}) for u, v in edge_set]

    # Convert to required format with strict types
    result = []
    for L, R in best_result:
        left = [int(v) for v in L]
        right = [int(v) for v in R]
        result.append([left, right])

    return result
