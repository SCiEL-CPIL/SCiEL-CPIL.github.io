import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']

    # Build adjacency as sets for fast membership
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Remaining edges as set of normalized tuples
    remaining = set()
    for u, v in edges:
        if u < v:
            remaining.add((u, v))
        else:
            remaining.add((v, u))

    def count_covered(L, R, rem):
        """Count how many edges in rem are covered by biclique (L, R)."""
        cnt = 0
        for u in L:
            for v in R:
                if u < v:
                    if (u, v) in rem:
                        cnt += 1
                else:
                    if (v, u) in rem:
                        cnt += 1
        return cnt

    def is_valid_biclique(L, R):
        for u in L:
            for v in R:
                if v not in adj[u]:
                    return False
        return True

    def expand_vertex_anchored(anchor, rem):
        """Build a biclique anchored at a vertex, growing both sides."""
        # Start: anchor on left, its neighbors in rem on right
        left = {anchor}
        right = set()
        for u, v in rem:
            if u == anchor:
                right.add(v)
            elif v == anchor:
                right.add(u)

        if not right:
            return None, None

        # Iteratively refine: filter to common neighbors, then expand
        for _ in range(30):
            # Filter left: must be adjacent to all in right
            new_left = set()
            for u in left:
                if right.issubset(adj[u]):
                    new_left.add(u)
            if new_left != left:
                left = new_left

            # Filter right: must be adjacent to all in left
            new_right = set()
            for v in right:
                if left.issubset(adj[v]):
                    new_right.add(v)
            if new_right != right:
                right = new_right

            # Expand left: add vertices adjacent to all in right
            expanded = False
            for w in range(n):
                if w not in left and w not in right:
                    if right.issubset(adj[w]):
                        # Check w has at least one remaining edge to right
                        has_edge = False
                        for v in right:
                            if (min(w, v), max(w, v)) in rem:
                                has_edge = True
                                break
                        if has_edge:
                            left.add(w)
                            expanded = True

            # Expand right: add vertices adjacent to all in left
            for w in range(n):
                if w not in left and w not in right:
                    if left.issubset(adj[w]):
                        has_edge = False
                        for u in left:
                            if (min(w, u), max(w, u)) in rem:
                                has_edge = True
                                break
                        if has_edge:
                            right.add(w)
                            expanded = True

            if not expanded and new_left == left and new_right == right:
                break

        if not left or not right:
            return None, None
        return left, right

    def expand_edge_anchored(u, v, rem):
        """Build a biclique from an edge (u,v)."""
        left = {u}
        right = {v}

        for _ in range(30):
            # Filter left: must be adjacent to all in right
            new_left = set()
            for x in left:
                if right.issubset(adj[x]):
                    new_left.add(x)
            left = new_left

            # Filter right: must be adjacent to all in left
            new_right = set()
            for y in right:
                if left.issubset(adj[y]):
                    new_right.add(y)
            right = new_right

            # Expand left
            for w in range(n):
                if w not in left and w not in right:
                    if right.issubset(adj[w]):
                        has_edge = False
                        for r in right:
                            if (min(w, r), max(w, r)) in rem:
                                has_edge = True
                                break
                        if has_edge:
                            left.add(w)

            # Expand right
            for w in range(n):
                if w not in left and w not in right:
                    if left.issubset(adj[w]):
                        has_edge = False
                        for l in left:
                            if (min(w, l), max(w, l)) in rem:
                                has_edge = True
                                break
                        if has_edge:
                            right.add(w)

        if not left or not right:
            return None, None
        return left, right

    def find_best_biclique(rem):
        """Try multiple strategies and return the biclique covering most edges."""
        if not rem:
            return None, None

        # Compute degrees in remaining graph
        deg = [0] * n
        for u, v in rem:
            deg[u] += 1
            deg[v] += 1

        best_L, best_R = None, None
        best_count = 0

        # Strategy 1: Vertex-anchored (top degree vertices)
        candidates = sorted(range(n), key=lambda x: -deg[x])
        for anchor in candidates[:20]:
            if deg[anchor] == 0:
                break
            L, R = expand_vertex_anchored(anchor, rem)
            if L and R and is_valid_biclique(L, R):
                cnt = count_covered(L, R, rem)
                if cnt > best_count:
                    best_count = cnt
                    best_L, best_R = L, R

        # Strategy 2: Edge-anchored (top degree edges)
        edge_list = sorted(rem, key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
        for u, v in edge_list[:80]:
            L, R = expand_edge_anchored(u, v, rem)
            if L and R and is_valid_biclique(L, R):
                cnt = count_covered(L, R, rem)
                if cnt > best_count:
                    best_count = cnt
                    best_L, best_R = L, R

        # Strategy 3: Random sampling of edges
        import random
        rng = random.Random(42)
        edge_list_all = list(rem)
        rng.shuffle(edge_list_all)
        for u, v in edge_list_all[:50]:
            L, R = expand_edge_anchored(u, v, rem)
            if L and R and is_valid_biclique(L, R):
                cnt = count_covered(L, R, rem)
                if cnt > best_count:
                    best_count = cnt
                    best_L, best_R = L, R

        if best_L is None:
            u, v = next(iter(rem))
            best_L, best_R = {u}, {v}

        return best_L, best_R

    # Main greedy cover
    bicliques = []

    while remaining:
        if time.time() > deadline - 5.0:
            break

        L, R = find_best_biclique(remaining)
        if not L or not R:
            u, v = next(iter(remaining))
            L, R = {u}, {v}

        # Collect edges covered
        to_remove = set()
        for u, v in remaining:
            if (u in L and v in R) or (u in R and v in L):
                to_remove.add((u, v))

        if not to_remove:
            u, v = next(iter(remaining))
            to_remove = {(u, v)}
            L, R = {u}, {v}

        bicliques.append([sorted(L), sorted(R)])
        remaining -= to_remove

    # Safety: cover remaining edges individually
    while remaining:
        if time.time() > deadline - 1.0:
            break
        u, v = next(iter(remaining))
        bicliques.append([[u], [v]])
        remaining.discard((u, v))

    # Force cover any leftover
    while remaining:
        u, v = next(iter(remaining))
        bicliques.append([[u], [v]])
        remaining.discard((u, v))

    # Local improvement: try to merge pairs of small bicliques
    if time.time() < deadline - 3.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            new_bicliques = list(bicliques)
            for i in range(len(new_bicliques)):
                if time.time() > deadline - 3.0:
                    break
                for j in range(i + 1, len(new_bicliques)):
                    Li, Ri = set(new_bicliques[i][0]), set(new_bicliques[i][1])
                    Lj, Rj = set(new_bicliques[j][0]), set(new_bicliques[j][1])

                    # Try merging: union of lefts and rights must form valid biclique
                    # Option A: Li ∪ Lj as left, Ri ∪ Rj as right
                    merged_L = Li | Lj
                    merged_R = Ri | Rj
                    if merged_L and merged_R and is_valid_biclique(merged_L, merged_R):
                        # Check no overlap in covered edges
                        edges_i = set()
                        for u in Li:
                            for v in Ri:
                                edges_i.add((min(u, v), max(u, v)))
                        edges_j = set()
                        for u in Lj:
                            for v in Rj:
                                edges_j.add((min(u, v), max(u, v)))
                        if not edges_i & edges_j:
                            new_bicliques.pop(j)
                            new_bicliques.pop(i)
                            new_bicliques.append([sorted(merged_L), sorted(merged_R)])
                            improved = True
                            break
                    if improved:
                        break

            bicliques = new_bicliques

    # Convert to proper types
    result = []
    for b in bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        if left and right:
            result.append([left, right])

    return result
