def construct():
    """Construct two disjoint K_{3,3} components on vertices 0..11.
    
    K_{3,3} is bipartite, triangle-free, has minimum degree 3,
    and all simple cycles have length 4 or 6 (exactly 2 distinct lengths).
    
    Analytical argument that 1 distinct cycle length is impossible with
    min-degree 3:
      - Girth 3: min-degree-3 graphs with girth 3 have cycles of length > 3
        (e.g., K4 has 3-cycles AND 4-cycles).
      - Girth 4: all-cycles-4 requires each vertex on exactly one 4-cycle,
        forcing degree 2. Contradicts min-degree 3.
      - Girth >= 5: all-cycles-k requires each vertex on exactly one k-cycle,
        forcing degree 2. Contradicts min-degree 3.
    Hence 2 is the global minimum.
    """
    import time
    import itertools

    deadline = time.time() + 85.0  # safety margin before 90s kill

    # --- Direct construction: two K_{3,3} ---
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]

    edges = []
    for u in A1:
        for v in B1:
            edges.append([int(u), int(v)])
    for u in A2:
        for v in B2:
            edges.append([int(u), int(v)])

    # --- Self-verification ---
    def verify(edge_list):
        # Check: exactly 12 vertices 0..11, no loops, no duplicates
        if len(edge_list) != 18:
            return False
        seen = set()
        for e in edge_list:
            u, v = e
            if not isinstance(u, int) or isinstance(u, bool):
                return False
            if not isinstance(v, int) or isinstance(v, bool):
                return False
            if u == v:
                return False
            if u < 0 or u > 11 or v < 0 or v > 11:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        # Check min degree >= 3
        degree = [0] * 12
        for u, v in edge_list:
            degree[u] += 1
            degree[v] += 1
        for d in degree:
            if d < 3:
                return False
        # Check triangle-free: no three mutually connected vertices
        adj = set(seen)
        for i in range(12):
            for j in range(i + 1, 12):
                if (i, j) in adj:
                    for k in range(j + 1, 12):
                        if (i, k) in adj and (j, k) in adj:
                            return False
        # Check contains a cycle: 18 edges, 12 vertices, connected components
        # K_{3,3} has 3+3-1=5 spanning tree edges, 9-5=4 extra edges -> cycles
        # Just confirm edge count > vertex count - number_of_components
        # Two components, each with 6 vertices: 12 - 2 = 10 < 18, so cycles exist
        return True

    if not verify(edges):
        # Should never happen, but fall through to search
        pass
    else:
        return edges

    # --- Fallback: fast random search for a score-1 graph (theoretically impossible) ---
    # This is a safety net only; the analytical construction above is returned if valid.
    import random

    def get_cycle_lengths(edge_list):
        """Return the set of distinct simple cycle lengths."""
        n = 12
        adj = [[] for _ in range(n)]
        for u, v in edge_list:
            adj[u].append(v)
            adj[v].append(u)

        lengths = set()

        def dfs_cycles(start, length_limit=12):
            # Find all simple cycles starting from 'start' using DFS
            # with backtracking; limit length to avoid explosion
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                if len(path) > length_limit:
                    continue
                for nb in adj[node]:
                    if nb == start and len(path) >= 4:
                        lengths.add(len(path))
                    elif nb not in visited and nb > start:
                        # Only count each cycle once (start is the smallest vertex)
                        new_visited = visited | {nb}
                        stack.append((nb, path + [nb], new_visited))

        for s in range(n):
            dfs_cycles(s)
        return lengths

    best_edges = edges  # default: the K_{3,3} solution
    best_count = 2

    # Very small random search: try random 3-regular graphs
    rng = random.Random(42)
    attempts = 0
    max_attempts = 200

    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        # Generate a random 3-regular graph on 12 vertices
        # Using the configuration model
        half_edges = list(range(12)) * 3  # 36 half-edges
        rng.shuffle(half_edges)
        # Pair them up
        raw_pairs = [(half_edges[i], half_edges[i + 1]) for i in range(0, 36, 2)]
        # Check for loops and multi-edges
        valid = True
        edge_set = set()
        for u, v in raw_pairs:
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        if not valid:
            continue
        # Convert to edge list
        cand = [[int(u), int(v)] for u, v in edge_set]
        # Check min degree
        degree = [0] * 12
        for u, v in cand:
            degree[u] += 1
            degree[v] += 1
        if min(degree) < 3:
            continue
        # Check triangle-free
        adj_set = set((min(u, v), max(u, v)) for u, v in cand)
        tri = False
        for i in range(12):
            for j in range(i + 1, 12):
                if (i, j) in adj_set:
                    for k in range(j + 1, 12):
                        if (i, k) in adj_set and (j, k) in adj_set:
                            tri = True
                            break
                    if tri:
                        break
            if tri:
                break
        if tri:
            continue
        # Count distinct cycle lengths
        cl = get_cycle_lengths(cand)
        if not cl:
            continue  # must contain a cycle
        cnt = len(cl)
        if cnt < best_count:
            best_count = cnt
            best_edges = cand
            if cnt == 1:
                return best_edges

    # Return the best found (should be the K_{3,3} with count=2)
    # Ensure strict types
    return [[int(u), int(v)] for u, v in best_edges]
