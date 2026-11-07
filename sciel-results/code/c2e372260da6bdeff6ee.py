"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import deque

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def hopcroft_karp(n_left, n_right, left_adj, right_adj):
        """
        Hopcroft-Karp for maximum bipartite matching.
        left_adj[i] = list of right nodes adjacent to left node i
        right_adj[j] = list of left nodes adjacent to right node j
        Returns (match_left, match_right) where match_left[i] = j or -1, match_right[j] = i or -1.
        """
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = deque()
            found = False
            for i in range(n_left):
                if match_left[i] == -1:
                    dist[i] = 0
                    queue.append(i)
                else:
                    dist[i] = -1
            while queue:
                u = queue.popleft()
                for v in left_adj[u]:
                    u2 = match_right[v]
                    if u2 == -1:
                        found = True
                    elif dist[u2] == -1:
                        dist[u2] = dist[u] + 1
                        queue.append(u2)
            return found

        def dfs(u):
            for v in left_adj[u]:
                u2 = match_right[v]
                if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs(u2)):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = -1
            return False

        while bfs():
            for i in range(n_left):
                if match_left[i] == -1:
                    dfs(i)

        return match_left, match_right

    def solve_instance(n, edges, lists):
        # Build bipartite graph:
        # Left side: (vertex, colour) pairs where colour is in lists[vertex]
        # Right side: edges (u, v)
        # Connect left (v, c) to right edge (u, w) if v is u or v and c != colour of the other endpoint
        # Actually, the standard reduction:
        # We want to assign a colour to each vertex such that adjacent vertices have different colours.
        # This is equivalent to: for each edge (u, v), the colour chosen for u must be different from the colour chosen for v.
        # 
        # A better approach using matching:
        # Since each vertex has a list of 3 colours, and the graph is bipartite, we can use the following:
        # Let's think of it differently. The problem is list colouring of a bipartite graph.
        # By Alon-Tarsi, it's always colourable.
        #
        # Alternative: Use a flow/matching approach.
        # For each vertex v, create 3 copies (v, c) for c in lists[v].
        # For each edge (u, v), we need to ensure that the colour chosen for u is not the same as for v.
        # This can be modeled as: we want to select exactly one colour-copy per vertex, such that no two adjacent vertices share the same colour.
        #
        # This is a constraint satisfaction problem. Let me use a different matching formulation.
        #
        # Actually, let me use a simpler approach: since the graph is bipartite and each vertex has 3 colours,
        # I can use a randomized approach with backtracking, or use the fact that list colouring of bipartite
        # graphs with lists of size >= 2 is always possible (this is a known result, stronger than Alon-Tarsi).
        #
        # Let me try a greedy approach with a smarter ordering, and if it fails, use backtracking.
        #
        # Actually, let me implement a proper backtracking solver with memoization on the bipartite structure.
        #
        # For efficiency, let me use a recursive backtracking with forward checking.

        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        # Map from position in order to vertex
        pos = {v: i for i, v in enumerate(order)}

        # We'll use backtracking with forward checking
        colour = [-1] * n
        # For each vertex, maintain the set of available colours (initially lists[v])
        avail = [set(l) for l in lists]

        def backtrack(idx):
            if idx == n:
                return True

            v = order[idx]
            # Try each available colour for v
            for c in sorted(avail[v]):
                # Check if c is compatible with all already-coloured neighbours
                conflict = False
                for u in adj[v]:
                    if colour[u] != -1 and colour[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue

                # Assign colour c to v
                colour[v] = c
                # Forward checking: remove c from available colours of uncoloured neighbours
                saved = []
                valid = True
                for u in adj[v]:
                    if colour[u] == -1 and c in avail[u]:
                        avail[u].remove(c)
                        saved.append(u)
                        if not avail[u]:
                            valid = False
                            break

                if valid and backtrack(idx + 1):
                    return True

                # Undo
                colour[v] = -1
                for u in saved:
                    avail[u].add(c)

            return False

        if backtrack(0):
            return [int(c) for c in colour]
        else:
            # Should not happen by Alon-Tarsi, but return a fallback
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        out.append(solve_instance(n, edges, lists))
    return out
# EVOLVE-BLOCK-END
