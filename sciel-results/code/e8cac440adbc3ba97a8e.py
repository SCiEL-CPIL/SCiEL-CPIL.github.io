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
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # MRV (Minimum Remaining Values) + forward checking backtracking
        colour = [-1] * n
        avail = [set(l) for l in lists]

        def backtrack():
            # Find uncoloured vertex with fewest available colours (MRV)
            min_count = 999
            v = -1
            for i in range(n):
                if colour[i] == -1:
                    cnt = len(avail[i])
                    if cnt == 0:
                        return False  # Dead end
                    if cnt < min_count:
                        min_count = cnt
                        v = i
            if v == -1:
                return True  # All coloured

            # Try each available colour for v
            for c in sorted(avail[v]):
                # Check compatibility with already-coloured neighbours
                conflict = False
                for u in adj[v]:
                    if colour[u] != -1 and colour[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue

                # Assign colour c to v
                colour[v] = c
                # Forward checking: remove c from uncoloured neighbours
                saved = []
                valid = True
                for u in adj[v]:
                    if colour[u] == -1 and c in avail[u]:
                        avail[u].remove(c)
                        saved.append(u)
                        if not avail[u]:
                            valid = False
                            break

                if valid and backtrack():
                    return True

                # Undo
                colour[v] = -1
                for u in saved:
                    avail[u].add(c)

            return False

        if backtrack():
            return [int(c) for c in colour]
        else:
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        out.append(solve_instance(n, edges, lists))
    return out
# EVOLVE-BLOCK-END
