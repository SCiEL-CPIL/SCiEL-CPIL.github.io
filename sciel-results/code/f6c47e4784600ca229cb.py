import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45  # safety margin before 60s limit

    n = 12

    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_cycle_of_length(adj, L):
        if L == 4:
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        common = adj[u] & adj[v]
                        if len(common) >= 2:
                            return True
            return False

        if L == 6:
            for start in range(n):
                stack = [(start, [start], {start})]
                while stack:
                    curr, path, visited = stack.pop()
                    if len(path) == L:
                        if start in adj[curr]:
                            return True
                        continue
                    for nb in adj[curr]:
                        if nb == start and len(path) < L - 1:
                            continue
                        if nb == start and len(path) == L - 1:
                            return True
                        if nb not in visited and nb > start:
                            stack.append((nb, path + [nb], visited | {nb}))
            return False

        for start in range(n):
            stack = [(start, [start], {start})]
            while stack:
                curr, path, visited = stack.pop()
                if len(path) == L:
                    if start in adj[curr]:
                        return True
                    continue
                for nb in adj[curr]:
                    if nb == start and len(path) < L - 1:
                        continue
                    if nb == start and len(path) == L - 1:
                        return True
                    if nb not in visited and nb > start:
                        stack.append((nb, path + [nb], visited | {nb}))
        return False

    def get_cycle_lengths(adj):
        lengths = set()
        for L in range(4, n + 1):
            if has_cycle_of_length(adj, L):
                lengths.add(L)
        return lengths

    def is_valid(edges):
        if len(edges) == 0:
            return False
        adj = build_adj(edges)
        for i in range(n):
            if len(adj[i]) < 3:
                return False
        for i in range(n):
            nbrs = list(adj[i])
            for j in range(len(nbrs)):
                for k in range(j + 1, len(nbrs)):
                    if nbrs[k] in adj[nbrs[j]]:
                        return False
        # Check for at least one cycle
        visited = [False] * n
        has_cycle = False
        for i in range(n):
            if not visited[i]:
                from collections import deque
                q = deque([i])
                visited[i] = True
                vertices_in_comp = 0
                edges_in_comp = 0
                while q:
                    curr = q.popleft()
                    vertices_in_comp += 1
                    for nb in adj[curr]:
                        if not visited[nb]:
                            visited[nb] = True
                            q.append(nb)
                        else:
                            edges_in_comp += 1
                edges_in_comp //= 2
                if edges_in_comp >= vertices_in_comp:
                    has_cycle = True
        if not has_cycle:
            return False
        return True

    best_edges = None
    best_score = float('inf')

    # Strategy 1: Clebsch graph vertex removal
    def clebsch_edges():
        verts = []
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    verts.append((a, b, c))
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                vi = verts[i]
                vj = verts[j]
                hamming = sum(1 for k in range(3) if vi[k] != vj[k])
                if hamming == 2:
                    edges.append([i, j])
        return edges

    cleb_edges = clebsch_edges()
    cleb_adj = [set() for _ in range(16)]
    for u, v in cleb_edges:
        cleb_adj[u].add(v)
        cleb_adj[v].add(u)

    all_vertices = list(range(16))

    for remove_set in combinations(all_vertices, 4):
        remove_set = frozenset(remove_set)
        keep_set = [v for v in all_vertices if v not in remove_set]

        valid = True
        for v in keep_set:
            deg = sum(1 for nb in cleb_adj[v] if nb not in remove_set)
            if deg < 3:
                valid = False
                break
        if not valid:
            continue

        old_to_new = {old: new for new, old in enumerate(keep_set)}
        new_edges = []
        for u, v in cleb_edges:
            if u not in remove_set and v not in remove_set:
                new_edges.append([old_to_new[u], old_to_new[v]])

        if len(new_edges) == 0:
            continue

        if not is_valid(new_edges):
            continue

        adj = build_adj(new_edges)
        lengths = get_cycle_lengths(adj)
        score = len(lengths)

        if score < best_score:
            best_score = score
            best_edges = [e[:] for e in new_edges]
            if best_score == 1:
                break

        if time.time() > deadline:
            break

    # Strategy 2: Random 3-regular bipartite search if needed
    if best_score > 1 and time.time() < deadline - 5:
        A = list(range(6))
        B = list(range(6, 12))
        random.seed(42)

        def random_3reg_bipartite():
            for _ in range(200):
                adj_a = {v: [] for v in A}
                adj_b = {v: [] for v in B}
                valid = True
                for a in A:
                    neighbors = random.sample(B, 3)
                    adj_a[a] = neighbors
                    for b in neighbors:
                        adj_b[b].append(a)
                for b in B:
                    if len(adj_b[b]) != 3:
                        valid = False
                        break
                if not valid:
                    continue
                edges = []
                for a in A:
                    for b in adj_a[a]:
                        edges.append([a, b])
                return edges
            return None

        for trial in range(5000):
            if time.time() > deadline - 2:
                break
            edges = random_3reg_bipartite()
            if edges is None:
                continue
            adj = build_adj(edges)
            # Quick 4-cycle check
            has_4 = False
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        if len(adj[u] & adj[v]) >= 2:
                            has_4 = True
                            break
                if has_4:
                    break
            if not has_4:
                continue
            lengths = get_cycle_lengths(adj)
            score = len(lengths)
            if score < best_score:
                best_score = score
                best_edges = [e[:] for e in edges]
                if best_score == 1:
                    break

    # Fallback
    if best_edges is None:
        base_edges = []
        for i in range(3):
            for j in range(3):
                base_edges.append([i, j + 3])
                base_edges.append([i + 6, j + 9])
        best_edges = base_edges

    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    return sorted(result)
