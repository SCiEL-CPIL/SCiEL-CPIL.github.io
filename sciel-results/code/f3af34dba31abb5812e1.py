"""Combined packing program: large-first backtracking with multi-seed restarts."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def _hopcroft_karp(adj, n_left, n_right):
    """Hopcroft-Karp max bipartite matching. adj[i] = list of right nodes for left node i."""
    from collections import deque
    INF = float('inf')
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        queue = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return matching, pair_u

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    trees = instance_trees(20, 611)
    n = 20

    # Precompute tree adjacency and BFS orders
    tree_info = []
    for idx in range(n):
        tree = trees[idx]
        k = len(tree) + 1
        if k == 1:
            tree_info.append((tree, [], [], [0]))
            continue
        adj = [[] for _ in range(k)]
        for u, v in tree:
            adj[u].append(v)
            adj[v].append(u)
        # BFS from vertex 0
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        queue = [0]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
        tree_info.append((tree, adj, parent, order))

    # All 190 edges of K20
    all_edges = set()
    for a in range(n):
        for b in range(a + 1, n):
            all_edges.add((a, b))

    # Sort trees by size descending for packing order
    tree_order = list(range(n))
    tree_order.sort(key=lambda i: len(trees[i]), reverse=True)

    def find_embedding(tree_idx, used_edges, rng):
        """Find injective embedding of tree tree_idx into K20 avoiding used_edges."""
        tree, adj, parent, order = tree_info[tree_idx]
        k = len(order)
        if k == 1:
            for hv in range(n):
                return [hv]
            return None

        host = [-1] * k
        used_hosts = set()

        def get_candidates(tv, p_tv, p_hv):
            cands = []
            for hv in range(n):
                if hv in used_hosts:
                    continue
                if p_tv != -1:
                    e = (min(hv, p_hv), max(hv, p_hv))
                    if e in used_edges:
                        continue
                cands.append(hv)
            return cands

        def count_future_options(hv):
            """Count how many unused edges emanate from hv to unused hosts."""
            cnt = 0
            for nh in range(n):
                if nh in used_hosts or nh == hv:
                    continue
                e = (min(hv, nh), max(hv, nh))
                if e not in used_edges:
                    cnt += 1
            return cnt

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == k:
                return True

            tv = order[idx]
            p_tv = parent[tv]
            p_hv = host[p_tv] if p_tv != -1 else -1

            cands = get_candidates(tv, p_tv, p_hv)
            if not cands:
                return False

            # Sort by minimum future options (MRV heuristic)
            if len(cands) > 2:
                cands.sort(key=count_future_options)

            # Add slight randomness to break symmetry
            if rng.random() < 0.2 and len(cands) > 3:
                cands[0], cands[1] = cands[1], cands[0]

            for hv in cands:
                host[tv] = hv
                used_hosts.add(hv)
                if backtrack(idx + 1):
                    return True
                used_hosts.remove(hv)
                host[tv] = -1

            return False

        if backtrack(0):
            return host[:]
        return None

    def solve_with_backtracking(seed, order):
        """
        Recursive backtracking over tree assignments.
        Tries to embed trees in given order; if a tree fails,
        backtracks to try different embedding for a previous tree.
        """
        rng = Random(seed)
        used = set()
        res = [None] * n

        def try_place(pos):
            """Try to place trees from position pos onward in order."""
            if time.time() > deadline:
                return False
            if pos == len(order):
                return True

            tree_idx = order[pos]
            tree, _, _, _ = tree_info[tree_idx]
            k = len(tree) + 1

            # Try multiple embeddings for this tree
            max_attempts = 8
            for attempt in range(max_attempts):
                embedding = find_embedding(tree_idx, used, rng)
                if embedding is None:
                    break

                # Check if this embedding is valid (all edges unused)
                valid = True
                for u, v in tree:
                    hu, hv = embedding[u], embedding[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used:
                        valid = False
                        break
                if not valid:
                    break

                # Place this tree
                res[tree_idx] = embedding
                for u, v in tree:
                    hu, hv = embedding[u], embedding[v]
                    e = (min(hu, hv), max(hu, hv))
                    used.add(e)

                # Recurse
                if try_place(pos + 1):
                    return True

                # Undo
                for u, v in tree:
                    hu, hv = embedding[u], embedding[v]
                    e = (min(hu, hv), max(hu, hv))
                    used.discard(e)
                res[tree_idx] = None

            return False

        if try_place(0):
            return res, sum(len(trees[i]) for i in range(n) if res[i] is not None)
        else:
            partial_score = sum(len(trees[i]) for i in range(n) if res[i] is not None)
            return res, partial_score

    best_res = None
    best_score = 0

    # Try multiple seeds and orders
    seed_counter = 0
    for attempt in range(200):
        if time.time() > deadline:
            break
        seed_counter += 1

        # Alternate between different orderings
        if attempt % 5 == 0:
            order = tree_order  # descending
        elif attempt % 5 == 1:
            order = tree_order[::-1]  # ascending
        else:
            rng_tmp = Random(attempt)
            order = list(range(n))
            rng_tmp.shuffle(order)

        res, score = solve_with_backtracking(seed_counter, order)

        if score > best_score:
            best_score = score
            best_res = res

        if best_score == 190:
            break

    if best_res is None:
        fallback = [None] * n
        fallback[0] = [0]
        fallback[1] = [0, 1]
        return fallback

    final_result = []
    for i in range(n):
        if best_res[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_res[i]])

    return final_result
