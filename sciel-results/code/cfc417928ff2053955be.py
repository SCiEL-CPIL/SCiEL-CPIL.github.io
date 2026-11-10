"""Combined program: path-centric CSP with Hall pruning + bipartite-matching fast path."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before the 90s kill

    n = P['n']
    edges = P['edges']
    es = set()
    for e in edges:
        es.add((e[0], e[1]))
        es.add((e[1], e[0]))

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]

    def bfs_path(start, end, forbidden, adj_list):
        """Shortest simple path start->end, forbidding interior vertices in `forbidden`."""
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj_list[u]:
                if v in visited:
                    continue
                if v != end and v in forbidden:
                    continue
                visited.add(v)
                parent[v] = u
                if v == end:
                    path = [v]
                    p = u
                    while p is not None:
                        path.append(p)
                        p = parent[p]
                    path.reverse()
                    return path
                queue.append(v)
        return None

    def bipartite_max_matching(adj_map):
        """Hopcroft-Karp. Returns (matching, unmatched_left)."""
        from collections import defaultdict
        matchL = {}
        matchR = {}
        dist = {}
        INF = float('inf')

        def bfs():
            dq = deque()
            for u in matchL:
                if matchL[u] is None:
                    dist[u] = 0
                    dq.append(u)
                else:
                    dist[u] = INF
            found = False
            while dq:
                u = dq.popleft()
                for v in adj_map[u]:
                    mu = matchR.get(v)
                    if mu is None:
                        found = True
                    elif dist[mu] == INF:
                        dist[mu] = dist[u] + 1
                        dq.append(mu)
            return found

        def dfs(u):
            for v in adj_map[u]:
                mu = matchR.get(v)
                if mu is None or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = INF
            return False

        while bfs():
            for u in list(matchL.keys()):
                if matchL[u] is None:
                    dfs(u)

        unmatched = [u for u in matchL if matchL[u] is None]
        return matchL, unmatched

    def hall_feasible(pairs, forbidden, adj_list, adj):
        """Hall's condition: for every subset S of pairs, |union of common-neighbors| >= |S|.
        Common-neighbor for pair (a,b) = N(a) & N(b) - forbidden - branches.
        Prune: if |S| > available interior count -> infeasible.
        Only check small subsets (|S|<=3) for tractability; returns False if violated."""
        avail_count = n - len(forbidden)
        if len(pairs) > avail_count:
            return False
        common = {}
        for (a, b) in pairs:
            cn = (adj[a] & adj[b]) - forbidden
            common[(a, b)] = cn
        # quick per-pair check
        for (a, b) in pairs:
            if len(common[(a, b)]) < 1:
                return False
        # Hall for pairs of pairs
        ps = list(pairs)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                union = common[ps[i]] | common[ps[j]]
                if len(union) < 2:
                    return False
        # Hall for triples (bounded)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                for k in range(j + 1, len(ps)):
                    union = common[ps[i]] | common[ps[j]] | common[ps[k]]
                    if len(union) < 3:
                        return False
        return True

    def try_branch_set(branches, seed_rng):
        """Return list of paths (one per unordered pair) or None."""
        if len(branches) < 2:
            return None
        k = len(branches)
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))

        # Fast path: all pairs share at least one common neighbor -> try full matching
        common = {}
        ok = True
        for (a, b) in pairs:
            cn = (adj[a] & adj[b]) - branch_set
            common[(a, b)] = cn
            if not cn:
                ok = False
                break
        if ok:
            # build adjacency map pair -> common neighbors
            adj_map = {}
            for idx, (a, b) in enumerate(pairs):
                adj_map[idx] = list(common[(a, b)])
            matchL, unmatched = bipartite_max_matching(adj_map)
            if not unmatched:
                paths = []
                for idx, (a, b) in enumerate(pairs):
                    w = matchL[idx]
                    paths.append([a, w, b])
                return paths

        # CSP backtracking over pairs, assigning interior vertex-disjoint paths
        used = set()

        # order pairs by "difficulty": fewest common neighbors first
        def pair_key(p):
            a, b = p
            cn = (adj[a] & adj[b]) - branch_set
            return len(cn)
        ordered = sorted(pairs, key=pair_key)

        def rec(remaining, used, acc):
            if time.time() > deadline:
                return None
            if not remaining:
                return acc
            # Hall check on remaining
            if not hall_feasible(remaining, used | branch_set, adj_list, adj):
                return None
            # MRV: pick pair with fewest candidate interiors
            def cand_count(p):
                a, b = p
                return len((adj[a] & adj[b]) - (used | branch_set))
            idx = min(range(len(remaining)), key=lambda i: cand_count(remaining[i]))
            p = remaining[idx]
            rest = remaining[:idx] + remaining[idx + 1:]
            a, b = p
            cands = sorted((adj[a] & adj[b]) - (used | branch_set),
                           key=lambda v: len(adj[v]))
            for w in cands:
                if time.time() > deadline:
                    return None
                # try length-2 path a-w-b
                new_used = used | {w}
                res = rec(rest, new_used, acc + [[a, w, b]])
                if res is not None:
                    return res
            # if no length-2 worked, try longer simple path (BFS) for this pair
            path = bfs_path(a, b, used | (branch_set - {a, b}), adj_list)
            if path is not None:
                interior = set(path[1:-1])
                if not (interior & used):
                    new_used = used | interior
                    res = rec(rest, new_used, acc + [path])
                    if res is not None:
                        return res
            return None

        # try natural ordered, then a few randomized orderings
        result = rec(ordered, set(), [])
        if result is not None:
            # reorder to match canonical pair order (combinations of sorted branches)
            canon = {}
            for path in result:
                key = tuple(sorted((path[0], path[-1])))
                canon[key] = path
            out = []
            for (a, b) in pairs:
                out.append(canon[tuple(sorted((a, b)))])
            return out

        rng = random.Random(seed_rng)
        for _ in range(4):
            if time.time() > deadline:
                break
            sh = ordered[:]
            rng.shuffle(sh)
            result = rec(sh, set(), [])
            if result is not None:
                canon = {}
                for path in result:
                    key = tuple(sorted((path[0], path[-1])))
                    canon[key] = path
                out = []
                for (a, b) in pairs:
                    out.append(canon[tuple(sorted((a, b)))])
                return out
        return None

    def find_max_clique_greedy():
        best = []
        for s in range(n):
            if time.time() > deadline:
                break
            clique = [s]
            cand = set(adj[s])
            while cand:
                bv = max(cand, key=lambda v: len(adj[v] & set(clique)))
                if len(adj[bv] & set(clique)) < len(clique):
                    break
                clique.append(bv)
                cand = cand & adj[bv]
                cand -= set(clique)
            if len(clique) > len(best):
                best = clique[:]
        return best

    # Start from a clique, then greedily expand branch set
    initial = find_max_clique_greedy()
    best_branches = None
    best_paths = None

    if initial:
        r = try_branch_set(sorted(initial), 1)
        if r is not None:
            best_branches = sorted(initial)
            best_paths = r

    # Greedy expansion: add vertices one at a time
    if best_branches is not None:
        current = best_branches[:]
        expanded = True
        while expanded and time.time() < deadline:
            expanded = False
            cur_set = set(current)
            cands = [v for v in range(n) if v not in cur_set]
            cands.sort(key=lambda v: len(adj[v]), reverse=True)
            for v in cands:
                if time.time() > deadline:
                    break
                nb = current + [v]
                r = try_branch_set(sorted(nb), 2)
                if r is not None:
                    current = sorted(nb)
                    best_branches = current[:]
                    best_paths = r
                    expanded = True
                    break

    # Fallback: any edge
    if best_branches is None or len(best_branches) < 2:
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [0, 1], 'paths': [[0, 1]]}

    # Reorder paths to canonical pair order for the final branch list
    branches = sorted(best_branches)
    canon = {}
    for path in best_paths:
        key = tuple(sorted((path[0], path[-1])))
        canon[key] = path
    final_paths = []
    for (a, b) in combinations(branches, 2):
        final_paths.append(canon[tuple(sorted((a, b)))])

    return {
        'branches': [int(b) for b in branches],
        'paths': [[int(x) for x in p] for p in final_paths]
    }
# EVOLVE-BLOCK-END
