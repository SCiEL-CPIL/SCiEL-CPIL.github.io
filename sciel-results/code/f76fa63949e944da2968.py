import time
import random
import sys

# EVOLVE-BLOCK-START
def _score(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(a) for a in adj]
    best = 0
    for u in range(n):
        for v in adj[u]:
            if u < v:
                for w in adj[u] & adj[v]:
                    s = deg[u] + deg[v] + deg[w]
                    if s > best:
                        best = s
    return best

def _valid(edges, n, target):
    if len(edges) != target:
        return False
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        if a >= n or b >= n:
            return False
    return True

def _seed_edges(n):
    # K_{8,8} + one intra edge -> 65 edges, score 24
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return edges

def construct():
    n = 16
    target = 65
    deadline = time.time() + 115.0   # safety margin under 150s wall-clock

    best_edges = _seed_edges(n)
    best_score = _score(best_edges, n)

    rng = random.Random(12345)

    def try_improve(edges, score):
        nonlocal best_edges, best_score
        if score < best_score:
            best_score = score
            best_edges = list(edges)
            return True
        return False

    # ---- local search: min-sum degree moves + edge swaps ----
    def local_search(edges):
        nonlocal best_score, best_edges
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        deg = [len(a) for a in adj]
        present = set((min(u, v), max(u, v)) for u, v in edges)
        score = _score(edges, n)
        try_improve(edges, score)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # 1) degree-matching moves (u,v)->(u,w): same count
            for u in range(n):
                for v in list(adj[u]):
                    if u < v and deg[u] < deg[v]:
                        # try moving v's edge to some w
                        for w in range(n):
                            if w == u or w == v or w in adj[u]:
                                continue
                            if deg[u] + 1 > deg[w]:
                                continue
                            # simulate
                            adj[u].discard(v); adj[v].discard(u)
                            adj[u].add(w); adj[w].add(u)
                            deg[u] += 1; deg[v] -= 1; deg[w] += 1
                            ns = _score_from(adj, deg, n)
                            if ns < score:
                                score = ns
                                improved = True
                                try_improve(list_of_edges(adj, n), score)
                            else:
                                adj[u].add(v); adj[v].add(u)
                                adj[u].discard(w); adj[w].discard(u)
                                deg[u] -= 1; deg[v] += 1; deg[w] -= 1
                    elif u < v and deg[u] > deg[v]:
                        for w in range(n):
                            if w == u or w == v or w in adj[v]:
                                continue
                            if deg[v] + 1 > deg[w]:
                                continue
                            adj[v].discard(u); adj[u].discard(v)
                            adj[v].add(w); adj[w].add(v)
                            deg[v] += 1; deg[u] -= 1; deg[w] += 1
                            ns = _score_from(adj, deg, n)
                            if ns < score:
                                score = ns
                                improved = True
                                try_improve(list_of_edges(adj, n), score)
                            else:
                                adj[v].add(u); adj[u].add(v)
                                adj[v].discard(w); adj[w].discard(v)
                                deg[v] -= 1; deg[u] += 1; deg[w] -= 1
            # 2) edge swaps: remove (a,b),(c,d) add (a,c),(b,d)
            if time.time() >= deadline:
                break
            e_list = list_of_edges(adj, n)
            for i in range(len(e_list)):
                a, b = e_list[i]
                for j in range(i + 1, len(e_list)):
                    c, d = e_list[j]
                    if len({a, b, c, d}) < 4:
                        continue
                    if (a, c) in present or (b, d) in present:
                        continue
                    # also avoid creating parallel
                    adj[a].discard(b); adj[b].discard(a)
                    adj[c].discard(d); adj[d].discard(c)
                    adj[a].add(c); adj[c].add(a)
                    adj[b].add(d); adj[d].add(b)
                    deg[a] += 0; deg[b] += 0; deg[c] += 0; deg[d] += 0  # degrees unchanged
                    ns = _score_from(adj, deg, n)
                    if ns < score:
                        score = ns
                        improved = True
                        present.discard((a, b)); present.discard((c, d))
                        present.add((min(a, c), max(a, c)))
                        present.add((min(b, d), max(b, d)))
                        try_improve(list_of_edges(adj, n), score)
                    else:
                        adj[a].add(b); adj[b].add(a)
                        adj[c].add(d); adj[d].add(c)
                        adj[a].discard(c); adj[c].discard(a)
                        adj[b].discard(d); adj[d].discard(b)
            # 3) random perturbation to escape
            if not improved and rng.random() < 0.3 and time.time() < deadline - 5:
                for _ in range(3):
                    e_list = list_of_edges(adj, n)
                    a, b = rng.choice(e_list)
                    c, d = rng.choice(e_list)
                    if len({a, b, c, d}) < 4:
                        continue
                    if (a, c) in present or (b, d) in present:
                        continue
                    adj[a].discard(b); adj[b].discard(a)
                    adj[c].discard(d); adj[d].discard(c)
                    adj[a].add(c); adj[c].add(a)
                    adj[b].add(d); adj[d].add(b)
                    present.discard((min(a, b), max(a, b)))
                    present.discard((min(c, d), max(c, d)))
                    present.add((min(a, c), max(a, c)))
                    present.add((min(b, d), max(b, d)))
                    score = _score_from(adj, deg, n)
                improved = True

    def list_of_edges(adj, n):
        out = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    out.append((u, v))
        return out

    def _score_from(adj, deg, n):
        best = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for w in adj[u] & adj[v]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > best:
                            best = s
        return best

    local_search(list(_seed_edges(n)))

    # a few restarts with random starting points
    start = time.time()
    while time.time() < deadline - 5:
        # random 65-edge graph
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)
        edges = [list(p) for p in all_pairs[:target]]
        if not _valid(edges, n, target):
            continue
        local_search(edges)

    if not _valid(best_edges, n, target):
        best_edges = _seed_edges(n)
        best_score = _score(best_edges, n)

    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
