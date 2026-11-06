"""Search for large (5,5)-free graphs using bitmask K5/indep5 counting,
randomized edge flips, and local repair. Returns best valid graph found."""
import time
import random

_DEADLINE = 330.0
_MARGIN = 3.0


def _quad_residues(q):
    return {(x * x) % q for x in range(1, q)}


def _paley_edges(q):
    r = _quad_residues(q)
    return [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in r]


def _build_adj_bitmask(n, edges):
    adj = [0] * n
    for i, j in edges:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj


def _count_k5(n, adj):
    cnt = 0
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common_ab = adj[a] & adj[b]
            for c in range(b + 1, n):
                if not (common_ab >> c) & 1:
                    continue
                common_abc = common_ab & adj[c]
                for d in range(c + 1, n):
                    if not (common_abc >> d) & 1:
                        continue
                    common_abcd = common_abc & adj[d]
                    # count remaining e > d in common_abcd
                    cnt += bin(common_abcd & ~((1 << (d + 1)) - 1)).count("1")
    return cnt


def _count_indep5(n, adj):
    cnt = 0
    for a in range(n):
        non_a = ~adj[a] & ((1 << n) - 1) & ~(1 << a)
        for b in range(a + 1, n):
            if not (non_a >> b) & 1:
                continue
            common = non_a & ~adj[b]
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common_abc = common & ~adj[c]
                for d in range(c + 1, n):
                    if not (common_abc >> d) & 1:
                        continue
                    common_abcd = common_abc & ~adj[d]
                    cnt += bin(common_abcd & ~((1 << (d + 1)) - 1)).count("1")
    return cnt


def _violations(n, adj):
    return _count_k5(n, adj) + _count_indep5(n, adj)


def _is_valid(n, adj):
    return _violations(n, adj) == 0


def _edges_from_adj(n, adj):
    return [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] >> j) & 1]


def _flip_edge(adj, n, i, j):
    if (adj[i] >> j) & 1:
        adj[i] &= ~(1 << j)
        adj[j] &= ~(1 << i)
    else:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)


def construct():
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN

    # Seed: Paley-37
    best_n = 37
    best_adj = _build_adj_bitmask(37, _paley_edges(37))
    best_edges = _edges_from_adj(best_n, best_adj)

    def now():
        return time.time()

    def local_search(n, adj, time_budget):
        t0 = now()
        cur_v = _violations(n, adj)
        if cur_v == 0:
            return adj, True
        improved = False
        while now() - t0 < time_budget and cur_v > 0:
            best_gain = 0
            best_move = None
            # sample random edges
            for _ in range(600):
                i = random.randrange(n)
                j = random.randrange(n)
                if i == j:
                    continue
                _flip_edge(adj, n, i, j)
                v = _violations(n, adj)
                dv = v - cur_v
                if dv < -best_gain:
                    best_gain = -dv
                    best_move = (i, j)
                    best_v = v
                else:
                    _flip_edge(adj, n, i, j)  # revert
            if best_move is None:
                break
            # best_move already applied
            cur_v = best_v
            improved = True
        return adj, improved

    # Try to grow from 37
    for target in range(38, min(47, 50)):
        if now() > deadline - 20:
            break
        found = False
        for attempt in range(40):
            if now() > deadline - 20:
                break
            random.seed((target, attempt))
            # Start from a random graph biased toward 50% density
            adj = [0] * target
            for i in range(target):
                for j in range(i + 1, target):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            adj, ok = local_search(target, adj, 2.0)
            if _violations(target, adj) == 0:
                best_n = target
                best_adj = adj
                best_edges = _edges_from_adj(target, adj)
                found = True
                break
        if found:
            continue
        else:
            break

    return best_n, best_edges
