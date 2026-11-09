import time
import random
from itertools import combinations

_TIME_LIMIT = 120.0
_DEADLINE = time.time() + _TIME_LIMIT - 20.0  # 20s safety margin
_RNG = random.Random(12345)


def _deadline_hit():
    return time.time() > _DEADLINE


def _has_k4(adj):
    n = len(adj)
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return True
    return False


def _has_indep6(adj):
    n = len(adj)
    for combo in combinations(range(n), 6):
        a, b, c, d, e, f = combo
        if (not adj[a][b] and not adj[a][c] and not adj[a][d] and
                not adj[a][e] and not adj[a][f] and
                not adj[b][c] and not adj[b][d] and not adj[b][e] and
                not adj[b][f] and
                not adj[c][d] and not adj[c][e] and not adj[c][f] and
                not adj[d][e] and not adj[d][f] and
                not adj[e][f]):
            return True
    return False


def _is_valid(adj):
    return (not _has_k4(adj)) and (not _has_indep6(adj))


def _edges_from_adj(adj):
    n = len(adj)
    return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j]]


def _polarity_adj(p):
    """K4-free self-complementary polarity graph on p vertices (p prime)."""
    sq = {x * x % p for x in range(1, p)}
    adj = [[False] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in sq:
                adj[i][j] = adj[j][i] = True
    return adj


def _largest_residue_sumset(p):
    """Largest subset of nonzero residues with no 3 summing to 0 mod p (== alpha of polarity graph)."""
    sq = sorted(x * x % p for x in range(1, p))
    best = 0
    for mask in range(1 << len(sq)):
        cnt = 0
        ok = True
        for a in range(len(sq)):
            if not (mask >> a) & 1:
                continue
            cnt += 1
            for b in range(a, len(sq)):
                if not (mask >> b) & 1:
                    continue
                for c in range(b, len(sq)):
                    if (mask >> c) & 1 and (sq[a] + sq[b] + sq[c]) % p == 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok and cnt > best:
            best = cnt
    return best


def _greedy_maximal_set(allowed, adj, base):
    """Greedy maximal subset of `allowed` that is triangle-free AND has no independent 5-set."""
    S = []
    Sset = set()
    for v in allowed:
        # triangle-free check: no edge among Sset that both touch v
        tri = False
        for u in Sset:
            if adj[u][v]:
                for w in Sset:
                    if w != u and adj[u][w]:
                        tri = True
                        break
            if tri:
                break
        if tri:
            continue
        # independence check: no independent 5-set in Sset ∪ {v}
        indep5 = False
        T = list(Sset) + [v]
        for combo in combinations(T, 5):
            a, b, c, d, e = combo
            if (not adj[a][b] and not adj[a][c] and not adj[a][d] and not adj[a][e] and
                    not adj[b][c] and not adj[b][d] and not adj[b][e] and
                    not adj[c][d] and not adj[c][e] and not adj[d][e]):
                indep5 = True
                break
        if indep5:
            continue
        S.append(v)
        Sset.add(v)
    return S


def _backtrack_max_set(allowed, adj, base, deadline):
    """Backtracking to find a large valid subset of `allowed` (time-boxed)."""
    best = []
    best_set = set()

    def rec(cand, cur, curset):
        nonlocal best, best_set
        if _deadline_hit():
            return
        if len(cur) > len(best):
            best = list(cur)
            best_set = set(cur)
        if not cand:
            return
        # branch: include cand[0]
        v = cand[0]
        rest = cand[1:]
        # can we include v?
        tri = False
        for u in curset:
            if adj[u][v]:
                for w in curset:
                    if w != u and adj[u][w]:
                        tri = True
                        break
            if tri:
                break
        if not tri:
            T = list(curset) + [v]
            indep5 = False
            for combo in combinations(T, 5):
                a, b, c, d, e = combo
                if (not adj[a][b] and not adj[a][c] and not adj[a][d] and not adj[a][e] and
                        not adj[b][c] and not adj[b][d] and not adj[b][e] and
                        not adj[c][d] and not adj[c][e] and not adj[d][e]):
                    indep5 = True
                    break
            if not indep5:
                cur.append(v)
                curset.add(v)
                rec(rest, cur, curset)
                cur.pop()
                curset.remove(v)
        # branch: exclude v
        rec(rest, cur, curset)

    rec(list(allowed), [], set())
    return best


def _try_grow(adj):
    """Try to add one vertex; return new adj if success else None."""
    n = len(adj)
    # find best vertex to connect to
    for v in range(n):
        if _deadline_hit():
            return None
        allowed = [u for u in range(n) if not adj[v][u]]
        # greedy candidate
        S = _greedy_maximal_set(allowed, adj, v)
        Sset = set(S)
        # try to improve with backtracking (time-boxed)
        if len(allowed) <= 22:
            S2 = _backtrack_max_set(allowed, adj, v, _DEADLINE)
            if len(S2) > len(S):
                S = S2
                Sset = set(S)
        if not S:
            continue
        # build new adj
        nn = n + 1
        new_adj = [row[:] for row in adj]
        for i in range(nn):
            new_adj.append([False] * nn)
        for u in S:
            new_adj[v][u] = new_adj[u][v] = True
        # verify validity (should hold by construction, but be safe)
        if _is_valid(new_adj):
            return new_adj
    return None


def _grow_loop(adj, max_iters=2000):
    cur = [row[:] for row in adj]
    improved = True
    it = 0
    while improved and it < max_iters and not _deadline_hit():
        it += 1
        improved = False
        # try adding a vertex
        new_adj = _try_grow(cur)
        if new_adj is not None:
            cur = new_adj
            improved = True
            continue
        # try flipping an edge (remove) to enable future growth
        # find a "useful" edge to remove: one whose removal doesn't create indep6
        # and may allow a larger S later. Simple heuristic: try removing a random edge.
        # We only accept if still valid.
        if len(cur) >= 10:
            n = len(cur)
            cand_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if cur[i][j]]
            _RNG.shuffle(cand_edges)
            for (i, j) in cand_edges[:40]:
                if _deadline_hit():
                    break
                cur[i][j] = cur[j][i] = False
                if _is_valid(cur):
                    # keep it, try to grow again
                    improved = True
                    break
                else:
                    cur[i][j] = cur[j][i] = True
    return cur


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT - 20.0

    # 1) Build base polarity graphs and pick the largest valid one.
    best_adj = None
    for p in [17, 19, 23, 29, 31, 37]:
        if _deadline_hit():
            break
        adj = _polarity_adj(p)
        if _is_valid(adj):
            if best_adj is None or p > len(best_adj):
                best_adj = adj

    if best_adj is None:
        # fallback: seed Paley 17
        best_adj = _polarity_adj(17)

    # 2) Grow via local search.
    grown = _grow_loop(best_adj)

    if not _is_valid(grown):
        # safety: return the valid base
        return {'n': len(best_adj), 'edges': _edges_from_adj(best_adj)}

    return {'n': len(grown), 'edges': _edges_from_adj(grown)}
