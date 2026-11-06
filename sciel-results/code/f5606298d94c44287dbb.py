# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 142.0   # safety margin vs 150s kill
    n = 40
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def creates_c6(adjb, u, v):
        """Bitset check: does adding edge (u,v) create a 6-cycle?
        A C6 through (u,v) needs a path of length 5 between u and v.
        Path: u - x1 - x2 - x3 - x4 - v  (x1 in N(u), x4 in N(v), distinct)."""
        Nu = adjb[u] & ~(1 << v)
        Nv = adjb[v] & ~(1 << u)
        mask_uv = ~((1 << u) | (1 << v))
        # x1 in N(u); x4 in N(v), x4 != x1
        x1 = Nu
        while x1:
            lsb = x1 & (-x1)
            x1 -= lsb
            i = lsb.bit_length() - 1
            # x4 candidates: in N(v), not u, not v, not i
            x4 = Nv & ~(1 << i)
            # x2 in N(i) \ {u,v,i}; x3 in N(x2) \ {u,v,i,x2}; need x3 ~ x4
            x2 = adjb[i] & mask_uv & ~(1 << i)
            while x2:
                lsb2 = x2 & (-x2)
                x2 -= lsb2
                j = lsb2.bit_length() - 1
                x3 = adjb[j] & mask_uv & ~(1 << i) & ~(1 << j)
                if x3 & x4:
                    return True
        return False

    def greedy_fill(adjb, adj):
        """Repeatedly add every edge that doesn't create a C6 until stable."""
        changed = True
        while changed:
            changed = False
            for u, v in all_pairs:
                if not (adjb[u] >> v) & 1:
                    if not creates_c6(adjb, u, v):
                        adjb[u] |= (1 << v)
                        adjb[v] |= (1 << u)
                        adj[u][v] = adj[v][u] = True
                        changed = True
        return adjb, adj

    def count(adjb):
        return sum(adjb[u] >> (u + 1) & ((1 << (n - u - 1)) - 1).bit_count()
                   for u in range(n))

    def edges_from_adjb(adjb):
        out = []
        for u in range(n):
            m = adjb[u] >> (u + 1)
            while m:
                lsb = m & (-m)
                m -= lsb
                out.append([u, lsb.bit_length() - 1])
        return out

    def local_search(adjb, adj):
        """1-swap and 2-swap hill climbing using spare budget."""
        ls_deadline = deadline - 1.0
        improved = True
        while improved and time.time() < ls_deadline:
            improved = False
            # 1-swap: remove one edge, greedily re-add
            if time.time() > ls_deadline - 0.5:
                break
            for u, v in all_pairs:
                if time.time() > ls_deadline:
                    break
                if (adjb[u] >> v) & 1:
                    c0 = count(adjb)
                    adjb[u] &= ~(1 << v)
                    adjb[v] &= ~(1 << u)
                    adj[u][v] = adj[v][u] = False
                    greedy_fill(adjb, adj)
                    c1 = count(adjb)
                    if c1 > c0:
                        improved = True
                    else:
                        # revert: rebuild from scratch is expensive; just keep if equal
                        # re-add to restore original state
                        pass
                    # ensure we don't lose progress: if c1 <= c0 we already lost an edge
                    # so we must restore. Simplest: rebuild from best-known.
                    if c1 <= c0:
                        # restore edge
                        adjb[u] |= (1 << v)
                        adjb[v] |= (1 << u)
                        adj[u][v] = adj[v][u] = True
                    # re-greedy to fill back
                    greedy_fill(adjb, adj)
                    if count(adjb) > c0:
                        improved = True

    best_adjb = None
    best_edges = None
    best_count = -1

    def evaluate(adjb, adj):
        nonlocal best_adjb, best_edges, best_count
        c = count(adjb)
        if c > best_count:
            best_count = c
            best_adjb = [x for x in adjb]
            best_edges = edges_from_adjb(adjb)

    # Primary: structured seed 8xK5 + fast greedy sweep
    adjb = [0] * n
    adj = [[False] * n for _ in range(n)]
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                adjb[u] |= (1 << v)
                adjb[v] |= (1 << u)
                adj[u][v] = adj[v][u] = True
    greedy_fill(adjb, adj)
    evaluate(adjb, adj)

    # Local search on the primary solution
    if time.time() < deadline - 10.0:
        local_search(adjb, adj)
        evaluate(adjb, adj)

    # Secondary: random restarts with greedy fill + light local search
    rng = random.Random(12345)
    for trial in range(200):
        if time.time() > deadline - 8.0:
            break
        adjb = [0] * n
        adj = [[False] * n for _ in range(n)]
        p = rng.uniform(0.4, 0.6)
        for u, v in all_pairs:
            if rng.random() < p:
                adjb[u] |= (1 << v)
                adjb[v] |= (1 << u)
                adj[u][v] = adj[v][u] = True
        greedy_fill(adjb, adj)
        evaluate(adjb, adj)
        if time.time() > deadline - 8.0:
            break
        # quick 1-swap pass
        c0 = count(adjb)
        for u, v in all_pairs:
            if time.time() > deadline - 6.0:
                break
            if (adjb[u] >> v) & 1:
                adjb[u] &= ~(1 << v)
                adjb[v] &= ~(1 << u)
                adj[u][v] = adj[v][u] = False
                greedy_fill(adjb, adj)
                if count(adjb) <= c0:
                    adjb[u] |= (1 << v)
                    adjb[v] |= (1 << u)
                    adj[u][v] = adj[v][u] = True
                    greedy_fill(adjb, adj)
        evaluate(adjb, adj)

    if best_edges is None:
        return []
    return best_edges
# EVOLVE-BLOCK-END
