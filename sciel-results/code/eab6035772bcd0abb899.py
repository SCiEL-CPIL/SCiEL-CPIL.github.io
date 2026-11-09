import time
import random
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # ---------- state (bit masks for speed) ----------
    adj = [0] * n

    def edge_count():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def has_edge(u, v):
        return bool(adj[u] & (1 << v))

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def can_add(u, v):
        # triangle-free: u and v must share no common neighbor
        return (adj[u] & adj[v]) == 0

    # ---------- 3-colorability (fast, bit-mask) ----------
    def is_3_colorable():
        color = [-1] * n
        # order vertices by degree descending for pruning
        order = sorted(range(n), key=lambda x: -bin(adj[x]).count('1'))

        def popcount_deg(v):
            return bin(adj[v]).count('1')

        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = 0
            m = adj[v]
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                if color[u] >= 0:
                    used |= (1 << color[u])
                m ^= lsb
            for c in range(3):
                if not (used & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False

        return backtrack(0)

    def is_not_3_colorable():
        return not is_3_colorable()

    # ---------- independent set helpers ----------
    def greedy_independent_set(active):
        """Greedy MIS with min-degree-in-remaining and random tie-break."""
        remaining = set(active)
        ind = set()
        while remaining:
            min_deg = min(len([w for w in remaining if has_edge(v, w)]) for v in remaining)
            cands = [v for v in remaining
                     if len([w for w in remaining if has_edge(v, w)]) == min_deg]
            v = random.choice(cands)
            ind.add(v)
            remaining.discard(v)
            for w in list(remaining):
                if has_edge(v, w):
                    remaining.discard(w)
        return ind

    def best_independent_set(active, attempts=20):
        best = set()
        for _ in range(attempts):
            s = greedy_independent_set(active)
            if len(s) > len(best):
                best = s
        return best

    # ---------- initial graph: Grötzsch (Mycielski of C5) on 0..10 ----------
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    for i in range(5):
        u, v = i, (i + 1) % 5
        add_edge(u, 5 + v)
        add_edge(v, 5 + u)
    for i in range(5):
        add_edge(5 + i, 10)

    best_count = edge_count()
    best_adj = list(adj)

    # ---------- Phase 1: MIS-based vertex addition (11..15) ----------
    for new_v in range(11, n):
        if time.time() > deadline - 25.0:
            break
        active = list(range(new_v))
        iset = best_independent_set(active, attempts=25)
        for u in iset:
            if can_add(new_v, u):
                add_edge(new_v, u)
        if edge_count() > best_count:
            best_count = edge_count()
            best_adj = list(adj)

    # ---------- Phase 2: greedy fill (triangle-free only) ----------
    def greedy_fill(max_time):
        changed = True
        while changed and time.time() < max_time:
            changed = False
            for u, v in all_edges:
                if time.time() > max_time:
                    break
                if has_edge(u, v):
                    continue
                if can_add(u, v):
                    add_edge(u, v)
                    changed = True
        return edge_count()

    c = greedy_fill(deadline - 20.0)
    if c > best_count:
        best_count = c
        best_adj = list(adj)

    # ---------- Phase 3: intensive local search (remove one, add one/two) ----------
    def local_search(max_time):
        cur = list(adj)
        cur_count = edge_count()
        best_local = list(cur)
        best_local_count = cur_count
        no_improve = 0

        while time.time() < max_time and no_improve < 400:
            existing = [(i, j) for i in range(n) for j in range(i + 1, n)
                        if cur[i] & (1 << j)]
            if not existing:
                break
            ui, vi = random.choice(existing)
            cur[ui] &= ~(1 << vi)
            cur[vi] &= ~(1 << ui)

            # candidate non-edges that keep triangle-free (against cur)
            cands = []
            for u, v in all_edges:
                if cur[u] & (1 << v):
                    continue
                if (cur[u] & cur[v]) == 0:
                    cands.append((u, v))
            random.shuffle(cands)

            # try to add one edge
            added = False
            for u, v in cands:
                if time.time() > max_time:
                    break
                cur[u] |= (1 << v)
                cur[v] |= (1 << u)
                added = True
                break

            # optionally add a second edge
            if added:
                cands2 = []
                for u, v in all_edges:
                    if cur[u] & (1 << v):
                        continue
                    if (cur[u] & cur[v]) == 0:
                        cands2.append((u, v))
                random.shuffle(cands2)
                for u, v in cands2[:60]:
                    if time.time() > max_time:
                        break
                    cur[u] |= (1 << v)
                    cur[v] |= (1 << u)

            new_count = edge_count()
            if new_count > best_local_count:
                best_local_count = new_count
                best_local = list(cur)
                no_improve = 0
            else:
                no_improve += 1
                # revert to previous (restore removed edge)
                cur[ui] |= (1 << vi)
                cur[vi] |= (1 << ui)
                # revert added edges: rebuild from best_local? cheaper: re-add removed, drop added
                # We'll just restore from a snapshot to be safe:
                cur = list(best_local)

        return best_local, best_local_count

    cur, lc = local_search(deadline - 12.0)
    if lc > best_count:
        best_count = lc
        best_adj = list(cur)

    # ---------- Phase 4: multi-start perturbation ----------
    for restart in range(6):
        if time.time() > deadline - 8.0:
            break
        cur = list(best_adj)
        existing = [(i, j) for i in range(n) for j in range(i + 1, n)
                    if cur[i] & (1 << j)]
        random.shuffle(existing)
        for e in existing[:3]:
            cur[e[0]] &= ~(1 << e[1])
            cur[e[1]] &= ~(1 << e[0])
        # fill
        changed = True
        while changed and time.time() < deadline - 8.0:
            changed = False
            for u, v in all_edges:
                if cur[u] & (1 << v):
                    continue
                if (cur[u] & cur[v]) == 0:
                    cur[u] |= (1 << v)
                    cur[v] |= (1 << u)
                    changed = True
        cur_count = sum(bin(cur[i]).count('1') for i in range(n)) // 2
        if cur_count > best_count:
            best_count = cur_count
            best_adj = list(cur)

    # ---------- final polish ----------
    cur = list(best_adj)
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        for u, v in all_edges:
            if cur[u] & (1 << v):
                continue
            if (cur[u] & cur[v]) == 0:
                cur[u] |= (1 << v)
                cur[v] |= (1 << u)
                changed = True
    final_count = sum(bin(cur[i]).count('1') for i in range(n)) // 2
    if final_count > best_count:
        best_count = final_count
        best_adj = list(cur)

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    return result
