import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin before the 150s kill

    n = 16
    full_mask = (1 << n) - 1

    # ---- Precompute Q3 structure (Q3 vertices = 3-bit numbers 0..7) ----
    # Q3 edges: pairs differing in exactly one bit.
    Q3_EDGES = []
    for i in range(8):
        for j in range(i + 1, 8):
            if (i ^ j) in (1, 2, 4):
                Q3_EDGES.append((i, j))
    Q3_NEIGHBORS = [[] for _ in range(8)]
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if j > i:
                Q3_NEIGHBORS[i].append(j)
                Q3_NEIGHBORS[j].append(i)

    # For each Q3 edge (a,b): the 6 remaining vertices in a greedy backtracking
    # order (most already-placed Q3-neighbors first) + per-position list of the
    # Q3-neighbors of that vertex that are among {a,b} or earlier in the order.
    EDGE_INFO = {}
    for a, b in Q3_EDGES:
        remaining = [v for v in range(8) if v != a and v != b]
        placed = {a, b}
        order = []
        for _ in range(6):
            best_v = -1
            best_count = -1
            for v in remaining:
                if v in placed:
                    continue
                count = sum(1 for nn in Q3_NEIGHBORS[v] if nn in placed)
                if count > best_count:
                    best_count = count
                    best_v = v
            order.append(best_v)
            placed.add(best_v)
        neighbor_info = []
        for i, v in enumerate(order):
            prior = {a, b} | set(order[:i])
            neighbor_info.append([nn for nn in Q3_NEIGHBORS[v] if nn in prior])
        EDGE_INFO[(a, b)] = (order, neighbor_info)

    # ---- Graph state: adjacency bitmasks ----
    adj = [0] * n

    def edge_count():
        return sum(adj[i].bit_count() for i in range(n)) // 2

    # ---- Q3 detection (existence) with bitmask intersection + backtracking ----
    def has_q3():
        for (a, b) in Q3_EDGES:
            order, neighbor_info = EDGE_INFO[(a, b)]
            # candidate graph-vertex pairs for the fixed Q3 edge (a,b)
            ca = full_mask
            cb = full_mask
            for i in range(n):
                if adj[i] & (1 << i):
                    continue
            # iterate ordered pairs (gu,gv), gu != gv
            for gu in range(n):
                row_a = adj[gu]
                for gv in range(n):
                    if gv == gu:
                        continue
                    if not (row_a >> gv) & 1:
                        continue
                    # mapping gu->a, gv->b
                    used = (1 << gu) | (1 << gv)
                    assigned = {a: gu, b: gv}

                    def bt(idx):
                        if idx == 6:
                            return True
                        qv = order[idx]
                        cands = (~used) & full_mask
                        for nqv in neighbor_info[idx]:
                            cands &= adj[assigned[nqv]]
                        if not cands:
                            return False
                        cc = cands
                        while cc:
                            lsb = cc & (-cc)
                            g2 = lsb.bit_length() - 1
                            cc ^= lsb
                            assigned[qv] = g2
                            used |= lsb
                            if bt(idx + 1):
                                return True
                            used &= ~lsb
                            del assigned[qv]
                        return False

                    if bt(0):
                        return True
                    # mapping gv->a, gu->b
                    used = (1 << gu) | (1 << gv)
                    assigned = {a: gv, b: gu}

                    def bt2(idx):
                        if idx == 6:
                            return True
                        qv = order[idx]
                        cands = (~used) & full_mask
                        for nqv in neighbor_info[idx]:
                            cands &= adj[assigned[nqv]]
                        if not cands:
                            return False
                        cc = cands
                        while cc:
                            lsb = cc & (-cc)
                            g2 = lsb.bit_length() - 1
                            cc ^= lsb
                            assigned[qv] = g2
                            used |= lsb
                            if bt2(idx + 1):
                                return True
                            used &= ~lsb
                            del assigned[qv]
                        return False

                    if bt2(0):
                        return True
        return False

    # ---- Find a concrete Q3 (returns list of 8 graph vertices, or None) ----
    def find_q3():
        for (a, b) in Q3_EDGES:
            order, neighbor_info = EDGE_INFO[(a, b)]
            for gu in range(n):
                row_a = adj[gu]
                for gv in range(n):
                    if gv == gu:
                        continue
                    if not (row_a >> gv) & 1:
                        continue
                    for (gu2, gv2, qa, qb) in ((gu, gv, a, b), (gv, gu, a, b)):
                        used = (1 << gu2) | (1 << gv2)
                        assigned = {qa: gu2, qb: gv2}

                        def bt(idx):
                            if idx == 6:
                                return True
                            qv = order[idx]
                            cands = (~used) & full_mask
                            for nqv in neighbor_info[idx]:
                                cands &= adj[assigned[nqv]]
                            if not cands:
                                return False
                            cc = cands
                            while cc:
                                lsb = cc & (-cc)
                                g2 = lsb.bit_length() - 1
                                cc ^= lsb
                                assigned[qv] = g2
                                used |= lsb
                                if bt(idx + 1):
                                    return True
                                used &= ~lsb
                                del assigned[qv]
                            return False

                        if bt(0):
                            return [assigned[v] for v in range(8)]
        return None

    # ---- Would adding edge (u,v) create a Q3? (only embeddings using u-v) ----
    def creates_q3(u, v):
        if u == v:
            return True
        for (a, b) in Q3_EDGES:
            order, neighbor_info = EDGE_INFO[(a, b)]
            for (gu, gv, qa, qb) in ((u, v, a, b), (v, u, a, b)):
                used = (1 << gu) | (1 << gv)
                assigned = {qa: gu, qb: gv}

                def bt(idx):
                    if idx == 6:
                        return True
                    qv = order[idx]
                    cands = (~used) & full_mask
                    for nqv in neighbor_info[idx]:
                        cands &= adj[assigned[nqv]]
                    if not cands:
                        return False
                    cc = cands
                    while cc:
                        lsb = cc & (-cc)
                        g2 = lsb.bit_length() - 1
                        cc ^= lsb
                        assigned[qv] = g2
                        used |= lsb
                        if bt(idx + 1):
                            return True
                        used &= ~lsb
                        del assigned[qv]
                    return False

                if bt(0):
                    return True
        return False

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # ---- Build a dense Q3-free starting graph via greedy edge addition ----
    random.seed(1)
    # candidate edges: all 120, shuffled, but try "easy" ones first is fine;
    # we just greedily add every edge that does not create a Q3.
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    for (u, v) in all_edges:
        if time.time() > deadline - 3:
            break
        if not creates_q3(u, v):
            add_edge(u, v)

    # ---- Improve with edge swaps (grow), falling back to deletions (safe) ----
    def q3_edges_of(vertices):
        # return the actual graph edges among the 8 Q3 vertices that are present
        out = []
        vs = set(vertices)
        for i in range(8):
            for j in range(i + 1, 8):
                a, b = vertices[i], vertices[j]
                if (a in vs) and (b in vs) and ((adj[a] >> b) & 1):
                    out.append((a, b))
        return out

    def try_swap():
        # find a Q3, remove one of its edges, then add a random non-Q3 edge
        for _ in range(200):
            if time.time() > deadline - 3:
                return False
            verts = find_q3()
            if verts is None:
                return False
            present = q3_edges_of(verts)
            if not present:
                # no actual edge among these (shouldn't happen for a real Q3)
                return False
            rem = present[random.randrange(len(present))]
            remove_edge(rem[0], rem[1])
            # try to add a random missing edge that doesn't create a Q3
            for _ in range(40):
                u = random.randrange(n)
                v = random.randrange(n)
                if u == v or (adj[u] >> v) & 1:
                    continue
                if not creates_q3(u, v):
                    add_edge(u, v)
                    return True
        return False

    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        # attempt a batch of swaps
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            if try_swap():
                improved = True
        if not improved:
            # no improving swap found -> delete one edge from a Q3 to make progress
            verts = find_q3()
            if verts is None:
                break
            present = q3_edges_of(verts)
            if not present:
                break
            rem = present[0]
            remove_edge(rem[0], rem[1])
            improved = True

    # ---- Safety: ensure the final graph is Q3-free (delete until clean) ----
    guard = 0
    while has_q3() and guard < 2000 and time.time() < deadline - 1:
        verts = find_q3()
        if verts is None:
            break
        present = q3_edges_of(verts)
        if not present:
            break
        rem = present[0]
        remove_edge(rem[0], rem[1])
        guard += 1

    # ---- Return edges as list of [u, v] ----
    edges = []
    for i in range(n):
        row = adj[i]
        for j in range(i + 1, n):
            if (row >> j) & 1:
                edges.append([i, j])
    return edges
