import time
import random

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.

    Strategy:
    1. Build deterministic polarity graphs of PG(2,q) for several q. These are
       C4-free by construction (polarity property).
    2. Use a fast bitmask branch-and-bound to compute independence number.
    3. Search for a large C4-free subgraph (by vertex deletion) whose
       independence number is < 11.
    4. Try to extend the best base by adding new vertices with edges that
       maintain C4-freeness and keep alpha < 11.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def popcount(x):
        return x.bit_count()

    # ---------------- C4-freeness check (bitmask) ----------------
    def is_c4_free(n, adj):
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    if popcount(ai & adj[j]) > 1:
                        return False
        return True

    # ---------------- Independence number via branch and bound ----------------
    def max_independent_set_size(n, adj):
        """Return the size of the maximum independent set."""
        best = [0]
        full = (1 << n) - 1

        def branch(cand, cur):
            if time_left() < 3:
                return
            if cand == 0:
                if cur > best[0]:
                    best[0] = cur
                return
            if cur + popcount(cand) <= best[0]:
                return
            # pick vertex with max degree in candidate subgraph
            best_v = -1
            best_deg = -1
            m = cand
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(cand & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            if best_v == -1:
                return
            vb = 1 << best_v
            # include
            branch(cand & ~adj[best_v] & ~vb, cur + 1)
            if best[0] >= 11:
                return
            # exclude
            branch(cand & ~vb, cur)

        branch(full, 0)
        return best[0]

    def has_independent_11(n, adj):
        # quick greedy lower bound
        def greedy_size():
            remaining = (1 << n) - 1
            s = 0
            while remaining:
                # pick vertex with min degree in remaining
                best_v = -1
                best_deg = 10**9
                m = remaining
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    deg = popcount(remaining & adj[v])
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    m -= lsb
                if best_v < 0:
                    break
                s += 1
                remaining &= ~adj[best_v] & ~(1 << best_v)
            return s
        if greedy_size() >= 11:
            return True
        # exact
        return max_independent_set_size(n, adj) >= 11

    # ---------------- Polarity graph construction ----------------
    def build_polarity(q):
        """Polarity graph of PG(2,q). Returns (n, adj, points)."""
        # all points (projective)
        pts = []
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv = pow(x, -1, q)
                        pts.append((1, (y * inv) % q, (z * inv) % q))
                    elif y != 0:
                        inv = pow(y, -1, q)
                        pts.append((0, 1, (z * inv) % q))
                    else:
                        pts.append((0, 0, 1))
        pts = list(set(pts))
        n = len(pts)

        def bilinear(u, v):
            return (u[0] * v[2] + u[1] * v[1] + u[2] * v[0]) % q

        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if bilinear(pts[i], pts[j]) == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return n, adj, pts

    # ---------------- Vertex-deletion subgraph search ----------------
    def find_subgraph(n, adj, target_alpha=11, max_iter=400):
        """
        Randomized greedy: keep vertices in a random order, include a vertex
        if it does not create a C4 with already-included vertices and keeps
        alpha < target_alpha. Try to maximize kept vertices.
        """
        best_n = 0
        best_adj = [0]

        for it in range(max_iter):
            if time_left() < 20:
                break
            order = list(range(n))
            random.shuffle(order)
            kept = []
            cand_adj = {}
            for v in order:
                # tentatively add v
                kept.append(v)
                # build adjacency restricted to kept
                # check C4-freeness of the induced subgraph on kept
                # and alpha
                if len(kept) <= best_n:
                    continue
                # quick C4 check on induced subgraph
                ok = True
                k = len(kept)
                # map kept indices
                # check pairs
                for a in range(k):
                    va = kept[a]
                    for b in range(a + 1, k):
                        vb = kept[b]
                        if adj[va] & (1 << vb):
                            common = 0
                            for c in range(k):
                                vc = kept[c]
                                if (adj[va] & (1 << vc)) and (adj[vb] & (1 << vc)):
                                    common += 1
                            if common > 1:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    kept.pop()
                    continue
                # alpha check (only when it might matter)
                # build compact adj
                idx = {v: i for i, v in enumerate(kept)}
                m = len(kept)
                cadj = [0] * m
                for i in range(m):
                    vi = kept[i]
                    mask = 0
                    for j in range(m):
                        if i != j and (adj[vi] & (1 << kept[j])):
                            mask |= (1 << j)
                    cadj[i] = mask
                if has_independent_11(m, cadj):
                    kept.pop()
                    continue
                # keep
            if len(kept) > best_n:
                best_n = len(kept)
                idx = {v: i for i, v in enumerate(kept)}
                m = best_n
                best_adj = [0] * m
                for i in range(m):
                    vi = kept[i]
                    mask = 0
                    for j in range(m):
                        if i != j and (adj[vi] & (1 << kept[j])):
                            mask |= (1 << j)
                    best_adj[i] = mask
        return best_n, best_adj

    # ---------------- Extension search ----------------
    def try_extend(n, adj, target_n):
        new_adj = adj[:]
        new_n = n
        for new_v in range(n, target_n):
            if time_left() < 25:
                break
            # try to connect new_v to a subset of existing vertices
            # such that C4-free and alpha<11
            best_conn = None
            # try isolated first
            trial = new_adj + [0]
            if not has_independent_11(new_n + 1, trial):
                best_conn = 0
            if best_conn is None:
                # try connecting to 1..3 existing vertices
                existing = list(range(new_n))
                random.shuffle(existing)
                found = False
                for r in range(1, 4):
                    if found:
                        break
                    # try random subsets of size r
                    for _ in range(30):
                        if time_left() < 25:
                            break
                        subset = random.sample(existing, r)
                        # C4 condition: for any two neighbors a,b of new_v,
                        # N(a) & N(b) must be empty (no common neighbor)
                        ok = True
                        for a in range(r):
                            for b in range(a + 1, r):
                                if new_adj[subset[a]] & new_adj[subset[b]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        trial = new_adj + [0]
                        for v in subset:
                            trial[v] |= (1 << new_n)
                            trial[new_n] |= (1 << v)
                        if not is_c4_free(new_n + 1, trial):
                            continue
                        if not has_independent_11(new_n + 1, trial):
                            best_conn = trial[new_n]
                            found = True
                            break
            if best_conn is not None:
                new_adj = new_adj + [best_conn]
                for v in range(new_n):
                    if best_conn & (1 << v):
                        new_adj[v] |= (1 << new_n)
                new_n += 1
            else:
                break
        return new_n, new_adj

    # ---------------- Main ----------------
    best_n = 0
    best_adj = [0]

    # Build several polarity bases and search subgraphs
    for q in [5, 7]:
        if time_left() < 40:
            break
        n, adj, pts = build_polarity(q)
        # quick check
        if not is_c4_free(n, adj):
            continue
        # search for large C4-free alpha<11 subgraph
        sn, sadj = find_subgraph(n, adj)
        if sn > best_n:
            best_n = sn
            best_adj = sadj[:]

    # Also try the PG(2,5) full graph directly if it qualifies
    if time_left() > 40:
        n5, adj5, _ = build_polarity(5)
        if is_c4_free(n5, adj5) and not has_independent_11(n5, adj5):
            if n5 > best_n:
                best_n = n5
                best_adj = adj5[:]

    # Try to extend the best base
    if time_left() > 30 and best_n < 43:
        en, eadj = try_extend(best_n, best_adj, 43)
        if en > best_n:
            best_n = en
            best_adj = eadj[:]

    # Final verification
    if best_n < 1:
        # fallback: minimal valid graph
        best_n = 1
        best_adj = [0]
    if not is_c4_free(best_n, best_adj):
        # fall back to a trivial valid graph
        best_n = 1
        best_adj = [0]
    if has_independent_11(best_n, best_adj):
        # fall back to a small graph with no independent 11-set
        # K_10 has alpha=1
        best_n = 10
        best_adj = [0] * 10
        for i in range(10):
            for j in range(i + 1, 10):
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)

    # Build edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
