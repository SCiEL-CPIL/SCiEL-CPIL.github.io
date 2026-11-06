import time
import sys
import random
import math

N = 18
FULL = (1 << N) - 1

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s wall-clock kill
    rng = random.Random(42)

    # ---------- bit-mask helpers ----------
    def has_k4(adj):
        for a in range(N):
            ma = adj[a]
            for b in range(a + 1, N):
                if (ma >> b) & 1:
                    common = ma & adj[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        m &= m - 1
                        if (adj[c] & m) != 0:
                            return True
        return False

    def get_triangles(adj):
        tris = []
        for a in range(N):
            ma = adj[a]
            for b in range(a + 1, N):
                if (ma >> b) & 1:
                    common = ma & adj[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        m &= m - 1
                        if c > b:
                            tris.append((a, b, c))
        return tris

    def max_triangle_free_size(adj):
        """Exact size of largest triangle-free induced subgraph (branch & bound)."""
        tris = get_triangles(adj)
        if not tris:
            return N
        tri_masks = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in tris]
        # Order triangles so vertices with high degree are hit first -> faster large-set discovery
        def tri_key(tm):
            return -((adj[(tm & -tm).bit_length() - 1]).bit_count())
        tri_masks.sort(key=tri_key)

        best = [0]

        def bnb(avail):
            cnt = avail.bit_count()
            if cnt <= best[0]:
                return
            tri = None
            for tm in tri_masks:
                if (tm & avail) == tm:
                    tri = tm
                    break
            if tri is None:
                if cnt > best[0]:
                    best[0] = cnt
                return
            a_bit = tri & -tri
            a = a_bit.bit_length() - 1
            rest = tri ^ a_bit
            b_bit = rest & -rest
            b = b_bit.bit_length() - 1
            c = (tri ^ a_bit ^ b_bit).bit_length() - 1
            da = (avail & adj[a]).bit_count()
            db = (avail & adj[b]).bit_count()
            dc = (avail & adj[c]).bit_count()
            verts = sorted([(da, a), (db, b), (dc, c)], reverse=True)
            for _, v in verts:
                bnb(avail & ~(1 << v))
                if cnt - 1 <= best[0]:
                    return

        bnb(FULL)
        return best[0]

    # ---------- graph builders ----------
    def make_paley17():
        adj = [0] * N
        residues = set([1, 2, 4, 8, 9, 13, 15, 16])
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def find_independent_sets(adj, size):
        """Return a list of independent sets of given size (as vertex tuples), via greedy enumeration."""
        sets = []
        def rec(start, chosen, avail_mask):
            if len(chosen) == size:
                sets.append(tuple(chosen))
                return
            if len(sets) >= 200:
                return
            for v in range(start, N):
                if (avail_mask >> v) & 1:
                    chosen.append(v)
                    # new avail = avail minus v and its neighbors (within > v)
                    new_avail = avail_mask & ~(1 << v)
                    for u in range(v + 1, N):
                        if (adj[v] >> u) & 1:
                            new_avail &= ~(1 << u)
                    rec(v + 1, chosen, new_avail)
                    chosen.pop()
        rec(0, [], (1 << 17) - 1)  # only vertices 0..16 for base graph
        return sets

    def attach(adj17, indep_set):
        """Build full graph: paley17 on 0..16, vertex 17 connected to indep_set."""
        adj = adj17[:]
        for v in indep_set:
            adj[17] |= (1 << v)
            adj[v] |= (1 << 17)
        return adj

    def flip(adj, u, v):
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)

    # ---------- build candidate seeds ----------
    paley17 = make_paley17()
    seeds = []

    # independent sets of various sizes in P(17)
    for sz in [1, 2, 3, 4]:
        for iset in find_independent_sets(paley17, sz):
            seeds.append(attach(paley17, iset))
            # also complement attachment (connect 17 to complement of iset within 0..16)
            comp = tuple(v for v in range(17) if v not in iset)
            seeds.append(attach(paley17, comp))

    # dedupe by edge mask
    seen = set()
    uniq_seeds = []
    for s in seeds:
        key = tuple(s[i] for i in range(N))
        if key not in seen:
            seen.add(key)
            uniq_seeds.append(s)

    # ---------- evaluate seeds ----------
    best_score = N + 1
    best_mask = None
    for s in uniq_seeds:
        if time.time() >= deadline:
            break
        if has_k4(s):
            continue
        sc = max_triangle_free_size(s)
        if sc < best_score:
            best_score = sc
            best_mask = s[:]

    if best_mask is None:
        best_mask = attach(paley17, (0,))
        best_score = max_triangle_free_size(best_mask)

    # ---------- local search (simulated annealing) from best ----------
    adj = best_mask[:]
    cur = best_score
    temperature = 3.0
    cooling = 0.995
    no_improve = 0
    max_no_improve = 400
    while time.time() < deadline and no_improve < max_no_improve:
        u, v = rng.sample(range(N), 2)
        flip(adj, u, v)
        if not has_k4(adj):
            flip(adj, u, v)
            no_improve += 1
            temperature *= cooling
            continue
        new_score = max_triangle_free_size(adj)
        delta = new_score - cur
        if delta <= 0 or (temperature > 0.05 and rng.random() < math.exp(-delta / temperature)):
            cur = new_score
            if cur < best_score:
                best_score = cur
                best_mask = adj[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            flip(adj, u, v)
            no_improve += 1
        temperature *= cooling

    # ---------- final single-flip polish ----------
    if time.time() < deadline - 5:
        adj = best_mask[:]
        cur = best_score
        for i in range(N):
            for j in range(i + 1, N):
                if time.time() >= deadline - 5:
                    break
                flip(adj, i, j)
                if not has_k4(adj):
                    flip(adj, i, j)
                    continue
                ns = max_triangle_free_size(adj)
                if ns < cur:
                    cur = ns
                    best_score = cur
                    best_mask = adj[:]
                flip(adj, i, j)

    # ---------- emit edges ----------
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges
