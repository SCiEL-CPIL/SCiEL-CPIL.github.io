import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    random.seed(42)

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total_edges = len(all_edges)  # 120
    edge_index = [[-1] * n for _ in range(n)]
    for idx, (u, v) in enumerate(all_edges):
        edge_index[u][v] = idx
        edge_index[v][u] = idx

    # ---------- Precompute Q3 masks over a fixed 8-vertex ordering ----------
    local_edge_idx = [[-1] * 8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i + 1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1

    # A Q3 on 8 vertices: bipartition into two 4-sets A,B with a perfect matching
    # (the matching = the "diagonal" edges crossing the bipartition). The cube
    # contains all edges within A, all within B, and the matching A-B edges.
    seen = set()
    q3_masks = []
    for A in itertools.combinations(range(8), 4):
        A_set = set(A)
        B = tuple(v for v in range(8) if v not in A_set)
        for perm in itertools.permutations(B):
            mask = 0
            for i in range(4):
                for j in range(4):
                    if i != j:
                        mask |= (1 << local_edge_idx[A[i]][B[j]])
            if mask not in seen:
                seen.add(mask)
                q3_masks.append(mask)

    # ---------- Precompute all 8-subsets ----------
    all_subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(all_subsets)

    subset_l2g = []
    for s in all_subsets:
        l2g = []
        for i in range(8):
            for j in range(i + 1, 8):
                l2g.append(edge_index[s[i]][s[j]])
        subset_l2g.append(l2g)

    # For each global edge, the list of subsets containing it
    edge_to_subsets = [[] for _ in range(total_edges)]
    for si, l2g in enumerate(subset_l2g):
        for g in l2g:
            edge_to_subsets[g].append(si)

    # ---------- Fast adjacency-bitmask Q3 check (inspiration 2) ----------
    def creates_q3(u, v, adj):
        # adj[x] = bitmask of neighbors of x. (u,v) assumed NOT present.
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        # a,b: other neighbors of u ; c,d: other neighbors of v
        for a in range(n):
            if a == u or a == v or not ((mask_u >> a) & 1):
                continue
            for b in range(a + 1, n):
                if b == u or b == v or not ((mask_u >> b) & 1):
                    continue
                # c in N(v) ∩ N(a), d in N(v) ∩ N(b), c != d, c,d not in {u,v,a,b}
                cand_c = mask_v & adj[a] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b))
                cand_d = mask_v & adj[b] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b))
                for c in range(n):
                    if not ((cand_c >> c) & 1):
                        continue
                    for d in range(n):
                        if d == c or not ((cand_d >> d) & 1):
                            continue
                        # e in N(a) ∩ N(b), f in N(c) ∩ N(d), (e,f) edge
                        cand_e = adj[a] & adj[b] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                        if cand_e == 0:
                            continue
                        for e in range(n):
                            if not ((cand_e >> e) & 1):
                                continue
                            cand_f = adj[c] & adj[d] & adj[e] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e))
                            if cand_f != 0:
                                return True
        return False

    # ---------- Greedy construction (current strategy) ----------
    G = (1 << total_edges) - 1  # start from K16
    adj = [0] * n
    for idx, (u, v) in enumerate(all_edges):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    edge_count = total_edges

    bad_subsets = set(range(num_subsets))

    while bad_subsets and time.time() < deadline - 25.0:
        si = random.choice(list(bad_subsets))
        l2g = subset_l2g[si]
        mask = 0
        for l in range(28):
            if (G >> l2g[l]) & 1:
                mask |= (1 << l)
        violated = None
        for qm in q3_masks:
            if (mask & qm) == qm:
                violated = qm
                break
        if violated is None:
            bad_subsets.discard(si)
            continue
        pattern_edges = [l2g[l] for l in range(28)
                         if ((violated >> l) & 1) and ((G >> l2g[l]) & 1)]
        if not pattern_edges:
            bad_subsets.discard(si)
            continue
        e_rm = random.choice(pattern_edges)
        G &= ~(1 << e_rm)
        edge_count -= 1
        u, v = all_edges[e_rm]
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        affected = set(edge_to_subsets[e_rm])
        new_bad = set()
        for sidx in bad_subsets:
            if sidx in affected:
                l2g2 = subset_l2g[sidx]
                m2 = 0
                for l in range(28):
                    if (G >> l2g2[l]) & 1:
                        m2 |= (1 << l)
                still = False
                for qm in q3_masks:
                    if (m2 & qm) == qm:
                        still = True
                        break
                if still:
                    new_bad.add(sidx)
            else:
                new_bad.add(sidx)
        bad_subsets = new_bad

    # ---------- Local search: re-add missing edges (inspiration 2 check) ----------
    if time.time() < deadline - 5.0:
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for e_idx in range(total_edges):
                if time.time() >= deadline - 5.0:
                    break
                if (G >> e_idx) & 1:
                    continue
                u, v = all_edges[e_idx]
                if creates_q3(u, v, adj):
                    continue
                G |= (1 << e_idx)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                changed = True

    # ---------- Final verification (safety net) ----------
    if time.time() < deadline - 1.0:
        for si in range(num_subsets):
            l2g = subset_l2g[si]
            mask = 0
            for l in range(28):
                if (G >> l2g[l]) & 1:
                    mask |= (1 << l)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    # remove one edge of the violated pattern
                    for l in range(28):
                        if (qm >> l) & 1 and (G >> l2g[l]) & 1:
                            G &= ~(1 << l2g[l])
                            u, v = all_edges[l2g[l]]
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            break
                    break

    result = []
    for u, v in all_edges:
        if (G >> edge_index[u][v]) & 1:
            result.append([u, v])
    return result
