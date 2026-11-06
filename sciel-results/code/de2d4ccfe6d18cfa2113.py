import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 117  # safety margin under 120s limit

    n = 40
    all_mask = (1 << n) - 1

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_pairs = len(all_pairs)

    # ---------- Fast C6 detection using precomputed neighbor intersections ----------
    # Key optimization: precompute N[w1] & N[w2] for all pairs to speed up C6 check

    def build_neighbor_intersections(adj):
        """Precompute N[w1] & N[w2] for all pairs (w1, w2)."""
        inter = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                v = adj[i] & adj[j]
                inter[i][j] = v
                inter[j][i] = v
        return inter

    def can_add_edge_fast(adj, inter, u, v):
        """Check if adding edge (u,v) creates a C6. Returns True if safe (no C6)."""
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        if nu == 0 or nv == 0:
            return True

        # Path: u-a-b-c-d-v
        # a in N(u)\{v}, d in N(v)\{u,a}
        # b in N(a)\{u,v,d}, c in N(d)\{u,v,a,b}, b~c
        # Optimization: for each (a,d), check if (N(a) & N(d)) has a vertex not in {u,v,a,d}
        # and that vertex is also in N(a)\{u,v,d} and N(d)\{u,v,a}
        
        am = nu
        while am:
            lb_a = am & (-am)
            ai = lb_a.bit_length() - 1
            am ^= lb_a

            dm = nv & ~lb_a
            while dm:
                lb_d = dm & (-dm)
                di = lb_d.bit_length() - 1
                dm ^= lb_d

                # Common neighbors of a and d, excluding u, v, a, d
                common = inter[ai][di] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_d
                if common:
                    # Need b in N(a)\{u,v,d} and c in N(d)\{u,v,a} with b~c
                    # But actually we need a path a-b-c-d, so b~a, b~c, c~d
                    # b in N(a)\{u,v,d}, c in N(d)\{u,v,a}, b~c
                    # If common has a vertex x, then x is in N(a) & N(d)
                    # But we need b~c where b in N(a) and c in N(d)
                    # The common neighbors give us candidates for b (or c)
                    # Actually: if x is in N(a) & N(d), then a-x-d is a path of length 2
                    # But we need a-b-c-d (length 3)
                    # So we need: b in N(a)\{u,v,d}, and N(b) & N(d) has a vertex c not in {u,v,a,b,d}
                    # OR equivalently: c in N(d)\{u,v,a}, and N(c) & N(a) has a vertex b not in {u,v,a,c,d}
                    
                    # Faster check: for each b in N(a)\{u,v,d}, check if N(b) & (N(d)\{u,v,a,b,d}) is non-empty
                    # Use precomputed intersections
                    b_cand = adj[ai] & ~(1 << u) & ~(1 << v) & ~lb_d
                    bm = b_cand
                    while bm:
                        lb_b = bm & (-bm)
                        bi = lb_b.bit_length() - 1
                        bm ^= lb_b
                        # c must be in N(d)\{u,v,a,b,d} and adjacent to b
                        c_cand = adj[di] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_b
                        if adj[bi] & c_cand:
                            return False
        return True

    def count_edges(adj):
        c = 0
        for i in range(n):
            c += adj[i].bit_count()
        return c // 2

    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def find_one_c6(adj):
        """Return one C6 as list of 6 vertices, or None."""
        for u in range(n):
            nu = adj[u]
            nv_cand = all_mask & ~(1 << u) & ~nu
            v = nv_cand
            while v:
                lb_v = v & (-v)
                vi = lb_v.bit_length() - 1
                v ^= lb_v

                am = nu
                while am:
                    lb_a = am & (-am)
                    ai = lb_a.bit_length() - 1
                    am ^= lb_a

                    dm = adj[vi] & ~lb_a
                    while dm:
                        lb_d = dm & (-dm)
                        di = lb_d.bit_length() - 1
                        dm ^= lb_d

                        b_cand = adj[ai] & ~(1 << u) & ~lb_v & ~lb_a & ~lb_d
                        c_cand = adj[di] & ~(1 << u) & ~lb_v & ~lb_a & ~lb_d

                        bm = b_cand
                        while bm:
                            lb_b = bm & (-bm)
                            bi = lb_b.bit_length() - 1
                            bm ^= lb_b
                            cm = adj[bi] & c_cand
                            if cm:
                                lb_c = cm & (-cm)
                                ci = lb_c.bit_length() - 1
                                return [u, ai, bi, ci, di, vi]
        return None

    # ---------- Greedy set-cover-inspired edge selection ----------
    # Score each non-edge by how "conflict-free" it is (fewer C6s it would create)
    # Then greedily add edges in order of lowest conflict score

    def greedy_set_cover_add(adj, inter, rng, budget_time):
        """Add edges using a set-cover-inspired heuristic: prioritize edges with fewer conflicts."""
        count = count_edges(adj)
        
        # Build list of non-edges with conflict scores
        non_edges = []
        for u, v in all_pairs:
            if not ((adj[u] >> v) & 1):
                non_edges.append((u, v))
        
        # Score: number of (a,d) pairs that would create a C6
        # Lower score = safer to add
        def conflict_score(u, v):
            nu = adj[u] & ~(1 << v)
            nv = adj[v] & ~(1 << u)
            score = 0
            am = nu
            while am:
                lb_a = am & (-am)
                ai = lb_a.bit_length() - 1
                am ^= lb_a
                dm = nv & ~lb_a
                while dm:
                    lb_d = dm & (-dm)
                    di = lb_d.bit_length() - 1
                    dm ^= lb_d
                    # Count common neighbors of a and d (excluding u,v,a,d)
                    common = inter[ai][di] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_d
                    if common:
                        score += 1
                        if score > 20:  # Early termination
                            return score
            return score
        
        # Sort by conflict score (ascending), with random tie-breaking
        scored_edges = []
        for u, v in non_edges:
            s = conflict_score(u, v)
            scored_edges.append((s, rng.random(), u, v))
        scored_edges.sort(key=lambda x: (x[0], x[1]))
        
        added = 0
        for s, _, u, v in scored_edges:
            if time.time() > budget_time:
                break
            if (adj[u] >> v) & 1:
                continue
            if can_add_edge_fast(adj, inter, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
                added += 1
                # Update inter for new edge
                for w in range(n):
                    if w != u and w != v:
                        if (adj[u] >> w) & 1:
                            inter[u][w] |= (1 << v)
                            inter[w][u] |= (1 << v)
                        if (adj[v] >> w) & 1:
                            inter[v][w] |= (1 << u)
                            inter[w][v] |= (1 << u)
        
        return adj, count, added

    # ---------- Local search with improved move evaluation ----------
    def local_search(adj, inter, rng, budget_time):
        """Local search with add, swap, and 2-opt moves using fast C6 check."""
        count = count_edges(adj)
        best_adj = adj[:]
        best_inter = [row[:] for row in inter]
        best_count = count

        existing = []
        non_existing = []
        for u, v in all_pairs:
            if (adj[u] >> v) & 1:
                existing.append((u, v))
            else:
                non_existing.append((u, v))

        no_improve = 0
        max_no_improve = 200

        while time.time() < budget_time:
            if not non_existing:
                break

            r = rng.random()
            if r < 0.5:
                # Add move
                idx = rng.randint(0, len(non_existing) - 1)
                u, v = non_existing[idx]
                if can_add_edge_fast(adj, inter, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
                    non_existing.pop(idx)
                    existing.append((u, v))
                    no_improve = 0
                    if count > best_count:
                        best_count = count
                        best_adj = adj[:]
                        best_inter = [row[:] for row in inter]
                else:
                    no_improve += 1
            elif r < 0.8:
                # Swap move
                if not existing or not non_existing:
                    no_improve += 1
                    continue
                idx_rem = rng.randint(0, len(existing) - 1)
                u1, v1 = existing[idx_rem]
                idx_add = rng.randint(0, len(non_existing) - 1)
                u2, v2 = non_existing[idx_add]

                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)

                if can_add_edge_fast(adj, inter, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)

                    existing.pop(idx_rem)
                    non_existing.append((u1, v1))
                    non_existing.pop(idx_add)
                    existing.append((u2, v2))
                    no_improve = 0
                    if count > best_count:
                        best_count = count
                        best_adj = adj[:]
                        best_inter = [row[:] for row in inter]
                else:
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    no_improve += 1
            else:
                # 2-opt: remove two, add two
                if len(existing) < 2 or len(non_existing) < 2:
                    no_improve += 1
                    continue
                idx_rem1 = rng.randint(0, len(existing) - 1)
                u1, v1 = existing[idx_rem1]
                idx_rem2 = rng.randint(0, len(existing) - 1)
                if idx_rem2 == idx_rem1:
                    no_improve += 1
                    continue
                u2, v2 = existing[idx_rem2]

                idx_add1 = rng.randint(0, len(non_existing) - 1)
                wa, xa = non_existing[idx_add1]
                idx_add2 = rng.randint(0, len(non_existing) - 1)
                if idx_add2 == idx_add1:
                    no_improve += 1
                    continue
                wb, xb = non_existing[idx_add2]

                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)

                ok1 = can_add_edge_fast(adj, inter, wa, xa)
                if ok1:
                    adj[wa] |= (1 << xa)
                    adj[xa] |= (1 << wa)
                    ok2 = can_add_edge_fast(adj, inter, wb, xb)
                    if ok2:
                        adj[wb] |= (1 << xb)
                        adj[xb] |= (1 << wb)

                        if idx_rem1 < idx_rem2:
                            existing.pop(idx_rem2)
                            existing.pop(idx_rem1)
                        else:
                            existing.pop(idx_rem1)
                            existing.pop(idx_rem2)
                        existing.append((wa, xa))
                        existing.append((wb, xb))

                        if idx_add1 < idx_add2:
                            non_existing.pop(idx_add2)
                            non_existing.pop(idx_add1)
                        else:
                            non_existing.pop(idx_add1)
                            non_existing.pop(idx_add2)
                        non_existing.append((u1, v1))
                        non_existing.append((u2, v2))

                        no_improve = 0
                        if count > best_count:
                            best_count = count
                            best_adj = adj[:]
                            best_inter = [row[:] for row in inter]
                    else:
                        adj[wa] &= ~(1 << xa)
                        adj[xa] &= ~(1 << wa)
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        no_improve += 1
                else:
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    no_improve += 1

            if no_improve > max_no_improve:
                no_improve = 0
                # Escape: try greedy set-cover on current state
                if time.time() < budget_time - 2:
                    adj, count, _ = greedy_set_cover_add(adj, inter, rng, budget_time - 1)
                    existing = []
                    non_existing = []
                    for u, v in all_pairs:
                        if (adj[u] >> v) & 1:
                            existing.append((u, v))
                        else:
                            non_existing.append((u, v))
                    if count > best_count:
                        best_count = count
                        best_adj = adj[:]
                        best_inter = [row[:] for row in inter]

        return best_adj, best_inter, best_count

    # ---------- Main search ----------
    best_adj = None
    best_count = 0
    best_inter = None

    # Phase 1: Structured seed + greedy set-cover
    # Start with 8 K5 blocks
    adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    inter = build_neighbor_intersections(adj)
    cnt = count_edges(adj)
    
    # Add inter-block edges using greedy set-cover
    rng1 = random.Random(42)
    adj, cnt, _ = greedy_set_cover_add(adj, inter, rng1, deadline - 60)
    
    if cnt > best_count:
        best_count = cnt
        best_adj = adj[:]
        best_inter = [row[:] for row in inter]

    # Phase 2: Try diverse random starts
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 40:
            break
        rng2 = random.Random(seed * 1000)
        adj2 = [0] * n
        # Random dense start
        for u, v in all_pairs:
            if rng2.random() < 0.3:
                adj2[u] |= (1 << v)
                adj2[v] |= (1 << u)
        
        # Prune C6s
        for _ in range(100):
            if time.time() > deadline - 35:
                break
            c6 = find_one_c6(adj2)
            if c6 is None:
                break
            idx = rng2.randint(0, 5)
            u, v = c6[idx], c6[(idx + 1) % 6]
            adj2[u] &= ~(1 << v)
            adj2[v] &= ~(1 << u)
        
        inter2 = build_neighbor_intersections(adj2)
        adj2, cnt2, _ = greedy_set_cover_add(adj2, inter2, rng2, deadline - 30)
        
        if cnt2 > best_count:
            best_count = cnt2
            best_adj = adj2[:]
            best_inter = [row[:] for row in inter2]

    # Phase 3: Local search from best
    if best_adj is None:
        best_adj = [0] * n
        best_inter = build_neighbor_intersections(best_adj)
        best_count = 0

    remaining = deadline - time.time() - 5
    if remaining > 10:
        ls_time = time.time() + remaining * 0.6
        best_adj, best_inter, best_count = local_search(best_adj, best_inter, random.Random(777), ls_time)

    # Phase 4: Final greedy set-cover polish
    if time.time() < deadline - 3:
        best_adj, best_count, _ = greedy_set_cover_add(best_adj, best_inter, random.Random(888), deadline - 2)

    # Phase 5: Final short local search
    if time.time() < deadline - 1:
        best_adj, best_inter, best_count = local_search(best_adj, best_inter, random.Random(999), deadline - 0.5)

    result = edges_from_adj(best_adj)
    return result
