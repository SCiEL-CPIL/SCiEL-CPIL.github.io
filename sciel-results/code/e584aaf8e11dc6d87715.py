import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def is_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):
                    return False
        return True

    def has_independent_set(adj_mask, n, target=11):
        sys.setrecursionlimit(100000)
        node_count = [0]
        max_nodes = 3500000
        deadline_local = deadline - 5

        # Precompute neighbor counts for ordering
        def dfs(candidates, size):
            if size >= target:
                return True
            if candidates == 0:
                return False
            if size + candidates.bit_count() < target:
                return False
            if time.time() > deadline_local:
                return False
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False

            # Choose vertex with minimum (remaining after include)
            c = candidates
            best_v = -1
            min_remaining = n + 1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                remaining = (candidates & ~(adj_mask[v] | (1 << v))).bit_count()
                if remaining < min_remaining:
                    min_remaining = remaining
                    best_v = v
                    if min_remaining <= 2:
                        break

            if best_v == -1:
                return False

            # Include branch first (more pruning)
            new_cand = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if dfs(new_cand, size + 1):
                return True

            # Exclude branch
            new_cand = candidates & ~(1 << best_v)
            if dfs(new_cand, size):
                return True

            return False

        all_cand = (1 << n) - 1
        return dfs(all_cand, 0)

    def build_c4_free_via_pairs(n, seed):
        """Build C4-free graph by selecting pairs with limited common neighbors."""
        rng = random.Random(seed)
        adj_mask = [0] * n
        # common_count[x] = number of common neighbors between vertex x and the "pivot"
        # Instead, track for each pair how many common neighbors they have
        # We build edge by edge

        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)

        for u, v in all_pairs:
            nu = adj_mask[u]
            nv = adj_mask[v]
            # Adding edge (u,v) creates C4 iff there exist x,y with
            # x in N(u), y in N(v), x != y, x~y, and x,y both adjacent to each other
            # More precisely: C4: u-x-v-y-u or u-x-y-v-u
            # Actually C4-free means no two vertices have >1 common neighbor.
            # So adding (u,v): check if u and v share >1 common neighbor already? No.
            # The condition is: after adding (u,v), no pair has >1 common neighbor.
            # Pairs to check: (u, x) for x in N(v), (v, x) for x in N(u),
            # and (x, y) for x in N(u), y in N(v).
            # (u,x): common neighbors after adding (u,v) = old_common(u,x) + (1 if v in N(x) and v in N(u) new)
            #   Actually v is now in N(u). So common(u,x) gains 1 if v in N(x).
            #   But v in N(x) means x in N(v). So for each x in N(v), common(u,x) increases by 1.
            #   We need common(u,x) <= 1 for all x.
            #   So for each x in N(v): if common(u,x) before >= 1, then after adding it's 2. BAD.
            # Similarly for (v,x) for x in N(u).
            # For (x,y) with x in N(u), y in N(v): common(x,y) gains 1 if u in N(y) and v in N(x).
            #   u in N(y) means y in N(u). v in N(x) means x in N(v).
            #   So for x in N(u) ∩ N(v) and y in N(v) ∩ N(u): common(x,y) gains 1.
            #   Actually both x and y are in N(u)∩N(v), and we need to check if common(x,y) becomes 2.

            # Simpler check: for each x in N(v), if (adj_mask[u] & adj_mask[x]).bit_count() >= 1:
            #   then after adding (u,v), common(u,x) = old + 1 (since v is now neighbor of u and v is neighbor of x)
            #   Wait, v is neighbor of x (x in N(v)) and v will be neighbor of u. So yes common(u,x) increases by 1.
            #   So if old common(u,x) >= 1, new common >= 2. Fail.
            # Similarly for x in N(u), check common(v,x).
            # For x in N(u), y in N(v): if x != y, common(x,y) increases by 1 iff u in N(y) and v in N(x).
            #   u in N(y) iff y in N(u). v in N(x) iff x in N(v).
            #   So for x in N(u) ∩ N(v), y in N(v) ∩ N(u), x != y: common(x,y) increases by 1.
            #   We need old common(x,y) <= 0 for all such pairs. I.e., no such x,y are adjacent.

            ok = True
            # Check u-x for x in N(v)
            c = nv
            while c:
                lsb = c & (-c)
                x = lsb.bit_length() - 1
                c ^= lsb
                if (nu & adj_mask[x]).bit_count() >= 1:
                    ok = False
                    break
            if ok:
                # Check v-x for x in N(u)
                c = nu
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if (nv & adj_mask[x]).bit_count() >= 1:
                        ok = False
                        break
            if ok:
                # Check x-y for x in N(u)∩N(v), y in N(v)∩N(u), x!=y
                common_nv_nu = nu & nv
                if common_nv_nu & (common_nv_nu - 1):
                    # More than one vertex in common
                    c1 = common_nv_nu
                    found = False
                    while c1 and not found:
                        lsb1 = c1 & (-c1)
                        x = lsb1.bit_length() - 1
                        c1 ^= lsb1
                        c2 = common_nv_nu & (~lsb1)
                        while c2:
                            lsb2 = c2 & (-c2)
                            y = lsb2.bit_length() - 1
                            c2 ^= lsb2
                            if adj_mask[x] & lsb2:
                                found = True
                                break
                    if found:
                        ok = False

            if ok:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)

        return adj_mask

    best_result = None

    for n in range(43, 29, -1):
        if n >= 40:
            max_seeds = 1800
        elif n >= 35:
            max_seeds = 1200
        else:
            max_seeds = 800

        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break

            adj_mask = build_c4_free_via_pairs(n, seed)

            if not is_c4_free(adj_mask, n):
                continue

            if not has_independent_set(adj_mask, n, 11):
                edges = []
                for u in range(n):
                    m = adj_mask[u]
                    while m:
                        lsb = m & (-m)
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        m ^= lsb
                return {'n': n, 'edges': edges}

        if time.time() > deadline - 40:
            break

    # Fallback: 10 disjoint triangles
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]

    return {'n': 30, 'edges': fallback_edges}
