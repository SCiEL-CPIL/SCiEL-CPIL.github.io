import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin against 90s kill

    N = 256
    all_nodes = list(range(2, N + 1))
    n_total = len(all_nodes)
    node_to_idx = {v: i for i, v in enumerate(all_nodes)}

    # Precompute conflict masks (lcm(a,b) <= 256  <=>  a*b <= 256*gcd(a,b))
    conflict_mask = [0] * n_total
    for i in range(n_total):
        a = all_nodes[i]
        mask = 0
        for j in range(n_total):
            if i == j:
                continue
            b = all_nodes[j]
            if a * b <= 256 * gcd(a, b):
                mask |= (1 << j)
        conflict_mask[i] = mask

    compat_mask = [0] * n_total
    full_mask = (1 << n_total) - 1
    for i in range(n_total):
        compat_mask[i] = full_mask & ~conflict_mask[i]

    # Number-theoretic reduction:
    # Primes p > 128 are compatible with EVERY node in [2,256]:
    #   gcd(p, y) in {1, p}; p>128 => no multiple of p in [2,256], so gcd=1,
    #   lcm(p, y) = p*y >= p*2 > 256.
    # So they are "universally compatible" and can always be included.
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    universal = [i for i in range(n_total) if all_nodes[i] > 128 and is_prime(all_nodes[i])]
    universal_set = set(universal)
    base_value = sum(1.0 / all_nodes[i] for i in universal)

    # Remaining (non-universal) nodes to solve as a weighted clique.
    rem = [i for i in range(n_total) if i not in universal_set]
    rem_pos = {i: k for k, i in enumerate(rem)}
    m = len(rem)

    # Build adjacency on the remaining nodes.
    adj = [0] * m
    for k, i in enumerate(rem):
        mask = compat_mask[i]
        am = 0
        for j in rem:
            if (mask >> j) & 1:
                am |= (1 << rem_pos[j])
        adj[k] = am

    weights = [1.0 / all_nodes[i] for i in rem]

    # ---------- Maximum weight clique (Tomita branch-and-bound) ----------
    best_clique = []
    best_weight = 0.0

    def solve():
        nonlocal best_clique, best_weight
        if m == 0:
            return
        # Order vertices by descending weight.
        order = sorted(range(m), key=lambda k: weights[k], reverse=True)
        pos = [0] * m
        for rank, k in enumerate(order):
            pos[k] = rank

        # P[i] = set of vertices with pos > pos[i] (bitmask), for coloring.
        P = [0] * m
        for i in range(m):
            p = 0
            for j in range(m):
                if pos[j] > pos[i]:
                    p |= (1 << j)
            P[i] = p

        # Precompute per-vertex candidate colors (bitmask over color classes).
        # A vertex v can use color c iff it is non-adjacent to every vertex
        # already in color c.
        max_colors = m
        color_classes = [0] * max_colors  # bitmask of vertices in each color
        vcolor = [0] * m
        for v in range(m):
            cm = 0
            for c in range(max_colors):
                cc = color_classes[c]
                if cc == 0:
                    cm |= (1 << c)
                    continue
                ok = True
                t = cc
                while t:
                    lsb = t & (-t)
                    u = lsb.bit_length() - 1
                    if (adj[v] >> u) & 1:
                        ok = False
                        break
                    t -= lsb
                if ok:
                    cm |= (1 << c)
            vcolor[v] = cm

        def color_bound(Pmask):
            # Sequential coloring of Pmask (in pos order) -> #colors = upper bound.
            if Pmask == 0:
                return 0
            # iterate vertices in increasing pos order
            verts = []
            t = Pmask
            while t:
                lsb = t & (-t)
                verts.append(lsb.bit_length() - 1)
                t -= lsb
            verts.sort(key=lambda x: pos[x])

            color_classes2 = [0] * m
            maxc = 0
            for v in verts:
                cm = vcolor[v]
                placed = False
                c = 0
                while cm:
                    lsb = cm & (-cm)
                    c = lsb.bit_length() - 1
                    cm -= lsb
                    cc = color_classes2[c]
                    ok = True
                    t2 = cc
                    while t2:
                        lsb2 = t2 & (-t2)
                        u = lsb2.bit_length() - 1
                        if (adj[v] >> u) & 1:
                            ok = False
                            break
                        t2 -= lsb2
                    if ok:
                        color_classes2[c] = cc | (1 << v)
                        if c + 1 > maxc:
                            maxc = c + 1
                        placed = True
                        break
                if not placed:
                    color_classes2[maxc] = (1 << v)
                    maxc += 1
            return maxc

        # Recursive branch-and-bound.
        def expand(Pmask, Qmask, cur_w):
            nonlocal best_clique, best_weight
            if (time.time() - start_time) > deadline:
                return
            # Pmask: candidate set (compatible with all of Q), ordered by pos desc
            # Qmask: current clique
            if Pmask == 0:
                if cur_w > best_weight:
                    best_weight = cur_w
                    best_clique = [j for j in range(m) if (Qmask >> j) & 1]
                return

            # Color bound on Pmask.
            cb = color_bound(Pmask)
            if cur_w + cb <= best_weight:
                return

            # Take vertices in descending pos order; process one at a time.
            t = Pmask
            while t:
                if (time.time() - start_time) > deadline:
                    return
                lsb = t & (-t)
                v = lsb.bit_length() - 1
                t -= lsb

                new_w = cur_w + weights[v]
                if new_w > best_weight:
                    best_weight = new_w
                    best_clique = [j for j in range(m) if ((Qmask | (1 << v)) >> j) & 1]

                new_P = Pmask & adj[v]
                # Only keep vertices in new_P that come AFTER v in pos order
                # (i.e., already processed ones are excluded automatically because
                #  Pmask shrinks as we go).
                if new_P:
                    expand(new_P, Qmask | (1 << v), new_w)

                # Remove v from Pmask for subsequent iterations.
                Pmask &= ~(1 << v)

        full = (1 << m) - 1
        expand(full, 0, 0.0)

    solve()

    # Assemble subset from clique + universal primes.
    subset = [all_nodes[i] for i in best_clique]
    subset += [all_nodes[i] for i in universal]
    best_value = base_value + best_weight

    # ---------- Local search polish (exact, on full graph) ----------
    def local_search(selected, d):
        selected = selected[:]
        used = 0
        for v in selected:
            used |= (1 << node_to_idx[v])

        improved = True
        while improved and (time.time() - start_time) < d:
            improved = False
            # 1. Add any compatible node.
            for i in range(n_total):
                if (used >> i) & 1:
                    continue
                if used & conflict_mask[i]:
                    continue
                selected.append(all_nodes[i])
                used |= (1 << i)
                improved = True
            # 2. 1-swap: replace a node with a smaller compatible one.
            non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
            for s_i in list(range(len(selected))):
                s = selected[s_i]
                si = node_to_idx[s]
                used_without_s = used & ~(1 << si)
                best_n = None
                best_gain = 0.0
                for n_i in non_selected:
                    n_val = all_nodes[n_i]
                    if n_val >= s:
                        continue
                    if used_without_s & conflict_mask[n_i]:
                        continue
                    gain = 1.0 / n_val - 1.0 / s
                    if gain > best_gain:
                        best_gain = gain
                        best_n = n_i
                if best_n is not None:
                    selected[s_i] = all_nodes[best_n]
                    used = used_without_s | (1 << best_n)
                    improved = True
                    non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
                    break

        return selected

    if (time.time() - start_time) < deadline - 3:
        polished = local_search(subset, deadline - 2)
        val = sum(1.0 / v for v in polished)
        if val > best_value:
            subset = polished
            best_value = val

    if not subset:
        return [256]

    return subset
