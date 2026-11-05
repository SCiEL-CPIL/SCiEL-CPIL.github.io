def construct():
    import time
    import random
    import sys
    import numpy as np

    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def time_up():
        return (time.time() - start_time) > DEADLINE

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def max_clique_bk(adj, n, active_mask):
        """Bron-Kerbosch with pivot and coloring to find max clique."""
        if active_mask == 0:
            return []

        best = [0]
        best_clique = []

        verts = []
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb

        if not verts:
            return []

        verts.sort(key=lambda v: (adj[v] & active_mask).bit_count(), reverse=True)

        def color_sort(P_mask):
            pm = P_mask
            verts_p = []
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                verts_p.append(idx)
                pm ^= lsb

            if not verts_p:
                return []

            verts_p.sort(key=lambda v: (adj[v] & P_mask).bit_count(), reverse=True)

            v_color = {}
            result = []
            for v in verts_p:
                used_colors = 0
                neigh = adj[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    if u in v_color:
                        used_colors |= (1 << v_color[u])
                    nm ^= lsb
                c = 1
                while (used_colors >> c) & 1:
                    c += 1
                v_color[v] = c
                result.append((c, v))

            result.sort(key=lambda x: x[0])
            return result

        def bk(R, P, X):
            nonlocal best, best_clique

            if P == 0 and X == 0:
                if len(R) > best[0]:
                    best[0] = len(R)
                    best_clique = list(R)
                return

            if time_up():
                return

            union = P | X
            pivot = -1
            max_deg = -1
            m = union
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                deg = (adj[u] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
                m ^= lsb

            if pivot == -1:
                candidates = P
            else:
                candidates = P & ~(adj[pivot])

            if candidates == 0:
                return

            colored = color_sort(candidates)
            for color, v in reversed(colored):
                if len(R) + color <= best[0]:
                    break

                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P ^= (1 << v)
                X |= (1 << v)

                if time_up():
                    return

        bk([], active_mask, 0)
        return best_clique

    def solve_instance(n, p, seed, time_budget=65.0):
        adj = generate_graph(n, p, seed)
        rem = adj[:]
        cliques = []

        active = 0
        for i in range(n):
            if rem[i]:
                active |= (1 << i)

        instance_start = time.time()
        while active and not time_up() and (time.time() - instance_start) < time_budget:
            clique = max_clique_bk(rem, n, active)

            if not clique:
                found = False
                m = active
                while m and not found:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    nb = rem[u] & active
                    if nb:
                        lsb2 = nb & -nb
                        v = lsb2.bit_length() - 1
                        clique = [u, v]
                        found = True
                    m ^= lsb
                if not found:
                    break

            cliques.append(clique)

            for u in clique:
                for v in clique:
                    if u != v:
                        rem[u] &= ~(1 << v)

            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)

        # Fallback
        if active:
            for u in range(n):
                nb = rem[u]
                while nb:
                    lsb = nb & -nb
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                        rem[u] &= ~(1 << v)
                        rem[v] &= ~(1 << u)
                    nb ^= lsb

        # Phase 2: Merge improvement
        if not time_up():
            improved = True
            while improved and not time_up():
                improved = False
                new_cliques = list(cliques)
                for i in range(len(new_cliques)):
                    if improved:
                        break
                    for j in range(i + 1, len(new_cliques)):
                        if improved:
                            break
                        c1 = set(new_cliques[i])
                        c2 = set(new_cliques[j])
                        union = c1 | c2
                        verts = list(union)
                        valid_clique = True
                        for a in range(len(verts)):
                            for b in range(a + 1, len(verts)):
                                if not (adj[verts[a]] & (1 << verts[b])):
                                    valid_clique = False
                                    break
                            if not valid_clique:
                                break

                        if not valid_clique:
                            continue

                        c1_only = c1 - c2
                        c2_only = c2 - c1

                        conflict = False
                        for k in range(len(new_cliques)):
                            if k == i or k == j:
                                continue
                            ck = set(new_cliques[k])
                            if (ck & c1_only) and (ck & c2_only):
                                conflict = True
                                break

                        if not conflict:
                            new_cliques[i] = verts
                            del new_cliques[j]
                            cliques = new_cliques
                            improved = True
                            break

        return cliques

    c1 = solve_instance(32, 0.6, 601, time_budget=60.0)
    c2 = solve_instance(64, 0.55, 602, time_budget=60.0)

    return [c1, c2]
