import time
import random
from collections import defaultdict

_DEADLINE = 128.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _deadline_hit():
    return _elapsed() > _DEADLINE


def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency
        adj_c = [0] * k
        adj_i = [0] * (n - k)

        for (u, v) in edges:
            if u < k and v < k:
                adj_c[u] |= (1 << v)
                adj_c[v] |= (1 << u)
            elif u < k and v >= k:
                idx_v = v - k
                adj_i[idx_v] |= (1 << u)
            elif u >= k and v < k:
                idx_v = u - k
                adj_i[idx_v] |= (1 << v)

        # uncovered tracks which C-C edges are still uncovered
        uncovered = [0] * k
        for u in range(k):
            uncovered[u] = adj_c[u]

        def _max_clique_greedy(cand_mask):
            """Find a large clique within cand_mask using greedy expansion."""
            if cand_mask == 0:
                return 0
            best = 0
            temp = cand_mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                # Greedy: start with v, add neighbors in order of degree
                sub = cand_mask & uncovered[v] & ~lsb
                # Sort candidates by degree in sub for better greedy
                cur = lsb
                cand = sub
                while cand:
                    # Pick vertex with highest degree in current candidate set
                    best_v = -1
                    best_deg = -1
                    t2 = cand
                    while t2:
                        lsb2 = t2 & -t2
                        v2 = lsb2.bit_length() - 1
                        t2 ^= lsb2
                        deg = (cand & uncovered[v2]).bit_count()
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v2
                    if best_v < 0:
                        break
                    cur |= (1 << best_v)
                    cand = cand & uncovered[best_v]
                    cand &= ~(1 << best_v)
                size = cur.bit_count()
                if size > best.bit_count():
                    best = cur
            return best

        def _cover_cross_for_vertex(idx):
            """Cover all cross edges for independent vertex k+idx."""
            remaining = adj_i[idx]
            while remaining:
                if _deadline_hit():
                    temp = remaining
                    while temp:
                        lsb = temp & -temp
                        u = lsb.bit_length() - 1
                        temp ^= lsb
                        yield [k + idx, u]
                    return

                # Find a clique in `remaining` that is also a clique in uncovered
                best_mask = 0
                temp = remaining
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    temp ^= lsb
                    # Neighbors of u in both remaining and uncovered
                    neigh = remaining & uncovered[u] & ~lsb
                    # Greedy expand
                    cur = lsb
                    cand = neigh
                    while cand:
                        # Pick vertex with highest degree in current candidate set
                        best_v = -1
                        best_deg = -1
                        t2 = cand
                        while t2:
                            lsb2 = t2 & -t2
                            v2 = lsb2.bit_length() - 1
                            t2 ^= lsb2
                            deg = (cand & uncovered[v2]).bit_count()
                            if deg > best_deg:
                                best_deg = deg
                                best_v = v2
                        if best_v < 0:
                            break
                        cur |= (1 << best_v)
                        cand = cand & uncovered[best_v]
                        cand &= ~(1 << best_v)
                    if cur.bit_count() > best_mask.bit_count():
                        best_mask = cur

                if best_mask == 0:
                    lsb = remaining & -remaining
                    best_mask = lsb

                # Form clique {k+idx} ∪ S
                clique = [k + idx]
                temp = best_mask
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    temp ^= lsb
                    clique.append(u)
                yield clique

                # Remove edges within best_mask from uncovered
                verts = []
                temp = best_mask
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    verts.append(u)
                    temp ^= lsb
                for i in range(len(verts)):
                    u = verts[i]
                    for j in range(i + 1, len(verts)):
                        w = verts[j]
                        uncovered[u] &= ~(1 << w)
                        uncovered[w] &= ~(1 << u)

                remaining &= ~best_mask

        cliques = []

        # Process independent vertices, sorted by degree descending
        i_indices = list(range(n - k))
        i_indices.sort(key=lambda idx: adj_i[idx].bit_count(), reverse=True)

        for idx in i_indices:
            for c in _cover_cross_for_vertex(idx):
                cliques.append(c)

        # Process remaining uncovered edges in C
        full_mask = (1 << k) - 1
        while True:
            if _deadline_hit():
                for u in range(k):
                    for w in range(u + 1, k):
                        if (uncovered[u] >> w) & 1:
                            cliques.append([u, w])
                            uncovered[u] &= ~(1 << w)
                            uncovered[w] &= ~(1 << u)
                break

            has_edge = False
            for u in range(k):
                if uncovered[u]:
                    has_edge = True
                    break
            if not has_edge:
                break

            S_mask = _max_clique_greedy(full_mask)
            if S_mask.bit_count() < 2:
                for u in range(k):
                    if uncovered[u]:
                        lsb = uncovered[u] & -uncovered[u]
                        w = lsb.bit_length() - 1
                        S_mask = (1 << u) | (1 << w)
                        break

            clique = []
            temp = S_mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                temp ^= lsb
                clique.append(u)
            cliques.append(clique)

            verts = []
            temp = S_mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                verts.append(u)
                temp ^= lsb
            for i in range(len(verts)):
                u = verts[i]
                for j in range(i + 1, len(verts)):
                    w = verts[j]
                    uncovered[u] &= ~(1 << w)
                    uncovered[w] &= ~(1 << u)

        # Local improvement: merge cliques whose union is a clique
        def _is_clique(verts_set):
            """Check if a set of vertices forms a clique in the original graph."""
            verts_list = list(verts_set)
            for a in range(len(verts_list)):
                ua = verts_list[a]
                for b in range(a + 1, len(verts_list)):
                    ub = verts_list[b]
                    if ua >= k and ub >= k:
                        return False  # Two independent vertices can't be adjacent
                    elif ua >= k:
                        if not (adj_i[ua - k] & (1 << ub)):
                            return False
                    elif ub >= k:
                        if not (adj_i[ub - k] & (1 << ua)):
                            return False
                    else:
                        if not (adj_c[ua] & (1 << ub)):
                            return False
            return True

        improved = True
        while improved and not _deadline_hit():
            improved = False
            new_cliques = []
            used = [False] * len(cliques)

            for i in range(len(cliques)):
                if used[i]:
                    continue
                cur = set(cliques[i])
                for j in range(i + 1, len(cliques)):
                    if used[j]:
                        continue
                    candidate = cur | set(cliques[j])
                    if _is_clique(candidate):
                        cur = candidate
                        used[j] = True
                        improved = True
                new_cliques.append(list(cur))
                used[i] = True

            if improved:
                cliques = new_cliques

        # Final type conversion
        result = []
        for c in cliques:
            result.append([int(x) for x in c])
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            out.append([list(e) for e in instance(*spec)])
        else:
            out.append(_solve_split(*spec))
    return out
