"""Combined program: MILP exact solve + bitmask BK + local search fallback."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math

    def bron_kerbosch_bitmask(n, adj_mask, cliques):
        """Bron-Kerbosch with pivoting, using bitmasks. Collects cliques of size >= 2."""
        def bk(R, P, X):
            if P == 0 and X == 0:
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            PX = P | X
            if PX == 0:
                return
            best_v = -1
            best_count = -1
            temp = PX
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                cnt = (P & adj_mask[v]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
                temp ^= lsb
            if best_v >= 0:
                candidates = P & ~adj_mask[best_v]
            else:
                candidates = P
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                bk(R | lsb, P & adj_mask[v], X & adj_mask[v])
                P ^= lsb
                X |= lsb
                temp ^= lsb

        bk(0, (1 << n) - 1, 0)

    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def bitmask_to_list(mask):
        return [i for i in range(mask.bit_length()) if mask & (1 << i)]

    def greedy_transversal(cliques_masks, n):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        if not cliques_masks:
            return 0
        m = len(cliques_masks)
        vertex_cliques = [[] for _ in range(n)]
        for ci, cm in enumerate(cliques_masks):
            temp = cm
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                vertex_cliques[v].append(ci)
                temp ^= lsb
        uncovered = [1] * m
        selected = 0
        while True:
            if all(c == 0 for c in uncovered):
                break
            best_v = -1
            best_count = -1
            for v in range(n):
                if selected & (1 << v):
                    continue
                count = sum(1 for ci in vertex_cliques[v] if uncovered[ci] > 0)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v < 0 or best_count <= 0:
                break
            selected |= (1 << best_v)
            for ci in vertex_cliques[best_v]:
                uncovered[ci] = 0
        return selected

    def local_search_reduce_fast(selected_mask, cliques_masks, n, max_iters=500):
        """Try to remove vertices that are not unique covers of any clique."""
        if not cliques_masks:
            return selected_mask
        m = len(cliques_masks)
        cover_count = [0] * m
        unique_cover = [-1] * m
        for ci, cm in enumerate(cliques_masks):
            in_clique = cm & selected_mask
            cnt = in_clique.bit_count()
            cover_count[ci] = cnt
            if cnt == 1:
                unique_cover[ci] = in_clique.bit_length() - 1
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for v in range(n):
                if not (selected_mask & (1 << v)):
                    continue
                is_unique = False
                for ci in range(m):
                    if unique_cover[ci] == v:
                        is_unique = True
                        break
                if is_unique:
                    continue
                selected_mask ^= (1 << v)
                for ci in range(m):
                    if cliques_masks[ci] & (1 << v):
                        cover_count[ci] -= 1
                        if cover_count[ci] == 1:
                            in_clique = cliques_masks[ci] & selected_mask
                            unique_cover[ci] = in_clique.bit_length() - 1 if in_clique else -1
                        elif cover_count[ci] <= 0:
                            unique_cover[ci] = -1
                        else:
                            unique_cover[ci] = -1
                improved = True
                break
        return selected_mask

    def solve_with_milp(n, cliques_masks, deadline):
        """Try to solve the minimum transversal via SciPy MILP. Returns mask or None."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np

            m = len(cliques_masks)
            if m == 0:
                return 0

            # Build constraint matrix A (m x n)
            A = np.zeros((m, n), dtype=np.float64)
            for ci, cm in enumerate(cliques_masks):
                temp = cm
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    A[ci, v] = 1.0
                    temp ^= lsb

            c = np.ones(n, dtype=np.float64)
            constraints = [LinearConstraint(A, lb=np.ones(m, dtype=np.float64), ub=np.inf)]
            integrality = np.ones(n, dtype=np.int8)
            bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

            if res.status == 0 and res.x is not None:
                mask = 0
                for v in range(n):
                    if res.x[v] > 0.5:
                        mask |= (1 << v)
                # Verify validity
                for cm in cliques_masks:
                    if not (cm & mask):
                        return None
                return mask
        except Exception:
            pass
        return None

    def solve_instance(n, p, seed, deadline):
        adj_mask = build_graph(n, p, seed)
        cliques = []
        bron_kerbosch_bitmask(n, adj_mask, cliques)

        if not cliques:
            return []

        # Try MILP first (exact or near-exact)
        if time.time() < deadline - 10:
            milp_mask = solve_with_milp(n, cliques, deadline)
            if milp_mask is not None:
                # Polish with local search reduction
                milp_mask = local_search_reduce_fast(milp_mask, cliques, n)
                return bitmask_to_list(milp_mask)

        # Fallback: greedy + local search
        mask = greedy_transversal(cliques, n)
        mask = local_search_reduce_fast(mask, cliques, n)
        return bitmask_to_list(mask)

    start_time = time.time()
    deadline = start_time + 130

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            out.append(list(range(n)))
            continue
        result = solve_instance(n, p, seed, deadline)
        out.append(result)

    return out
# EVOLVE-BLOCK-END
