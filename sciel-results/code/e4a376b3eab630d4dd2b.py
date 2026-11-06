"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
            # Pivot: choose v in P|X maximizing |P & adj[v]|
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
            # Candidates: P minus neighbors of pivot
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

    def greedy_transversal(cliques_masks, n, mode="count"):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques.
        mode: 'count' = number of cliques, 'size' = sum of clique sizes, 'inverse' = prefer smaller cliques"""
        if not cliques_masks:
            return 0

        uncovered = set(cliques_masks)
        selected = 0

        while uncovered:
            best_v = -1
            best_score = -1
            for v in range(n):
                if selected & (1 << v):
                    continue
                score = 0
                for cm in uncovered:
                    if cm & (1 << v):
                        if mode == "count":
                            score += 1
                        elif mode == "size":
                            score += cm.bit_count()
                        elif mode == "inverse":
                            score += 1.0 / max(1, cm.bit_count())
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v < 0 or best_score <= 0:
                break
            selected |= (1 << best_v)
            uncovered = {cm for cm in uncovered if not (cm & (1 << best_v))}

        return selected

    def local_search_reduce(selected_mask, cliques_masks, n, max_iters=500):
        """Try to remove vertices; if it breaks coverage, stop."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for v in range(n):
                if not (selected_mask & (1 << v)):
                    continue
                trial = selected_mask ^ (1 << v)
                ok = True
                for cm in cliques_masks:
                    if not (cm & trial):
                        ok = False
                        break
                if ok:
                    selected_mask = trial
                    improved = True
                    break
        return selected_mask

    def local_search_swap(selected_mask, cliques_masks, n, max_iters=200):
        """Try swapping: remove one vertex, add another if it helps reduce total size."""
        # This is less useful for minimization since we're already trying to remove
        # But can help escape local optima by temporarily increasing then decreasing
        return selected_mask

    def solve_instance(n, p, seed, deadline):
        adj_mask = build_graph(n, p, seed)
        cliques = []
        bron_kerbosch_bitmask(n, adj_mask, cliques)

        if not cliques:
            return []

        best_mask = None
        best_size = n + 1

        # Try multiple greedy strategies
        for mode in ["count", "size", "inverse"]:
            mask = greedy_transversal(cliques, n, mode)
            mask = local_search_reduce(mask, cliques, n)
            s = mask.bit_count()
            if s < best_size:
                best_size = s
                best_mask = mask

        # Additional local search passes
        if best_mask is not None:
            # Try more aggressive reduction with different orderings
            for _ in range(3):
                if time.time() > deadline:
                    break
                improved = True
                while improved:
                    improved = False
                    for v in range(n):
                        if not (best_mask & (1 << v)):
                            continue
                        trial = best_mask ^ (1 << v)
                        ok = True
                        for cm in cliques:
                            if not (cm & trial):
                                ok = False
                                break
                        if ok:
                            best_mask = trial
                            improved = True
                            break
                s = best_mask.bit_count()
                if s < best_size:
                    best_size = s

        return [i for i in range(n) if best_mask & (1 << i)]

    start_time = time.time()
    deadline = start_time + 140

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices (guaranteed valid but large)
            out.append(list(range(n)))
            continue

        result = solve_instance(n, p, seed, deadline)
        out.append(result)

    return out
# EVOLVE-BLOCK-END
