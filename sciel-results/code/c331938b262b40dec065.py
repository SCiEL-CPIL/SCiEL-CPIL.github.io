import itertools
import time

def construct():
    deadline = time.time() + 140.0
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    nT = len(all_triples)
    masks = [0] * nT
    for idx, (a, b, c) in enumerate(all_triples):
        masks[idx] = (1 << a) | (1 << b) | (1 << c)
    # Precompute linear-incompatibility: edge i conflicts with j iff they share >=2 points
    conflict = [0] * nT
    for i in range(nT):
        mi = masks[i]
        for j in range(i + 1, nT):
            if (mi & masks[j]) & ((mi & masks[j]) - 1):
                conflict[i] |= 1 << j
                conflict[j] |= 1 << i

    # ---------- independence number via branch and bound ----------
    def alpha_leq(edges, k):
        # returns True if independence number <= k
        m = len(edges)
        if m == 0:
            return True
        E = [edges[i] for i in range(m)]
        best = 0
        # order vertices by degree (heuristic)
        deg = [0] * n
        for e in E:
            for v in range(n):
                if (e >> v) & 1:
                    deg[v] += 1
        order = sorted(range(n), key=lambda v: -deg[v])
        pos = [0] * n
        for p, v in enumerate(order):
            pos[v] = p
        # branch on a vertex: include (remove it and its edges) or exclude
        def rec(mask, depth, count):
            nonlocal best
            if count + (n - depth) <= best:
                return
            # find a set of remaining vertices with no full edge; count is chosen so far
            # upper bound: count + remaining vertices
            # prune: if count + (n - depth) <= best -> handled above
            # pick next vertex to branch on: first in order not decided
            # We track decided via depth over 'order'
            if depth == n:
                if count > best:
                    best = count
                return
            v = order[depth]
            if (mask >> v) & 1 == 0:
                # vertex v already removed (by an edge we included) -> cannot include
                rec(mask, depth + 1, count)
                return
            # Option A: exclude v (v not in independent set)
            rec(mask, depth + 1, count)
            # Option B: include v -> remove v and all edges containing v
            newmask = mask
            for e in E:
                if (e >> v) & 1:
                    newmask &= ~e
            rec(newmask, depth + 1, count + 1)
        rec((1 << n) - 1, 0, 0)
        return best <= k

    def has_indep_set(edges, k):
        # returns True if there exists an independent set of size k (alpha >= k)
        m = len(edges)
        if m == 0:
            return True
        E = [edges[i] for i in range(m)]
        found = False
        # try to build independent set of size k
        def rec(available, need, count):
            nonlocal found
            if found:
                return
            if count == k:
                found = True
                return
            if need < 0:
                return
            # choose vertices from 'available' bitmask
            # greedy pick highest-degree? just iterate
            while available and not found:
                # pick a vertex
                v = (available & -available).bit_length() - 1
                available &= available - 1
                # include v: remove v and its edges from available vertices
                newavail = available
                for e in E:
                    if (e >> v) & 1:
                        newavail &= ~e
                rec(newavail, need - 1, count + 1)
        rec((1 << n) - 1, k, 0)
        return found

    # ---------- evaluation ----------
    def evaluate(edge_indices):
        # returns (alpha, is_linear)
        edges = [masks[i] for i in edge_indices]
        # linear check
        seen = 0
        linear = True
        for e in edges:
            if (seen & e) & ((seen & e) - 1):
                linear = False
                break
            seen |= e
        if not linear:
            return None, False
        # independence: prove <=5
        if alpha_leq(edges, 5):
            return 5, True
        # else prove >=6
        if has_indep_set(edges, 6):
            return 6, True
        # fallback: compute exact by trying
        # (should not happen)
        return 6, True

    # ---------- greedy construction ----------
    def greedy(seed):
        import random
        rng = random.Random(seed)
        order = list(range(nT))
        rng.shuffle(order)
        chosen = []
        chosen_set = set()
        for i in order:
            if i in chosen_set:
                continue
            # check linear with all chosen
            mi = masks[i]
            ok = True
            for j in chosen:
                if (mi & masks[j]) & ((mi & masks[j]) - 1):
                    ok = False
                    break
            if ok:
                chosen.append(i)
                chosen_set.add(i)
        return chosen

    best_alpha = None
    best_edges = None

    seeds = list(range(4000))
    # shuffle seeds for variety
    import random
    random.Random(123).shuffle(seeds)

    for s in seeds:
        if time.time() > deadline:
            break
        chosen = greedy(s)
        if len(chosen) < 14:
            continue
        alpha, linear = evaluate(chosen)
        if not linear:
            continue
        if best_alpha is None or alpha < best_alpha:
            best_alpha = alpha
            best_edges = chosen[:]
            if best_alpha <= 5:
                # found alpha 5; try to improve (lower is better) - cannot go below 5 easily
                # but keep searching briefly for robustness
                continue

    # local search: try to reduce alpha by swapping edges
    if best_edges is not None and best_alpha is not None and best_alpha > 5:
        # attempt hill-climb to push alpha down
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur = best_edges[:]
            for a in range(len(cur)):
                if time.time() > deadline:
                    break
                for b in range(len(cur)):
                    if b == a:
                        continue
                    for cand in range(nT):
                        if cand in cur:
                            continue
                        # swap cur[b] with cand
                        trial = cur[:]
                        trial[b] = cand
                        alpha, linear = evaluate(trial)
                        if linear and alpha is not None and alpha < best_alpha:
                            best_alpha = alpha
                            best_edges = trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    if best_edges is None:
        # fallback: a valid linear system (5 disjoint triples)
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return [list(all_triples[i]) for i in best_edges]
