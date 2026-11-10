import time
import random

_GLOBAL_START = time.time()
_GLOBAL_BUDGET = 145.0  # safety margin under the 150 s wall-clock kill

def _popcount(x):
    return x.bit_count()

def lagrangian_round(cliques, cover_bm, all_bm, n, rng, time_budget):
    """Lagrangian relaxation: dualize clique constraints, optimize multipliers,
    round to feasible set, then local-search polish."""
    m = len(cliques)
    # Initial multipliers: 1.0 for all cliques
    lam = [1.0] * m
    best_set = None
    best_size = 10**9
    step = 1.0
    last_best = 10**9
    stagnant = 0
    t0 = time.time()

    while time.time() - t0 < time_budget:
        # --- Primal subproblem: minimize sum_v x_v + sum_i lam_i * max(0, 1 - sum_{v in C_i} x_v)
        # With x in {0,1}, the penalty for clique i is lam_i if no vertex in C_i is selected.
        # Marginal utility of selecting vertex v: 1 - sum_{i: v in C_i, clique i uncovered} lam_i
        # Greedy: repeatedly pick vertex with highest marginal utility (net benefit).
        # But we need feasibility, so we do: pick vertices by descending "net weight",
        # then ensure all cliques are hit.
        
        # Compute for each vertex its "cost" = 1 - sum of lam over cliques it covers
        # (negative cost = beneficial to pick)
        vertex_weight = [1.0] * n
        for ci in range(m):
            if lam[ci] == 0:
                continue
            for v in cliques[ci]:
                vertex_weight[v] -= lam[ci]
        
        # Greedy: sort by weight (most negative first = most beneficial), pick until all covered
        order = sorted(range(n), key=lambda v: vertex_weight[v])
        selected = set()
        sel_bm = 0
        for v in order:
            if sel_bm == all_bm:
                break
            new_bm = sel_bm | cover_bm[v]
            if new_bm != sel_bm:
                selected.add(v)
                sel_bm = new_bm
        
        # Ensure feasibility (should already be, but just in case)
        rem = all_bm & ~sel_bm
        while rem:
            best_v = -1
            best_c = 0
            for v in range(n):
                if v in selected:
                    continue
                cv = _popcount(cover_bm[v] & rem)
                if cv > best_c:
                    best_c = cv
                    best_v = v
            if best_v == -1:
                break
            selected.add(best_v)
            sel_bm |= cover_bm[best_v]
            rem = all_bm & ~sel_bm
        
        sz = len(selected)
        if sz < best_size:
            best_size = sz
            best_set = set(selected)
            stagnant = 0
        else:
            stagnant += 1
        
        # --- Subgradient update: g_i = 1 - sum_{v in C_i} x_v (violation)
        # step size: step = alpha * (UB - LB) / ||g||^2
        violations = [0.0] * m
        for ci in range(m):
            hit = False
            for v in cliques[ci]:
                if v in selected:
                    hit = True
                    break
            if not hit:
                violations[ci] = 1.0
        
        g_norm_sq = sum(g * g for g in violations)
        if g_norm_sq == 0:
            break  # feasible
        
        # Upper bound = best_size, lower bound = sum(lam) (dual objective)
        dual_obj = sum(lam)
        ub = best_size
        gap = max(ub - dual_obj, 0.01)
        alpha = 1.0
        step = alpha * gap / g_norm_sq
        # Cap step to avoid oscillation
        step = min(step, 2.0)
        
        for i in range(m):
            lam[i] = max(0.0, lam[i] + step * violations[i])
        
        # Decay alpha if stagnant
        if stagnant > 20:
            alpha *= 0.95
            step = alpha * gap / g_norm_sq
            step = min(step, 1.0)
            stagnant = 0
        
        if stagnant > 100:
            break
    
    return best_set if best_set is not None else None

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X):
        if not P and not X:
            if len(R) >= 2:
                yield tuple(sorted(R))
            return
        if P or X:
            # Pivot selection: pick vertex with max degree in P|X
            u = max(P | X, key=lambda v: len(P & adj[v]))
            candidates = P - adj[u]
        else:
            candidates = set()
        for v in candidates:
            yield from bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        return list(bron_kerbosch(adj, set(), set(range(n)), set()))

    def make_cover_bitmasks(cliques, n):
        m = len(cliques)
        cover_bm = [0] * n
        for ci, c in enumerate(cliques):
            bit = 1 << ci
            for v in c:
                cover_bm[v] |= bit
        all_bm = (1 << m) - 1
        return cover_bm, all_bm

    def greedy_cover(cliques, cover_bm, all_bm, rng):
        rem = all_bm
        selected = set()
        while rem:
            best_v = -1
            best_c = 0
            ties = []
            for v in range(len(cover_bm)):
                cv = _popcount(cover_bm[v] & rem)
                if cv <= 0:
                    continue
                if cv > best_c:
                    best_c = cv
                    best_v = v
                    ties = [v]
                elif cv == best_c:
                    ties.append(v)
            if best_v == -1:
                break
            if len(ties) > 1:
                best_v = rng.choice(ties)
            selected.add(best_v)
            rem &= ~cover_bm[best_v]
        return selected

    def local_search_fast(selected, cliques, cover_bm, all_bm, n, rng, iters):
        selected = set(selected)
        sel_bm = 0
        for v in selected:
            sel_bm |= cover_bm[v]
        
        for _ in range(iters):
            improved = False
            
            # 1) Try removal
            for v in list(selected):
                rest_bm = sel_bm & ~cover_bm[v]
                if rest_bm == all_bm:
                    selected.discard(v)
                    sel_bm = rest_bm
                    improved = True
                    break
            if improved:
                continue
            
            # 2) 1-exchange: remove v, add w if it covers more
            best_gain = 0
            best_v = -1
            best_w = -1
            for v in selected:
                rest_bm = sel_bm & ~cover_bm[v]
                need = all_bm & ~rest_bm
                if need == 0:
                    continue
                for w in range(n):
                    if w in selected:
                        continue
                    if (cover_bm[w] & need) == need:
                        new_bm = rest_bm | cover_bm[w]
                        gain = _popcount(new_bm ^ sel_bm)
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                            best_w = w
            
            if best_w >= 0:
                selected.discard(best_v)
                selected.add(best_w)
                sel_bm = (sel_bm & ~cover_bm[best_v]) | cover_bm[best_w]
                continue
            
            # 3) 2-exchange: remove v1, v2, add w1, w2 if it reduces size or same size with better coverage
            best_2x_gain = 0
            best_v1 = -1
            best_v2 = -1
            best_w1 = -1
            best_w2 = -1
            sel_list = list(selected)
            for i, v1 in enumerate(sel_list):
                rest1 = sel_bm & ~cover_bm[v1]
                for j, v2 in enumerate(sel_list):
                    if j <= i:
                        continue
                    rest2 = rest1 & ~cover_bm[v2]
                    need2 = all_bm & ~rest2
                    if need2 == 0:
                        # Can remove both
                        best_2x_gain = max(best_2x_gain, 2)
                        best_v1, best_v2 = v1, v2
                        best_w1, best_w2 = -1, -1
                        continue
                    for w1 in range(n):
                        if w1 in selected:
                            continue
                        if not (cover_bm[w1] & need2):
                            continue
                        for w2 in range(n):
                            if w2 in selected or w2 == w1:
                                continue
                            new_bm = rest2 | cover_bm[w1] | cover_bm[w2]
                            if new_bm == all_bm:
                                gain = _popcount(new_bm ^ sel_bm)
                                if gain > best_2x_gain:
                                    best_2x_gain = gain
                                    best_v1, best_v2 = v1, v2
                                    best_w1, best_w2 = w1, w2
            
            if best_2x_gain > 0:
                if best_w1 == -1:
                    # Remove both
                    selected.discard(best_v1)
                    selected.discard(best_v2)
                    sel_bm = sel_bm & ~cover_bm[best_v1] & ~cover_bm[best_v2]
                else:
                    selected.discard(best_v1)
                    selected.discard(best_v2)
                    selected.add(best_w1)
                    selected.add(best_w2)
                    sel_bm = (sel_bm & ~cover_bm[best_v1] & ~cover_bm[best_v2]) | cover_bm[best_w1] | cover_bm[best_w2]
                continue
            
            # 4) 1-swap for diversity (same size, different coverage profile)
            did = False
            for v in list(selected):
                rest_bm = sel_bm & ~cover_bm[v]
                for w in range(n):
                    if w in selected:
                        continue
                    new_bm = rest_bm | cover_bm[w]
                    if new_bm == all_bm:
                        selected.discard(v)
                        selected.add(w)
                        sel_bm = new_bm
                        did = True
                        break
                if did:
                    break
            if not did:
                break
        
        return selected

    def exact_branch_and_bound(cliques, cover_bm, all_bm, budget_fn, limit=5000000):
        n = len(cover_bm)
        m = len(cliques)
        best_size = [10**9]
        best_set = [set()]
        solved = [False]
        nodes = [0]
        stop = [False]

        def branch(selected, sel_bm):
            if stop[0]:
                return
            if sel_bm == all_bm:
                if len(selected) < best_size[0]:
                    best_size[0] = len(selected)
                    best_set[0] = set(selected)
                solved[0] = True
                return
            nodes[0] += 1
            if nodes[0] > limit:
                stop[0] = True
                return
            if not budget_fn():
                stop[0] = True
                return

            # Lower bound: greedy packing of disjoint uncovered cliques
            unhit = all_bm & ~sel_bm
            lb = 0
            rem = unhit
            while rem:
                best_ci = -1
                best_size_ci = 10**9
                for ci in range(m):
                    if (rem >> ci) & 1:
                        sz = len(cliques[ci])
                        if sz < best_size_ci:
                            best_size_ci = sz
                            best_ci = ci
                if best_ci == -1:
                    break
                lb += 1
                rep = cliques[best_ci]
                new_rem = rem
                for v in rep:
                    new_rem &= ~cover_bm[v]
                rem = new_rem
                if not rem:
                    break

            if len(selected) + lb >= best_size[0]:
                return

            # Choose a minimal uncovered clique (by size)
            unhit_ci_list = []
            for ci in range(m):
                if (unhit >> ci) & 1:
                    unhit_ci_list.append(ci)
            if not unhit_ci_list:
                solved[0] = True
                return
            rep_ci = min(unhit_ci_list, key=lambda i: len(cliques[i]))

            # Candidate vertices: those in rep_ci, ordered by descending coverage of unhit cliques
            verts = list(cliques[rep_ci])
            verts.sort(key=lambda v: _popcount(cover_bm[v] & unhit), reverse=True)

            for v in verts:
                if v in selected:
                    continue
                new_bm = sel_bm | cover_bm[v]
                selected.add(v)
                branch(selected, new_bm)
                selected.discard(v)
                if stop[0]:
                    return

        branch(set(), 0)
        if best_size[0] < 10**9:
            return best_size[0], best_set[0], solved[0]
        return None, None, False

    def solve_instance(n, p, seed, time_budget):
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        if not cliques:
            return []
        cover_bm, all_bm = make_cover_bitmasks(cliques, n)

        # Greedy + local search baseline
        best = None
        rng = random.Random(777)
        g = greedy_cover(cliques, cover_bm, all_bm, rng)
        g = local_search_fast(g, cliques, cover_bm, all_bm, n, rng, iters=50)
        best = set(g)

        # Time budget for exact + extra local search
        def budget_fn():
            return (time.time() - _GLOBAL_START) < _GLOBAL_BUDGET

        # Exact branch-and-bound (with node limit and time guard)
        esize, eset, solved = exact_branch_and_bound(cliques, cover_bm, all_bm, budget_fn)
        if eset and (best is None or len(eset) < len(best)):
            best = set(eset)

        # If exact solved, we're done (optimal)
        if solved:
            return sorted(best)

        # Lagrangian relaxation with subgradient optimization
        lr_time = max(1.0, min(8.0, (_GLOBAL_BUDGET - (time.time() - _GLOBAL_START)) * 0.25))
        lr_set = lagrangian_round(cliques, cover_bm, all_bm, n, random.Random(42), lr_time)
        if lr_set is not None:
            lr_set = local_search_fast(lr_set, cliques, cover_bm, all_bm, n, random.Random(43), iters=80)
            if len(lr_set) < len(best):
                best = set(lr_set)

        # Use remaining time for more greedy restarts + local search with 2-exchange
        for it in range(500):
            if (time.time() - _GLOBAL_START) >= _GLOBAL_BUDGET:
                break
            rr = random.Random(100000 + it)
            cand = greedy_cover(cliques, cover_bm, all_bm, rr)
            cand = local_search_fast(cand, cliques, cover_bm, all_bm, n, rr, iters=60)
            if best is None or len(cand) < len(best):
                best = set(cand)
            # also try local search starting from current best with fresh rng
            cand2 = local_search_fast(set(best), cliques, cover_bm, all_bm, n, rr, iters=60)
            if len(cand2) < len(best):
                best = set(cand2)
            # Periodically re-run Lagrangian with different seeds
            if it % 50 == 0 and (time.time() - _GLOBAL_START) < _GLOBAL_BUDGET - 5.0:
                lr_time2 = max(0.5, min(4.0, (_GLOBAL_BUDGET - (time.time() - _GLOBAL_START)) * 0.15))
                lr_set2 = lagrangian_round(cliques, cover_bm, all_bm, n, random.Random(500 + it), lr_time2)
                if lr_set2 is not None:
                    lr_set2 = local_search_fast(lr_set2, cliques, cover_bm, all_bm, n, random.Random(600 + it), iters=60)
                    if len(lr_set2) < len(best):
                        best = set(lr_set2)
        return sorted(best)

    out = []
    instances = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))
    per_instance = (_GLOBAL_BUDGET - (time.time() - _GLOBAL_START)) / len(instances)
    for i, (n, p, seed) in enumerate(instances):
        # hard cap: leave at least a small margin for the rest
        remaining = _GLOBAL_BUDGET - (time.time() - _GLOBAL_START)
        tb = max(1.0, min(per_instance, remaining - 2.0))
        res = solve_instance(n, p, seed, tb)
        out.append(res)
        # if we're running out, just fill the rest cheaply
        if (time.time() - _GLOBAL_START) > _GLOBAL_BUDGET - 1.0:
            for (n2, p2, s2) in instances[i + 1:]:
                out.append(sorted(range(n2)))  # safe (valid) but large
            break
    return out
