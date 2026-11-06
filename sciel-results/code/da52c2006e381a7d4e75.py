import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 16


def construct():
    start_time = time.time()
    DEADLINE = 128.0  # safety margin before 150s kill

    def now():
        return time.time()

    # Precompute pair index for fast linearity checks
    def pair_idx(a, b):
        if a > b:
            a, b = b, a
        return a * 16 + b

    def make_pairset(triples):
        ps = set()
        for t in triples:
            a, b, c = t
            ps.add(pair_idx(a, b))
            ps.add(pair_idx(a, c))
            ps.add(pair_idx(b, c))
        return ps

    def is_linear(triples):
        ps = set()
        for t in triples:
            a, b, c = t
            for p in ((a, b), (a, c), (b, c)):
                idx = pair_idx(p[0], p[1])
                if idx in ps:
                    return False
                ps.add(idx)
        return True

    # Fast independence-number upper bound via maximal clique on the
    # hypergraph-complement (independent set = clique in complement).
    def indep_upper_bound(triples):
        # mask of points
        mask = (1 << N) - 1
        # For each point, list of triples containing it
        point_triples = [[] for _ in range(N)]
        for t in triples:
            for p in t:
                point_triples[p].append(t)
        # Precompute for each point the set of points that appear together with it
        # in some triple (these cannot both be in an independent set)
        conflict = [0] * N
        for t in triples:
            a, b, c = t
            conflict[a] |= (1 << b) | (1 << c)
            conflict[b] |= (1 << a) | (1 << c)
            conflict[c] |= (1 << a) | (1 << b)

        # Greedy branch-and-bound for max clique in complement graph.
        # A set S is independent iff no two of its points are in conflict.
        best = [0]
        order = list(range(N))

        def branch(cand_mask, size):
            # cand_mask: candidate points that can extend current clique
            # size: current clique size
            if size + cand_mask.bit_count() <= best[0]:
                return
            if cand_mask == 0:
                if size > best[0]:
                    best[0] = size
                return
            # pick a pivot point
            pivot = cand_mask & (-cand_mask)
            pb = pivot.bit_length() - 1
            # candidates compatible with pivot
            compat = cand_mask & ~conflict[pb]
            # branching: include pivot or exclude it
            # include pivot
            branch(compat, size + 1)
            # exclude pivot
            branch(cand_mask & ~pivot, size)

        branch(mask, 0)
        return best[0]

    def independence_number(triples):
        # Exact via ILP; use upper bound to prune
        ub = indep_upper_bound(triples)
        n = N
        A_ub_rows = []
        for t in triples:
            row = np.zeros(n)
            for p in t:
                row[p] = 1.0
            A_ub_rows.append(row)
        if A_ub_rows:
            A_ub = np.array(A_ub_rows)
            b_ub = np.array([2.0] * len(triples))
        else:
            A_ub = np.zeros((1, n))
            b_ub = np.array([1.0])
        lb = np.zeros(n)
        ubv = np.ones(n)
        integrality = np.ones(n)
        c = np.ones(n)
        try:
            res = milp(c=-c,
                       constraints=LinearConstraint(A_ub, -np.inf, b_ub),
                       integrality=integrality,
                       bounds=Bounds(lb, ubv))
            if res.status == 0:
                val = int(round(-res.fun))
                # ILP is exact for this small problem; clamp by ub for safety
                return min(val, ub)
        except Exception:
            pass
        # Fallback: use the branch-and-bound upper bound as an estimate
        # (it is actually exact for max clique, hence exact for independence)
        return ub

    best_triples = None
    best_indep = N

    all_triples = list(combinations(range(N), 3))

    def greedy_pack(rng, order=None):
        used_pairs = set()
        triples = []
        if order is None:
            order = list(range(len(all_triples)))
            rng.shuffle(order)
        for oi in order:
            t = all_triples[oi]
            a, b, c = t
            idxs = (pair_idx(a, b), pair_idx(a, c), pair_idx(b, c))
            if any(ix in used_pairs for ix in idxs):
                continue
            triples.append(list(t))
            used_pairs.update(idxs)
        return triples

    def local_search(triples, rng, budget):
        # Returns improved triples (smaller independence) if found.
        triples = [list(t) for t in triples]
        cur_indep = independence_number(triples)
        best_local = cur_indep
        best_local_triples = [list(t) for t in triples]
        t0 = now()
        # Candidate pool: all triples not currently used
        used_pairs = make_pairset(triples)
        used_set = set(tuple(t) for t in triples)
        pool = [t for t in all_triples if tuple(t) not in used_set]

        while now() - t0 < budget and best_local > 3:
            # Try a random move: add a triple, remove a triple, or swap
            move = rng.random()
            improved = False
            if move < 0.4:
                # add
                if not pool:
                    continue
                t = rng.choice(pool)
                a, b, c = t
                idxs = (pair_idx(a, b), pair_idx(a, c), pair_idx(b, c))
                if any(ix in used_pairs for ix in idxs):
                    continue
                new_triples = triples + [list(t)]
                ni = independence_number(new_triples)
                if ni < best_local:
                    best_local = ni
                    best_local_triples = new_triples
                    triples = new_triples
                    used_pairs.update(idxs)
                    used_set.add(tuple(t))
                    pool.remove(t)
                    improved = True
            elif move < 0.6:
                # remove
                if not triples:
                    continue
                i = rng.randrange(len(triples))
                t = triples[i]
                a, b, c = t
                idxs = (pair_idx(a, b), pair_idx(a, c), pair_idx(b, c))
                new_triples = triples[:i] + triples[i + 1:]
                ni = independence_number(new_triples)
                if ni < best_local:
                    best_local = ni
                    best_local_triples = new_triples
                    triples = new_triples
                    for ix in idxs:
                        used_pairs.discard(ix)
                    used_set.discard(tuple(t))
                    pool.append(t)
                    improved = True
            else:
                # swap: remove one, add one
                if not triples or not pool:
                    continue
                i = rng.randrange(len(triples))
                t_rm = triples[i]
                a, b, c = t_rm
                idxs_rm = (pair_idx(a, b), pair_idx(a, c), pair_idx(b, c))
                # temporarily remove
                new_triples = triples[:i] + triples[i + 1:]
                tmp_pairs = set(used_pairs)
                for ix in idxs_rm:
                    tmp_pairs.discard(ix)
                # pick an add candidate compatible with tmp_pairs
                cands = []
                for t in pool:
                    x, y, z = t
                    if (pair_idx(x, y) not in tmp_pairs and
                            pair_idx(x, z) not in tmp_pairs and
                            pair_idx(y, z) not in tmp_pairs):
                        cands.append(t)
                if not cands:
                    continue
                t_add = rng.choice(cands)
                x, y, z = t_add
                idxs_add = (pair_idx(x, y), pair_idx(x, z), pair_idx(y, z))
                new_triples.append(list(t_add))
                ni = independence_number(new_triples)
                if ni < best_local:
                    best_local = ni
                    best_local_triples = new_triples
                    triples = new_triples
                    used_pairs = tmp_pairs
                    used_pairs.update(idxs_add)
                    used_set.discard(tuple(t_rm))
                    used_set.add(tuple(t_add))
                    pool.remove(t_add)
                    pool.append(t_rm)
                    improved = True
            if not improved:
                # occasional random restart of the local search neighborhood
                if rng.random() < 0.1:
                    break
        return best_local_triples, best_local

    # Phase 1: greedy packings from random orders
    attempts = 0
    max_greedy = 600
    while attempts < max_greedy and now() - start_time < DEADLINE - 20:
        attempts += 1
        rng = random.Random(attempts * 7919 + 13)
        order = list(range(len(all_triples)))
        rng.shuffle(order)
        triples = greedy_pack(rng, order)
        if not triples:
            continue
        indep = independence_number(triples)
        if indep < best_indep:
            best_indep = indep
            best_triples = triples
        if best_indep <= 4:
            break

    # Phase 2: local search on the best found and on fresh packings
    if best_triples is None:
        best_triples = greedy_pack(random.Random(1))
        best_indep = independence_number(best_triples)

    ls_budget = 4.0
    ls_iter = 0
    max_ls = 40
    while ls_iter < max_ls and now() - start_time < DEADLINE - 5:
        ls_iter += 1
        rng = random.Random(100000 + ls_iter * 31)
        if ls_iter == 1:
            start_triples = best_triples
        else:
            order = list(range(len(all_triples)))
            rng.shuffle(order)
            start_triples = greedy_pack(rng, order)
        if not start_triples:
            continue
        refined, indep = local_search(start_triples, rng, ls_budget)
        if indep < best_indep:
            best_indep = indep
            best_triples = refined
        if best_indep <= 3:
            break

    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return best_triples
