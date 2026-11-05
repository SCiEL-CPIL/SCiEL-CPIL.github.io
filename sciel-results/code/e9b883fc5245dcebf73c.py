import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def is_sum_free(S):
        s = set(S)
        L = list(S)
        for i in range(len(L)):
            a = L[i]
            for j in range(len(L)):
                b = L[j]
                if (a + b) in s:
                    return False
        return True

    def build_conflict_structure(V):
        """
        Build conflict graph and hyperedges for sum-free constraint.
        For each element x, conflicts are:
          - a where a+x in V (x is the 'c' in a+b=c)
          - a where x-a in V (x is the 'a' in a+b=c, b=x-a)
          - 2x in V (a=b=x)
        We model this as: x conflicts with y if adding both to S would violate.
        Actually, the constraint is: no a,b,c in S with a+b=c.
        So x and y conflict if there exists z in V such that x+y=z, or x+z=y, or y+z=x.
        But for greedy, we check: can we add x to S?
        x can be added if for all a in S: (a+x) not in S and (x-a) not in S and 2x not in S.
        
        For the conflict graph (pairwise): x and y are incompatible if:
          - x+y in V (then if both in S, and x+y also in S, violation; but x+y might not be in S)
          - Actually pairwise incompatibility is not well-defined for sum-free.
          - The constraint is 3-way: a+b=c. So we need hyperedges.
        
        For greedy, we just need: given current S, can x be added?
        can_add(S, x): for all a in S, (a+x) not in S and (x-a) not in S, and 2x not in S.
        
        Precompute for each x, the set of "forbidden pairs" but that's complex.
        Instead, precompute for each x:
          - sum_partners[x] = {a in V : a+x in V}  (x is the second addend)
          - diff_partners[x] = {a in V : x-a in V}  (x is the first addend)
          - double_in_V = (2*x in V)
        """
        V_set = set(V)
        sum_partners = defaultdict(set)  # x -> {a : a+x in V}
        diff_partners = defaultdict(set)  # x -> {a : x-a in V}
        
        for x in V:
            for a in V:
                if a + x in V_set:
                    sum_partners[x].add(a)
                if x - a in V_set:
                    diff_partners[x].add(a)
        
        return V_set, sum_partners, diff_partners

    def can_add_fast(S, x, sum_partners, diff_partners, V_set):
        """Check if x can be added to sum-free set S."""
        # For each a in S, check if a+x in S or x-a in S
        for a in S:
            if (a + x) in S or (x - a) in S:
                return False
        if (2 * x) in S:
            return False
        return True

    def conflict_count_fast(S, x, sum_partners, diff_partners, V_set):
        """Count conflicts of x with current S."""
        cnt = 0
        for a in S:
            if (a + x) in S or (x - a) in S:
                cnt += 1
        if (2 * x) in S:
            cnt += 1
        return cnt

    def greedy_from_order(order, V_set, sum_partners, diff_partners):
        S = set()
        for x in order:
            if x not in S and can_add_fast(S, x, sum_partners, diff_partners, V_set):
                S.add(x)
        return S

    def greedy_min_conflict(V_set, sum_partners, diff_partners, max_passes=5, rng=None):
        S = set()
        candidates = list(V_set)
        if rng is None:
            rng = random.Random()
        for _ in range(max_passes):
            if not candidates:
                break
            candidates.sort(key=lambda x: (conflict_count_fast(S, x, sum_partners, diff_partners, V_set), abs(x), x))
            added_any = False
            new_candidates = []
            for x in candidates:
                if x not in S and can_add_fast(S, x, sum_partners, diff_partners, V_set):
                    S.add(x)
                    added_any = True
                elif x not in S:
                    new_candidates.append(x)
            candidates = new_candidates
            if not added_any:
                break
        return S

    def try_add_remaining(S, V_set, sum_partners, diff_partners, deadline):
        added = True
        while added and time.time() < deadline:
            added = False
            remaining = [x for x in V_set if x not in S]
            remaining.sort(key=lambda x: (conflict_count_fast(S, x, sum_partners, diff_partners, V_set), abs(x), x))
            for x in remaining:
                if time.time() >= deadline:
                    break
                if x not in S and can_add_fast(S, x, sum_partners, diff_partners, V_set):
                    S.add(x)
                    added = True
        return S

    def local_search_1opt(S, V_set, sum_partners, diff_partners, budget, rng):
        """Try removing one element and re-greedy filling."""
        best = set(S)
        best_size = len(best)
        t_start = time.time()
        S_list = list(best)
        rng.shuffle(S_list)
        for a in S_list:
            if time.time() - t_start > budget:
                break
            trial = set(best)
            trial.discard(a)
            # Re-greedy fill
            remaining = [x for x in V_set if x not in trial]
            remaining.sort(key=lambda x: (conflict_count_fast(trial, x, sum_partners, diff_partners, V_set), abs(x), x))
            for x in remaining:
                if x not in trial and can_add_fast(trial, x, sum_partners, diff_partners, V_set):
                    trial.add(x)
            if len(trial) > best_size:
                best = trial
                best_size = len(best)
        return best

    def local_search_2opt(S, V_set, sum_partners, diff_partners, budget, rng):
        """Try removing two elements and re-greedy filling."""
        best = set(S)
        best_size = len(best)
        t_start = time.time()
        S_list = list(best)
        rng.shuffle(S_list)
        n = len(S_list)
        for i in range(n):
            if time.time() - t_start > budget:
                break
            a = S_list[i]
            for j in range(i + 1, n):
                if time.time() - t_start > budget:
                    break
                b = S_list[j]
                trial = set(best)
                trial.discard(a)
                trial.discard(b)
                remaining = [x for x in V_set if x not in trial]
                remaining.sort(key=lambda x: (conflict_count_fast(trial, x, sum_partners, diff_partners, V_set), abs(x), x))
                for x in remaining:
                    if x not in trial and can_add_fast(trial, x, sum_partners, diff_partners, V_set):
                        trial.add(x)
                if len(trial) > best_size:
                    best = trial
                    best_size = len(best)
                    break
            if time.time() - t_start > budget:
                break
        return best

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V = list(elements)
        V_set = set(V)
        n = len(V)
        best = set()
        best_size = 0
        rng = random.Random(seed_off)

        # Precompute conflict structure
        V_set, sum_partners, diff_partners = build_conflict_structure(V)

        # Precompute potential degree for ordering
        potential_degree = {}
        for x in V:
            deg = 0
            for a in V:
                if a == x:
                    continue
                if (a + x) in V_set or (x - a) in V_set:
                    deg += 1
            if (2 * x) in V_set:
                deg += 1
            potential_degree[x] = deg

        # Candidate structural orders
        orders = []

        # A. By value descending (largest first)
        orders.append(sorted(V, key=lambda x: x, reverse=True))
        # B. By value ascending
        orders.append(sorted(V, key=lambda x: x))
        # C. By |x| descending
        orders.append(sorted(V, key=lambda x: abs(x), reverse=True))
        # D. Odds first, then evens by |x|
        odds = [x for x in V if x % 2 != 0]
        evens = [x for x in V if x % 2 == 0]
        orders.append(sorted(odds, key=lambda x: abs(x), reverse=True) + sorted(evens, key=lambda x: abs(x), reverse=True))
        # E. Positives by value desc, then negatives by value asc
        pos = sorted([x for x in V if x > 0], reverse=True)
        neg = sorted([x for x in V if x < 0])
        orders.append(pos + neg)
        # F. Negatives first (most negative), then positives
        neg2 = sorted([x for x in V if x < 0], reverse=True)
        pos2 = sorted([x for x in V if x > 0])
        orders.append(neg2 + pos2)
        # G. Upper value interval only: top 60% by value
        sv = sorted(V)
        k = int(n * 0.6)
        orders.append(sv[n-k:])
        # H. Top 60% by |x|
        sm = sorted(V, key=lambda x: abs(x), reverse=True)
        orders.append(sm[:k])
        # I. By potential degree ascending (least constrained first)
        orders.append(sorted(V, key=lambda x: (potential_degree[x], abs(x), x)))
        # J. By potential degree descending
        orders.append(sorted(V, key=lambda x: (-potential_degree[x], abs(x), x)))
        # K. Random orders
        for _ in range(10):
            rand_order = V[:]
            rng.shuffle(rand_order)
            orders.append(rand_order)

        # Run greedy on each order
        for order in orders:
            if time.time() - t_start > budget:
                break
            S = greedy_from_order(order, V_set, sum_partners, diff_partners)
            if len(S) > best_size:
                best_size = len(S)
                best = set(S)

        # Min-conflict greedy
        if time.time() - t_start < budget:
            S = greedy_min_conflict(V_set, sum_partners, diff_partners, max_passes=5, rng=rng)
            if len(S) > best_size:
                best_size = len(S)
                best = set(S)

        # Try adding remaining
        best = try_add_remaining(best, V_set, sum_partners, diff_partners, t_start + budget)
        best_size = len(best)

        # 1-opt local search
        if time.time() - t_start < budget * 0.7:
            ls_budget = min(budget * 0.2, 8)
            best = local_search_1opt(best, V_set, sum_partners, diff_partners, ls_budget, rng)
            best_size = len(best)
            best = try_add_remaining(best, V_set, sum_partners, diff_partners, t_start + budget)

        # 2-opt local search
        if time.time() - t_start < budget * 0.85:
            ls_budget = min(budget * 0.15, 6)
            best = local_search_2opt(best, V_set, sum_partners, diff_partners, ls_budget, rng)
            best_size = len(best)
            best = try_add_remaining(best, V_set, sum_partners, diff_partners, t_start + budget)

        return list(best)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    # time budgets, leave margin
    budgets = [12, 30, 70]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            # fallback: odds
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        # verify
        if not is_sum_free(res):
            res = [x for x in elems if x % 2 != 0]
        results.append(res)

    return results
