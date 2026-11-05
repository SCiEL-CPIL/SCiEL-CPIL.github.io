import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 138  # safety margin before 150s kill

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

    def build_hyperedges(V_set):
        """Build all triples (a,b,c) with a+b=c, a,b,c in V_set."""
        V_list = list(V_set)
        V_s = set(V_list)
        triples = []
        for a in V_list:
            for b in V_list:
                c = a + b
                if c in V_s:
                    triples.append((a, b, c))
        return triples

    def greedy_add(S, candidates, V_set):
        """Add elements from candidates to S in the given order if they don't violate sum-free."""
        s = set(S)
        for x in candidates:
            if x in s:
                continue
            ok = True
            for a in s:
                if (a + x) in s or (x - a) in s:
                    ok = False
                    break
            if ok and (2 * x) in s:
                ok = False
            if ok:
                s.add(x)
        return s

    def hypergraph_removal(V_set, seed_off, budget):
        """
        Start with all elements. Iteratively remove the element whose removal
        breaks the most active hyperedges. This directly optimizes for the
        number of elements kept in a sum-free set.
        """
        t_start = time.time()
        rng = random.Random(seed_off)
        V_list = list(V_set)
        V_s = set(V_list)
        n = len(V_list)

        # Build all triples
        triples = []
        for a in V_list:
            for b in V_list:
                c = a + b
                if c in V_s:
                    triples.append((a, b, c))

        if not triples:
            return list(V_list)

        # Map each element to the indices of triples it participates in
        elem_triples = {v: [] for v in V_list}
        for idx, (a, b, c) in enumerate(triples):
            elem_triples[a].append(idx)
            if b != a:
                elem_triples[b].append(idx)
            elem_triples[c].append(idx)

        active = set(V_list)
        active_triples = set(range(len(triples)))

        # Iteratively remove the element whose removal breaks the most active triples
        removed = 0
        max_removals = int(n * 0.7)  # don't remove too many
        while removed < max_removals and time.time() - t_start < budget:
            # For each active element, count how many active triples it's in
            best_elem = None
            best_count = -1
            for v in active:
                cnt = sum(1 for idx in elem_triples[v] if idx in active_triples)
                if cnt > best_count:
                    best_count = cnt
                    best_elem = v
            if best_count <= 0:
                break
            # Remove best_elem
            active.discard(best_elem)
            removed += 1
            # Deactivate all triples involving best_elem
            for idx in elem_triples[best_elem]:
                a, b, c = triples[idx]
                if a in active and b in active and c in active:
                    pass
                else:
                    active_triples.discard(idx)

        return list(active)

    def local_search(S, V_set, budget, seed_off):
        """
        Local search: try 1-swap and 2-swap moves.
        1-swap: remove one element from S, greedily refill, accept if size grows.
        2-swap: remove two elements from S, greedily refill, accept if size grows.
        """
        t_start = time.time()
        rng = random.Random(seed_off)
        S = set(S)
        best_size = len(S)

        def try_refill(trial):
            """Greedily add remaining elements to trial."""
            remaining = [x for x in V_set if x not in trial]
            # Sort by a heuristic: prefer elements with fewer conflicts
            def conflict_score(x):
                cnt = 0
                for a in trial:
                    if (a + x) in trial or (x - a) in trial:
                        cnt += 1
                if (2 * x) in trial:
                    cnt += 1
                return cnt
            remaining.sort(key=lambda x: (conflict_score(x), abs(x), x))
            for x in remaining:
                if x in trial:
                    continue
                ok = True
                for a in trial:
                    if (a + x) in trial or (x - a) in trial:
                        ok = False
                        break
                if ok and (2 * x) in trial:
                    ok = False
                if ok:
                    trial.add(x)
            return trial

        # 1-swap
        improved = True
        while improved and time.time() - t_start < budget * 0.6:
            improved = False
            S_list = list(S)
            rng.shuffle(S_list)
            for a in S_list:
                if time.time() - t_start > budget * 0.6:
                    break
                trial = set(S)
                trial.discard(a)
                trial = try_refill(trial)
                if len(trial) > best_size:
                    S = trial
                    best_size = len(S)
                    improved = True
                    break

        # 2-swap
        improved = True
        while improved and time.time() - t_start < budget:
            improved = False
            S_list = list(S)
            rng.shuffle(S_list)
            for i in range(len(S_list)):
                if time.time() - t_start > budget:
                    break
                a = S_list[i]
                for j in range(i + 1, len(S_list)):
                    if time.time() - t_start > budget:
                        break
                    b = S_list[j]
                    trial = set(S)
                    trial.discard(a)
                    trial.discard(b)
                    trial = try_refill(trial)
                    if len(trial) > best_size:
                        S = trial
                        best_size = len(S)
                        improved = True
                        break
                if improved:
                    break

        return list(S)

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V = list(elements)
        V_set = set(V)
        n = len(V)
        best = set()
        best_size = 0
        rng = random.Random(seed_off)

        # --- Phase 1: Hypergraph removal (multiple random restarts) ---
        # Run several random restarts of the removal heuristic
        for restart in range(5):
            if time.time() - t_start > budget * 0.4:
                break
            sub_budget = min(budget * 0.08, 5)
            S = hypergraph_removal(V_set, seed_off + restart * 100, sub_budget)
            if len(S) > best_size:
                best_size = len(S)
                best = set(S)

        # --- Phase 2: Structural initializations ---
        # Odds
        S_odds = [x for x in V if x % 2 != 0]
        if len(S_odds) > best_size:
            best_size = len(S_odds)
            best = set(S_odds)

        # Positives only
        S_pos = [x for x in V if x > 0]
        if len(S_pos) > best_size:
            best_size = len(S_pos)
            best = set(S_pos)

        # Negatives only
        S_neg = [x for x in V if x < 0]
        if len(S_neg) > best_size:
            best_size = len(S_neg)
            best = set(S_neg)

        # Top half by value
        sv = sorted(V)
        S_top = sv[n // 2:]
        if len(S_top) > best_size:
            best_size = len(S_top)
            best = set(S_top)

        # Bottom half by value
        S_bot = sv[:n // 2]
        if len(S_bot) > best_size:
            best_size = len(S_bot)
            best = set(S_bot)

        # Random initializations
        for _ in range(5):
            if time.time() - t_start > budget * 0.5:
                break
            S_rand = rng.sample(V, n // 2)
            if len(S_rand) > best_size:
                best_size = len(S_rand)
                best = set(S_rand)

        # --- Phase 3: Greedy add from best seed ---
        # Try to add more elements to the current best
        if time.time() - t_start < budget * 0.7:
            remaining = [x for x in V if x not in best]
            remaining.sort(key=lambda x: abs(x), reverse=True)
            best = set(greedy_add(best, remaining, V_set))
            best_size = len(best)

        # --- Phase 4: Local search ---
        if time.time() - t_start < budget:
            ls_budget = min(budget * 0.3, 20)
            S_ls = local_search(list(best), V_set, ls_budget, seed_off + 500)
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best = set(S_ls)

        # --- Phase 5: Final greedy add ---
        if time.time() - t_start < budget:
            remaining = [x for x in V if x not in best]
            remaining.sort(key=lambda x: abs(x), reverse=True)
            best = set(greedy_add(best, remaining, V_set))
            best_size = len(best)

        return list(best)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    budgets = [10, 28, 65]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        if not is_sum_free(res):
            res = [x for x in elems if x % 2 != 0]
        results.append(res)

    return results
