"""Planted 3-SAT solver: WalkSAT + DLS + component decomposition via union-find."""
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 290.0

    # Preprocess: 0-based vars, sign as +1/-1
    clause_var_ids = []
    clause_signs = []
    for clause in clauses:
        vlist = [abs(l) - 1 for l in clause]
        slist = [1 if l > 0 else -1 for l in clause]
        clause_var_ids.append(vlist)
        clause_signs.append(slist)

    # Build variable -> list of (clause_idx, sign)
    var_occurrences = [[] for _ in range(n)]
    for ci in range(m):
        for i in range(len(clause_var_ids[ci])):
            v = clause_var_ids[ci][i]
            s = clause_signs[ci][i]
            var_occurrences[v].append((ci, s))

    # --- Component detection via union-find on clause incidence graph ---
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for ci in range(m):
        vlist = clause_var_ids[ci]
        if len(vlist) > 1:
            for i in range(1, len(vlist)):
                union(vlist[0], vlist[i])

    # Group variables by component
    comp_map = {}
    for v in range(n):
        root = find(v)
        if root not in comp_map:
            comp_map[root] = []
        comp_map[root].append(v)

    components = list(comp_map.values())
    comp_sizes = [len(c) for c in components]

    # If only one component, skip decomposition overhead
    if len(components) == 1:
        components = [list(range(n))]

    # --- Core WalkSAT solver for a sub-problem ---
    # Maps global var indices to local 0..k-1
    def solve_component(comp_vars, time_budget, rng):
        """Solve a component. Returns dict var->bool or None if failed."""
        k = len(comp_vars)
        var_to_local = {v: i for i, v in enumerate(comp_vars)}
        local_to_global = comp_vars

        # Find clauses that involve only vars in this component
        comp_clause_indices = []
        for ci in range(m):
            vlist = clause_var_ids[ci]
            if all(v in var_to_local for v in vlist):
                comp_clause_indices.append(ci)

        cm = len(comp_clause_indices)
        if cm == 0:
            # No clauses: all vars can be arbitrary
            return {v: False for v in comp_vars}

        # Build local clause structure
        local_clause_vars = []
        local_clause_signs = []
        local_var_occ = [[] for _ in range(k)]
        for ci in comp_clause_indices:
            vlist = [var_to_local[v] for v in clause_var_ids[ci]]
            slist = clause_signs[ci]
            local_clause_vars.append(vlist)
            local_clause_signs.append(slist)
            for i in range(len(vlist)):
                local_var_occ[vlist[i]].append((len(local_clause_vars) - 1, slist[i]))

        # If very small, brute force
        if k <= 12:
            best_assign = None
            best_unsat = cm + 1
            for trial in range(min(2000, 2 ** k)):
                if time.time() > time_budget:
                    break
                assign = np.array([rng.random() < 0.5 for _ in range(k)], dtype=bool)
                unsat = 0
                for ci in range(cm):
                    vlist = local_clause_vars[ci]
                    slist = local_clause_signs[ci]
                    sat = False
                    for i in range(len(vlist)):
                        if (assign[vlist[i]] and slist[i] == 1) or (not assign[vlist[i]] and slist[i] == -1):
                            sat = True
                            break
                    if not sat:
                        unsat += 1
                if unsat == 0:
                    return {local_to_global[v]: bool(assign[v]) for v in range(k)}
                if unsat < best_unsat:
                    best_unsat = unsat
                    best_assign = assign.copy()
            if best_assign is not None:
                return {local_to_global[v]: bool(best_assign[v]) for v in range(k)}
            return None

        # WalkSAT on the component
        assignment = np.array([rng.random() < 0.5 for _ in range(k)], dtype=bool)

        def init_sat_count(assign):
            cnt = np.zeros(cm, dtype=np.int32)
            for ci in range(cm):
                vlist = local_clause_vars[ci]
                slist = local_clause_signs[ci]
                c = 0
                for i in range(len(vlist)):
                    if (assign[vlist[i]] and slist[i] == 1) or (not assign[vlist[i]] and slist[i] == -1):
                        c += 1
                cnt[ci] = c
            return cnt

        def compute_delta(v, assign, sat_cnt):
            delta = 0
            vval = assign[v]
            for (aci, s) in local_var_occ[v]:
                lit_true = (vval and s == 1) or ((not vval) and s == -1)
                if lit_true:
                    if sat_cnt[aci] == 1:
                        delta += 1
                else:
                    if sat_cnt[aci] == 0:
                        delta -= 1
            return delta

        def do_flip(v, assign, sat_cnt):
            old = assign[v]
            assign[v] = not old
            for (aci, s) in local_var_occ[v]:
                was = (old and s == 1) or ((not old) and s == -1)
                is_now = (assign[v] and s == 1) or ((not assign[v]) and s == -1)
                if was and not is_now:
                    sat_cnt[aci] -= 1
                elif (not was) and is_now:
                    sat_cnt[aci] += 1

        clause_sat = init_sat_count(assignment)
        unsat = int(np.sum(clause_sat == 0))
        if unsat == 0:
            return {local_to_global[v]: bool(assignment[v]) for v in range(k)}

        var_activity = np.zeros(k, dtype=np.float64)
        decay = 0.995
        best_assign = assignment.copy()
        best_unsat = unsat

        max_restarts = 200
        base_steps = 2000
        p_good = 0.6

        for restart in range(max_restarts):
            if time.time() > time_budget:
                break
            if restart > 0:
                assignment = np.array([rng.random() < 0.5 for _ in range(k)], dtype=bool)
                clause_sat = init_sat_count(assignment)
                unsat = int(np.sum(clause_sat == 0))
                if unsat == 0:
                    return {local_to_global[v]: bool(assignment[v]) for v in range(k)}
                if unsat < best_unsat:
                    best_unsat = unsat
                    best_assign = assignment.copy()

            steps = base_steps if unsat > 10 else base_steps * 2
            stagnant = 0
            for step in range(steps):
                if time.time() > time_budget:
                    break
                if unsat == 0:
                    return {local_to_global[v]: bool(assignment[v]) for v in range(k)}

                unsat_idx = np.where(clause_sat == 0)[0]
                if len(unsat_idx) == 0:
                    return {local_to_global[v]: bool(assignment[v]) for v in range(k)}
                ci = int(unsat_idx[rng.randint(0, len(unsat_idx) - 1)])
                vlist = local_clause_vars[ci]
                slist = local_clause_signs[ci]
                clen = len(vlist)

                if rng.random() < p_good:
                    candidates = []
                    for i in range(clen):
                        v = vlist[i]
                        s = slist[i]
                        is_sat = (assignment[v] and s == 1) or (not assignment[v] and s == -1)
                        if not is_sat:
                            candidates.append(v)
                    if candidates:
                        if len(candidates) > 1:
                            weights = [var_activity[v] + 1.0 for v in candidates]
                            total_w = sum(weights)
                            r = rng.random() * total_w
                            cum = 0
                            v = candidates[-1]
                            for j, cv in enumerate(candidates):
                                cum += weights[j]
                                if r <= cum:
                                    v = cv
                                    break
                        else:
                            v = candidates[0]
                    else:
                        v = vlist[rng.randint(0, clen - 1)]
                else:
                    if clen > 1:
                        weights = [var_activity[vlist[i]] + 1.0 for i in range(clen)]
                        total_w = sum(weights)
                        r = rng.random() * total_w
                        cum = 0
                        v = vlist[-1]
                        for i in range(clen):
                            cum += weights[i]
                            if r <= cum:
                                v = vlist[i]
                                break
                    else:
                        v = vlist[0]

                delta = compute_delta(v, assignment, clause_sat)
                if delta <= 0 or (delta > 0 and rng.random() < 0.25):
                    do_flip(v, assignment, clause_sat)
                    new_unsat = unsat + delta
                    if new_unsat < unsat:
                        stagnant = 0
                    else:
                        stagnant += 1
                    unsat = new_unsat
                    for i in range(clen):
                        var_activity[vlist[i]] *= decay
                        var_activity[vlist[i]] += 1.0
                    if unsat < best_unsat:
                        best_unsat = unsat
                        best_assign = assignment.copy()
                else:
                    stagnant += 1
                if stagnant > 150 and unsat > 5:
                    break
            var_activity *= 0.95

        if best_unsat == 0:
            return {local_to_global[v]: bool(best_assign[v]) for v in range(k)}
        return {local_to_global[v]: bool(best_assign[v]) for v in range(k)}

    # --- Solve each component ---
    total_comp_time = deadline - time.time() - 10.0  # reserve 10s for final polish
    # Allocate time proportional to component size (with minimum)
    total_size = sum(comp_sizes)
    comp_times = []
    for sz in comp_sizes:
        t = max(0.5, total_comp_time * sz / total_size)
        comp_times.append(t)

    rng = random.Random(42)
    full_assignment = [False] * n

    for idx, comp in enumerate(components):
        if time.time() > deadline - 5.0:
            break
        budget = min(comp_times[idx], deadline - time.time() - 3.0)
        if budget < 0.1:
            continue
        result = solve_component(comp, time.time() + budget, rng)
        if result is not None:
            for v, val in result.items():
                full_assignment[v] = val

    # --- Final full-instance WalkSAT polish ---
    if time.time() < deadline - 2.0:
        # Check current state
        clause_sat = np.zeros(m, dtype=np.int32)
        for ci in range(m):
            vlist = clause_var_ids[ci]
            slist = clause_signs[ci]
            c = 0
            for i in range(len(vlist)):
                v = vlist[i]
                s = slist[i]
                if (full_assignment[v] and s == 1) or (not full_assignment[v] and s == -1):
                    c += 1
            clause_sat[ci] = c
        unsat = int(np.sum(clause_sat == 0))

        if unsat > 0:
            # Run global WalkSAT from current state
            assignment = np.array(full_assignment, dtype=bool)

            def g_init_sat_count(assign):
                cnt = np.zeros(m, dtype=np.int32)
                for ci in range(m):
                    vlist = clause_var_ids[ci]
                    slist = clause_signs[ci]
                    c = 0
                    for i in range(len(vlist)):
                        if (assign[vlist[i]] and slist[i] == 1) or (not assign[vlist[i]] and slist[i] == -1):
                            c += 1
                    cnt[ci] = c
                return cnt

            def g_compute_delta(v, assign, sat_cnt):
                delta = 0
                vval = assign[v]
                for (aci, s) in var_occurrences[v]:
                    lit_true = (vval and s == 1) or ((not vval) and s == -1)
                    if lit_true:
                        if sat_cnt[aci] == 1:
                            delta += 1
                    else:
                        if sat_cnt[aci] == 0:
                            delta -= 1
                return delta

            def g_do_flip(v, assign, sat_cnt):
                old = assign[v]
                assign[v] = not old
                for (aci, s) in var_occurrences[v]:
                    was = (old and s == 1) or ((not old) and s == -1)
                    is_now = (assign[v] and s == 1) or ((not assign[v]) and s == -1)
                    if was and not is_now:
                        sat_cnt[aci] -= 1
                    elif (not was) and is_now:
                        sat_cnt[aci] += 1

            clause_sat = g_init_sat_count(assignment)
            unsat = int(np.sum(clause_sat == 0))

            var_activity = np.zeros(n, dtype=np.float64)
            decay = 0.995
            best_assign = assignment.copy()
            best_unsat = unsat
            p_good = 0.6
            base_steps = 3000
            max_restarts = 5000

            for restart in range(max_restarts):
                if time.time() > deadline - 1.0:
                    break
                if restart > 0:
                    assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
                    clause_sat = g_init_sat_count(assignment)
                    unsat = int(np.sum(clause_sat == 0))
                    if unsat == 0:
                        return assignment.tolist()
                    if unsat < best_unsat:
                        best_unsat = unsat
                        best_assign = assignment.copy()

                steps = base_steps if unsat > 20 else base_steps * 2
                if unsat <= 5:
                    steps = base_steps * 4
                stagnant = 0
                for step in range(steps):
                    if time.time() > deadline - 1.0:
                        break
                    if unsat == 0:
                        return assignment.tolist()

                    unsat_idx = np.where(clause_sat == 0)[0]
                    if len(unsat_idx) == 0:
                        return assignment.tolist()
                    ci = int(unsat_idx[rng.randint(0, len(unsat_idx) - 1)])
                    vlist = clause_var_ids[ci]
                    slist = clause_signs[ci]
                    clen = len(vlist)

                    if rng.random() < p_good:
                        candidates = []
                        for i in range(clen):
                            v = vlist[i]
                            s = slist[i]
                            is_sat = (assignment[v] and s == 1) or (not assignment[v] and s == -1)
                            if not is_sat:
                                candidates.append(v)
                        if candidates:
                            if len(candidates) > 1:
                                weights = [var_activity[v] + 1.0 for v in candidates]
                                total_w = sum(weights)
                                r = rng.random() * total_w
                                cum = 0
                                v = candidates[-1]
                                for j, cv in enumerate(candidates):
                                    cum += weights[j]
                                    if r <= cum:
                                        v = cv
                                        break
                            else:
                                v = candidates[0]
                        else:
                            v = vlist[rng.randint(0, clen - 1)]
                    else:
                        if clen > 1:
                            weights = [var_activity[vlist[i]] + 1.0 for i in range(clen)]
                            total_w = sum(weights)
                            r = rng.random() * total_w
                            cum = 0
                            v = vlist[-1]
                            for i in range(clen):
                                cum += weights[i]
                                if r <= cum:
                                    v = vlist[i]
                                    break
                        else:
                            v = vlist[0]

                    delta = g_compute_delta(v, assignment, clause_sat)
                    if delta <= 0 or (delta > 0 and rng.random() < 0.25):
                        g_do_flip(v, assignment, clause_sat)
                        new_unsat = unsat + delta
                        if new_unsat < unsat:
                            stagnant = 0
                        else:
                            stagnant += 1
                        unsat = new_unsat
                        for i in range(clen):
                            var_activity[vlist[i]] *= decay
                            var_activity[vlist[i]] += 1.0
                        if unsat < best_unsat:
                            best_unsat = unsat
                            best_assign = assignment.copy()
                    else:
                        stagnant += 1
                    if stagnant > 200 and unsat > 10:
                        break
                var_activity *= 0.95

            if best_unsat == 0:
                return best_assign.tolist()
            return best_assign.tolist()

    return full_assignment
