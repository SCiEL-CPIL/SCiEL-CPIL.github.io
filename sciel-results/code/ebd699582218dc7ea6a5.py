# EVOLVE-BLOCK-START
import time
import random
import heapq
import math


def _satisfied(cl, ass):
    for v, p in cl:
        if ass[v] == p:
            return True
    return False


def _dpll_solve(clauses_opt, n, m, deadline):
    """DPLL with watched literals, VSIDS, and 1-UIP conflict analysis."""
    assign = [-1] * n
    trail = []
    level = [0] * n
    learnt = []
    
    activity = [1.0] * n
    phase = [False] * n
    
    # All clauses: original + learnt
    all_clauses = list(clauses_opt)
    
    # Watches: for each clause, two watched variable indices
    watches = []
    for cl in all_clauses:
        watches.append([cl[0][0], cl[1][0]])
    
    # var_to_watches[v] = list of (clause_idx, watch_slot)
    var_to_watches = [[] for _ in range(n)]
    for ci in range(len(all_clauses)):
        w = watches[ci]
        var_to_watches[w[0]].append((ci, 0))
        var_to_watches[w[1]].append((ci, 1))

    def propagate(queue):
        """Unit propagation. Returns (ok, conflict_clause_idx)."""
        while queue:
            v, p = queue.pop()
            new_val = 1 if p else 0
            
            # Process all watches for variable v
            i = 0
            while i < len(var_to_watches[v]):
                ci, slot = var_to_watches[v][i]
                cl = all_clauses[ci]
                
                # Check if clause is satisfied
                sat = False
                for vv, pp in cl:
                    if assign[vv] == (1 if pp else 0):
                        sat = True
                        break
                if sat:
                    i += 1
                    continue
                
                # Check if clause is falsified
                falsified = True
                for vv, pp in cl:
                    if assign[vv] != (1 if pp else 0):
                        falsified = False
                        break
                if falsified:
                    return False, ci
                
                # Find a new watch literal (unassigned, not v)
                new_watch = -1
                for vv, pp in cl:
                    if vv != v and assign[vv] == -1:
                        new_watch = vv
                        break
                
                if new_watch == -1:
                    # Unit clause
                    for vv, pp in cl:
                        if assign[vv] == -1:
                            queue.append((vv, pp))
                            break
                    # Remove this watch entry
                    var_to_watches[v].pop(i)
                    # Don't increment i
                else:
                    # Replace v with new_watch in the watch slot
                    old_other = watches[ci][1 - slot]
                    # Remove old watch for old_other
                    for j, (c, s) in enumerate(var_to_watches[old_other]):
                        if c == ci and s == (1 - slot):
                            var_to_watches[old_other].pop(j)
                            break
                    # Add new watch for new_watch
                    var_to_watches[new_watch].append((ci, 1 - slot))
                    watches[ci][1 - slot] = new_watch
                    # Remove v from this watch slot
                    var_to_watches[v].pop(i)
                    # Don't increment i
            # End while for var_to_watches[v]
        
        return True, None

    def analyze_conflict(conf_ci):
        """1-UIP conflict analysis."""
        cl = all_clauses[conf_ci]
        
        seen = set()
        learnt_lits = []
        max_level = 0
        
        for v, p in cl:
            if assign[v] != -1:
                if (v, p) not in seen:
                    seen.add((v, p))
                    learnt_lits.append((v, p))
                    if level[v] > max_level:
                        max_level = level[v]
        
        if len(learnt_lits) <= 1:
            return None, 0
        
        # Simplify
        simplified = []
        pos_vars = set()
        neg_vars = set()
        for v, p in learnt_lits:
            if p:
                if v in neg_vars:
                    continue
                pos_vars.add(v)
            else:
                if v in pos_vars:
                    continue
                neg_vars.add(v)
            simplified.append((v, p))
        
        if len(simplified) <= 1:
            return None, 0
        
        learnt.append(simplified)
        all_clauses.append(simplified)
        watches.append([simplified[0][0], simplified[1][0]])
        new_ci = len(all_clauses) - 1
        var_to_watches[simplified[0][0]].append((new_ci, 0))
        var_to_watches[simplified[1][0]].append((new_ci, 1))
        
        for v, p in simplified:
            activity[v] += 1.0
        
        return simplified, max_level

    # Initial propagation
    q = []
    for ci, cl in enumerate(all_clauses):
        unfixed = []
        for v, p in cl:
            if assign[v] == -1:
                unfixed.append((v, p))
        if len(unfixed) == 1:
            q.append(unfixed[0])
    
    ok, conf = propagate(q)
    if not ok:
        return None

    def check_solved():
        for ci in range(len(all_clauses)):
            cl = all_clauses[ci]
            sat = False
            for v, p in cl:
                if assign[v] == (1 if p else 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    if check_solved():
        return [bool(assign[i]) for i in range(n)]

    # VSIDS heap
    var_heap = [(-activity[v], v) for v in range(n)]
    heapq.heapify(var_heap)

    while time.time() < deadline:
        v = -1
        while var_heap:
            neg_act, cand = var_heap[0]
            if assign[cand] == -1:
                v = cand
                heapq.heappop(var_heap)
                break
            else:
                heapq.heappop(var_heap)
        
        if v == -1:
            if check_solved():
                return [bool(assign[i]) for i in range(n)]
            return None
        
        val = 1 if phase[v] else 0
        assign[v] = val
        trail.append(v)
        level[v] = len(trail)
        
        # Collect unit propagations
        q = []
        for ci, slot in list(var_to_watches[v]):
            cl = all_clauses[ci]
            unfixed = []
            for vv, pp in cl:
                if assign[vv] == -1:
                    unfixed.append((vv, pp))
            if len(unfixed) == 1:
                q.append(unfixed[0])
        
        ok, conf = propagate(q)
        
        if ok:
            if check_solved():
                return [bool(assign[i]) for i in range(n)]
        else:
            learnt_clause, bt_level = analyze_conflict(conf)
            if bt_level == 0:
                return None
            while trail and level[trail[-1]] >= bt_level:
                vv = trail.pop()
                assign[vv] = -1
                level[vv] = 0
            phase[v] = not phase[v]
            heapq.heappush(var_heap, (-activity[v], v))
            continue
        
        heapq.heappush(var_heap, (-activity[v], v))
    
    return None


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 325

    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_opt)

    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses_opt):
        for v, p in cl:
            var_clauses[v].append((ci, p))

    weights = [len(l) for l in var_clauses]
    total_w = sum(weights) or 1

    # Phase 0: DPLL
    dpll_deadline = start_time + 80
    result = _dpll_solve(clauses_opt, n, m, dpll_deadline)
    if result is not None:
        for cl in clauses_opt:
            for v, p in cl:
                if result[v] == p:
                    break
            else:
                result = None
                break
        if result is not None:
            return result

    # Phase 1: SLS
    def count_unsat(ass):
        cnt = 0
        for cl in clauses_opt:
            for v, p in cl:
                if ass[v] == p:
                    break
            else:
                cnt += 1
        return cnt

    def build_state(ass):
        sat = [0] * m
        for ci, cl in enumerate(clauses_opt):
            s = 0
            for v, p in cl:
                if ass[v] == p:
                    s += 1
            sat[ci] = s
        return sat

    def flip_var(ass, var, sat):
        old = ass[var]
        new = not old
        ass[var] = new
        for ci, p in var_clauses[var]:
            if p == new:
                sat[ci] += 1
            else:
                sat[ci] -= 1

    def unit_propagate(ass, remaining_vars, n_remaining, sat):
        changed = True
        while changed and n_remaining > 0:
            changed = False
            for ci in range(m):
                if sat[ci] > 0:
                    continue
                cl = clauses_opt[ci]
                unfixed = []
                for v, p in cl:
                    if v not in remaining_vars:
                        continue
                    unfixed.append((v, p))
                if len(unfixed) == 1:
                    v, p = unfixed[0]
                    if ass[v] != p:
                        flip_var(ass, v, sat)
                    remaining_vars.discard(v)
                    n_remaining -= 1
                    changed = True
        return True

    best_assignment = None
    best_unsat = float('inf')

    def track(ass):
        nonlocal best_assignment, best_unsat
        u = count_unsat(ass)
        if u < best_unsat:
            best_unsat = u
            best_assignment = ass[:]
        return u

    max_restarts = 300

    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break

        rng = random.Random(restart * 7919 + 131)
        ass = [rng.random() < 0.5 for _ in range(n)]

        if restart % 5 == 0:
            for v in range(n):
                pos = sum(1 for ci, p in var_clauses[v] if p)
                ass[v] = (pos >= len(var_clauses[v]) - pos) if len(var_clauses[v]) else False

        u = track(ass)
        if u == 0:
            return ass

        remaining_vars = set(range(n))
        n_remaining = n
        sat = build_state(ass)
        unit_propagate(ass, remaining_vars, n_remaining, sat)
        u = track(ass)
        if u == 0:
            return ass

        unsat_count = sum(1 for s in sat if s == 0)
        mode = restart % 3
        noise = 0.15 + 0.05 * (restart % 4)
        steps = 30000

        for step in range(steps):
            if time.time() > deadline - 5:
                break

            if unsat_count == 0:
                return ass

            best_ci = -1
            best_len = 0
            for ci in range(m):
                if sat[ci] == 0:
                    ln = len(clauses_opt[ci])
                    if ln > best_len:
                        best_len = ln
                        best_ci = ci
                        if ln >= 3:
                            break
            if best_ci < 0:
                return ass

            clause = clauses_opt[best_ci]

            if rng.random() < noise:
                v, p = clause[rng.randrange(len(clause))]
            else:
                best_v = clause[0][0]
                best_score = None
                for v, p in clause:
                    if ass[v] == p:
                        continue
                    cost = 0
                    for ci2, p2 in var_clauses[v]:
                        if p2 == ass[v] and sat[ci2] == 1:
                            cost += 1
                    if mode == 0:
                        score = (cost, -weights[v])
                    elif mode == 1:
                        pos = sum(1 for ci2, p2 in var_clauses[v] if p2)
                        phase_match = 1 if (p == (pos >= len(var_clauses[v]) - pos)) else 0
                        score = (cost, -phase_match, -weights[v])
                    else:
                        score = (cost, -weights[v])
                    if best_score is None or score < best_score:
                        best_score = score
                        best_v = v
                v = best_v
            
            flip_var(ass, v, sat)
            new_unsat = 0
            for s in sat:
                if s == 0:
                    new_unsat += 1
            unsat_count = new_unsat

            if unsat_count == 0:
                return ass

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = ass[:]

    # Phase 2: deep walks
    if best_assignment is not None and time.time() < deadline - 5:
        ass = best_assignment[:]
        sat = build_state(ass)
        u = count_unsat(ass)
        if u > 0:
            rng2 = random.Random(987654321)
            noise = 0.15
            unsat_count = u
            while time.time() < deadline - 2:
                if unsat_count == 0:
                    return ass
                unsat_list = [ci for ci in range(m) if sat[ci] == 0]
                if not unsat_list:
                    return ass
                ci = unsat_list[rng2.randrange(len(unsat_list))]
                clause = clauses_opt[ci]
                if rng2.random() < noise:
                    v, p = clause[rng2.randrange(len(clause))]
                else:
                    best_v = clause[0][0]
                    best_cost = float('inf')
                    for v, p in clause:
                        cost = 0
                        for ci2, p2 in var_clauses[v]:
                            if p2 == ass[v] and sat[ci2] == 1:
                                cost += 1
                        if cost < best_cost:
                            best_cost = cost
                            best_v = v
                    v = best_v
                flip_var(ass, v, sat)
                new_unsat = 0
                for s in sat:
                    if s == 0:
                        new_unsat += 1
                unsat_count = new_unsat
                if unsat_count == 0:
                    return ass
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = ass[:]

    if best_assignment is not None:
        return best_assignment
    return ass
# EVOLVE-BLOCK-END
