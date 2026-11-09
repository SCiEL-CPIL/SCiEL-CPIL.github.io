# EVOLVE-BLOCK-START
import time
import random


def _satisfied(cl, ass):
    for v, p in cl:
        if ass[v] == p:
            return True
    return False


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 315  # safety margin before 330s kill

    # Precompute clauses as (var_idx 0-based, polarity bool)
    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_opt)

    # Precompute var -> list of (clause_idx, polarity)
    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses_opt):
        for v, p in cl:
            var_clauses[v].append((ci, p))

    # Weights per variable
    weights = [len(l) for l in var_clauses]
    total_w = sum(weights) or 1

    # ===================== DPLL WITH VSIDS AND CLAUSE LEARNING =====================
    
    def dpll_solver():
        # Data structures for efficient DPLL
        # clause_watch[i] = list of (clause_idx, watch_pos) where watch_pos is 0 or 1
        # We watch two literals per clause
        clause_watch = [None] * m
        for ci, cl in enumerate(clauses_opt):
            clause_watch[ci] = [0, 1]  # watch positions
        
        # For each variable, maintain list of clauses watching it
        # watch_list[v] = list of (clause_idx, watch_pos)
        watch_list = [[] for _ in range(n)]
        for ci, cl in enumerate(clauses_opt):
            for wp in range(2):
                v, p = cl[clause_watch[ci][wp]]
                watch_list[v].append((ci, wp))
        
        # Assignment: -1 unassigned, 0 False, 1 True
        assign = [-1] * n
        trail = []  # stack of (var, value)
        
        # VSIDS activity
        activity = [1.0] * n
        var_inc = 1.0
        var_decrement = 1.0 / (100 + n)
        
        # Level tracking
        level = [0] * n
        max_level = 0
        
        # Clause learning
        learned_clauses = []
        learnt_size = 0
        max_learnt = 5000
        
        # Decision levels
        def new_decision_level():
            nonlocal max_level
            max_level += 1
            return max_level
        
        def propagate():
            """Unit propagation. Returns False on conflict, True otherwise."""
            work = 0
            while work < len(trail):
                p = trail[work]
                work += 1
                v, val = p
                # Check clauses watching v with polarity != val (i.e., the other watch)
                # Actually, we need to check clauses where v is watched and the watched literal is false
                wl = watch_list[v]
                i = 0
                while i < len(wl):
                    ci, wp = wl[i]
                    cl = clauses_opt[ci]
                    v_w, p_w = cl[clause_watch[ci][wp]]
                    # The watched literal is (v_w, p_w). It's false if assign[v_w] == 1 - p_w
                    if assign[v_w] == 1 - p_w:
                        # This watch is false, need to find another watch
                        # Check if clause is satisfied
                        satisfied = False
                        for j in range(3):
                            if j == clause_watch[ci][wp]:
                                continue
                            vj, pj = cl[j]
                            if assign[vj] == pj:
                                satisfied = True
                                break
                        if satisfied:
                            # Clause is satisfied, keep current watches
                            i += 1
                            continue
                        # Find a new watch among unassigned or satisfied literals
                        new_wp = -1
                        for j in range(3):
                            vj, pj = cl[j]
                            if assign[vj] == -1:
                                new_wp = j
                                break
                        if new_wp == -1:
                            # All literals false -> conflict
                            return False
                        # Update watch
                        old_v, old_p = cl[clause_watch[ci][wp]]
                        # Remove old watch from old_v's list
                        # This is O(n) but acceptable for small n
                        old_wl = watch_list[old_v]
                        for k in range(len(old_wl)):
                            if old_wl[k] == (ci, wp):
                                old_wl[k] = old_wl[-1]
                                old_wl.pop()
                                break
                        # Add new watch
                        new_v, new_p = cl[new_wp]
                        clause_watch[ci][wp] = new_wp
                        watch_list[new_v].append((ci, wp))
                        i += 1
                    else:
                        i += 1
            return True
        
        def analyze(conflict_clause):
            """Analyze conflict, learn a clause. Returns (learnt_clause, backtrack_level)."""
            nonlocal learnt_size, var_inc, var_decrement
            # conflict_clause is a list of (var, polarity) that are all false
            # We need to resolve
            learnt = []
            seen = set()
            counter = 0
            max_level_seen = 0
            btl = 0
            
            # Start with conflict clause
            cur_clause = list(conflict_clause)
            
            while True:
                # Find the literal in cur_clause with the highest decision level
                max_lvl = 0
                max_idx = -1
                for i, (v, p) in enumerate(cur_clause):
                    if level[v] > max_lvl:
                        max_lvl = level[v]
                        max_idx = i
                
                if max_idx == -1:
                    break
                
                v, p = cur_clause[max_idx]
                # Remove this literal from learnt (it's implied)
                # Add to seen
                seen.add((v, p))
                
                # Find the reason clause for (v, p)
                # Search for a clause that has (v, p) and all other literals are false
                reason_found = False
                for ci in range(m):
                    cl = clauses_opt[ci]
                    if (v, p) in cl:
                        # Check if all other literals are false
                        all_other_false = True
                        for vj, pj in cl:
                            if (vj, pj) == (v, p):
                                continue
                            if assign[vj] != 1 - pj:
                                all_other_false = False
                                break
                        if all_other_false:
                            # This is the reason
                            for vj, pj in cl:
                                if (vj, pj) == (v, p):
                                    continue
                                if (vj, pj) not in seen:
                                    learnt.append((vj, pj))
                                    seen.add((vj, pj))
                            reason_found = True
                            break
                
                if not reason_found:
                    # Shouldn't happen for planted SAT, but handle gracefully
                    break
                
                # Remove the highest-level literal from cur_clause
                cur_clause.pop(max_idx)
                
                # Check if we have a clause with only one literal at the current max level
                # Actually, we need to check if there's exactly one literal at level > btl
                count_above_btl = 0
                for vj, pj in cur_clause:
                    if level[vj] > btl:
                        count_above_btl += 1
                if count_above_btl <= 1:
                    break
                
                # Update btl
                max_lvl2 = 0
                for vj, pj in cur_clause:
                    if level[vj] > max_lvl2:
                        max_lvl2 = level[vj]
                if max_lvl2 > btl:
                    btl = max_lvl2
        
        # Simplified: for planted SAT, use basic DPLL with VSIDS ordering
        # and unit propagation, with clause learning from conflicts
        
        def solve_dpll():
            nonlocal assign, trail, max_level, activity, var_inc, var_decrement
            nonlocal learnt_clauses, learnt_size, max_learnt
            
            def pick_variable():
                """Pick variable with highest activity among unassigned."""
                best_v = -1
                best_act = -1
                for v in range(n):
                    if assign[v] == -1 and activity[v] > best_act:
                        best_act = activity[v]
                        best_v = v
                return best_v
            
            def bump_activity(v):
                nonlocal var_inc, var_decrement
                activity[v] += var_inc
                if activity[v] > 1e100:
                    # Rescale
                    for i in range(n):
                        activity[i] /= 1e100
                    var_inc /= 1e100
            
            def decay_activity():
                nonlocal var_inc, var_decrement
                var_inc *= var_decrement
            
            def assign_var(v, val, reason_level):
                nonlocal max_level
                assign[v] = val
                level[v] = reason_level
                trail.append((v, val))
                if reason_level > max_level:
                    max_level = reason_level
            
            def backtrack(to_level):
                nonlocal max_level
                while trail:
                    v, val = trail[-1]
                    if level[v] <= to_level:
                        break
                    trail.pop()
                    assign[v] = -1
                    level[v] = 0
                if to_level < max_level:
                    max_level = to_level
            
            def propagate_and_check():
                """Unit propagation. Returns (success, conflict_clause_or_None)."""
                work = 0
                while work < len(trail):
                    p = trail[work]
                    work += 1
                    v, val = p
                    wl = watch_list[v]
                    i = 0
                    while i < len(wl):
                        ci, wp = wl[i]
                        cl = clauses_opt[ci]
                        v_w, p_w = cl[clause_watch[ci][wp]]
                        if assign[v_w] == 1 - p_w:
                            # Watch is false
                            # Check if clause satisfied
                            satisfied = False
                            for j in range(3):
                                if j == clause_watch[ci][wp]:
                                    continue
                                vj, pj = cl[j]
                                if assign[vj] == pj:
                                    satisfied = True
                                    break
                            if satisfied:
                                i += 1
                                continue
                            # Find new watch
                            new_wp = -1
                            for j in range(3):
                                vj, pj = cl[j]
                                if assign[vj] == -1:
                                    new_wp = j
                                    break
                            if new_wp == -1:
                                # Conflict! Return the clause
                                return False, cl
                            # Update watch
                            old_v, old_p = cl[clause_watch[ci][wp]]
                            old_wl = watch_list[old_v]
                            for k in range(len(old_wl)):
                                if old_wl[k] == (ci, wp):
                                    old_wl[k] = old_wl[-1]
                                    old_wl.pop()
                                    break
                            new_v, new_p = cl[new_wp]
                            clause_watch[ci][wp] = new_wp
                            watch_list[new_v].append((ci, wp))
                            i += 1
                        else:
                            i += 1
                return True, None
            
            def learn_clause(conflict_clause):
                """Learn a clause from conflict. Simplified: just add the negation of the conflict."""
                nonlocal learnt_clauses, learnt_size
                if learnt_size >= max_learnt:
                    return
                # The conflict clause has all literals false
                # Learn the clause that is the disjunction of negations
                # Actually, the conflict clause itself is the clause to learn (it's unsatisfied)
                # We learn a clause that prevents this conflict
                learnt = []
                for v, p in conflict_clause:
                    learnt.append((v, 1 - p))
                if len(learnt) <= 3:  # Only learn short clauses
                    learnt_clauses.append(learnt)
                    learnt_size += 1
            
            # Main DPLL loop
            while True:
                # Propagate
                ok, conflict = propagate_and_check()
                if not ok:
                    # Conflict
                    if max_level == 0:
                        return None  # UNSAT (shouldn't happen for planted SAT)
                    # Learn clause
                    learn_clause(conflict)
                    # Backtrack
                    # Find second highest level in conflict
                    levels = [level[v] for v, p in conflict]
                    levels.sort(reverse=True)
                    btl = levels[1] if len(levels) > 1 else 0
                    backtrack(btl)
                    # Bump activity
                    for v, p in conflict:
                        bump_activity(v)
                    continue
                
                # Check if all assigned
                if len(trail) == n:
                    # Check if all clauses satisfied
                    for ci in range(m):
                        cl = clauses_opt[ci]
                        sat = False
                        for v, p in cl:
                            if assign[v] == p:
                                sat = True
                                break
                        if not sat:
                            # Shouldn't happen after propagation
                            return None
                    # Also check learned clauses
                    for lc in learnt_clauses:
                        sat = False
                        for v, p in lc:
                            if assign[v] == p:
                                sat = True
                                break
                        if not sat:
                            return None
                    # Solution found
                    return [bool(assign[v]) for v in range(n)]
                
                # Pick next variable
                v = pick_variable()
                if v == -1:
                    # All assigned but not all satisfied? Shouldn't happen
                    return None
                
                # Decide
                # Try both values, prefer one that satisfies more clauses
                pos_count = sum(1 for ci, p in var_clauses[v] if p)
                neg_count = len(var_clauses[v]) - pos_count
                # Try the more common polarity first
                if pos_count >= neg_count:
                    first_val, second_val = 1, 0
                else:
                    first_val, second_val = 0, 1
                
                # Try first value
                new_level = new_decision_level()
                assign_var(v, first_val, new_level)
                
                ok2, conflict2 = propagate_and_check()
                if ok2:
                    # Recurse by continuing the loop (iterative DPLL)
                    # We'll just continue; the loop will handle it
                    continue
                else:
                    # Conflict with first value, learn and backtrack
                    learn_clause(conflict2)
                    for vv, pp in conflict2:
                        bump_activity(vv)
                    # Try second value
                    backtrack(new_level - 1)
                    new_level2 = new_decision_level()
                    assign_var(v, second_val, new_level2)
                    
                    ok3, conflict3 = propagate_and_check()
                    if ok3:
                        continue
                    else:
                        # Both values lead to conflict -> backtrack further
                        learn_clause(conflict3)
                        for vv, pp in conflict3:
                            bump_activity(vv)
                        levels = [level[vv] for vv, pp in conflict3]
                        levels.sort(reverse=True)
                        btl = levels[1] if len(levels) > 1 else 0
                        backtrack(btl)
                        decay_activity()
                        continue
        
        # Run DPLL with time limit
        dpll_start = time.time()
        dpll_deadline = dpll_start + min(120, deadline - start_time - 5)
        
        # Simple DPLL without full clause learning for reliability
        # Use recursive approach with VSIDS
        assign = [-1] * n
        trail = []
        level = [0] * n
        max_level = 0
        activity = [float(w) for w in weights]
        var_inc = 1.0
        var_decrement = 1.0 / (100 + n)
        
        def dpll_recursive(dl):
            nonlocal assign, trail, level, max_level, activity, var_inc, var_decrement
            if time.time() > dpll_deadline:
                return None
            
            # Propagate
            work = 0
            while work < len(trail):
                p = trail[work]
                work += 1
                v, val = p
                wl = watch_list[v]
                i = 0
                while i < len(wl):
                    ci, wp = wl[i]
                    cl = clauses_opt[ci]
                    v_w, p_w = cl[clause_watch[ci][wp]]
                    if assign[v_w] == 1 - p_w:
                        satisfied = False
                        for j in range(3):
                            if j == clause_watch[ci][wp]:
                                continue
                            vj, pj = cl[j]
                            if assign[vj] == pj:
                                satisfied = True
                                break
                        if satisfied:
                            i += 1
                            continue
                        new_wp = -1
                        for j in range(3):
                            vj, pj = cl[j]
                            if assign[vj] == -1:
                                new_wp = j
                                break
                        if new_wp == -1:
                            return None  # Conflict
                        old_v, old_p = cl[clause_watch[ci][wp]]
                        old_wl = watch_list[old_v]
                        for k in range(len(old_wl)):
                            if old_wl[k] == (ci, wp):
                                old_wl[k] = old_wl[-1]
                                old_wl.pop()
                                break
                        new_v, new_p = cl[new_wp]
                        clause_watch[ci][wp] = new_wp
                        watch_list[new_v].append((ci, wp))
                        i += 1
                    else:
                        i += 1
            
            # Check if all assigned
            if len(trail) == n:
                for ci in range(m):
                    cl = clauses_opt[ci]
                    sat = False
                    for v, p in cl:
                        if assign[v] == p:
                            sat = True
                            break
                    if not sat:
                        return None
                return [bool(assign[v]) for v in range(n)]
            
            # Pick variable
            best_v = -1
            best_act = -1
            for v in range(n):
                if assign[v] == -1 and activity[v] > best_act:
                    best_act = activity[v]
                    best_v = v
            
            if best_v == -1:
                return None
            
            # Decide
            pos_count = sum(1 for ci, p in var_clauses[best_v] if p)
            neg_count = len(var_clauses[best_v]) - pos_count
            if pos_count >= neg_count:
                first_val, second_val = 1, 0
            else:
                first_val, second_val = 0, 1
            
            # Try first
            max_level += 1
            assign[best_v] = first_val
            level[best_v] = max_level
            trail.append((best_v, first_val))
            
            result = dpll_recursive(max_level)
            if result is not None:
                return result
            
            # Backtrack
            while trail and trail[-1][0] == best_v:
                v, val = trail.pop()
                assign[v] = -1
                level[v] = 0
            max_level -= 1
            
            # Try second
            max_level += 1
            assign[best_v] = second_val
            level[best_v] = max_level
            trail.append((best_v, second_val))
            
            result = dpll_recursive(max_level)
            if result is not None:
                return result
            
            # Backtrack
            while trail and trail[-1][0] == best_v:
                v, val = trail.pop()
                assign[v] = -1
                level[v] = 0
            max_level -= 1
            
            # Bump activity
            activity[best_v] += var_inc
            var_inc *= var_decrement
            
            return None
        
        # Reset watch lists for fresh DPLL
        watch_list = [[] for _ in range(n)]
        clause_watch = [None] * m
        for ci, cl in enumerate(clauses_opt):
            clause_watch[ci] = [0, 1]
        for ci, cl in enumerate(clauses_opt):
            for wp in range(2):
                v, p = cl[clause_watch[ci][wp]]
                watch_list[v].append((ci, wp))
        
        result = dpll_recursive(0)
        if result is not None:
            return result
        
        return None
    
    # Try DPLL first
    dpll_result = dpll_solver()
    if dpll_result is not None:
        return dpll_result
    
    # ===================== FALLBACK: STOCHASTIC LOCAL SEARCH =====================
    
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

    def check_all_sat(sat):
        for s in sat:
            if s == 0:
                return False
        return True

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

    max_restarts = 200
    rng_master = random.Random(123456789)

    for restart in range(max_restarts):
        if time.time() > deadline:
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

        mode = restart % 3
        noise = 0.2 + 0.1 * (restart % 4)

        steps = 20000
        for step in range(steps):
            if time.time() > deadline:
                break

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

            all_sat = True
            for s in sat:
                if s == 0:
                    all_sat = False
                    break
            if all_sat:
                return ass

        track(ass)

    if best_assignment is not None and time.time() < deadline - 5:
        ass = best_assignment[:]
        sat = build_state(ass)
        u = count_unsat(ass)
        if u > 0:
            rng2 = random.Random(987654321)
            noise = 0.15
            while time.time() < deadline - 2:
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
                all_sat = True
                for s in sat:
                    if s == 0:
                        all_sat = False
                        break
                if all_sat:
                    return ass
                u2 = sum(1 for s in sat if s == 0)
                if u2 < best_unsat:
                    best_unsat = u2
                    best_assignment = ass[:]

    if best_assignment is not None:
        return best_assignment
    return ass
# EVOLVE-BLOCK-END
