# EVOLVE-BLOCK-START
import time
import random
import heapq


def _satisfied(cl, ass):
    for v, p in cl:
        if ass[v] == p:
            return True
    return False


def _dpll_solve(clauses_opt, var_clauses, n, m, deadline):
    """
    Clause-learning DPLL with VSIDS variable ordering.
    Returns assignment list or None if timeout/UNSAT.
    """
    # Assignment: -1 = unassigned, 0 = False, 1 = True
    assign = [-1] * n
    
    # Clause state: for each clause, track which literals are satisfied/unsatisfied
    # sat_count[ci] = number of satisfied literals
    sat_count = [0] * m
    
    # For unit propagation, we need to track unfixed literals per clause
    # We'll use a worklist approach
    
    # VSIDS activity
    activity = [1.0] * n
    # Heap for variable selection: (negative_activity, var)
    var_heap = [(-activity[v], v) for v in range(n)]
    heapq.heapify(var_heap)
    # Track if var is in heap (for efficiency, we'll rebuild periodically)
    
    # Decision level stack
    decision_level = [0] * n  # level at which var was assigned
    trail = []  # (var, value, level)
    
    # Learnt clauses
    learnt_clauses = []
    learnt_sat_count = [0] * len(learnt_clauses)  # will update
    
    # Conflict count for VSIDS decay
    conflict_count = 0
    
    # Phase saving
    phase = [True] * n  # default phase
    
    # Unit propagation queue
    unit_queue = []  # (clause_idx, is_learnt)
    
    def propagate():
        """Perform unit propagation. Returns True if no conflict, False if conflict."""
        nonlocal unit_queue, trail, assign, sat_count, learnt_clauses, learnt_sat_count
        while unit_queue:
            ci, is_learnt = unit_queue.pop()
            if is_learnt:
                if ci >= len(learnt_clauses):
                    continue
                cl = learnt_clauses[ci]
                # Check if clause is already satisfied
                satisfied = False
                for v, p in cl:
                    if assign[v] != -1 and assign[v] == (1 if p else 0):
                        satisfied = True
                        break
                if satisfied:
                    continue
                # Count unfixed and check for unit
                unfixed = []
                for v, p in cl:
                    if assign[v] == -1:
                        unfixed.append((v, p))
                if len(unfixed) == 0:
                    return False  # Conflict
                elif len(unfixed) == 1:
                    v, p = unfixed[0]
                    if assign[v] == -1:
                        assign[v] = 1 if p else 0
                        trail.append((v, assign[v], len(trail)))
                        # Add to unit queue all clauses containing this var
                        for ci2, p2 in var_clauses[v]:
                            unit_queue.append((ci2, False))
                        for li in range(len(learnt_clauses)):
                            for v2, p2 in learnt_clauses[li]:
                                if v2 == v:
                                    unit_queue.append((li, True))
                                    break
                    break
            else:
                if ci >= m:
                    continue
                cl = clauses_opt[ci]
                # Check if clause is already satisfied
                satisfied = False
                for v, p in cl:
                    if assign[v] != -1 and assign[v] == (1 if p else 0):
                        satisfied = True
                        break
                if satisfied:
                    continue
                # Count unfixed and check for unit
                unfixed = []
                for v, p in cl:
                    if assign[v] == -1:
                        unfixed.append((v, p))
                if len(unfixed) == 0:
                    return False  # Conflict
                elif len(unfixed) == 1:
                    v, p = unfixed[0]
                    if assign[v] == -1:
                        assign[v] = 1 if p else 0
                        trail.append((v, assign[v], len(trail)))
                        # Add to unit queue all clauses containing this var
                        for ci2, p2 in var_clauses[v]:
                            unit_queue.append((ci2, False))
                        for li in range(len(learnt_clauses)):
                            for v2, p2 in learnt_clauses[li]:
                                if v2 == v:
                                    unit_queue.append((li, True))
                                    break
                    break
        return True
    
    def select_variable():
        """Select next variable using VSIDS."""
        nonlocal var_heap, activity
        # Clean up heap: remove assigned variables
        while var_heap:
            neg_act, v = var_heap[0]
            if assign[v] == -1:
                return v
            heapq.heappop(var_heap)
        return -1
    
    def backtrack_to_level(level):
        """Undo assignments back to given level."""
        nonlocal trail, assign
        while trail and trail[-1][2] >= level:
            v, val, lvl = trail.pop()
            assign[v] = -1
    
    def analyze_conflict(ci):
        """Analyze conflict and produce learnt clause. Returns (learnt_clause, backtrack_level)."""
        nonlocal trail, assign, activity, conflict_count
        # Find the conflict clause
        if ci < m:
            cl = clauses_opt[ci]
        else:
            cl = learnt_clauses[ci - m]
        
        # Build implication graph (simplified)
        # Find all variables assigned in this conflict path
        conflict_vars = set()
        for v, p in cl:
            if assign[v] != -1:
                conflict_vars.add(v)
        
        # Find the highest decision level among conflict variables
        max_level = 0
        for v in conflict_vars:
            # Find level from trail
            for tv, tv_val, tv_lvl in trail:
                if tv == v:
                    if tv_lvl > max_level:
                        max_level = tv_lvl
                    break
        
        # Simplified: learn a clause with the conflict literals
        # This is a very simplified analysis - in practice we'd do full 1UIP
        learnt = []
        for v, p in cl:
            if assign[v] != -1:
                # Add negation of the assignment
                neg_p = not p
                learnt.append((v, neg_p))
        
        # Remove duplicates
        seen = set()
        unique_learnt = []
        for v, p in learnt:
            key = (v, p)
            if key not in seen:
                seen.add(key)
                unique_learnt.append((v, p))
        
        if len(unique_learnt) > 1:
            learnt_clauses.append(unique_learnt)
            # Add to unit queue if it's unit
            unfixed = [x for x in unique_learnt if assign[x[0]] == -1]
            if len(unfixed) == 1:
                unit_queue.append((len(learnt_clauses) - 1, True))
        
        # VSIDS bump
        conflict_count += 1
        if conflict_count % 100 == 0:
            for i in range(n):
                activity[i] *= 0.95
        
        for v, p in unique_learnt:
            activity[v] += 1.0
            if assign[v] == -1:
                var_heap.append((-activity[v], v))
        
        return unique_learnt, max_level
    
    # Initial propagation
    # Add all clauses to unit queue
    for ci in range(m):
        unit_queue.append((ci, False))
    
    if not propagate():
        return None
    
    # Check if all satisfied
    def all_satisfied():
        for ci in range(m):
            cl = clauses_opt[ci]
            for v, p in cl:
                if assign[v] != -1 and assign[v] == (1 if p else 0):
                    break
            else:
                return False
        return True
    
    if all_satisfied():
        # Build assignment
        result = []
        for v in range(n):
            if assign[v] == -1:
                result.append(phase[v])
            else:
                result.append(assign[v] == 1)
        return result
    
    # Main search loop
    while time.time() < deadline:
        v = select_variable()
        if v == -1:
            break
        
        # Try phase first
        try_phase = phase[v]
        
        # Assign and propagate
        assign[v] = 1 if try_phase else 0
        trail.append((v, assign[v], len(trail)))
        
        # Add to unit queue
        for ci2, p2 in var_clauses[v]:
            unit_queue.append((ci2, False))
        for li in range(len(learnt_clauses)):
            for v2, p2 in learnt_clauses[li]:
                if v2 == v:
                    unit_queue.append((li, True))
                    break
        
        ok = propagate()
        
        if ok:
            if all_satisfied():
                result = []
                for vi in range(n):
                    if assign[vi] == -1:
                        result.append(phase[vi])
                    else:
                        result.append(assign[vi] == 1)
                return result
        else:
            # Conflict
            # Find which clause caused conflict
            conflict_ci = -1
            # Check learnt clauses first
            for li in range(len(learnt_clauses) - 1, -1, -1):
                cl = learnt_clauses[li]
                all_unsat = True
                for v2, p2 in cl:
                    if assign[v2] != -1 and assign[v2] == (1 if p2 else 0):
                        all_unsat = False
                        break
                    if assign[v2] == -1:
                        all_unsat = False
                        break
                if all_unsat:
                    conflict_ci = m + li
                    break
            
            if conflict_ci == -1:
                for ci in range(m):
                    cl = clauses_opt[ci]
                    all_unsat = True
                    for v2, p2 in cl:
                        if assign[v2] != -1 and assign[v2] == (1 if p2 else 0):
                            all_unsat = False
                            break
                        if assign[v2] == -1:
                            all_unsat = False
                            break
                    if all_unsat:
                        conflict_ci = ci
                        break
            
            if conflict_ci != -1:
                learnt, bt_level = analyze_conflict(conflict_ci)
                backtrack_to_level(bt_level)
            else:
                # No clear conflict, just backtrack
                backtrack_to_level(len(trail) - 1)
        
        # Try other phase if first failed
        if assign[v] == -1:
            assign[v] = 0 if try_phase else 1
            trail.append((v, assign[v], len(trail)))
            phase[v] = not try_phase
            
            for ci2, p2 in var_clauses[v]:
                unit_queue.append((ci2, False))
            for li in range(len(learnt_clauses)):
                for v2, p2 in learnt_clauses[li]:
                    if v2 == v:
                        unit_queue.append((li, True))
                        break
            
            ok = propagate()
            if ok:
                if all_satisfied():
                    result = []
                    for vi in range(n):
                        if assign[vi] == -1:
                            result.append(phase[vi])
                        else:
                            result.append(assign[vi] == 1)
                    return result
            else:
                # Conflict on second try too
                conflict_ci = -1
                for li in range(len(learnt_clauses) - 1, -1, -1):
                    cl = learnt_clauses[li]
                    all_unsat = True
                    for v2, p2 in cl:
                        if assign[v2] != -1 and assign[v2] == (1 if p2 else 0):
                            all_unsat = False
                            break
                        if assign[v2] == -1:
                            all_unsat = False
                            break
                    if all_unsat:
                        conflict_ci = m + li
                        break
                
                if conflict_ci == -1:
                    for ci in range(m):
                        cl = clauses_opt[ci]
                        all_unsat = True
                        for v2, p2 in cl:
                            if assign[v2] != -1 and assign[v2] == (1 if p2 else 0):
                                all_unsat = False
                                break
                            if assign[v2] == -1:
                                all_unsat = False
                                break
                        if all_unsat:
                            conflict_ci = ci
                            break
                
                if conflict_ci != -1:
                    learnt, bt_level = analyze_conflict(conflict_ci)
                    backtrack_to_level(bt_level)
                else:
                    backtrack_to_level(len(trail) - 1)
        
        # If still unassigned, it's truly unassigned (shouldn't happen in SAT)
    
    return None


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 320  # safety margin before 330s kill

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

    # Phase 0: Try DPLL with clause learning
    dpll_deadline = start_time + 60  # Give DPLL 60 seconds
    result = _dpll_solve(clauses_opt, var_clauses, n, m, dpll_deadline)
    if result is not None:
        # Verify
        for cl in clauses_opt:
            for v, p in cl:
                if result[v] == p:
                    break
            else:
                result = None
                break
        if result is not None:
            return result

    # Fallback to SLS if DPLL didn't solve
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
        # sat[v] = number of satisfied literals in clause v
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

    def unsat_clauses_list(sat):
        return [ci for ci in range(m) if sat[ci] == 0]

    def unit_propagate(ass, remaining_vars, n_remaining, sat):
        # Repeat: find clauses with exactly one unfixed unsat literal.
        # Returns True if no contradiction (ass updated in place).
        # Uses remaining_vars set.
        changed = True
        while changed and n_remaining > 0:
            changed = False
            # For each clause still unsat, count unfixed literals
            for ci in range(m):
                if sat[ci] > 0:
                    continue
                cl = clauses_opt[ci]
                unfixed = []
                for v, p in cl:
                    if v not in remaining_vars:
                        # fixed: if it's satisfied, clause would be sat; but sat==0 so it's unsat
                        continue
                    unfixed.append((v, p))
                if len(unfixed) == 1:
                    v, p = unfixed[0]
                    if ass[v] != p:
                        flip_var(ass, v, sat)
                    remaining_vars.discard(v)
                    n_remaining -= 1
                    changed = True
        return True  # no contradiction handling needed (planted SAT)

    # Phase 1: unit propagation on a random seed to get a good starting point
    best_assignment = None
    best_unsat = float('inf')

    def track(ass):
        nonlocal best_assignment, best_unsat
        u = count_unsat(ass)
        if u < best_unsat:
            best_unsat = u
            best_assignment = ass[:]
        return u

    def quick_check(ass):
        for cl in clauses_opt:
            for v, p in cl:
                if ass[v] == p:
                    break
            else:
                return False
        return True

    max_restarts = 200
    rng_master = random.Random(123456789)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(restart * 7919 + 131)
        ass = [rng.random() < 0.5 for _ in range(n)]

        # Try to bias toward positive literals (planted instances often)
        # Occasionally use phase bias: prefer the literal that appears more
        if restart % 5 == 0:
            # phase save bias: set var to polarity with more occurrences
            for v in range(n):
                pos = sum(1 for ci, p in var_clauses[v] if p)
                ass[v] = (pos >= len(var_clauses[v]) - pos) if len(var_clauses[v]) else False

        u = track(ass)
        if u == 0:
            return ass

        # Unit propagation to reduce
        remaining_vars = set(range(n))
        n_remaining = n
        sat = build_state(ass)
        unit_propagate(ass, remaining_vars, n_remaining, sat)
        u = track(ass)
        if u == 0:
            return ass

        # Determine heuristic mode for this restart
        mode = restart % 3  # 0: min-break, 1: phase-satisfy, 2: max-weight
        noise = 0.2 + 0.1 * (restart % 4)

        steps = 20000
        for step in range(steps):
            if time.time() > deadline:
                break

            # Find an unsatisfied clause - use longest unsat for stronger guidance
            best_ci = -1
            best_len = 0
            # Sample to find a long unsat clause quickly
            # Check all but stop early is expensive; do a partial scan
            # We'll scan but only track; m is moderate.
            for ci in range(m):
                if sat[ci] == 0:
                    ln = len(clauses_opt[ci])
                    if ln > best_len:
                        best_len = ln
                        best_ci = ci
                        if ln >= 3:
                            # 3 is max for 3-SAT, can break
                            break
            if best_ci < 0:
                return ass

            clause = clauses_opt[best_ci]

            # Choose variable
            if rng.random() < noise:
                v, p = clause[rng.randrange(len(clause))]
            else:
                # Evaluate each literal
                best_v = clause[0][0]
                best_score = None
                for v, p in clause:
                    # new value would be p (to satisfy this literal)
                    if ass[v] == p:
                        # already satisfied? shouldn't be in unsat clause but be safe
                        continue
                    # cost = number of currently-satisfied other clauses broken
                    cost = 0
                    for ci2, p2 in var_clauses[v]:
                        if p2 == ass[v] and sat[ci2] == 1:
                            cost += 1
                    # benefit weight
                    if mode == 0:
                        score = (cost, -weights[v])
                    elif mode == 1:
                        # prefer phase: lower cost, then prefer polarity that matches majority
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

            u = 0
            # quick check via sat
            # Instead of full recount, track if all >0
            all_sat = True
            for s in sat:
                if s == 0:
                    all_sat = False
                    break
            if all_sat:
                return ass

        track(ass)

    # Phase 2: if time remains, do targeted deep walks from best_assignment
    if best_assignment is not None and time.time() < deadline - 5:
        ass = best_assignment[:]
        sat = build_state(ass)
        u = count_unsat(ass)
        if u > 0:
            rng2 = random.Random(987654321)
            noise = 0.15
            while time.time() < deadline - 2:
                # find unsat clauses
                unsat_list = [ci for ci in range(m) if sat[ci] == 0]
                if not unsat_list:
                    return ass
                # pick a random unsat clause, but bias to shorter ones (easier)
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
                # quick full check
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
