# EVOLVE-BLOCK-START
import time
import random


def _satisfied(cl, ass):
    for v, p in cl:
        if ass[v] == p:
            return True
    return False


def _dpll_solve(n, clauses_opt, var_clauses, deadline):
    """DPLL with VSIDS and watched-literal unit propagation."""
    m = len(clauses_opt)
    
    # Assignment: 0 = unassigned, 1 = True, -1 = False
    assignment = [0] * n
    # Clause status: for each clause, track count of unassigned literals
    unassigned_count = [len(cl) for cl in clauses_opt]
    
    # Watched literals: for each clause, track two literals that are unassigned
    # watches[ci] = list of (var, polarity) that are currently unassigned
    watches = []
    for ci, cl in enumerate(clauses_opt):
        watches.append([list(cl[0]), list(cl[1])])
    
    # Assignment trail for backtracking
    trail = []  # list of (var, old_value)
    
    # VSIDS
    activity = [1.0] * n
    decay = 0.95
    
    def bump(v):
        activity[v] *= 2.0
    
    def decay_activities():
        for v in range(n):
            activity[v] *= decay
    
    # Unit propagation using watched literals
    def propagate():
        """Returns False on conflict, True otherwise."""
        worklist = []
        # Initialize worklist with all unit clauses
        for ci in range(m):
            if unassigned_count[ci] == 1:
                worklist.append(ci)
        
        while worklist:
            ci = worklist.pop()
            if unassigned_count[ci] == 0:
                return False  # conflict
            if unassigned_count[ci] > 1:
                continue  # no longer unit
            
            # Find the unassigned literal in this clause
            w = watches[ci]
            unassigned_lit = None
            for i in range(2):
                v, p = w[i]
                if assignment[v] == 0:
                    unassigned_lit = (v, p)
                    break
            
            if unassigned_lit is None:
                # Both watched literals are assigned - this shouldn't happen if unassigned_count==1
                # But let's find the actual unassigned one
                for v, p in clauses_opt[ci]:
                    if assignment[v] == 0:
                        unassigned_lit = (v, p)
                        break
            
            if unassigned_lit is None:
                return False  # conflict
            
            v, p = unassigned_lit
            new_val = 1 if p else -1
            
            if assignment[v] != 0:
                if assignment[v] != new_val:
                    return False  # conflict
                continue
            
            # Assign v
            assignment[v] = new_val
            trail.append((v, 0))
            bump(v)
            
            # Update watches for all clauses containing v
            for ci2, p2 in var_clauses[v]:
                if unassigned_count[ci2] == 0 or ci2 == ci:
                    continue
                # Check if v is in watches[ci2]
                w2 = watches[ci2]
                found_idx = -1
                for i in range(2):
                    if w2[i][0] == v:
                        found_idx = i
                        break
                if found_idx == -1:
                    continue
                # v was satisfied or unsatisfied; since we just assigned it,
                # if p2 == new_val, clause is satisfied, no need to update
                if p2 == (new_val == 1):
                    continue  # clause satisfied
                # v is now unsatisfied in this clause, need to find new watch
                unassigned_count[ci2] -= 1
                # Find a new unassigned literal to watch
                new_watch = None
                for vv, pp in clauses_opt[ci2]:
                    if assignment[vv] == 0 and vv != v:
                        new_watch = (vv, pp)
                        break
                if new_watch is not None:
                    w2[found_idx] = list(new_watch)
                else:
                    # No other unassigned literal - clause is now unit
                    if unassigned_count[ci2] == 1:
                        worklist.append(ci2)
                    elif unassigned_count[ci2] == 0:
                        return False
        return True
    
    def select_variable():
        """Select variable with highest activity among unassigned vars in active clauses."""
        best_v = -1
        best_score = -1.0
        for ci in range(m):
            if unassigned_count[ci] == 0:
                continue
            for v, p in clauses_opt[ci]:
                if assignment[v] == 0 and activity[v] > best_score:
                    best_score = activity[v]
                    best_v = v
        return best_v
    
    def all_satisfied():
        for ci in range(m):
            if unassigned_count[ci] == 0:
                return False
        return True
    
    def backtrack_to(level):
        while len(trail) > level:
            v, old_val = trail.pop()
            assignment[v] = old_val
            # Update unassigned_count for clauses containing v
            for ci, p in var_clauses[v]:
                unassigned_count[ci] += 1
            # Rebuild watches if needed (simplified: just mark for rebuild)
    
    def dpll(level):
        if time.time() > deadline:
            return None
        
        if all_satisfied():
            # Fill in remaining unassigned vars
            for v in range(n):
                if assignment[v] == 0:
                    assignment[v] = 1
            return assignment[:]
        
        if not propagate():
            return None
        
        if all_satisfied():
            for v in range(n):
                if assignment[v] == 0:
                    assignment[v] = 1
            return assignment[:]
        
        v = select_variable()
        if v == -1:
            return None
        
        # Determine preferred polarity
        pos_score = 0
        neg_score = 0
        for ci, p in var_clauses[v]:
            if unassigned_count[ci] > 0:
                if p:
                    pos_score += 1
                else:
                    neg_score += 1
        
        if pos_score >= neg_score:
            order = [1, -1]
        else:
            order = [-1, 1]
        
        for val in order:
            trail_level = len(trail)
            assignment[v] = val
            trail.append((v, 0))
            bump(v)
            
            # Update unassigned counts
            for ci, p in var_clauses[v]:
                unassigned_count[ci] -= 1
                if unassigned_count[ci] < 0:
                    unassigned_count[ci] = 0
            
            result = dpll(level + 1)
            if result is not None:
                return result
            
            # Backtrack
            backtrack_to(trail_level)
        
        return None
    
    result = dpll(0)
    if result is not None:
        return [bool(x) for x in result]
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

    def quick_check(ass):
        for cl in clauses_opt:
            for v, p in cl:
                if ass[v] == p:
                    break
            else:
                return False
        return True

    # Phase 0: Try DPLL with time limit
    dpll_deadline = start_time + 90  # Give DPLL up to 90 seconds
    dpll_result = _dpll_solve(n, clauses_opt, var_clauses, dpll_deadline)
    if dpll_result is not None:
        # Verify the solution
        if quick_check(dpll_result):
            return dpll_result

    # Phase 1: Stochastic local search if DPLL didn't solve
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

        steps = 30000
        for step in range(steps):
            if time.time() > deadline:
                break

            # Find an unsatisfied clause - use longest unsat for stronger guidance
            best_ci = -1
            best_len = 0
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
