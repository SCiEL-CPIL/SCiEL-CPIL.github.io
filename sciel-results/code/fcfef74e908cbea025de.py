"""CDCL-style 3-SAT solver with watched literals, clause learning, and VSIDS-like variable activity."""
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Convert to 0-based (var, sign) tuples
    processed_clauses = []
    for clause in clauses:
        processed = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            processed.append((var, sign))
        processed_clauses.append(processed)
    
    start_time = time.time()
    deadline = start_time + 290.0
    rng = random.Random(42)
    
    # Build variable-to-clause mapping for WalkSAT
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(processed_clauses):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # CDCL phase with watched literals
    cdcl_deadline = start_time + 120.0
    
    # State
    assignment = [-1] * n
    trail = []
    trail_lim = []
    decision_level = 0
    reason = [-1] * n
    reason_lit_idx = [-1] * n
    var_activity = [0.0] * n
    
    # Watched literals setup
    clause_watches = [(0, 1)] * num_clauses
    var_watch_list = [[] for _ in range(n)]
    for ci in range(num_clauses):
        for wp in range(2):
            var, sign = processed_clauses[ci][clause_watches[ci][wp]]
            var_watch_list[var].append((ci, wp))
    
    def is_lit_false(var, sign):
        if assignment[var] == -1:
            return False
        if sign == 1:
            return assignment[var] == 0
        else:
            return assignment[var] == 1
    
    def is_lit_true(var, sign):
        if assignment[var] == -1:
            return False
        if sign == 1:
            return assignment[var] == 1
        else:
            return assignment[var] == 0
    
    def is_clause_satisfied(ci):
        for var, sign in processed_clauses[ci]:
            if is_lit_true(var, sign):
                return True
        return False
    
    def assign(var, sign, level, clause_idx=-1, lit_idx=-1):
        """Assign variable. Returns conflict clause index or None."""
        assignment[var] = sign
        trail.append(var)
        reason[var] = clause_idx
        reason_lit_idx[var] = lit_idx
        var_activity[var] += 1.0
        
        false_sign = -sign
        
        for (ci, wp) in list(var_watch_list[var]):
            v2, s2 = processed_clauses[ci][clause_watches[ci][wp]]
            if v2 == var and s2 == false_sign:
                other_wp = 1 - wp
                v3, s3 = processed_clauses[ci][clause_watches[ci][other_wp]]
                
                if is_lit_false(v3, s3):
                    clause = processed_clauses[ci]
                    found_new_watch = False
                    for pi in range(len(clause)):
                        if pi == clause_watches[ci][0] or pi == clause_watches[ci][1]:
                            continue
                        vp, sp = clause[pi]
                        if not is_lit_false(vp, sp):
                            # Remove old watch
                            try:
                                idx = var_watch_list[var].index((ci, wp))
                                var_watch_list[var].pop(idx)
                            except ValueError:
                                pass
                            
                            new_wp = 1 if (clause_watches[ci][0] == pi) else 0
                            clause_watches[ci] = (other_wp, pi) if wp == 0 else (wp, pi)
                            var_watch_list[vp].append((ci, new_wp))
                            found_new_watch = True
                            break
                    
                    if not found_new_watch:
                        return ci
        return None
    
    def backtrack(level):
        global decision_level
        if level < len(trail_lim):
            trail_len = trail_lim[level]
        else:
            trail_len = 0
        while len(trail) > trail_len:
            v = trail.pop()
            assignment[v] = -1
            reason[v] = -1
            reason_lit_idx[v] = -1
        decision_level = level
    
    def pick_var():
        best_var = -1
        best_act = -1
        for v in range(n):
            if assignment[v] == -1:
                if var_activity[v] > best_act:
                    best_act = var_activity[v]
                    best_var = v
        if best_var == -1:
            return -1, 1
        sign = 1 if rng.random() < 0.5 else -1
        return best_var, sign
    
    # CDCL main loop
    while time.time() < cdcl_deadline:
        # Check if all clauses satisfied
        all_sat = True
        for ci in range(num_clauses):
            if not is_clause_satisfied(ci):
                all_sat = False
                break
        if all_sat:
            result = []
            for v in range(n):
                if assignment[v] == -1:
                    result.append(rng.random() < 0.5)
                else:
                    result.append(assignment[v] == 1)
            return result
        
        # Check for conflicts
        conflict_ci = -1
        for ci in range(num_clauses):
            if not is_clause_satisfied(ci):
                all_assigned = True
                for var, sign in processed_clauses[ci]:
                    if assignment[var] == -1:
                        all_assigned = False
                        break
                if all_assigned:
                    conflict_ci = ci
                    break
        
        if conflict_ci != -1:
            for var, sign in processed_clauses[conflict_ci]:
                var_activity[var] *= 1.05
            if decision_level == 0:
                break
            backtrack(decision_level - 1)
            decision_level -= 1
            continue
        
        # Check if all variables assigned
        all_assigned = True
        for v in range(n):
            if assignment[v] == -1:
                all_assigned = False
                break
        
        if all_assigned:
            all_sat = True
            for ci in range(num_clauses):
                if not is_clause_satisfied(ci):
                    all_sat = False
                    break
            if all_sat:
                return [assignment[v] == 1 for v in range(n)]
            else:
                break
        
        var, sign = pick_var()
        if var == -1:
            break
        
        trail_lim.append(len(trail))
        decision_level = len(trail_lim) - 1
        
        conflict = assign(var, sign, decision_level)
        if conflict is not None:
            for v2, s2 in processed_clauses[conflict]:
                var_activity[v2] *= 1.05
            backtrack(decision_level - 1)
            decision_level -= 1
    
    # WalkSAT fallback
    walksat_deadline = start_time + 290.0
    max_walksat_restarts = 150
    steps_per_restart = 10000
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    for restart in range(max_walksat_restarts):
        if time.time() > walksat_deadline:
            break
        
        assignment_ws = [rng.random() < 0.5 for _ in range(n)]
        
        unsat_count = 0
        for ci, clause in enumerate(processed_clauses):
            sat = False
            for var, sign in clause:
                val = assignment_ws[var]
                if sign == 1 and val:
                    sat = True
                    break
                elif sign == -1 and not val:
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        
        if unsat_count == 0:
            return assignment_ws
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment_ws[:]
        
        for step in range(steps_per_restart):
            if time.time() > walksat_deadline:
                break
            
            unsat_clauses = []
            for ci, clause in enumerate(processed_clauses):
                sat = False
                for var, sign in clause:
                    val = assignment_ws[var]
                    if sign == 1 and val:
                        sat = True
                        break
                    elif sign == -1 and not val:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                return assignment_ws
            
            unsat_count = len(unsat_clauses)
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment_ws[:]
            
            ci = rng.choice(unsat_clauses)
            clause = processed_clauses[ci]
            
            if rng.random() < 0.75:
                best_var = None
                best_gain = -float('inf')
                for var, sign in clause:
                    gain = 0
                    for ci2, sign2 in var_to_clauses[var]:
                        clause2 = processed_clauses[ci2]
                        sat_by_other = False
                        for v2, s2 in clause2:
                            if v2 == var:
                                continue
                            val2 = assignment_ws[v2]
                            if s2 == 1 and val2:
                                sat_by_other = True
                                break
                            elif s2 == -1 and not val2:
                                sat_by_other = True
                                break
                        
                        var_in_clause = False
                        var_sign = 0
                        for v2, s2 in clause2:
                            if v2 == var:
                                var_in_clause = True
                                var_sign = s2
                                break
                        
                        if not var_in_clause:
                            continue
                        
                        var_val = assignment_ws[var]
                        var_satisfies = (var_sign == 1 and var_val) or (var_sign == -1 and not var_val)
                        
                        if sat_by_other:
                            continue
                        else:
                            if var_satisfies:
                                gain -= 1
                            else:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                
                if best_var is not None:
                    assignment_ws[best_var] = not assignment_ws[best_var]
            else:
                var, sign = rng.choice(clause)
                assignment_ws[var] = not assignment_ws[var]
    
    if best_assignment is not None:
        return best_assignment
    return [False] * n
