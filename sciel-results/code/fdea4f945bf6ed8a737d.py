import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    timeout = 325.0  # 5s margin

    # Convert to 0-based, (var, sign)
    proc_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            pc.append((v, s))
        # Remove duplicates
        pc = list(set(pc))
        if not pc:
            return [False] * n
        proc_clauses.append(pc)
        
    m = len(proc_clauses)
    if m == 0:
        return [True] * n

    # Helper: Check if assignment satisfies all clauses
    def is_sat(assignment):
        for c in proc_clauses:
            sat = False
            for v, s in c:
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    # Helper: Unit Propagation
    # Returns (ok, assignment, new_clauses)
    def unit_propagate(clauses, assignment):
        # assignment: list of None/True/False
        # clauses: list of list of (v, s)
        
        # Simplify based on current assignment
        new_clauses = []
        unit_queue = []
        
        for c in clauses:
            sc = []
            sat = False
            for v, s in c:
                if assignment[v] is not None:
                    if assignment[v] == (s == 1):
                        sat = True
                        break
                    else:
                        continue
                else:
                    sc.append((v, s))
            
            if sat:
                continue
            elif not sc:
                return False, assignment, []
            elif len(sc) == 1:
                unit_queue.append(sc[0])
            else:
                new_clauses.append(sc)
                
        # Process units
        while unit_queue:
            v, s = unit_queue.pop()
            if assignment[v] is not None:
                if assignment[v] != (s == 1):
                    return False, assignment, []
                else:
                    continue
                    
            assignment[v] = (s == 1)
            
            rem_clauses = []
            for c in new_clauses:
                sc = []
                sat = False
                for vv, ss in c:
                    if vv == v:
                        if assignment[v] == (ss == 1):
                            sat = True
                            break
                        else:
                            continue
                    else:
                        sc.append((vv, ss))
                        
                if sat:
                    continue
                elif not sc:
                    return False, assignment, []
                elif len(sc) == 1:
                    unit_queue.append(sc[0])
                else:
                    rem_clauses.append(sc)
                    
            new_clauses = rem_clauses
            
        return True, assignment, new_clauses

    # Helper: Pure Literal Elimination
    def pure_literal(clauses, assignment):
        if not clauses:
            return clauses, assignment, False
            
        pos = set()
        neg = set()
        for c in clauses:
            for v, s in c:
                if s == 1:
                    pos.add(v)
                else:
                    neg.add(v)
                    
        pure = (pos - neg) | (neg - pos)
        if not pure:
            return clauses, assignment, False
            
        changed = False
        new_clauses = []
        for c in clauses:
            sc = []
            sat = False
            for v, s in c:
                if v in pure:
                    if assignment[v] is None:
                        assignment[v] = (s == 1)
                        changed = True
                    if assignment[v] == (s == 1):
                        sat = True
                        break
                    else:
                        continue
                else:
                    sc.append((v, s))
                    
            if sat:
                continue
            else:
                new_clauses.append(sc)
                
        return new_clauses, assignment, changed

    # DPLL
    def dpll(clauses, assignment, depth, max_depth, max_nodes, node_counter):
        if node_counter[0] > max_nodes:
            return None
        node_counter[0] += 1
        
        if time.time() - start_time > timeout:
            return None
            
        if not clauses:
            return assignment.copy()
            
        ok, assignment, clauses = unit_propagate(clauses, assignment)
        if not ok:
            return None
        if not clauses:
            return assignment.copy()
            
        clauses, assignment, changed = pure_literal(clauses, assignment)
        if changed:
            ok, assignment, clauses = unit_propagate(clauses, assignment)
            if not ok:
                return None
            if not clauses:
                return assignment.copy()
                
        if not clauses:
            return assignment.copy()
            
        if depth > max_depth:
            return None
            
        # Select variable
        var_counts = {}
        for c in clauses:
            for v, s in c:
                var_counts[v] = var_counts.get(v, 0) + 1
                
        if not var_counts:
            return assignment.copy()
            
        max_c = max(var_counts.values())
        cands = [v for v, c in var_counts.items() if c == max_c]
        v = random.choice(cands)
        
        # Try True
        assignment[v] = True
        sol = dpll(clauses, assignment, depth + 1, max_depth, max_nodes, node_counter)
        if sol:
            return sol
            
        # Try False
        assignment[v] = False
        sol = dpll(clauses, assignment, depth + 1, max_depth, max_nodes, node_counter)
        if sol:
            return sol
            
        assignment[v] = None
        return None

    # WalkSAT with Phase Saving
    def walksat(assignment, phase, time_limit):
        # Build var_to_clauses
        v2c = [[] for _ in range(n)]
        for i, c in enumerate(proc_clauses):
            for v, s in c:
                v2c[v].append((i, s))
                
        # Init sat counts
        sat_counts = [0] * m
        for i, c in enumerate(proc_clauses):
            for v, s in c:
                if assignment[v] == (s == 1):
                    sat_counts[i] += 1
                    
        unsat = sum(1 for c in sat_counts if c == 0)
        best_unsat = unsat
        best_assign = assignment.copy()
        
        weight = 0.5
        max_flips = 10000
        
        for i in range(max_flips):
            if time.time() > time_limit:
                break
                
            if unsat == 0:
                return assignment, phase
                
            if unsat < best_unsat:
                best_unsat = unsat
                best_assign = assignment.copy()
                
            # Pick unsat clause
            unsat_idxs = [j for j, c in enumerate(sat_counts) if c == 0]
            if not unsat_idxs:
                return assignment, phase
            ci = random.choice(unsat_idxs)
            clause = proc_clauses[ci]
            
            # Pick var
            if random.random() < weight:
                v, s = random.choice(clause)
            else:
                best_v, best_s = None, None
                min_damage = float('inf')
                for v, s in clause:
                    damage = 0
                    for ci2, s2 in v2c[v]:
                        if ci2 != ci and sat_counts[ci2] > 0:
                            # If assignment[v] satisfies ci2, flipping breaks it
                            if assignment[v] == (s2 == 1):
                                damage += 1
                    # Tie break with phase
                    if damage < min_damage or (damage == min_damage and assignment[v] != phase[v]):
                        min_damage = damage
                        best_v, best_s = v, s
                v, s = best_v, best_s
                
            # Flip
            assignment[v] = not assignment[v]
            phase[v] = assignment[v]
            
            # Update counts
            for ci2, s2 in v2c[v]:
                if assignment[v] == (s2 == 1):
                    sat_counts[ci2] += 1
                else:
                    sat_counts[ci2] -= 1
                    
            unsat = sum(1 for c in sat_counts if c == 0)
            
        return best_assign, phase

    # Main
    # 1. Preprocess
    init_assign = [None] * n
    ok, init_assign, simp_clauses = unit_propagate(proc_clauses, init_assign)
    if not ok:
        return [False] * n
    simp_clauses, init_assign, _ = pure_literal(simp_clauses, init_assign)
    ok, init_assign, simp_clauses = unit_propagate(simp_clauses, init_assign)
    if not ok:
        return [False] * n
        
    if not simp_clauses:
        final = [init_assign[i] if init_assign[i] is not None else True for i in range(n)]
        return final

    # 2. Try DPLL
    node_counter = [0]
    sol = dpll(simp_clauses, init_assign, 0, 50, 500, node_counter)
    if sol:
        final = [sol[i] if sol[i] is not None else True for i in range(n)]
        if is_sat(final):
            return final

    # 3. WalkSAT
    random.seed(42)
    phase = [random.random() < 0.5 for _ in range(n)]
    ws_deadline = start_time + timeout - 5.0
    
    for i in range(20):
        if time.time() > ws_deadline:
            break
            
        # Initialize assignment using phase
        assign = [phase[i] for i in range(n)]
        
        if is_sat(assign):
            return assign
            
        assign, phase = walksat(assign, phase, ws_deadline)
        
        if is_sat(assign):
            return assign

    # Return best effort
    return assign
