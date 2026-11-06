import numpy as np
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    C = len(clauses)
    
    # Preprocessing: build efficient data structures
    # clause_vars[i] = list of (var_idx, sign) for clause i
    # var_to_clauses[v] = list of (clause_idx, pos) for variable v
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        cv = []
        for j, lit in enumerate(clause):
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((v, sign))
            var_to_clauses[v].append((i, j))
        clause_vars.append(cv)
    
    # Precompute for each variable and each clause it appears in,
    # the other two literals (var, sign) to speed up delta computation
    # var_other_lits[v] = dict: clause_idx -> (other_var1, other_sign1, other_var2, other_sign2)
    var_other_lits = [{} for _ in range(n)]
    for v in range(n):
        for (ci, pos) in var_to_clauses[v]:
            others = []
            for p in range(3):
                if p != pos:
                    others.append(clause_vars[ci][p])
            var_other_lits[v][ci] = (others[0][0], others[0][1], others[1][0], others[1][1])
    
    # Fast evaluation: returns (num_unsat, unsat_list, unsat_set)
    def fast_evaluate(assignment):
        num_unsat = 0
        unsat_clauses = []
        unsat_set = set()
        for c in range(C):
            cv = clause_vars[c]
            sat = False
            for (v, s) in cv:
                val = assignment[v]
                if (s == 1 and val) or (s == -1 and not val):
                    sat = True
                    break
            if not sat:
                num_unsat += 1
                unsat_clauses.append(c)
                unsat_set.add(c)
        return num_unsat, unsat_clauses, unsat_set
    
    # Compute delta for flipping variable v given current assignment
    def compute_delta(v, assignment, unsat_set):
        delta = 0
        for (ci, pos) in var_to_clauses[v]:
            o1v, o1s, o2v, o2s = var_other_lits[v][ci]
            
            o1_sat = (o1s == 1 and assignment[o1v]) or (o1s == -1 and not assignment[o1v])
            o2_sat = (o2s == 1 and assignment[o2v]) or (o2s == -1 and not assignment[o2v])
            
            if o1_sat and o2_sat:
                if ci in unsat_set:
                    delta += 1
            else:
                if not o1_sat and not o2_sat:
                    if ci not in unsat_set:
                        delta -= 1
        return delta
    
    # WalkSAT with min-conflicts style heuristic
    def walk_sat(assignment, num_unsat, unsat_clauses, unsat_set, rng, max_steps, deadline):
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            if num_unsat == 0:
                return True
            
            # Pick a random unsatisfied clause
            c = unsat_clauses[rng.randrange(num_unsat)]
            cv = clause_vars[c]
            
            # For each variable in this clause, compute delta
            best_v = -1
            best_score = -1e9
            for (v, s) in cv:
                delta = compute_delta(v, assignment, unsat_set)
                noise = rng.random() * 0.5
                score = delta + noise
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                break
            
            # Flip best_v
            assignment[best_v] = not assignment[best_v]
            
            # Update num_unsat and unsat tracking incrementally
            affected = []
            for (ci, pos) in var_to_clauses[best_v]:
                affected.append(ci)
            
            new_num_unsat = num_unsat
            new_unsat_clauses = []
            new_unsat_set = set()
            
            # Process all clauses: check affected ones, keep others
            for ci in range(C):
                if ci in unsat_set:
                    # Was unsat, check if still unsat
                    if ci in affected:
                        cv2 = clause_vars[ci]
                        sat = False
                        for (v2, s2) in cv2:
                            val = assignment[v2]
                            if (s2 == 1 and val) or (s2 == -1 and not val):
                                sat = True
                                break
                        if not sat:
                            new_unsat_clauses.append(ci)
                            new_unsat_set.add(ci)
                    else:
                        new_unsat_clauses.append(ci)
                        new_unsat_set.add(ci)
                else:
                    # Was sat, check if became unsat (only if affected)
                    if ci in affected:
                        cv2 = clause_vars[ci]
                        sat = False
                        for (v2, s2) in cv2:
                            val = assignment[v2]
                            if (s2 == 1 and val) or (s2 == -1 and not val):
                                sat = True
                                break
                        if not sat:
                            new_unsat_clauses.append(ci)
                            new_unsat_set.add(ci)
            
            unsat_clauses = new_unsat_clauses
            unsat_set = new_unsat_set
            num_unsat = len(unsat_clauses)
            
            if num_unsat == 0:
                return True
        
        return False
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    rng = random.Random(42)
    
    best_num_unsat = C + 1
    best_assignment = None
    
    # Multiple restarts with increasing step limits
    restart = 0
    while time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Evaluate
        num_unsat, unsat_clauses, unsat_set = fast_evaluate(assignment)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment = assignment[:]
        
        # WalkSAT with adaptive step limit
        base_steps = 5000 + restart * 1000
        max_steps = min(base_steps, 100000)
        
        solved = walk_sat(assignment, num_unsat, unsat_clauses, unsat_set, rng, max_steps, deadline)
        
        if solved:
            return assignment
        
        restart += 1
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
