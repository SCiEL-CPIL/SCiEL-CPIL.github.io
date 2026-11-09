import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin

    rng = random.Random(42)

    # Preprocess: convert to 0-based
    # clause_vars[c] = list of (var_idx, sign)
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    for c, clause in enumerate(clauses):
        cv = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((var, sign))
            var_to_clauses[var].append(c)
        clause_vars.append(cv)

    def initialize_state(assignment):
        """
        Returns clause_unsat_count and unsat_clauses list.
        clause_unsat_count[c] = number of literals in clause c that are False.
        """
        clause_unsat_count = [0] * m
        unsat_clauses = []
        
        for c in range(m):
            count = 0
            for var, sign in clause_vars[c]:
                # Literal is unsatisfied if (assignment[var] and sign==-1) or (not assignment[var] and sign==1)
                # Which is equivalent to: assignment[var] != (sign == 1)
                if assignment[var] != (sign == 1):
                    count += 1
            clause_unsat_count[c] = count
            if count == len(clause_vars[c]):
                unsat_clauses.append(c)
                
        return clause_unsat_count, unsat_clauses

    best_assignment = None
    best_unsat_count = m + 1

    # Main search loop with restarts
    while time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        clause_unsat_count, unsat_clauses = initialize_state(assignment)
        
        if not unsat_clauses:
            return assignment

        if len(unsat_clauses) < best_unsat_count:
            best_unsat_count = len(unsat_clauses)
            best_assignment = assignment[:]

        # Local search loop
        steps = 0
        max_steps = 20000  # Reduced steps per restart to allow more restarts if needed
        
        while unsat_clauses and steps < max_steps and time.time() < deadline:
            # Pick a random unsatisfied clause
            c = rng.choice(unsat_clauses)
            
            # Verify it's still unsat (safety check)
            if clause_unsat_count[c] < len(clause_vars[c]):
                try:
                    idx = unsat_clauses.index(c)
                    unsat_clauses.pop(idx)
                except ValueError:
                    pass
                continue

            # Min-Conflicts: Find var in c that minimizes total unsat clauses after flip
            best_var = None
            best_total_unsat = m + 1
            
            for var, sign in clause_vars[c]:
                # Simulate flip of var
                delta = 0
                for c2 in var_to_clauses[var]:
                    old_count = clause_unsat_count[c2]
                    old_unsat = (old_count == len(clause_vars[c2]))
                    
                    # Find sign of var in c2
                    sign_in_c2 = 0
                    for v2, s2 in clause_vars[c2]:
                        if v2 == var:
                            sign_in_c2 = s2
                            break
                    
                    # Determine if literal for var in c2 is currently satisfied
                    # Current val = assignment[var]
                    # Lit sat if assignment[var] == (sign_in_c2 == 1)
                    is_lit_sat = (assignment[var] == (sign_in_c2 == 1))
                    
                    # If we flip var, lit sat status inverts
                    # If it was sat, it becomes unsat -> count +1
                    # If it was unsat, it becomes sat -> count -1
                    if is_lit_sat:
                        new_count = old_count + 1
                    else:
                        new_count = old_count - 1
                    
                    new_unsat = (new_count == len(clause_vars[c2]))
                    
                    if old_unsat and not new_unsat:
                        delta -= 1
                    elif not old_unsat and new_unsat:
                        delta += 1
                
                total_after = len(unsat_clauses) + delta
                if total_after < best_total_unsat:
                    best_total_unsat = total_after
                    best_var = var
                    if best_total_unsat == 0:
                        break

            if best_var is None:
                # Fallback (should not happen)
                var, _ = rng.choice(clause_vars[c])
                best_var = var

            # Flip best_var
            assignment[best_var] = not assignment[best_var]
            
            # Update state
            for c2 in var_to_clauses[best_var]:
                old_count = clause_unsat_count[c2]
                old_unsat = (old_count == len(clause_vars[c2]))
                
                sign_in_c2 = 0
                for v2, s2 in clause_vars[c2]:
                    if v2 == best_var:
                        sign_in_c2 = s2
                        break
                
                # New value is not assignment[best_var] (since we just flipped it, 
                # wait, assignment[best_var] IS the new value)
                # So we check if the literal is satisfied with the NEW value
                is_lit_sat = (assignment[best_var] == (sign_in_c2 == 1))
                
                if is_lit_sat:
                    new_count = old_count - 1
                else:
                    new_count = old_count + 1
                
                clause_unsat_count[c2] = new_count
                new_unsat = (new_count == len(clause_vars[c2]))
                
                if old_unsat and not new_unsat:
                    try:
                        idx = unsat_clauses.index(c2)
                        unsat_clauses.pop(idx)
                    except ValueError:
                        pass
                elif not old_unsat and new_unsat:
                    unsat_clauses.append(c2)
            
            steps += 1
            
            if not unsat_clauses:
                return assignment

            if steps % 50 == 0:
                if len(unsat_clauses) < best_unsat_count:
                    best_unsat_count = len(unsat_clauses)
                    best_assignment = assignment[:]

        # End of restart
        if len(unsat_clauses) < best_unsat_count:
            best_unsat_count = len(unsat_clauses)
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return [False] * n
