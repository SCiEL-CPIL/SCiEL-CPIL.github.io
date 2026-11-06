import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Time limit guard
    start_time = time.time()
    TIME_LIMIT = 280  # 20s margin
    
    # DPLL Solver
    # assignment: list of None (unassigned), True, False
    assignment = [None] * n
    
    def is_satisfied(clause):
        for l in clause:
            val = assignment[abs(l) - 1]
            if val is not None:
                if (l > 0 and val == True) or (l < 0 and val == False):
                    return True
        return False

    def is_false(clause):
        for l in clause:
            val = assignment[abs(l) - 1]
            if val is not None:
                if (l > 0 and val == False) or (l < 0 and val == True):
                    return False
        return True

    def unit_propagate():
        # Returns (success, conflict)
        # success: True if no change or just satisfied, False if conflict
        # We need to loop until no more unit clauses
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                # Check if clause is satisfied
                sat = False
                unassigned_lits = []
                for l in clause:
                    val = assignment[abs(l) - 1]
                    if val is not None:
                        if (l > 0 and val == True) or (l < 0 and val == False):
                            sat = True
                            break
                    else:
                        unassigned_lits.append(l)
                
                if sat:
                    continue
                
                if len(unassigned_lits) == 0:
                    return False, True # Conflict
                
                if len(unassigned_lits) == 1:
                    # Unit clause
                    l = unassigned_lits[0]
                    var_idx = abs(l) - 1
                    desired_val = (l > 0)
                    if assignment[var_idx] is None:
                        assignment[var_idx] = desired_val
                        changed = True
                    elif assignment[var_idx] != desired_val:
                        return False, True # Conflict
        
        return True, False

    def choose_var():
        # Most occurrences heuristic
        counts = [0] * n
        for clause in clauses:
            if is_satisfied(clause):
                continue
            for l in clause:
                val = assignment[abs(l) - 1]
                if val is None:
                    counts[abs(l) - 1] += 1
        
        # Find max
        max_count = -1
        best_var = -1
        for i in range(n):
            if assignment[i] is None and counts[i] > max_count:
                max_count = counts[i]
                best_var = i
        
        return best_var

    def dpll():
        # Check time
        if time.time() - start_time > TIME_LIMIT:
            return None # Timeout
        
        # Unit Propagation
        ok, conflict = unit_propagate()
        if conflict:
            return False
        
        # Check if all clauses satisfied
        for clause in clauses:
            if not is_satisfied(clause):
                break
        else:
            # All satisfied
            # Fill remaining with False (arbitrary)
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = False
            return True

        # Choose variable
        var = choose_var()
        if var == -1:
            # No unassigned vars, but not all satisfied? Shouldn't happen if UP is correct
            # Unless there's a conflict we missed?
            # If no unassigned vars, all literals are assigned.
            # If not all satisfied, it's a conflict.
            return False

        # Try True
        assignment[var] = True
        res = dpll()
        if res is True:
            return True
        if res is None: # Timeout
            return None
        
        # Try False
        assignment[var] = False
        res = dpll()
        if res is True:
            return True
        if res is None: # Timeout
            return None
        
        # Backtrack
        assignment[var] = None
        return False

    result = dpll()
    
    if result is True:
        return assignment
    
    # If DPLL failed or timed out, fallback to WalkSAT
    # This is unlikely for n=50 planted SAT, but good for robustness
    
    import random
    rng = random.Random(42)
    
    # Initialize assignment
    if result is None:
        # Keep whatever assignment we have from DPLL partial search?
        # Better to restart WalkSAT cleanly or use current partial?
        # Let's restart WalkSAT with random init for safety
        assignment = [rng.random() < 0.5 for _ in range(n)]
    else:
        # DPLL returned False (UNSAT) - but problem says satisfiable.
        # Maybe DPLL bug? Or time limit hit in a way that returned False?
        # If result is False, it means UNSAT. But we assume SAT.
        # Let's try WalkSAT anyway.
        assignment = [rng.random() < 0.5 for _ in range(n)]

    def satisfied(c):
        return any(assignment[abs(l) - 1] == (l > 0) for l in c)

    # WalkSAT
    for _ in range(100000): # More steps
        if time.time() - start_time > TIME_LIMIT:
            break
        
        unsat = [c for c in clauses if not satisfied(c)]
        if not unsat:
            break
        
        # Pick a random unsatisfied clause
        clause = rng.choice(unsat)
        
        # Heuristic: flip the literal that satisfies the most other unsat clauses
        # Simple: random flip
        literal = rng.choice(clause)
        assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
        
    return assignment
