# EVOLVE-BLOCK-START
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Convert to 0-based internal representation
    # Clause is a list of (var, sign) where sign is 1 for positive, -1 for negative
    # var is 0..n-1
    # We use a list of lists for clauses.
    # Each literal is (var, is_pos). 
    # For efficiency in unit propagation, we can track literal counts.
    
    m = len(clauses)
    if m == 0:
        return [True] * n
        
    # Precompute clause data
    # We will represent the problem state in a way that supports efficient backtracking.
    # However, for n=50, a simple recursive DPLL with copying the clause list might be too slow if it branches a lot.
    # Better: Use a trail-based approach or just simple recursion with immutable clause lists if branching is limited.
    # Given n=50, simple recursion with clause filtering is often fast enough for planted SATs.
    
    # Let's use a more efficient structure:
    # Clauses as lists of signed integers (1-based, +/-).
    # We'll implement a DPLL solver.
    
    def dpll(clauses, assignment):
        # assignment: dict var -> bool
        # clauses: list of lists of signed ints
        
        # 1. Check for empty clause (conflict)
        # 2. Unit propagation
        # 3. Pure literal elimination
        # 4. Variable selection and branching
        
        # We'll do unit propagation first, which may simplify clauses to empty (sat) or empty list (unsat).
        
        changed = True
        while changed:
            changed = False
            new_clauses = []
            
            # First pass: handle unit clauses and empty clauses
            # We need to be careful with order.
            
            # Let's collect unit assignments
            unit_assignments = {}
            
            for c in clauses:
                if len(c) == 0:
                    return None # Conflict
                if len(c) == 1:
                    lit = c[0]
                    var = abs(lit)
                    val = lit > 0
                    if var in unit_assignments:
                        if unit_assignments[var] != val:
                            return None # Conflict
                    else:
                        unit_assignments[var] = val
            
            if not unit_assignments:
                # Check for pure literals? 
                # Pure literal elimination: if a variable appears only as positive, set to True.
                # This is an optimization.
                var_signs = {}
                for c in clauses:
                    for lit in c:
                        var = abs(lit)
                        sign = 1 if lit > 0 else -1
                        if var not in var_signs:
                            var_signs[var] = set()
                        var_signs[var].add(sign)
                
                for var, signs in var_signs.items():
                    if var in assignment:
                        continue
                    if len(signs) == 1:
                        if 1 in signs:
                            unit_assignments[var] = True
                        else:
                            unit_assignments[var] = False
                if unit_assignments:
                    changed = True
            
            if not unit_assignments:
                break
            
            # Apply assignments
            for var, val in unit_assignments.items():
                if var in assignment:
                    if assignment[var] != val:
                        return None
                else:
                    assignment[var] = val
            
            # Filter clauses
            for c in clauses:
                new_c = []
                for lit in c:
                    var = abs(lit)
                    if var in assignment:
                        if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                            # Clause is satisfied
                            new_c = None
                            break
                        else:
                            # Literal is false, remove it
                            continue
                    else:
                        new_c.append(lit)
                
                if new_c is None:
                    continue # Satisfied, drop clause
                else:
                    new_clauses.append(new_c)
            
            clauses = new_clauses
            
            if not clauses:
                return assignment # All clauses satisfied
        
        if not clauses:
            return assignment
        
        # Select variable to branch on
        # Heuristic: pick variable with highest occurrence count in remaining clauses
        var_count = {}
        for c in clauses:
            for lit in c:
                var = abs(lit)
                var_count[var] = var_count.get(var, 0) + 1
        
        if not var_count:
            return assignment
            
        # Pick variable with max count
        best_var = max(var_count, key=var_count.get)
        
        # Branch: try True first, then False
        # Try True
        assignment[best_var] = True
        # Filter clauses for this assignment
        new_clauses_true = []
        for c in clauses:
            new_c = []
            sat = False
            for lit in c:
                var = abs(lit)
                if var == best_var:
                    if lit > 0:
                        sat = True
                        break
                    else:
                        continue
                else:
                    new_c.append(lit)
            if sat:
                continue
            new_clauses_true.append(new_c)
        
        res = dpll(new_clauses_true, assignment)
        if res is not None:
            return res
            
        # Backtrack: try False
        assignment[best_var] = False
        new_clauses_false = []
        for c in clauses:
            new_c = []
            sat = False
            for lit in c:
                var = abs(lit)
                if var == best_var:
                    if lit < 0:
                        sat = True
                        break
                    else:
                        continue
                else:
                    new_c.append(lit)
            if sat:
                continue
            new_clauses_false.append(new_c)
        
        res = dpll(new_clauses_false, assignment)
        if res is not None:
            return res
            
        return None

    # Initial call
    # Clauses are 1-based signed ints
    initial_clauses = [list(c) for c in clauses]
    assignment = {}
    
    # Increase recursion limit just in case, though n=50 is small
    sys.setrecursionlimit(10000)
    
    result = dpll(initial_clauses, assignment)
    
    if result is None:
        # Should not happen for satisfiable instances, but return default
        return [False] * n
        
    # Convert dict to list
    final_assignment = [result.get(i, False) for i in range(1, n+1)]
    return final_assignment
# EVOLVE-BLOCK-END
