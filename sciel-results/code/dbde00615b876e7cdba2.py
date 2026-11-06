"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-process clauses: convert to 0-based, remove duplicates/tautologies
    processed_clauses = []
    for c in clauses:
        seen_vars = set()
        new_c = []
        is_tautology = False
        for l in c:
            var_idx = abs(l) - 1
            if var_idx in seen_vars:
                # Check for tautology: if we already added the opposite literal
                # This check is slightly expensive, but n is small
                # Simpler: just check if -l is in new_c
                if -l in new_c:
                    is_tautology = True
                    break
            seen_vars.add(var_idx)
            new_c.append(l)
        if is_tautology:
            continue
        if len(new_c) == 0:
            return [False] * n # Empty clause -> UNSAT, but task says SAT
        processed_clauses.append(new_c)
        
    clauses = processed_clauses

    # DPLL Solver with Unit Propagation
    # To avoid recursion depth issues and overhead, we use a stack-based approach 
    # or just recursive with increased limit. n=50 is small enough for recursion.
    sys.setrecursionlimit(10000)
    
    def dpll(clauses, assignment):
        # 1. Unit Propagation
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for c in clauses:
                sat = False
                unit_lit = None
                unassigned_count = 0
                falsified = True
                
                for l in c:
                    var_idx = abs(l) - 1
                    if assignment[var_idx] is None:
                        unassigned_count += 1
                        falsified = False
                        if unassigned_count == 1:
                            unit_lit = l
                        else:
                            unit_lit = None
                    else:
                        if assignment[var_idx] == (l > 0):
                            sat = True
                            break
                        # else: literal is false
                
                if sat:
                    continue
                
                if unassigned_count == 0:
                    # All literals false -> UNSAT
                    return None
                
                if unassigned_count == 1:
                    # Unit clause
                    var_idx = abs(unit_lit) - 1
                    val = unit_lit > 0
                    if assignment[var_idx] is not None and assignment[var_idx] != val:
                        return None # Conflict
                    if assignment[var_idx] is None:
                        assignment[var_idx] = val
                        changed = True
                    # Note: We don't add this clause to new_clauses because it's satisfied
                    # by the assignment we just made.
                else:
                    new_clauses.append(c)
            
            clauses = new_clauses
            
        # 2. Check for empty clause (UNSAT)
        # If any clause has 0 unassigned literals and wasn't satisfied, it's UNSAT.
        # The unit propagation loop above handles this: if unassigned_count==0 and not sat, returns None.
        # So if we get here, no UNSAT detected by unit prop.
        
        # 3. If no clauses, SAT
        if not clauses:
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = False
            return assignment

        # 4. Branching
        # Heuristic: Pick variable with highest occurrence count in remaining clauses
        var_counts = [0] * n
        for c in clauses:
            for l in c:
                var_idx = abs(l) - 1
                if assignment[var_idx] is None:
                    var_counts[var_idx] += 1
        
        max_var = -1
        max_count = -1
        for i in range(n):
            if assignment[i] is None and var_counts[i] > max_count:
                max_count = var_counts[i]
                max_var = i
        
        if max_var == -1:
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = False
            return assignment

        # Try True
        assignment[max_var] = True
        # Pass copies to avoid side effects in backtracking
        result = dpll(clauses[:], assignment[:])
        if result is not None:
            return result
            
        # Try False
        assignment[max_var] = False
        result = dpll(clauses[:], assignment[:])
        if result is not None:
            return result
            
        # Backtrack
        assignment[max_var] = None
        return None

    initial_assignment = [None] * n
    result = dpll(clauses, initial_assignment)
    
    if result is None:
        # Fallback: Random assignment (should not happen for satisfiable instances)
        return [False] * n
        
    return result
# EVOLVE-BLOCK-END
