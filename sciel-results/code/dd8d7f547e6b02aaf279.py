"""Rewritten program using a complete DPLL solver with unit propagation and pure literal elimination."""
import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Convert clauses to sets of signed integers for faster manipulation
    # Also handle potential duplicates or tautologies early
    clean_clauses = []
    for c in clauses:
        s = set(c)
        # Check for tautology: if x and -x are in the clause, it's always true
        is_tautology = False
        for lit in s:
            if -lit in s:
                is_tautology = True
                break
        if not is_tautology:
            clean_clauses.append(s)
            
    # If no clauses, any assignment works
    if not clean_clauses:
        return [False] * n

    # DPLL Solver
    # assignment: dict var_index (1-based) -> bool
    assignment = {}
    
    def propagate(clauses, assignment):
        """
        Perform unit propagation and pure literal elimination.
        Returns (new_clauses, new_assignment, satisfied)
        satisfied is True if all clauses are satisfied.
        If a conflict is found (empty clause), returns None.
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            
            # 1. Unit Propagation
            unit_vars = {}
            non_unit_clauses = []
            
            for clause in clauses:
                # Check if clause is already satisfied
                satisfied = False
                # Check if clause is empty (conflict)
                if len(clause) == 0:
                    return None, None, False # Conflict
                
                # Determine if it's a unit clause
                unassigned_lits = [l for l in clause if abs(l) not in assignment]
                
                if len(unassigned_lits) == 0:
                    # All literals assigned. Check if satisfied.
                    # We already know it's not empty.
                    # If it's not satisfied, it's a conflict.
                    # But wait, if all are assigned and it's not satisfied, that means all are False.
                    # This is a conflict.
                    # Actually, if we maintain the invariant that we remove satisfied clauses, 
                    # we shouldn't have this. But let's be safe.
                    is_sat = any(assignment[abs(l)] == (l > 0) for l in clause)
                    if not is_sat:
                        return None, None, False # Conflict
                    # Else it's satisfied, discard it
                    continue
                    
                if len(unassigned_lits) == 1:
                    lit = unassigned_lits[0]
                    var = abs(lit)
                    val = lit > 0
                    # If var is already assigned, check consistency
                    if var in assignment:
                        if assignment[var] != val:
                            return None, None, False # Conflict
                        # Else it's satisfied, discard clause
                        continue
                    # Assign the variable
                    assignment[var] = val
                    changed = True
                    # Discard this clause as it is now satisfied
                    continue
                else:
                    # Keep the clause, but we might want to remove assigned literals?
                    # For efficiency in pure literal check, it's better to keep full clauses
                    # or just track unassigned. Let's keep the set but filter assigned later?
                    # Actually, standard DPLL keeps the full clause and checks assignment on the fly.
                    # But for pure literal elimination, we need to see all occurrences.
                    non_unit_clauses.append(clause)
            
            clauses = non_unit_clauses
            
            if not clauses:
                return clauses, assignment, True # All satisfied
                
            # 2. Pure Literal Elimination
            # Count occurrences of positive and negative literals for unassigned vars
            pos_count = {i: 0 for i in range(1, n+1)}
            neg_count = {i: 0 for i in range(1, n+1)}
            
            for clause in clauses:
                for lit in clause:
                    var = abs(lit)
                    if var in assignment:
                        continue # Already assigned, handled above
                    if lit > 0:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1
                        
            pure_found = False
            for var in range(1, n+1):
                if var in assignment:
                    continue
                if pos_count[var] > 0 and neg_count[var] == 0:
                    assignment[var] = True
                    pure_found = True
                    changed = True
                elif neg_count[var] > 0 and pos_count[var] == 0:
                    assignment[var] = False
                    pure_found = True
                    changed = True
            
            if pure_found:
                # Need to re-evaluate clauses because assignments changed
                # The loop `while changed` will handle this
                continue
                
            # If no change, break
            if not changed:
                break
                
        return clauses, assignment, False

    def dpll(clauses, assignment):
        # Check for conflict (empty clause)
        if any(len(c) == 0 for c in clauses):
            return False
            
        # Check if all satisfied
        all_sat = True
        for c in clauses:
            sat = False
            for l in c:
                if abs(l) in assignment:
                    if assignment[abs(l)] == (l > 0):
                        sat = True
                        break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return True

        # Propagate
        res = propagate(clauses, assignment)
        if res is None:
            return False
        new_clauses, new_assignment, is_sat = res
        if is_sat:
            return True
        clauses = new_clauses
        assignment = new_assignment
        
        if not clauses:
            return True

        # Variable Selection
        # Choose variable with highest frequency in remaining clauses
        var_freq = {}
        for c in clauses:
            for l in c:
                v = abs(l)
                if v not in assignment:
                    var_freq[v] = var_freq.get(v, 0) + 1
        
        if not var_freq:
            return True # All assigned, and we didn't return conflict/sat earlier? 
                        # Actually if all assigned and not sat, we would have caught it in 'all_sat' check or propagate.
                        # But if clauses remain, it means they are not satisfied. 
                        # Wait, if all vars assigned, and clause not satisfied, it's a conflict.
                        # The 'all_sat' check at the top handles this.
                        # If we reach here, there are unassigned vars.
                        
        var = max(var_freq, key=var_freq.get)
        
        # Branch: Try True
        assignment[var] = True
        # Filter clauses
        # We can just pass the same clauses and let propagate handle it, 
        # but it's faster to filter satisfied clauses and remove var from others
        new_clauses_true = []
        for c in clauses:
            # Remove var and -var
            new_c = set(c)
            new_c.discard(var)
            new_c.discard(-var)
            if len(new_c) == 0:
                continue # Satisfied
            new_clauses_true.append(new_c)
        
        if dpll(new_clauses_true, assignment):
            return True
            
        # Branch: Try False
        assignment[var] = False
        new_clauses_false = []
        for c in clauses:
            new_c = set(c)
            new_c.discard(var)
            new_c.discard(-var)
            if len(new_c) == 0:
                continue # Satisfied
            new_clauses_false.append(new_c)
        
        if dpll(new_clauses_false, assignment):
            return True
            
        return False

    # Start DPLL
    # Make a deep copy of initial clauses to avoid mutation issues in backtracking?
    # My dpll function creates new lists for branches, so it's safe.
    # But `assignment` is a dict passed by reference. 
    # In the branch calls, I modify `assignment` directly.
    # I need to save/restore assignment state or pass a copy.
    # Let's fix the dpll function to use a local copy of assignment for branching.
    
    def dpll_safe(clauses, assignment):
        if any(len(c) == 0 for c in clauses):
            return False
            
        all_sat = True
        for c in clauses:
            sat = False
            for l in c:
                if abs(l) in assignment:
                    if assignment[abs(l)] == (l > 0):
                        sat = True
                        break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return True

        res = propagate(clauses, assignment)
        if res is None:
            return False
        new_clauses, new_assignment, is_sat = res
        if is_sat:
            return True
        clauses = new_clauses
        assignment = new_assignment
        
        if not clauses:
            return True

        var_freq = {}
        for c in clauses:
            for l in c:
                v = abs(l)
                if v not in assignment:
                    var_freq[v] = var_freq.get(v, 0) + 1
        
        if not var_freq:
            return True

        var = max(var_freq, key=var_freq.get)
        
        # Branch True
        assignment_copy = dict(assignment)
        assignment_copy[var] = True
        new_clauses_true = []
        for c in clauses:
            new_c = set(c)
            new_c.discard(var)
            new_c.discard(-var)
            if len(new_c) == 0:
                continue
            new_clauses_true.append(new_c)
        
        if dpll_safe(new_clauses_true, assignment_copy):
            # Update the main assignment dict with the solution
            assignment.update(assignment_copy)
            return True
            
        # Branch False
        assignment_copy = dict(assignment)
        assignment_copy[var] = False
        new_clauses_false = []
        for c in clauses:
            new_c = set(c)
            new_c.discard(var)
            new_c.discard(-var)
            if len(new_c) == 0:
                continue
            new_clauses_false.append(new_c)
        
        if dpll_safe(new_clauses_false, assignment_copy):
            assignment.update(assignment_copy)
            return True
            
        return False

    success = dpll_safe(clean_clauses, assignment)
    
    if success:
        # Construct final list
        result = []
        for i in range(1, n + 1):
            if i in assignment:
                result.append(assignment[i])
            else:
                # Should not happen if complete, but default to False
                result.append(False)
        return result
    else:
        # Should not happen for satisfiable instances, but return a default
        # or try a simple random assignment as fallback
        return [False] * n

