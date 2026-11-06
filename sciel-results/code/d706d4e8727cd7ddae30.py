# EVOLVE-BLOCK-START
def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # If no clauses, any assignment works
    if not clauses:
        return [False] * n

    # Preprocess: Remove tautologies and duplicate literals
    processed_clauses = []
    for clause in clauses:
        seen = set()
        new_clause = []
        is_tautology = False
        for l in clause:
            idx = abs(l) - 1
            if idx in seen:
                is_tautology = True
                break
            seen.add(idx)
            new_clause.append(l)
        if not is_tautology and new_clause:
            processed_clauses.append(new_clause)
            
    if not processed_clauses:
        return [False] * n

    # DPLL Solver
    # assignment: list of 0 (False), 1 (True), -1 (Unassigned)
    # We use 0/1 for booleans and -1 for unassigned to make checks fast
    
    def propagate(clauses, assignment):
        """
        Perform unit propagation.
        Returns (new_clauses, assignment, success).
        If success is False, a contradiction was found.
        """
        # We iterate until no more unit clauses are found
        # To avoid infinite loops, we track if we made progress
        changed = True
        while changed:
            changed = False
            new_clauses = []
            
            # First pass: find units and propagate
            # We need to be careful to process all units that might appear
            # A simple approach: loop through clauses, if unit, assign, mark to remove.
            # Then rebuild clause list.
            
            unit_vars = {} # idx -> value (0 or 1)
            
            for c in clauses:
                # Check if clause is satisfied
                is_sat = False
                # Check if clause is unit
                unassigned_lits = []
                
                for l in c:
                    idx = abs(l) - 1
                    val = assignment[idx]
                    
                    if val != -1:
                        # Check if this literal is satisfied
                        # l > 0 means variable must be True (1)
                        # l < 0 means variable must be False (0)
                        expected_val = 1 if l > 0 else 0
                        if val == expected_val:
                            is_sat = True
                            break
                    else:
                        unassigned_lits.append(l)
                
                if is_sat:
                    continue # Clause is satisfied, remove it
                
                if len(unassigned_lits) == 0:
                    # Clause is falsified
                    return None, None, False
                
                if len(unassigned_lits) == 1:
                    # Unit clause
                    l = unassigned_lits[0]
                    idx = abs(l) - 1
                    val = 1 if l > 0 else 0
                    unit_vars[idx] = val
            
            if unit_vars:
                # Apply assignments
                for idx, val in unit_vars.items():
                    if assignment[idx] != -1:
                        if assignment[idx] != val:
                            return None, None, False # Contradiction
                    else:
                        assignment[idx] = val
                        changed = True
                
                # Rebuild clauses after assignments
                # We do this in the next iteration of the while loop
                # But we must update the 'clauses' variable for the next check
                # Actually, the loop above already identified units based on the OLD assignment.
                # We need to rebuild the clause list with the NEW assignments to find new units.
                
                # Let's rebuild clauses now
                rebuilt_clauses = []
                for c in clauses:
                    is_sat = False
                    new_c = []
                    for l in c:
                        idx = abs(l) - 1
                        val = assignment[idx]
                        if val != -1:
                            expected_val = 1 if l > 0 else 0
                            if val == expected_val:
                                is_sat = True
                                break
                        else:
                            new_c.append(l)
                    if not is_sat:
                        if not new_c:
                            return None, None, False
                        rebuilt_clauses.append(new_c)
                clauses = rebuilt_clauses
            else:
                break
        
        return clauses, assignment, True

    def pure_literal_elimination(clauses, assignment):
        """
        Apply pure literal elimination.
        Returns (new_clauses, assignment, changed).
        """
        pos_occurs = set()
        neg_occurs = set()
        
        for c in clauses:
            for l in c:
                idx = abs(l) - 1
                if assignment[idx] != -1:
                    continue
                if l > 0:
                    pos_occurs.add(idx)
                else:
                    neg_occurs.add(idx)
        
        pure_vars = {}
        for idx in pos_occurs:
            if idx not in neg_occurs:
                pure_vars[idx] = 1
        for idx in neg_occurs:
            if idx not in pos_occurs:
                pure_vars[idx] = 0
        
        if not pure_vars:
            return clauses, assignment, False
        
        # Apply pure literals
        for idx, val in pure_vars.items():
            assignment[idx] = val
        
        # Rebuild clauses
        new_clauses = []
        for c in clauses:
            is_sat = False
            new_c = []
            for l in c:
                idx = abs(l) - 1
                val = assignment[idx]
                if val != -1:
                    expected_val = 1 if l > 0 else 0
                    if val == expected_val:
                        is_sat = True
                        break
                else:
                    new_c.append(l)
            if not is_sat:
                if not new_c:
                    return None, None, True # Contradiction (shouldn't happen with pure lit if consistent)
                new_clauses.append(new_c)
                
        return new_clauses, assignment, True

    def choose_variable(clauses, assignment):
        """
        Heuristic to choose a variable to branch on.
        Picks the variable that appears most frequently in the remaining clauses.
        """
        counts = {}
        for c in clauses:
            for l in c:
                idx = abs(l) - 1
                if assignment[idx] == -1:
                    counts[idx] = counts.get(idx, 0) + 1
        
        if not counts:
            return -1
            
        # Pick the variable with the highest count
        # Tie-breaking: lower index
        max_count = -1
        best_var = -1
        for idx, count in counts.items():
            if count > max_count:
                max_count = count
                best_var = idx
            elif count == max_count and idx < best_var:
                best_var = idx
                
        return best_var

    def dpll(clauses, assignment):
        """
        Recursive DPLL.
        Returns assignment list if satisfiable, None otherwise.
        """
        # 1. Unit Propagation
        clauses, assignment, success = propagate(clauses, assignment)
        if not success:
            return None
        
        if not clauses:
            # All clauses satisfied
            # Fill remaining unassigned variables with False (0)
            for i in range(n):
                if assignment[i] == -1:
                    assignment[i] = 0
            return assignment
        
        # 2. Pure Literal Elimination
        clauses, assignment, changed = pure_literal_elimination(clauses, assignment)
        if changed:
            # Recurse immediately to handle any new units or empties
            return dpll(clauses, assignment)
            
        # 3. Branching
        var = choose_variable(clauses, assignment)
        if var == -1:
            # No unassigned variables in clauses, but clauses exist?
            # This implies a contradiction was missed or logic error.
            # However, if clauses exist and no unassigned vars, they must be falsified.
            return None
            
        # Try True (1)
        assignment[var] = 1
        result = dpll(clauses, assignment)
        if result is not None:
            return result
        
        # Try False (0)
        assignment[var] = 0
        result = dpll(clauses, assignment)
        if result is not None:
            return result
        
        # Backtrack
        assignment[var] = -1
        return None

    initial_assignment = [-1] * n
    result = dpll(processed_clauses, initial_assignment)
    
    if result is None:
        # Fallback for unsatisfiable (should not happen per task description)
        return [False] * n
        
    # Convert -1/0/1 to False/True
    # 0 -> False, 1 -> True
    return [bool(x) for x in result]
# EVOLVE-BLOCK-END
