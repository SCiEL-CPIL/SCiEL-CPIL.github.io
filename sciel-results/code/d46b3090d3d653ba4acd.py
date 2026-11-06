import time
import sys

def solve(instance):
    """
    Solves a 3-SAT instance using DPLL with Unit Propagation and Pure Literal Elimination.
    Returns a list of booleans representing the satisfying assignment.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 280 seconds to leave margin for overhead
    deadline = time.time() + 280
    
    # Preprocessing: Normalize clauses
    # Remove tautologies (clause has both x and -x) and duplicates
    processed_clauses = []
    seen = set()
    for c in clauses:
        # Check for tautology
        is_taut = False
        lit_set = set()
        for l in c:
            if -l in lit_set:
                is_taut = True
                break
            lit_set.add(l)
        if is_taut:
            continue
        
        # Check for duplicates (frozenset of literals)
        key = frozenset(c)
        if key in seen:
            continue
        seen.add(key)
        processed_clauses.append(c)
        
    clauses = processed_clauses
    
    # If no clauses, any assignment works
    if not clauses:
        return [True] * n

    # DPLL Solver
    # State: 
    # assignment: list of 0 (unassigned), 1 (True), -1 (False)
    # active_clauses: list of clauses not yet satisfied
    
    def unit_propagate(assignment, active_clauses):
        """
        Performs unit propagation.
        Returns (success, new_assignment, new_active_clauses, contradiction_flag)
        """
        changed = True
        while changed:
            changed = False
            new_active = []
            for c in active_clauses:
                # Check if clause is satisfied
                sat = False
                unit_lit = None
                unassigned_count = 0
                for l in c:
                    var = abs(l)
                    val = assignment[var-1]
                    if val == 1:
                        if l > 0:
                            sat = True
                            break
                    elif val == -1:
                        if l < 0:
                            sat = True
                            break
                    else:
                        unassigned_count += 1
                        unit_lit = l
                
                if sat:
                    continue # Clause satisfied, remove it
                
                if unassigned_count == 0:
                    return False, assignment, [], True # Contradiction: clause is falsified
                
                if unassigned_count == 1:
                    # Unit clause, force assignment
                    var = abs(unit_lit)
                    target_val = 1 if unit_lit > 0 else -1
                    if assignment[var-1] == 0:
                        assignment[var-1] = target_val
                        changed = True
                    elif assignment[var-1] != target_val:
                        return False, assignment, [], True # Contradiction
                    # Clause becomes satisfied by this assignment, so don't add to new_active
                    continue
                
                new_active.append(c)
            active_clauses = new_active
        return True, assignment, active_clauses, False

    def pure_literal_eliminate(assignment, active_clauses):
        """
        Assigns pure literals.
        Returns (new_assignment, new_active_clauses)
        """
        # Determine which literals are pure
        # sign_count[var][sign] where sign is 1 for positive, -1 for negative
        # Actually, just track if a variable appears only positively or only negatively in active clauses
        
        var_pos = [0] * n
        var_neg = [0] * n
        
        for c in active_clauses:
            for l in c:
                var = abs(l) - 1
                if l > 0:
                    var_pos[var] += 1
                else:
                    var_neg[var] += 1
        
        changed = True
        while changed:
            changed = False
            new_active = []
            for c in active_clauses:
                sat = False
                for l in c:
                    var = abs(l) - 1
                    val = assignment[var]
                    if val == 1 and l > 0:
                        sat = True
                        break
                    if val == -1 and l < 0:
                        sat = True
                        break
                if sat:
                    continue
                new_active.append(c)
            
            active_clauses = new_active
            
            # Now check for pure literals in the remaining active clauses
            var_pos = [0] * n
            var_neg = [0] * n
            for c in active_clauses:
                for l in c:
                    var = abs(l) - 1
                    if l > 0:
                        var_pos[var] += 1
                    else:
                        var_neg[var] += 1
            
            for var in range(n):
                if assignment[var] == 0:
                    if var_pos[var] > 0 and var_neg[var] == 0:
                        # Pure positive
                        assignment[var] = 1
                        changed = True
                    elif var_neg[var] > 0 and var_pos[var] == 0:
                        # Pure negative
                        assignment[var] = -1
                        changed = True
                        
            if changed:
                # Re-filter active clauses because assignment changed
                # This loop structure handles it by restarting the while loop
                pass
        
        return assignment, active_clauses

    def select_variable(assignment, active_clauses):
        """
        Selects the next variable to branch on using MRV (Minimum Remaining Values).
        If tie, use Max Degree.
        """
        if not active_clauses:
            return -1 # No unassigned variables in active clauses, or all satisfied
        
        # Count unassigned occurrences per variable
        unassigned_count = [0] * n
        degree = [0] * n
        
        for c in active_clauses:
            for l in c:
                var = abs(l) - 1
                if assignment[var] == 0:
                    unassigned_count[var] += 1
                    degree[var] += 1
        
        best_var = -1
        min_count = n + 1
        max_deg = -1
        
        for var in range(n):
            if assignment[var] == 0 and unassigned_count[var] > 0:
                if unassigned_count[var] < min_count or (unassigned_count[var] == min_count and degree[var] > max_deg):
                    min_count = unassigned_count[var]
                    max_deg = degree[var]
                    best_var = var
        
        return best_var

    def dpll(assignment, active_clauses):
        """
        Recursive DPLL.
        Returns True if solution found, False otherwise.
        Modifies assignment in place if successful.
        """
        if time.time() > deadline:
            return False # Timeout, treat as failure for this branch
        
        # 1. Check for contradiction (empty clause)
        # 2. Check for success (no active clauses)
        if not active_clauses:
            return True
        
        # 3. Unit Propagation
        ok, assignment, active_clauses, contradiction = unit_propagate(assignment, active_clauses)
        if contradiction:
            return False
        if not active_clauses:
            return True
            
        # 4. Pure Literal Elimination
        assignment, active_clauses = pure_literal_eliminate(assignment, active_clauses)
        if not active_clauses:
            return True
            
        # 5. Select Variable
        var = select_variable(assignment, active_clauses)
        if var == -1:
            # No unassigned variables in active clauses? 
            # This implies all variables in active clauses are assigned, but clauses aren't satisfied?
            # This should have been caught by unit propagation (unassigned_count 0 -> contradiction)
            # Unless a clause has 0 literals? No, 3-SAT.
            return False
        
        # 6. Branch
        # Try True first
        assignment[var] = 1
        if dpll(assignment, active_clauses):
            return True
            
        # Try False
        assignment[var] = -1
        if dpll(assignment, active_clauses):
            return True
            
        # Backtrack
        assignment[var] = 0
        return False

    # Initialize
    assignment = [0] * n
    
    # Start DPLL
    if dpll(assignment, clauses):
        # Convert 1/-1 to True/False
        return [val == 1 for val in assignment]
    else:
        # If DPLL fails (timeout or unsat), fallback to a quick random assignment 
        # or the current partial assignment filled randomly.
        # Since the problem is planted satisfiable, DPLL should succeed.
        # If it times out, we return a random assignment as a best-effort guess.
        import random
        rng = random.Random(42)
        return [rng.random() < 0.5 for _ in range(n)]

