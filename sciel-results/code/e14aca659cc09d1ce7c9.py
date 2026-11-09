"""
Synthetic planted 3-SAT solver (50 variables)
Uses a deterministic DPLL solver with unit propagation and pure literal elimination.
"""

import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Preprocess: Convert to 0-based indices
    # Clause is a list of signed 1-based literals.
    # We will maintain a list of clauses, where each clause is a set of (var, sign) pairs.
    # var is 0-based, sign is +1 or -1.
    
    # For speed and simplicity with N=50, we can represent the state as:
    # assignment: list of 0, 1, or None (unassigned)
    # active_clauses: list of sets of (var, sign)
    
    # However, a more efficient representation for DPLL in Python for small N:
    # We can just copy the list of clauses and filter them as we assign.
    # Since N=50 and max clauses is likely around 42*50 = 2100 (ratio 4.2), 
    # copying lists of sets is cheap.
    
    # Initial clauses: list of sets of (var, sign)
    initial_clauses = []
    for c in clauses:
        cs = set()
        for l in c:
            var = abs(l) - 1
            sign = 1 if l > 0 else -1
            cs.add((var, sign))
        initial_clauses.append(cs)
        
    # We need to be careful with time.
    # DPLL with unit propagation and pure literal elimination.
    
    # To ensure we don't hit recursion limits or time out, we use an iterative approach 
    # or a recursive one with a depth limit and time check.
    # Given N=50, recursion depth is at most 50, which is safe.
    
    # Variable ordering heuristic: 
    # We'll pick the variable that appears in the most clauses (in the current subproblem).
    
    def dpll(assignment, clauses):
        """
        assignment: list of length n, values are 0, 1, or None
        clauses: list of sets of (var, sign)
        Returns True if satisfiable, updating assignment in place if so.
        """
        # 1. Check for empty clause (contradiction)
        # 2. Check if all clauses are satisfied
        
        # Filter clauses based on current assignment
        new_clauses = []
        for c in clauses:
            is_sat = False
            unsat_literals = set()
            for var, sign in c:
                if assignment[var] is None:
                    unsat_literals.add((var, sign))
                else:
                    # Check if this literal is satisfied
                    if (assignment[var] == 1 and sign == 1) or (assignment[var] == 0 and sign == -1):
                        is_sat = True
                        break
            if not is_sat:
                if len(unsat_literals) == 0:
                    # Contradiction: clause is empty and not satisfied
                    return False
                new_clauses.append(unsat_literals)
                
        # If no clauses left, it's satisfiable
        if not new_clauses:
            return True
            
        # 3. Unit Propagation
        # Find clauses with exactly one unassigned literal
        changed = True
        while changed:
            changed = False
            next_clauses = []
            for c in new_clauses:
                if len(c) == 1:
                    var, sign = next(iter(c))
                    # Assign this variable
                    if assignment[var] is not None:
                        if (assignment[var] == 1 and sign == 1) or (assignment[var] == 0 and sign == -1):
                            continue # Already satisfied
                        else:
                            return False # Conflict
                    # Assign
                    assignment[var] = 1 if sign == 1 else 0
                    changed = True
                else:
                    next_clauses.append(c)
            new_clauses = next_clauses
            
            if not new_clauses:
                return True
                
        # 4. Pure Literal Elimination
        # Find variables that appear with only one sign in the remaining clauses
        var_signs = {}
        for c in new_clauses:
            for var, sign in c:
                if var not in var_signs:
                    var_signs[var] = set()
                var_signs[var].add(sign)
                
        pure_vars = []
        for var, signs in var_signs.items():
            if len(signs) == 1:
                pure_vars.append(var)
                
        if pure_vars:
            # Assign all pure literals
            for var in pure_vars:
                sign = next(iter(var_signs[var]))
                assignment[var] = 1 if sign == 1 else 0
            # Recurse/Filter again
            return dpll(assignment, new_clauses)
            
        # 5. Choose a variable to branch on
        # Heuristic: Variable with highest frequency in remaining clauses
        var_freq = {}
        for c in new_clauses:
            for var, sign in c:
                var_freq[var] = var_freq.get(var, 0) + 1
                
        if not var_freq:
            return True
            
        # Pick max frequency
        best_var = max(var_freq, key=lambda v: var_freq[v])
        
        # Branch 1: best_var = True
        assignment[best_var] = 1
        if dpll(assignment, new_clauses):
            return True
            
        # Branch 2: best_var = False
        assignment[best_var] = 0
        if dpll(assignment, new_clauses):
            return True
            
        # Backtrack
        assignment[best_var] = None
        return False

    # Initialize assignment
    assignment = [None] * n
    
    # Time limit guard
    start_time = time.time()
    deadline = start_time + 290
    
    # Since DPLL is deterministic and N is small, we just run it.
    # However, if the instance is somehow hard (which shouldn't happen for planted SAT with N=50),
    # we might need to break out. But DPLL is complete, so it will either find a solution or prove UNSAT.
    # The problem states instances are satisfiable.
    
    # To prevent potential worst-case exponential blowup (though unlikely for N=50 planted),
    # we can add a simple time check inside the recursion if needed, but standard DPLL 
    # with unit propagation is very fast for this size.
    
    # Let's run the solver
    # We need to handle the case where the recursion might be deep or slow.
    # For N=50, it should be instantaneous.
    
    satisfiable = dpll(assignment, initial_clauses)
    
    if satisfiable:
        # Ensure all variables are assigned (in case some were left unassigned but not needed)
        # Though dpll returns True only if all clauses are satisfied, unassigned vars can remain None.
        # We should assign them arbitrarily or leave them? 
        # The interface requires a list of booleans.
        # If a variable is unassigned, it doesn't affect the satisfaction of any clause 
        # (because if it did, it would have been involved in a unit clause or pure literal step 
        # or branching step that resolved the clauses).
        # Actually, if a variable is not in any remaining clause, its value doesn't matter.
        # We can assign False or True arbitrarily.
        final_assignment = []
        for val in assignment:
            if val is None:
                final_assignment.append(False)
            else:
                final_assignment.append(val == 1)
        return final_assignment
    else:
        # Should not happen for satisfiable instances
        # Return a random assignment as fallback
        return [False] * n

