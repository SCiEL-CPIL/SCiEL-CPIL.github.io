"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict


def solve(instance):
    """
    Solve a planted satisfiable 3-CNF instance.
    Uses a DPLL solver with unit propagation, pure literal elimination,
    and a hybrid fallback to WalkSAT if DPLL times out.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: use most of the 330s budget, leave safety margin
    deadline = time.time() + 325.0
    
    # Preprocess: convert to 0-based, keep track
    # We'll work with a mutable structure for DPLL
    
    def dpll(clauses, assignment):
        """
        DPLL with unit propagation and pure literal elimination.
        clauses: list of lists of signed 0-based literals (or 1-based, we'll use 1-based for consistency)
        assignment: dict or list; we'll use a list of length n with None/True/False
        Returns: assignment if satisfiable, None if unsatisfiable or timeout
        """
        # We'll use a recursive approach with explicit stack to avoid recursion limits
        # But for 50 vars, recursion is fine. Let's use recursion with a time check.
        
        # Convert clauses to use 0-based indices for faster lookup
        # Actually, let's keep 1-based in the input but convert internally
        
        # State:
        # - current clauses (list of lists of signed literals, 1-based)
        # - assignment: list of length n, each element is None, True, or False
        # - we track which variables are assigned
        
        # For efficiency, we'll maintain:
        # - clauses as list of lists of (var, sign) where var is 0-based, sign is +1 or -1
        # - assignment as list of bools or None
        
        # Let's restructure: convert input clauses to 0-based
        def to_internal(clauses_in):
            internal = []
            for c in clauses_in:
                ic = []
                for lit in c:
                    var = abs(lit) - 1  # 0-based
                    sign = 1 if lit > 0 else -1
                    ic.append((var, sign))
                internal.append(ic)
            return internal
        
        clauses_int = to_internal(clauses)
        
        # assignment: list of None/True/False, length n
        assign = [None] * n
        
        def is_satisfied(clause, a):
            for var, sign in clause:
                if a[var] is not None:
                    if (sign == 1 and a[var]) or (sign == -1 and not a[var]):
                        return True
            return False
        
        def unit_propagate(clauses_int, a):
            """
            Perform unit propagation. Returns (success, new_assignment, remaining_clauses).
            If a clause becomes empty, return (False, None, None).
            """
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses_int:
                    # Check if clause is satisfied
                    sat = False
                    # Check if clause is unit
                    unassigned = []
                    for var, sign in clause:
                        if a[var] is not None:
                            if (sign == 1 and a[var]) or (sign == -1 and not a[var]):
                                sat = True
                                break
                        else:
                            unassigned.append((var, sign))
                    if sat:
                        continue
                    if len(unassigned) == 0:
                        # Empty clause: contradiction
                        return (False, None, None)
                    if len(unassigned) == 1:
                        # Unit clause
                        var, sign = unassigned[0]
                        a[var] = (sign == 1)
                        changed = True
                        # This clause is now satisfied, don't add it
                    else:
                        new_clauses.append(unassigned)
                clauses_int = new_clauses
            return (True, a, clauses_int)
        
        def pure_literal_elimination(clauses_int, a):
            """
            Eliminate pure literals. Returns (new_assignment, remaining_clauses).
            A literal is pure if it appears only with one polarity.
            """
            # Count occurrences of each literal
            pos_count = [0] * n
            neg_count = [0] * n
            for clause in clauses_int:
                for var, sign in clause:
                    if sign == 1:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1
            
            # Assign pure literals
            changed = False
            for var in range(n):
                if a[var] is not None:
                    continue
                if pos_count[var] == 0 and neg_count[var] > 0:
                    a[var] = False
                    changed = True
                elif neg_count[var] == 0 and pos_count[var] > 0:
                    a[var] = True
                    changed = True
            
            if not changed:
                return (a, clauses_int)
            
            # Remove satisfied clauses and re-propagate
            new_clauses = []
            for clause in clauses_int:
                sat = False
                unassigned = []
                for var, sign in clause:
                    if a[var] is not None:
                        if (sign == 1 and a[var]) or (sign == -1 and not a[var]):
                            sat = True
                            break
                    else:
                        unassigned.append((var, sign))
                if sat:
                    continue
                if len(unassigned) == 0:
                    return (None, None)  # Contradiction
                new_clauses.append(unassigned)
            
            # Unit propagate again
            success, a, clauses_int = unit_propagate(new_clauses, a)
            if not success:
                return (None, None)
            return (a, clauses_int)
        
        # Main DPLL loop
        # We'll use a stack-based approach to avoid deep recursion issues
        # Stack entries: (clauses_int, assign_copy, var_to_branch)
        
        # Actually, for 50 vars, let's just use recursion with a time check
        import sys
        sys.setrecursionlimit(10000)
        
        def dpll_recursive(clauses_int, a):
            if time.time() > deadline:
                return None  # Timeout
            
            # Check if all clauses satisfied
            if not clauses_int:
                return a[:]
            
            # Unit propagation
            success, a, clauses_int = unit_propagate(clauses_int, a)
            if not success:
                return None
            
            # Check if all clauses satisfied
            if not clauses_int:
                return a[:]
            
            # Pure literal elimination
            a, clauses_int = pure_literal_elimination(clauses_int, a)
            if a is None:
                return None
            
            # Check if all clauses satisfied
            if not clauses_int:
                return a[:]
            
            # Choose a variable to branch on
            # Pick the variable that appears most frequently in the remaining clauses
            var_freq = [0] * n
            for clause in clauses_int:
                for var, sign in clause:
                    var_freq[var] += 1
            
            # Find unassigned variable with highest frequency
            best_var = -1
            best_freq = -1
            for var in range(n):
                if a[var] is None and var_freq[var] > best_freq:
                    best_freq = var_freq[var]
                    best_var = var
            
            if best_var == -1:
                # No unassigned variables, but clauses remain? Shouldn't happen
                return a[:]
            
            # Try True first
            a[best_var] = True
            result = dpll_recursive(clauses_int, a)
            if result is not None:
                return result
            
            # Try False
            a[best_var] = False
            result = dpll_recursive(clauses_int, a)
            if result is not None:
                return result
            
            # Backtrack
            a[best_var] = None
            return None
        
        # Run DPLL
        result = dpll_recursive(clauses_int, assign)
        if result is not None:
            return result
        
        # DPLL failed (timeout or unsat). Fall back to WalkSAT
        # But first, check if we have a partial assignment that's close
        # Just run WalkSAT from scratch with the remaining time
        return None
    
    # Try DPLL first
    result = dpll(clauses, [None] * n)
    if result is not None:
        # Verify the solution
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if (lit > 0 and result[var]) or (lit < 0 and not result[var]):
                    sat = True
                    break
            if not sat:
                break
        else:
            return result
    
    # Fallback: WalkSAT with the remaining time
    rng = random.Random(0)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def satisfied(clause, a):
        return any(a[abs(l) - 1] == (l > 0) for l in clause)
    
    steps = 0
    max_steps = 50000
    while time.time() < deadline and steps < max_steps:
        unsat = [c for c in clauses if not satisfied(c, assignment)]
        if not unsat:
            break
        clause = rng.choice(unsat)
        literal = rng.choice(clause)
        assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
        steps += 1
    
    # Final check
    for clause in clauses:
        if not satisfied(clause, assignment):
            break
    else:
        return assignment
    
    return assignment
# EVOLVE-BLOCK-END
