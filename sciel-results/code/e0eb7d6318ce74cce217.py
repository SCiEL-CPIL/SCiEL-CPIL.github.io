"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time


def solve(instance):
    """
    Deterministic DPLL solver with unit propagation and pure literal elimination,
    followed by a WalkSAT fallback for robustness within the time budget.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: allow ~290s of the 330s wall-clock limit
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Convert to 0-based indices for internal use
    # Each literal is stored as (var, sign) where var is 0-based, sign is +1 or -1
    # We'll work with signed 1-based literals as given, but use 0-based for assignment
    
    # Preprocess: build clause lists with 0-based variable indices
    # clause is list of signed ints (1-based)
    # We'll keep them as-is and convert on the fly for assignment lookup
    
    def check_satisfied(clause, assignment):
        for lit in clause:
            var = abs(lit) - 1
            if assignment[var] == (lit > 0):
                return True
        return False
    
    def dpll(clauses, assignment, depth=0):
        """
        Recursive DPLL with unit propagation and pure literal elimination.
        assignment: list of (bool or None) for each variable.
        Returns a complete satisfying assignment or None if UNSAT (or timeout).
        """
        nonlocal deadline
        
        if time.time() > deadline:
            return None
        
        # Work with a copy of assignment
        if assignment is None:
            assignment = [None] * n
        else:
            assignment = assignment[:]
        
        # Unit propagation and pure literal elimination
        changed = True
        while changed:
            changed = False
            
            # Filter out satisfied clauses and unit clauses
            new_clauses = []
            unit_clauses = []
            
            for clause in clauses:
                # Check if clause is satisfied
                sat = False
                remaining = []
                for lit in clause:
                    var = abs(lit) - 1
                    if assignment[var] is not None:
                        if assignment[var] == (lit > 0):
                            sat = True
                            break
                    else:
                        remaining.append(lit)
                
                if sat:
                    continue
                elif len(remaining) == 0:
                    # Clause is falsified
                    return None
                elif len(remaining) == 1:
                    unit_clauses.append(remaining[0])
                    # Also add to new_clauses for tracking, but we'll handle units
                    new_clauses.append(remaining)
                else:
                    new_clauses.append(remaining)
            
            # Apply unit clauses
            for lit in unit_clauses:
                var = abs(lit) - 1
                val = (lit > 0)
                if assignment[var] is not None:
                    if assignment[var] != val:
                        return None
                else:
                    assignment[var] = val
                    changed = True
            
            # Pure literal elimination
            if not changed:
                # Find pure literals
                positive = set()
                negative = set()
                for clause in new_clauses:
                    for lit in clause:
                        var = abs(lit) - 1
                        if assignment[var] is None:
                            if lit > 0:
                                positive.add(var)
                            else:
                                negative.add(var)
                
                # A variable is pure if it appears only in one polarity
                for var in positive:
                    if var not in negative and assignment[var] is None:
                        assignment[var] = True
                        changed = True
                
                for var in negative:
                    if var not in positive and assignment[var] is None:
                        assignment[var] = False
                        changed = True
            
            clauses = new_clauses
        
        # Check if all clauses are satisfied
        if not clauses:
            # Fill in remaining unassigned variables arbitrarily
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = False
            return assignment
        
        # Choose a branching variable
        # Heuristic: pick variable from the shortest clause
        min_clause = min(clauses, key=len)
        # Pick the first literal's variable
        lit = min_clause[0]
        var = abs(lit) - 1
        
        # Try both values
        for val in [True, False]:
            new_assignment = assignment[:]
            new_assignment[var] = val
            
            # Recurse
            result = dpll(clauses, new_assignment, depth + 1)
            if result is not None:
                return result
        
        return None
    
    # First, try DPLL
    result = dpll(clauses, None)
    
    if result is not None:
        # Verify the result
        for clause in clauses:
            if not check_satisfied(clause, result):
                break
        else:
            return result
    
    # Fallback: WalkSAT if DPLL failed or timed out
    import random
    rng = random.Random(42)
    
    # Start from a random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def count_unsatisfied(clause, assignment):
        count = 0
        for lit in clause:
            var = abs(lit) - 1
            if assignment[var] == (lit > 0):
                count += 1
        return count
    
    # WalkSAT with restarts
    max_steps_per_restart = 5000
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        if restart > 0:
            # Random restart
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat = []
            for c in clauses:
                if not check_satisfied(c, assignment):
                    unsat.append(c)
            
            if not unsat:
                return assignment
            
            # Pick a random unsatisfied clause
            clause = rng.choice(unsat)
            
            # Pick a literal to flip
            # Weight by number of satisfied literals in the clause
            weights = []
            for lit in clause:
                var = abs(lit) - 1
                # If flipping this literal would satisfy the clause, give it higher weight
                if not assignment[var] == (lit > 0):
                    # This literal is currently false, flipping it would make it true
                    weights.append(10)
                else:
                    # This literal is currently true, flipping it would make it false
                    weights.append(1)
            
            total = sum(weights)
            r = rng.random() * total
            cumulative = 0
            for i, lit in enumerate(clause):
                cumulative += weights[i]
                if r <= cumulative:
                    var = abs(lit) - 1
                    assignment[var] = not assignment[var]
                    break
    
    # If we get here, return the current best assignment (may not be satisfying)
    # But we should try to return something valid
    return assignment
    
# EVOLVE-BLOCK-END
