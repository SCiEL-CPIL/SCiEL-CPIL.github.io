# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # safety margin before 300s limit
    
    # Preprocess: convert clauses to 0-based signed literals
    # Store as list of tuples of (var_idx, sign) where sign is +1 or -1
    # Actually let's keep them as signed literals but 0-based internally
    # literal l (1-based): var = abs(l)-1, is_positive = l > 0
    # We'll store clauses as list of (var, is_pos) tuples for efficiency
    
    clauses_internal = []
    for c in clauses:
        clauses_internal.append(tuple((abs(l) - 1, l > 0) for l in c))
    
    m = len(clauses_internal)
    
    # Build index: for each variable, which clauses contain it
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses_internal):
        for var, _ in c:
            var_to_clauses[var].append(i)
    
    # GLARE-style solver with unit propagation and adaptive restarts
    best_assignment = None
    best_unsat_count = m + 1
    
    rng = random.Random(42)
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        cnt = 0
        for c in clauses_internal:
            sat = False
            for var, is_pos in c:
                if assignment[var] == is_pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    def unit_propagate(assignment, clauses_to_check):
        """
        Perform unit propagation on a copy of the assignment.
        Returns (new_assignment, changed_vars, success) where success is False if contradiction.
        """
        changed = set()
        # Work with a queue of unit clauses
        # A clause is unit if exactly one literal is unassigned (False in our context means not set)
        # Actually in this SAT context, all variables are assigned. Unit propagation here means:
        # if a clause has exactly one literal that is False (unsatisfied) and the others are True,
        # we must flip that one literal to satisfy the clause.
        # More generally: if a clause has k unsatisfied literals, and k==1, we can flip that literal.
        
        # Let's implement iterative unit propagation
        # For each clause, check if it's unit (exactly one false literal)
        # If so, flip that variable
        
        modified = list(assignment)
        queue = []
        
        # Initial pass: find all unit clauses
        for ci in clauses_to_check:
            c = clauses_internal[ci]
            false_lits = []
            true_count = 0
            for var, is_pos in c:
                if modified[var] == is_pos:
                    true_count += 1
                else:
                    false_lits.append((var, is_pos))
            if true_count == 0 and len(false_lits) == 1:
                # This clause is unit: must set false_lits[0]
                var, is_pos = false_lits[0]
                if modified[var] != is_pos:
                    modified[var] = is_pos
                    changed.add(var)
                    queue.append(var)
        
        # Propagate
        qi = 0
        while qi < len(queue):
            var = queue[qi]
            qi += 1
            for ci in var_to_clauses[var]:
                c = clauses_internal[ci]
                false_lits = []
                true_count = 0
                for v, is_pos in c:
                    if modified[v] == is_pos:
                        true_count += 1
                    else:
                        false_lits.append((v, is_pos))
                if true_count == 0 and len(false_lits) == 1:
                    v, is_pos = false_lits[0]
                    if modified[v] != is_pos:
                        modified[v] = is_pos
                        changed.add(v)
                        queue.append(v)
        
        return modified, changed, True
    
    def glare_step(assignment, unsat_clauses):
        """
        GLARE-style step: for each unsatisfied clause, if it's unit, flip the single false literal.
        Otherwise, use a weighted random selection among the false literals in the clause.
        Returns the new assignment and the variable that was flipped.
        """
        if not unsat_clauses:
            return assignment, None
        
        # Try unit propagation first on unsatisfied clauses
        unit_flips = []
        for ci in unsat_clauses:
            c = clauses_internal[ci]
            false_vars = []
            for var, is_pos in c:
                if assignment[var] != is_pos:
                    false_vars.append((var, is_pos))
            if len(false_vars) == 1:
                unit_flips.append(false_vars[0])
        
        if unit_flips:
            # Flip a random unit literal
            var, is_pos = rng.choice(unit_flips)
            new_assignment = list(assignment)
            new_assignment[var] = is_pos
            return new_assignment, var
        
        # No unit clauses; pick a random unsatisfied clause and flip a literal
        ci = rng.choice(unsat_clauses)
        c = clauses_internal[ci]
        false_vars = []
        for var, is_pos in c:
            if assignment[var] != is_pos:
                false_vars.append((var, is_pos))
        
        if not false_vars:
            # Shouldn't happen for unsatisfied clause
            return assignment, None
        
        # Weighted selection: prefer variables that appear in more unsatisfied clauses
        # Compute weights based on how many unsatisfied clauses each false literal is in
        weights = []
        for var, is_pos in false_vars:
            w = 1.0
            # Count how many unsatisfied clauses contain this variable
            for ci2 in unsat_clauses:
                if ci2 != ci:
                    for v, p in clauses_internal[ci2]:
                        if v == var:
                            w += 1.0
                            break
            weights.append(w)
        
        # Normalize and pick
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        chosen_idx = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                chosen_idx = i
                break
        
        var, is_pos = false_vars[chosen_idx]
        new_assignment = list(assignment)
        new_assignment[var] = is_pos
        return new_assignment, var
    
    # Main search loop with adaptive restarts
    max_restarts = 500
    base_steps = 5000
    step_growth = 1.15
    
    restart = 0
    while restart < max_restarts and time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial unit propagation
        assignment, _, _ = unit_propagate(assignment, range(m))
        
        # Count initial unsatisfied
        unsat_count = count_unsat(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assignment)
        
        steps = int(base_steps * (step_growth ** restart))
        steps = min(steps, 100000)  # Cap steps per restart
        
        for step in range(steps):
            if time.time() >= deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for ci, c in enumerate(clauses_internal):
                sat = False
                for var, is_pos in c:
                    if assignment[var] == is_pos:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                return assignment
            
            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assignment = list(assignment)
            
            # GLARE step
            assignment, _ = glare_step(assignment, unsat_clauses)
        
        restart += 1
    
    # Return best found assignment
    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
