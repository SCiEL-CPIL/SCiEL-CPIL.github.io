# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict


def solve(instance):
    """
    Hybrid SAT solver:
    1. Preprocessing: pure literal elimination + unit propagation
    2. Clause-weighted WalkSAT with dynamic noise on reduced instance
    3. Multiple restarts with increasing budgets
    """
    deadline = time.time() + 280  # 280s safety margin out of 300s
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert to 0-based representation
    # literal l: variable index = abs(l) - 1, positive if l > 0
    clauses_0 = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    
    # --- Preprocessing: Pure Literal Elimination + Unit Propagation ---
    # We'll work with a mutable copy
    assignment = [None] * n  # None = unassigned, True/False = assigned
    clauses_work = [list(c) for c in clauses_0]
    
    changed = True
    while changed:
        changed = False
        
        # Remove clauses that are satisfied or have unit literals
        new_clauses = []
        for clause in clauses_work:
            # Check if clause is already satisfied
            sat = False
            unit_lit = None
            remaining = []
            for (var, pos) in clause:
                if assignment[var] is not None:
                    if assignment[var] == pos:
                        sat = True
                        break
                    else:
                        continue
                else:
                    remaining.append((var, pos))
            
            if sat:
                continue
            if not remaining:
                # Empty clause -> UNSAT (shouldn't happen for planted instances)
                return [random.random() < 0.5 for _ in range(n)]
            if len(remaining) == 1:
                var, pos = remaining[0]
                if assignment[var] is None:
                    assignment[var] = pos
                    changed = True
                continue
            new_clauses.append(remaining)
        
        clauses_work = new_clauses
        
        # Pure literal elimination
        # For each unassigned variable, check if it appears with only one polarity
        var_polarities = defaultdict(set)
        for clause in clauses_work:
            for (var, pos) in clause:
                if assignment[var] is None:
                    var_polarities[var].add(pos)
        
        pure_vars = []
        for var, polys in var_polarities.items():
            if len(polys) == 1:
                pos = list(polys)[0]
                if assignment[var] is None:
                    assignment[var] = pos
                    pure_vars.append((var, pos))
                    changed = True
        
        # Re-filter clauses after pure literal assignment
        if pure_vars:
            new_clauses = []
            for clause in clauses_work:
                sat = False
                remaining = []
                for (var, pos) in clause:
                    if assignment[var] is not None:
                        if assignment[var] == pos:
                            sat = True
                            break
                        else:
                            continue
                    else:
                        remaining.append((var, pos))
                if not sat:
                    if not remaining:
                        return [random.random() < 0.5 for _ in range(n)]
                    if len(remaining) == 1:
                        var, pos = remaining[0]
                        if assignment[var] is None:
                            assignment[var] = pos
                            changed = True
                        continue
                    new_clauses.append(remaining)
            clauses_work = new_clauses
    
    # Fill any unassigned variables randomly
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = random.random() < 0.5
    
    # Check if already solved
    def check_satisfied(assn, cls):
        for clause in cls:
            sat = False
            for (var, pos) in clause:
                if assn[var] == pos:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    if check_satisfied(assignment, clauses_0):
        return assignment
    
    # --- Clause-weighted WalkSAT on full instance ---
    # Use the original clauses for final check, but work with 0-based
    rng = random.Random(42)
    
    best_assignment = list(assignment)
    best_score = sum(1 for c in clauses_0 if all(assignment[v] == p for v, p in c) or any(assignment[v] == p for v, p in c))
    # Actually let's just count unsatisfied clauses
    def count_unsat(assn, cls):
        cnt = 0
        for clause in cls:
            sat = False
            for (var, pos) in clause:
                if assn[var] == pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    best_unsat = count_unsat(assignment, clauses_0)
    
    if best_unsat == 0:
        return assignment
    
    # Build clause weights based on frequency of variables
    var_count = np.zeros(n)
    for clause in clauses_0:
        for (var, pos) in clause:
            var_count[var] += 1
    
    # WalkSAT with clause weighting and dynamic noise
    max_time = deadline
    restart_budgets = [5000, 10000, 20000, 40000, 80000, 150000]
    
    for restart_idx, budget in enumerate(restart_budgets):
        if time.time() > max_time - 10:
            break
        
        # Initialize assignment
        if restart_idx == 0:
            curr = list(assignment)
        else:
            curr = [rng.random() < 0.5 for _ in range(n)]
        
        curr_unsat = count_unsat(curr, clauses_0)
        if curr_unsat < best_unsat:
            best_unsat = curr_unsat
            best_assignment = list(curr)
            if best_unsat == 0:
                return best_assignment
        
        # Dynamic noise parameter
        noise_start = 0.5
        noise_end = 0.1
        
        for step in range(budget):
            if time.time() > max_time - 5:
                return best_assignment
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for c in clauses_0:
                sat = False
                for (var, pos) in c:
                    if curr[var] == pos:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(c)
            
            if not unsat_clauses:
                return curr
            
            # Progressively reduce noise
            progress = step / budget
            noise = noise_start + (noise_end - noise_start) * progress
            
            # Weighted clause selection
            # Clauses with more unsatisfied literals (closer to being satisfied) get higher weight
            weights = []
            for c in unsat_clauses:
                # Weight by how many literals are "almost" satisfied
                w = 1.0
                for (var, pos) in c:
                    if curr[var] != pos:
                        # Variable is flipped from what clause wants
                        w *= 1.0  # All are unsatisfied by definition
                # Add variable frequency weight
                var_w = sum(var_count[v] for v, p in c)
                weights.append(var_w)
            
            # Normalize weights
            total_w = sum(weights)
            if total_w == 0:
                weights = [1.0] * len(unsat_clauses)
                total_w = len(unsat_clauses)
            probs = [w / total_w for w in weights]
            
            # Select clause
            r = rng.random()
            cumulative = 0
            selected_clause = unsat_clauses[0]
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    selected_clause = unsat_clauses[i]
                    break
            
            # Select variable to flip
            # With probability (1-noise), flip a variable that appears in the clause
            # With probability noise, flip a random variable
            if rng.random() < (1 - noise):
                # Choose from clause variables, weighted by how many other clauses they appear in
                var_options = []
                for (var, pos) in selected_clause:
                    var_options.append(var)
                # Weight by inverse of how many clauses this var appears in (prefer less constrained)
                var_weights = [1.0 / max(1, var_count[v]) for v in var_options]
                total_vw = sum(var_weights)
                if total_vw == 0:
                    var_weights = [1.0] * len(var_options)
                    total_vw = len(var_options)
                vprobs = [w / total_vw for w in var_weights]
                
                r2 = rng.random()
                cum2 = 0
                flip_var = var_options[0]
                for i, p in enumerate(vprobs):
                    cum2 += p
                    if r2 <= cum2:
                        flip_var = var_options[i]
                        break
            else:
                flip_var = rng.randint(0, n - 1)
            
            # Flip the variable
            curr[flip_var] = not curr[flip_var]
            
            # Update unsat count
            new_unsat = count_unsat(curr, clauses_0)
            if new_unsat < best_unsat:
                best_unsat = new_unsat
                best_assignment = list(curr)
                if best_unsat == 0:
                    return best_assignment
            
            curr_unsat = new_unsat
    
    return best_assignment
# EVOLVE-BLOCK-END
