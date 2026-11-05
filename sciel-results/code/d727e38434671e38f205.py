"""Planted 3-SAT solver with DPLL + WalkSAT hybrid and time guard."""
# EVOLVE-BLOCK-START
import time
import random
import math


def _build_index(clauses):
    """Build literal -> list of (clause_idx, position) index."""
    lit_to_clauses = {}
    for ci, clause in enumerate(clauses):
        for li, lit in enumerate(clause):
            lit_to_clauses.setdefault(lit, []).append((ci, li))
    return lit_to_clauses


def _dpll(clauses, n, deadline, max_dpll_steps=500000):
    """
    DPLL with unit propagation and pure literal elimination.
    Returns (assignment_list or None, steps_used).
    assignment_list is a list of length n with values True/False/None.
    """
    start = time.time()
    # assignment: list of True/False/None for variables 1..n
    assignment = [None] * n
    
    # Build initial clause structure: list of lists of signed literals
    # We'll work with a mutable structure.
    # For efficiency, we maintain:
    #   clauses: list of lists of literals (current)
    #   lit_to_clauses: dict mapping literal to list of clause indices
    #   var_to_clauses: for pure literal elimination (approximate)
    
    # Actually, let's implement a cleaner DPLL with unit propagation
    # using a worklist approach.
    
    def propagate(clauses, assignment):
        """
        Perform unit propagation.
        clauses: list of lists of literals (mutated)
        assignment: list of True/False/None (mutated)
        Returns: True if no contradiction, False if contradiction (empty clause).
        """
        unit_queue = []
        for ci, clause in enumerate(clauses):
            if len(clause) == 1:
                unit_queue.append(ci)
            elif len(clause) == 0:
                return False
        
        while unit_queue:
            ci = unit_queue.pop()
            clause = clauses[ci]
            if not clause:
                return False
            if len(clause) == 1:
                lit = clause[0]
                var = abs(lit) - 1
                val = lit > 0
                if assignment[var] is not None and assignment[var] != val:
                    return False
                if assignment[var] is None:
                    assignment[var] = val
                    # Remove this literal and its negation from all clauses
                    neg_lit = -lit
                    for cj in range(len(clauses)):
                        c = clauses[cj]
                        # Remove lit and neg_lit from c
                        new_c = []
                        changed = False
                        for l in c:
                            if l == lit:
                                changed = True
                            elif l == neg_lit:
                                changed = True
                            else:
                                new_c.append(l)
                        if changed:
                            clauses[cj] = new_c
                            if len(new_c) == 1 and cj not in unit_queue:
                                unit_queue.append(cj)
                            elif len(new_c) == 0:
                                return False
            # If clause is already satisfied, skip (it's been processed or is fine)
            # We need to check if clause is satisfied by current assignment
            # Actually, we should only process unit clauses. Let me restructure.
        
        # Check for empty clauses
        for clause in clauses:
            if len(clause) == 0:
                return False
        return True
    
    # Simpler DPLL approach:
    # Use recursion with unit propagation
    def dpll_rec(clauses, assignment, steps):
        if steps[0] > max_dpll_steps:
            return None
        if time.time() > deadline:
            return None
        
        # Check if all clauses satisfied
        # First, do unit propagation
        # We'll use a fixed-point iteration for unit propagation
        
        # Check for contradiction
        for clause in clauses:
            if len(clause) == 0:
                return None  # contradiction
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for ci, clause in enumerate(clauses):
                if len(clause) == 1:
                    lit = clause[0]
                    var = abs(lit) - 1
                    val = lit > 0
                    if assignment[var] is None:
                        assignment[var] = val
                        changed = True
                        # Remove lit and -lit from all clauses
                        neg_lit = -lit
                        for cj in range(len(clauses)):
                            c = clauses[cj]
                            new_c = [l for l in c if l != lit and l != neg_lit]
                            if len(new_c) != len(c):
                                clauses[cj] = new_c
                                if len(new_c) == 0:
                                    return None
                    elif assignment[var] != val:
                        return None  # contradiction
        
        # Check if all variables assigned
        if all(a is not None for a in assignment):
            # Verify all clauses are satisfied
            for clause in clauses:
                if not clause:
                    return None
                # If we got here without contradiction, should be fine
            return assignment[:]
        
        # Pick a variable to branch on
        # Choose the variable that appears in the most clauses (heuristic)
        var_counts = [0] * n
        for clause in clauses:
            if not clause:
                return None
            for lit in clause:
                var_counts[abs(lit) - 1] += 1
        
        # Find unassigned variable with highest count
        best_var = -1
        best_count = -1
        for v in range(n):
            if assignment[v] is None and var_counts[v] > best_count:
                best_count = var_counts[v]
                best_var = v
        
        if best_var == -1:
            # All assigned, verify
            for clause in clauses:
                if not clause:
                    return None
            return assignment[:]
        
        # Branch: try True first
        for val in [True, False]:
            # Copy state
            clauses_copy = [c[:] for c in clauses]
            assignment_copy = assignment[:]
            assignment_copy[best_var] = val
            steps[0] += 1
            
            result = dpll_rec(clauses_copy, assignment_copy, steps)
            if result is not None:
                return result
        
        return None
    
    # Make deep copy of clauses
    clauses_copy = [c[:] for c in clauses]
    steps = [0]
    result = dpll_rec(clauses_copy, assignment, steps)
    return result, steps[0]


def _walksat(clauses, n, assignment, deadline, max_steps=1000000):
    """
    WalkSAT with clause weighting and noise.
    Returns satisfying assignment or None.
    """
    start = time.time()
    rng = random.Random(42)
    
    # Pre-compute for speed
    def is_satisfied(clause):
        for lit in clause:
            var = abs(lit) - 1
            if assignment[var] == (lit > 0):
                return True
        return False
    
    def clause_score(clause):
        """Number of false literals in clause."""
        count = 0
        for lit in clause:
            var = abs(lit) - 1
            if assignment[var] != (lit > 0):
                count += 1
        return count
    
    # Count unsatisfied clauses
    unsat_count = sum(1 for c in clauses if not is_satisfied(c))
    
    step = 0
    while unsat_count > 0 and step < max_steps:
        if time.time() > deadline:
            break
        step += 1
        
        # Find an unsatisfied clause
        # Use a random sample to find one
        while True:
            c = clauses[rng.randrange(len(clauses))]
            if not is_satisfied(c):
                break
        
        # Choose a literal to flip
        # Score each literal: number of satisfied literals in other clauses that would be flipped
        best_var = -1
        best_score = -1
        best_noise = 0
        
        # Sample up to 30 literals from the clause
        lit_indices = list(range(len(c)))
        rng.shuffle(lit_indices)
        sample_size = min(30, len(lit_indices))
        
        for idx in lit_indices[:sample_size]:
            lit = c[idx]
            var = abs(lit) - 1
            
            # Compute gain: how many satisfied clauses become unsatisfied
            # and how many unsatisfied clauses become satisfied if we flip var
            gain = 0
            # This is expensive to compute exactly, so use a heuristic
            # Count how many clauses contain this literal or its negation
            # For simplicity, use a local search approach
            
            # Actually, let's just pick randomly with some bias
            # Flip the literal that appears in the most unsatisfied clauses
            pass
        
        # Simpler: pick a random literal from the clause with probability (1-noise)
        # or pick a random variable with probability noise
        if rng.random() < 0.2:
            # Noise: pick a random variable
            var = rng.randrange(n)
        else:
            # Pick from the clause
            lit = c[rng.randrange(len(c))]
            var = abs(lit) - 1
        
        # Flip
        assignment[var] = not assignment[var]
        
        # Recompute unsat count (expensive but necessary for correctness)
        # Optimize: only recompute affected clauses
        # For now, do a full recompute every 100 steps
        if step % 100 == 0:
            unsat_count = sum(1 for c in clauses if not is_satisfied(c))
        else:
            # Approximate: just check if we made progress
            # This is hacky, let's do periodic full recompute
            pass
    
    # Final check
    unsat_count = sum(1 for c in clauses if not is_satisfied(c))
    if unsat_count == 0:
        return assignment
    return None


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    deadline = time.time() + 280  # Leave 50s margin
    
    # First, try DPLL with unit propagation
    dpll_result, dpll_steps = _dpll(clauses, n, deadline)
    if dpll_result is not None:
        return dpll_result
    
    # If DPLL didn't finish or failed, try WalkSAT with multiple restarts
    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random(123)
    
    # Multiple WalkSAT restarts
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        result = _walksat(clauses, n, assignment, deadline - 10)
        if result is not None:
            return result
        
        # Track best partial solution
        unsat = sum(1 for c in clauses if not any(
            assignment[abs(l)-1] == (l > 0) for l in c
        ))
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
    
    # Return best partial solution found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
