"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285.0
    m = len(clauses)
    
    # Precompute variable-to-clause incidence for fast delta computation
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            vi = abs(lit) - 1
            var_clauses[vi].append(ci)
    
    # Precompute literal satisfaction check helpers
    # For each clause, store (var_idx, is_positive) for each literal
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            vi = abs(lit) - 1
            pos = 1 if lit > 0 else 0
            cv.append((vi, pos))
        clause_vars.append(cv)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for ci in range(m):
            sat = False
            for vi, pos in clause_vars[ci]:
                if assignment[vi] == (pos == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses."""
        for ci in range(m):
            sat = False
            for vi, pos in clause_vars[ci]:
                if assignment[vi] == (pos == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def compute_delta(var_idx, assignment, unsat_set):
        """Compute delta (change in unsat count) if we flip var_idx.
        Returns positive delta if flipping increases unsat count."""
        delta = 0
        for ci in var_clauses[var_idx]:
            if ci in unsat_set:
                # Clause is currently unsatisfied
                # If flipping var_idx satisfies this clause, delta decreases
                # Check if any other literal in this clause is already satisfied
                # (it can't be, since clause is unsatisfied)
                # So flipping this var will satisfy the clause -> delta -= 1
                # But we need to check: after flip, is the clause satisfied?
                # Since clause is unsat, all its literals are false.
                # If we flip var_idx, the literal for var_idx becomes true (if sign matches) or false.
                # Actually we need to check the sign of the literal in this clause.
                # Let's just check: after flip, is clause satisfied?
                # We'll handle this more carefully.
                pass
            else:
                # Clause is currently satisfied
                # If flipping var_idx makes this clause unsatisfied, delta increases
                # This happens if the literal for var_idx in this clause was the only
                # satisfied literal in the clause
                # Check: is the literal for var_idx satisfied in this clause?
                lit_satisfied = False
                for vi, pos in clause_vars[ci]:
                    if vi == var_idx:
                        if assignment[vi] == (pos == 1):
                            lit_satisfied = True
                        break
                if lit_satisfied:
                    # Check if any other literal in this clause is satisfied
                    other_sat = False
                    for vi2, pos2 in clause_vars[ci]:
                        if vi2 != var_idx and assignment[vi2] == (pos2 == 1):
                            other_sat = True
                            break
                    if not other_sat:
                        delta += 1
        return delta
    
    # More efficient delta: for a flip of var v:
    # For each clause C containing v:
    #   If C is unsat: flipping v may satisfy C (if the literal v in C becomes true)
    #     -> delta -= 1 (reduces unsat count)
    #   If C is sat: flipping v may unsatisfy C (if v's literal was the only sat literal)
    #     -> delta += 1
    # Net delta = (# clauses that become unsat) - (# clauses that become sat)
    
    def fast_delta(var_idx, assignment, unsat_indices):
        """Compute net change in unsat count when flipping var_idx."""
        delta = 0
        for ci in var_clauses[var_idx]:
            if ci in unsat_indices:
                # Currently unsat. After flip, will it be sat?
                # The literal for var_idx in clause ci: check its sign
                # If after flip, that literal becomes true, clause becomes sat
                for vi, pos in clause_vars[ci]:
                    if vi == var_idx:
                        # After flip, assignment[var_idx] changes
                        new_val = not assignment[var_idx]
                        if new_val == (pos == 1):
                            delta -= 1  # clause becomes satisfied
                        break
            else:
                # Currently sat. After flip, will it be unsat?
                # Only if var_idx's literal was the only satisfied literal
                for vi, pos in clause_vars[ci]:
                    if vi == var_idx:
                        # Check if this literal is currently satisfied
                        if assignment[var_idx] == (pos == 1):
                            # Check if any other literal is satisfied
                            other_sat = False
                            for vi2, pos2 in clause_vars[ci]:
                                if vi2 != var_idx and assignment[vi2] == (pos2 == 1):
                                    other_sat = True
                                    break
                            if not other_sat:
                                delta += 1  # clause becomes unsatisfied
                        break
        return delta
    
    # Main search: multiple restarts with Min-Conflicts / WalkSAT hybrid
    for attempt in range(500):
        if time.time() >= deadline:
            break
        
        rng = random.Random(attempt * 7919 + 12345)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # Find unsatisfied clauses
        unsat_indices = set()
        for ci in range(m):
            sat = False
            for vi, pos in clause_vars[ci]:
                if assignment[vi] == (pos == 1):
                    sat = True
                    break
            if not sat:
                unsat_indices.add(ci)
        
        max_steps = 2000
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            
            if not unsat_indices:
                return assignment
            
            # Min-Conflicts: pick variable that minimizes delta
            # With some randomness to escape local minima
            # Choose a random unsatisfied clause, then among its variables, pick best
            unsat_list = list(unsat_indices)
            ci = unsat_list[rng.randrange(len(unsat_list))]
            clause = clause_vars[ci]
            
            # Compute delta for each variable in the chosen clause
            best_var = -1
            best_delta = float('inf')
            candidates = []
            
            for vi, pos in clause:
                d = fast_delta(vi, assignment, unsat_indices)
                candidates.append((d, vi))
                if d < best_delta:
                    best_delta = d
                    best_var = vi
            
            # With probability 0.2, pick a random candidate instead of the best
            # This provides noise to escape local minima
            if rng.random() < 0.2 and len(candidates) > 1:
                best_var = candidates[rng.randrange(len(candidates))][1]
            
            # Flip the chosen variable
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            # Update unsat_indices incrementally
            # For each clause containing best_var, recheck satisfaction
            for ci2 in var_clauses[best_var]:
                # Determine if clause is now satisfied
                sat = False
                for vi, pos in clause_vars[ci2]:
                    if assignment[vi] == (pos == 1):
                        sat = True
                        break
                if sat:
                    unsat_indices.discard(ci2)
                else:
                    unsat_indices.add(ci2)
            
            # Check if solved
            if not unsat_indices:
                return assignment
            
            # Update best
            current_unsat = len(unsat_indices)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
        
        # After max_steps, if not solved, try a few more with different noise
        # Actually just continue to next restart
    
    # Final check
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
