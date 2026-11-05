"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause structure for fast updates
    # For each clause, store (var_idx, is_positive) pairs
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
    
    # For each variable, store list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))
    
    # State: assignment as list of bools
    # clause_unsat_count[ci] = number of falsified literals in clause ci
    # A clause is satisfied iff clause_unsat_count[ci] < 3 (i.e., at least one literal is true)
    # Actually for 3-SAT, a clause is unsat iff all 3 literals are falsified, i.e., unsat_count == 3
    
    def init_state(rng):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_unsat_count = [0] * m
        for ci, cv in enumerate(clause_vars):
            count = 0
            for vi, pos in cv:
                # Literal is falsified if assignment[vi] != pos
                if assignment[vi] != pos:
                    count += 1
            clause_unsat_count[ci] = count
        return assignment, clause_unsat_count
    
    def count_unsat(clause_unsat_count):
        # Count clauses where all literals are falsified
        cnt = 0
        for c in clause_unsat_count:
            if c >= 3:  # All 3 literals falsified
                cnt += 1
        return cnt
    
    def find_unsat_clause(clause_unsat_count):
        # Find index of an unsatisfied clause
        for ci in range(m):
            if clause_unsat_count[ci] >= 3:
                return ci
        return -1
    
    def flip_var(vi, assignment, clause_unsat_count):
        """Flip variable vi and update clause_unsat_count incrementally."""
        old_val = assignment[vi]
        assignment[vi] = not old_val
        for ci, pos in var_clauses[vi]:
            # Before flip: literal (vi, pos) was falsified iff old_val != pos
            # After flip: literal (vi, pos) is falsified iff (not old_val) != pos
            if old_val == pos:
                # Was satisfied, now falsified
                clause_unsat_count[ci] += 1
            else:
                # Was falsified, now satisfied
                clause_unsat_count[ci] -= 1
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random(0)
    
    # Multiple restarts with increasing step budgets
    max_restarts = 50
    base_steps = 5000
    max_steps = 50000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        assignment, clause_unsat_count = init_state(rng)
        current_unsat = count_unsat(clause_unsat_count)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = list(assignment)
        
        # WalkSAT with adaptive flip selection
        # Weight: for each variable in an unsat clause, score = number of satisfied literals it would break
        # We want to flip a variable that minimizes new unsatisfied clauses
        steps_in_restart = base_steps + restart * 2000
        steps_in_restart = min(steps_in_restart, max_steps)
        
        for step in range(steps_in_restart):
            if time.time() > deadline:
                break
            
            # Find an unsatisfied clause
            unsat_idx = find_unsat_clause(clause_unsat_count)
            if unsat_idx == -1:
                # All satisfied
                return assignment
            
            # Get the clause
            cv = clause_vars[unsat_idx]
            
            # For each variable in the clause, compute the "damage" of flipping it
            # Damage = number of currently-satisfied clauses that would become unsatisfied
            best_var = -1
            best_damage = float('inf')
            best_score = float('inf')
            
            for vi, pos in cv:
                # If we flip vi, what happens to clauses containing vi?
                damage = 0
                for cj, cpos in var_clauses[vi]:
                    if cj == unsat_idx:
                        continue
                    # Check if clause cj is currently satisfied
                    if clause_unsat_count[cj] < 3:
                        # Currently satisfied. Would flipping vi make it unsatisfied?
                        # Clause cj has literal (vi, cpos).
                        # Currently satisfied means at least one literal is true.
                        # If (vi, cpos) is currently true (assignment[vi] == cpos), 
                        # then after flip it becomes false. If all other literals are also false, clause becomes unsat.
                        if assignment[vi] == cpos:
                            # This literal is currently true. After flip it becomes false.
                            # Check if all other literals in cj are false
                            # clause_unsat_count[cj] counts falsified literals.
                            # If assignment[vi] == cpos, this literal is NOT counted in unsat_count.
                            # After flip, it WILL be counted. So new unsat count = old + 1.
                            # Clause becomes unsat iff new count >= 3, i.e., old count >= 2.
                            if clause_unsat_count[cj] >= 2:
                                damage += 1
                        # If assignment[vi] != cpos, literal is currently false, after flip becomes true, 
                        # so clause stays satisfied (or becomes more satisfied). No damage.
                
                # Score: primarily damage, secondarily random tiebreaker
                # Use a weighted score to allow some randomness
                score = damage + rng.random() * 0.1
                
                if score < best_score or (score == best_score and damage < best_damage):
                    best_score = score
                    best_damage = damage
                    best_var = vi
            
            if best_var == -1:
                # Fallback: pick random variable from clause
                best_var = rng.choice(cv)[0]
            
            # Flip the variable
            flip_var(best_var, assignment, clause_unsat_count)
            
            # Update unsat count
            current_unsat = count_unsat(clause_unsat_count)
            
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = list(assignment)
        
        # Check time after restart
        if time.time() > deadline:
            break
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random assignment
    rng2 = random.Random(42)
    return [rng2.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
