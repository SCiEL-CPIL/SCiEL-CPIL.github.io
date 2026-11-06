# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses_raw = instance["clauses"]
    m = len(clauses_raw)
    
    # Preprocess: convert to 0-based indices with sign
    # clause_vars[i] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses_raw:
        cv = []
        for l in c:
            if l > 0:
                cv.append((l - 1, True))
            else:
                cv.append((-l - 1, False))
        clause_vars.append(cv)
    
    # Build variable-to-clauses index
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, _ in cv:
            var_clauses[var_idx].append(i)
    
    # Time limit: 290 seconds (safety margin from 300s limit)
    deadline = time.time() + 290.0
    
    def is_satisfied(assignment, clause):
        for var_idx, is_pos in clause:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    def count_unsat(assignment):
        count = 0
        for c in clause_vars:
            if not is_satisfied(assignment, c):
                count += 1
        return count
    
    def evaluate_clause(assignment, clause):
        """Return (num_unsat_literals, best_var_for_flip)"""
        num_unsat = 0
        best_var = -1
        best_score = -1
        for var_idx, is_pos in clause:
            if assignment[var_idx] != is_pos:
                num_unsat += 1
                # Score: number of clauses this variable appears in
                score = len(var_clauses[var_idx])
                if score > best_score:
                    best_score = score
                    best_var = var_idx
        return num_unsat, best_var
    
    def walk_sat(assignment, max_steps):
        """
        WalkSAT with adaptive noise.
        Returns (assignment, solved)
        """
        rng = random.Random(0)
        unsat_count = count_unsat(assignment)
        
        if unsat_count == 0:
            return assignment, True
        
        # Track best unsat count
        best_unsat = unsat_count
        best_assignment = assignment[:]
        
        # Noise parameter: start low, increase if stuck
        noise = 0.25
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for c in clause_vars:
                if not is_satisfied(assignment, c):
                    unsat_clauses.append(c)
            
            if not unsat_clauses:
                return assignment, True
            
            # Pick a random unsatisfied clause
            clause = unsat_clauses[rng.randrange(len(unsat_clauses))]
            
            # Evaluate each literal in the clause
            candidates = []
            for var_idx, is_pos in clause:
                # Count how many clauses would be satisfied/unsatisfied if we flip this variable
                gain = 0
                loss = 0
                for ci in var_clauses[var_idx]:
                    # Check if clause ci is currently satisfied
                    currently_sat = is_satisfied(assignment, clause_vars[ci])
                    # After flip, would it be satisfied?
                    # The literal in clause ci for this variable
                    lit_val = None
                    for v, p in clause_vars[ci]:
                        if v == var_idx:
                            lit_val = p
                            break
                    if lit_val is None:
                        continue
                    # After flip, assignment[var_idx] becomes not assignment[var_idx]
                    new_val = not assignment[var_idx]
                    would_be_sat = (new_val == lit_val)
                    # Check if clause would be satisfied (any literal true)
                    clause_would_be_sat = False
                    for v2, p2 in clause_vars[ci]:
                        if v2 == var_idx:
                            if new_val == p2:
                                clause_would_be_sat = True
                                break
                        else:
                            if assignment[v2] == p2:
                                clause_would_be_sat = True
                                break
                    
                    if currently_sat and not clause_would_be_sat:
                        loss += 1
                    elif not currently_sat and clause_would_be_sat:
                        gain += 1
                
                candidates.append((gain - loss, var_idx))
            
            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])
            
            # Choose with noise: with probability noise, pick random; else pick best
            if rng.random() < noise:
                choice = candidates[rng.randrange(len(candidates))][1]
            else:
                # Pick from top candidates
                top_k = min(3, len(candidates))
                choice = candidates[rng.randrange(top_k)][1]
            
            # Flip the variable
            assignment[choice] = not assignment[choice]
            
            # Update unsat count
            new_unsat = count_unsat(assignment)
            if new_unsat < best_unsat:
                best_unsat = new_unsat
                best_assignment = assignment[:]
            
            if new_unsat == 0:
                return assignment, True
            
            # Adaptive noise: increase if stuck
            if step % 1000 == 0:
                noise = min(0.5, noise + 0.05)
        
        return best_assignment, (best_unsat == 0)
    
    # Multiple restarts with different seeds
    base_seed = 42
    restart = 0
    best_result = None
    best_unsat = float('inf')
    
    while time.time() < deadline - 5.0:
        # Create initial assignment
        rng_init = random.Random(base_seed + restart)
        assignment = [rng_init.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT with a step budget
        # Use more steps for earlier restarts, less for later
        max_steps = 50000 if restart < 5 else 30000
        assignment, solved = walk_sat(assignment, max_steps)
        
        if solved:
            return assignment
        
        unsat = count_unsat(assignment)
        if unsat < best_unsat:
            best_unsat = unsat
            best_result = assignment[:]
        
        # If we've made progress, continue; otherwise restart
        restart += 1
        base_seed += 1
    
    # Return best found
    if best_result is not None:
        return best_result
    return [False] * n

# EVOLVE-BLOCK-END
