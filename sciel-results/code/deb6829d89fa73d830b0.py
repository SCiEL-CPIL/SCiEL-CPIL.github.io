import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Preprocess: Build variable-to-clause index
    var_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_clauses[abs(lit) - 1].append(i)
            
    # Initialize assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Initialize clause satisfaction status
    clause_unsat_count = [0] * len(clauses)
    unsat_clauses = []
    
    for c_idx, clause in enumerate(clauses):
        count = 0
        for lit in clause:
            var = abs(lit) - 1
            val = assignment[var]
            is_true = (lit > 0)
            if val != is_true:
                count += 1
        clause_unsat_count[c_idx] = count
        if count > 0:
            unsat_clauses.append(c_idx)
            
    total_unsat = len(unsat_clauses)
    
    start_time = time.time()
    timeout = 320.0
    
    # WalkSAT parameters
    p = 0.5 
    
    steps_since_progress = 0
    best_total_unsat = total_unsat
    
    while total_unsat > 0:
        if time.time() - start_time > timeout:
            break
            
        if not unsat_clauses:
            break
            
        # Pick a random unsatisfied clause
        c_idx = rng.choice(unsat_clauses)
        clause = clauses[c_idx]
        
        # Choose a variable to flip
        if rng.random() < p:
            lit = rng.choice(clause)
        else:
            # MinOnes heuristic: pick var in clause that minimizes new unsat clauses
            best_lit = None
            min_new_unsat = float('inf')
            for lit in clause:
                var = abs(lit) - 1
                # Count how many other clauses would become unsatisfied
                # This requires checking if var is the only true literal in any satisfied clause
                new_unsat = 0
                for j in var_clauses[var]:
                    if j == c_idx:
                        continue
                    if clause_unsat_count[j] == 0:
                        # Clause j is satisfied. Check if var is true in j
                        # Find lit in j for var
                        for l in clauses[j]:
                            if abs(l) == var + 1:
                                is_true = (l > 0)
                                if assignment[var] == is_true:
                                    # Var is true in j. If we flip it, j becomes unsatisfied
                                    # Only if it was the ONLY true literal.
                                    # clause_unsat_count[j] is 0, so all literals are true.
                                    # So flipping any literal makes it unsatisfied.
                                    new_unsat += 1
                                break
                if new_unsat < min_new_unsat:
                    min_new_unsat = new_unsat
                    best_lit = lit
            lit = best_lit
            
        var = abs(lit) - 1
        old_val = assignment[var]
        new_val = not old_val
        assignment[var] = new_val
        
        # Update counts
        for j in var_clauses[var]:
            # Find literal in clause j corresponding to var
            lit_in_j = None
            for l in clauses[j]:
                if abs(l) == var + 1:
                    lit_in_j = l
                    break
            
            if lit_in_j is None:
                continue
                
            is_true = (lit_in_j > 0)
            was_true = old_val == is_true
            is_now_true = new_val == is_true
            
            if was_true and not is_now_true:
                clause_unsat_count[j] += 1
                if clause_unsat_count[j] == 1:
                    unsat_clauses.append(j)
                    total_unsat += 1
            elif not was_true and is_now_true:
                clause_unsat_count[j] -= 1
                if clause_unsat_count[j] == 0:
                    # Remove from unsat_clauses
                    # O(M) operation
                    if j in unsat_clauses:
                        unsat_clauses.remove(j)
                    total_unsat -= 1

        if total_unsat < best_total_unsat:
            best_total_unsat = total_unsat
            steps_since_progress = 0
        else:
            steps_since_progress += 1
            
        if steps_since_progress > 2000:
            # Restart
            assignment = [rng.random() < 0.5 for _ in range(n)]
            clause_unsat_count = [0] * len(clauses)
            unsat_clauses = []
            for c_idx, clause in enumerate(clauses):
                count = 0
                for lit in clause:
                    var = abs(lit) - 1
                    val = assignment[var]
                    is_true = (lit > 0)
                    if val != is_true:
                        count += 1
                clause_unsat_count[c_idx] = count
                if count > 0:
                    unsat_clauses.append(c_idx)
            total_unsat = len(unsat_clauses)
            best_total_unsat = total_unsat
            steps_since_progress = 0

    return assignment
