import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-process clauses for faster access
    # Store as (var_index, is_positive)
    processed_clauses = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        p_clause = []
        for lit in clause:
            var = abs(lit) - 1
            is_pos = lit > 0
            p_clause.append((var, is_pos))
            var_to_clauses[var].append(i)
        processed_clauses.append(p_clause)
        
    m = len(clauses)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters
    max_steps_per_run = 5000
    noise_prob = 0.5
    
    while time.time() < deadline:
        # Random initialization
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Helper to check if clause is satisfied
        def is_satisfied(ci, assign):
            for var, is_pos in processed_clauses[ci]:
                if assign[var] == is_pos:
                    return True
            return False
        
        # Helper to count unsatisfied clauses
        def count_unsat(assign):
            return sum(0 if is_satisfied(i, assign) else 1 for i in range(m))
        
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        steps = 0
        last_improvement = 0
        min_unsat_in_run = current_unsat
        
        while steps < max_steps_per_run and time.time() < deadline:
            # Check if solved
            if current_unsat == 0:
                return assignment
            
            if current_unsat < min_unsat_in_run:
                min_unsat_in_run = current_unsat
                last_improvement = steps
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                    
            # If stuck for a while, restart
            if steps - last_improvement > 1000:
                break
            
            # Pick a random unsatisfied clause
            # To speed up, we could maintain a list of unsat clauses, 
            # but for n=50, finding one is O(m). 
            # Let's find one randomly.
            unsat_clauses = [i for i in range(m) if not is_satisfied(i, assignment)]
            if not unsat_clauses:
                return assignment
            target_clause = random.choice(unsat_clauses)
            
            # Evaluate flipping each variable in the target clause
            best_var = None
            best_score = -float('inf')
            candidates = []
            
            for var, is_pos in processed_clauses[target_clause]:
                # Calculate score if we flip var
                # Score = (clauses that become sat) - (clauses that become unsat)
                # Note: The target clause itself will become satisfied (since it's currently unsat and we flip one of its literals).
                # So it contributes +1 to the gain.
                
                score = 0
                
                # Check all clauses containing this variable
                for ci in var_to_clauses[var]:
                    # Determine current status
                    cur_sat = is_satisfied(ci, assignment)
                    
                    # Determine status if we flip var
                    # If assignment[var] is True, it becomes False.
                    # If assignment[var] is False, it becomes True.
                    # We need to check if the clause is satisfied by any other literal, 
                    # or if this literal was the one satisfying it.
                    
                    # Simpler: just construct the new assignment state for this variable?
                    # No, that's expensive. 
                    # Logic:
                    # Let new_val = not assignment[var]
                    # Clause is satisfied if any (assign[lit_var] == lit_is_pos) is True.
                    # For clauses NOT containing var, status doesn't change.
                    # For clauses containing var:
                    #   If cur_sat is True:
                    #     Check if var was the ONLY satisfying literal.
                    #     If yes, it becomes unsat (score -1).
                    #     If no, it stays sat (score 0).
                    #   If cur_sat is False:
                    #     Check if flipping var makes it sat.
                    #     i.e., is (new_val == is_pos_in_this_clause)?
                    #     If yes, it becomes sat (score +1).
                    #     If no, it stays unsat (score 0).
                    
                    new_val = not assignment[var]
                    
                    # Find the literal of var in this clause
                    # We know var is in this clause. We need to know its sign in this specific clause.
                    # We can iterate processed_clauses[ci] to find var.
                    # Since clauses are small, this is fine.
                    var_sign_in_ci = None
                    for v, s in processed_clauses[ci]:
                        if v == var:
                            var_sign_in_ci = s
                            break
                    
                    if cur_sat:
                        # Check if var is the only satisfier
                        only_satisfier = True
                        for v, s in processed_clauses[ci]:
                            if v != var and assignment[v] == s:
                                only_satisfier = False
                                break
                        if only_satisfier:
                            score -= 1
                    else:
                        # Check if flipping var satisfies it
                        if var_sign_in_ci is not None and new_val == var_sign_in_ci:
                            score += 1
                
                candidates.append((score, var))
                
                if score > best_score:
                    best_score = score
                    best_var = var
            
            # Decide which variable to flip
            if random.random() < noise_prob:
                # Pick best
                var_to_flip = best_var
            else:
                # Pick random from clause
                var_to_flip = random.choice([v for v, s in processed_clauses[target_clause]])
            
            # Flip
            assignment[var_to_flip] = not assignment[var_to_flip]
            
            # Update current_unsat
            # Instead of recounting all, we can update incrementally, 
            # but recounting is O(m) which is small.
            current_unsat = count_unsat(assignment)
            steps += 1
            
        # End of run, restart if not solved
        
    return best_assignment
