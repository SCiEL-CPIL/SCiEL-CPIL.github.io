<<<<<<< SEARCH
        # Adaptive parameters
        # For planted 3-SAT, the ratio of clauses to variables matters
        # Typical planted instances have m/n around 4-5
        m = num_clauses
        ratio = m / n if n > 0 else 0
        
        # Number of steps per restart
        # More steps for harder instances
        if ratio < 4.0:
            steps_per_restart = 5000
        elif ratio < 5.0:
            steps_per_restart = 10000
        else:
            steps_per_restart = 20000
        
        # Track best within this restart
        restart_best_unsat = unsat_count
        restart_best_assignment = list(assignment)
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Recompute unsat clauses
            unsat_clauses = []
            for clause in clauses_opt:
                sat = False
                for var_idx, polarity in clause:
                    if assignment[var_idx] == polarity:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(clause)
            
            if not unsat_clauses:
                return assignment
            
            current_unsat = len(unsat_clauses)
            if current_unsat < restart_best_unsat:
                restart_best_unsat = current_unsat
                restart_best_assignment = list(assignment)
            
            # Hybrid step
            hybrid_step(assignment, unsat_clauses, rng)
=======
        # Adaptive parameters
        # For planted 3-SAT, the ratio of clauses to variables matters
        # Typical planted instances have m/n around 4-5
        m = num_clauses
        ratio = m / n if n > 0 else 0
        
        # Number of steps per restart
        # Reduced to ensure we finish within time limit with more restarts
        if ratio < 4.0:
            steps_per_restart = 3000
        elif ratio < 5.0:
            steps_per_restart = 5000
        else:
            steps_per_restart = 8000
            
        # Track best within this restart
        restart_best_unsat = unsat_count
        restart_best_assignment = list(assignment)
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Recompute unsat clauses
            unsat_clauses = []
            for clause in clauses_opt:
                sat = False
                for var_idx, polarity in clause:
                    if assignment[var_idx] == polarity:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(clause)
            
            if not unsat_clauses:
                return assignment
            
            current_unsat = len(unsat_clauses)
            if current_unsat < restart_best_unsat:
                restart_best_unsat = current_unsat
                restart_best_assignment = list(assignment)
            
            # Fast WalkSAT step
            fast_walksat_step(assignment, unsat_clauses, rng)
>>>>>>> REPLACE

Also increase `max_restarts` to take advantage of the faster steps.

