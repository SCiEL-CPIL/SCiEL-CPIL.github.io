import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290
    
    # Pre-compute variable to clauses mapping
    # var_to_clauses[i] = list of clause indices
    var_to_clauses = [[] for _ in range(n)]
    # clause_literal_map[c] = {var_idx: literal_value}
    clause_literal_map = [{} for _ in range(len(clauses))]
    
    for i, clause in enumerate(clauses):
        for l in clause:
            v = abs(l) - 1
            var_to_clauses[v].append(i)
            clause_literal_map[i][v] = l
            
    rng = random.Random(0)
    best_assignment = None
    best_unsat = float('inf')
    
    # WalkSAT Parameters
    p_noise = 0.2  # Reduced noise for more directed search
    steps_per_restart = 20000
    max_restarts = 2000
    
    # Activity tracking for VSIDS-like variable selection
    activity = [1.0] * n
    decay = 0.99
    
    def is_satisfied(clause, assignment):
        for l in clause:
            v = assignment[abs(l)-1]
            if (l > 0 and v) or (l < 0 and not v):
                return True
        return False

    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize unsat set
        unsat_indices = set()
        for i, c in enumerate(clauses):
            if not is_satisfied(c, assignment):
                unsat_indices.add(i)
                
        if not unsat_indices:
            return assignment
            
        cur_unsat = len(unsat_indices)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
        # Local search
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            if not unsat_indices:
                return assignment
                
            # Pick a random unsat clause
            c_idx = rng.choice(list(unsat_indices))
            clause = clauses[c_idx]
            
            # Evaluate flip for each literal in the clause
            best_var = None
            best_score = float('inf')
            best_act = -1.0
            
            for l in clause:
                var_idx = abs(l) - 1
                
                # Calculate score: number of clauses that will become unsat (broken)
                # Note: The current clause will be satisfied.
                # Other clauses containing var_idx:
                # If they were satisfied, check if they remain satisfied.
                # If they were unsat, they will become satisfied (gain).
                
                # We want to minimize breakage.
                # Breakage = number of satisfied clauses that become unsat.
                
                # Hypothetical new value
                # The literal l is either var_idx or ~var_idx.
                # If l > 0, literal is var. Flipping var changes literal value.
                # If l < 0, literal is ~var. Flipping var changes literal value.
                
                # In all cases, flipping var_idx changes the truth value of the literal l.
                # Since the current clause is unsat, all its literals are False.
                # So flipping var_idx will make literal l True.
                
                # Check other clauses containing var_idx
                broken = 0
                for other_c_idx in var_to_clauses[var_idx]:
                    if other_c_idx == c_idx:
                        continue
                    
                    other_clause = clauses[other_c_idx]
                    # Was this clause satisfied?
                    # If it was unsat, flipping var_idx might satisfy it (if the literal matches the flip) or not.
                    # Actually, if it was unsat, all literals are False.
                    # The literal for var_idx in this clause is some l_other.
                    # Flipping var_idx changes the value of l_other.
                    # If l_other becomes True, the clause becomes Satisfied. (Gain)
                    # If l_other becomes False (was it True? No, clause was unsat so it was False), it stays False.
                    # So unsat clauses never break, they can only be fixed.
                    
                    # Therefore, we only care about SATISFIED clauses that might break.
                    
                    if other_c_idx in unsat_indices:
                        # It's unsat, so it won't break. It might become sat.
                        # We don't need to count this as "broken".
                        continue
                    
                    # It is currently satisfied.
                    # Check if it remains satisfied after flipping var_idx.
                    # It remains satisfied if there is at least one other true literal.
                    
                    # Quick check:
                    # Iterate through literals in other_clause
                    # If any literal (other than the one with var_idx) is true, it stays sat.
                    # The literal with var_idx might become false.
                    
                    # Find the literal in other_clause that corresponds to var_idx
                    # There is exactly one.
                    
                    other_satisfied_without_var = False
                    for ol in other_clause:
                        o_var = abs(ol) - 1
                        if o_var == var_idx:
                            continue
                        # Check if this other literal is currently true
                        o_val = assignment[o_var]
                        if (ol > 0 and o_val) or (ol < 0 and not o_val):
                            other_satisfied_without_var = True
                            break
                            
                    if not other_satisfied_without_var:
                        # This clause relied solely on var_idx to be satisfied.
                        # Flipping var_idx will make that literal false.
                        # So the clause becomes unsat.
                        broken += 1
                        
                # Score is 'broken'
                # Tie-breaker: prefer variable with higher activity
                if broken < best_score or (broken == best_score and activity[var_idx] > best_act):
                    best_score = broken
                    best_var = var_idx
                    best_act = activity[var_idx]
                    
            # Decide flip
            if rng.random() < p_noise:
                # Random flip from clause
                l = rng.choice(clause)
                var_idx = abs(l) - 1
            else:
                var_idx = best_var
                
            if var_idx is None:
                continue
                
            # Perform flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update activity
            for l in clause:
                v = abs(l) - 1
                activity[v] *= decay
            activity[var_idx] += 1.0
            
            # Update unsat_indices
            # We need to update the status of all clauses containing var_idx
            
            for c_idx in var_to_clauses[var_idx]:
                c = clauses[c_idx]
                sat = is_satisfied(c, assignment)
                if sat:
                    if c_idx in unsat_indices:
                        unsat_indices.remove(c_idx)
                else:
                    if c_idx not in unsat_indices:
                        unsat_indices.add(c_idx)
                        
        # End local search loop
        if time.time() > deadline - 5:
            break

    if best_assignment:
        return best_assignment
    return [False] * n
