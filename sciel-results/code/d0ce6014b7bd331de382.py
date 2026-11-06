"""Improved 3-SAT solver for planted satisfiable instances."""
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)
    
    start_time = time.time()
    time_limit = 290.0
    
    # Convert to 0-based, store as (var, sign) where sign=1 for positive, -1 for negative
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, 1 if l > 0 else -1) for l in c]
        clause_vars.append(cv)
    
    m = len(clause_vars)
    
    # Build variable-to-clauses index
    var_to_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var_idx, sign in cv:
            var_to_clauses[var_idx].append(ci)
    
    # Precompute for each clause, for each literal position, the variable and sign
    # Also build a fast structure: for each variable, list of (clause_idx, sign_in_clause)
    var_clause_signs = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var_idx, sign in cv:
            var_clause_signs[var_idx].append((ci, sign))
    
    def is_satisfied_fast(ci, assignment):
        """Check if clause ci is satisfied by assignment."""
        cv = clause_vars[ci]
        for var_idx, sign in cv:
            if assignment[var_idx] == (sign == 1):
                return True
        return False
    
    def count_unsat(assignment):
        return sum(1 for i in range(m) if not is_satisfied_fast(i, assignment))
    
    def random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]
    
    def compute_damage(var_idx, new_val, assignment, current_unsat_set=None):
        """
        Compute the damage of setting var_idx to new_val.
        Damage = number of currently satisfied clauses that would become unsatisfied.
        Gain = number of currently unsatisfied clauses that would become satisfied.
        Net change in unsat count = -gain + damage.
        We want to minimize net change (i.e., minimize damage - gain).
        """
        damage = 0
        gain = 0
        new_val_bool = new_val
        
        for ci, sign in var_clause_signs[var_idx]:
            # After setting var_idx to new_val_bool:
            # The literal (var_idx, sign) is satisfied if new_val_bool == (sign == 1)
            literal_satisfied = (new_val_bool == (sign == 1))
            
            if literal_satisfied:
                # This clause will be satisfied after the flip
                # Check if it's currently satisfied
                if not is_satisfied_fast(ci, assignment):
                    gain += 1
            else:
                # This clause's literal for var_idx is NOT satisfied after flip
                # Check if this clause is currently satisfied and var_idx is the only satisfying literal
                if is_satisfied_fast(ci, assignment):
                    # Check if var_idx currently satisfies this clause
                    current_val = assignment[var_idx]
                    if current_val == (sign == 1):
                        # var_idx currently satisfies this clause
                        # After flip, check if any other literal in the clause satisfies it
                        cv = clause_vars[ci]
                        still_satisfied = False
                        for v, s in cv:
                            if v == var_idx:
                                continue
                            if assignment[v] == (s == 1):
                                still_satisfied = True
                                break
                        if not still_satisfied:
                            damage += 1
        
        return damage - gain
    
    def walksat(assignment, max_steps, p_noise):
        """
        WalkSAT with damage-minimizing variable selection.
        p_noise: probability of making a random flip within the clause.
        """
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment, True
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return assignment, False
            
            # Find an unsatisfied clause
            # Use random sampling for speed when m is large
            if m <= 500:
                unsat_clauses = [i for i in range(m) if not is_satisfied_fast(i, assignment)]
                if not unsat_clauses:
                    return assignment, True
                chosen = rng.choice(unsat_clauses)
            else:
                # Sample until we find an unsatisfied clause
                chosen = None
                for _ in range(20):
                    c = rng.randrange(m)
                    if not is_satisfied_fast(c, assignment):
                        chosen = c
                        break
                if chosen is None:
                    # Fallback: scan
                    for i in range(m):
                        if not is_satisfied_fast(i, assignment):
                            chosen = i
                            break
                    if chosen is None:
                        return assignment, True
            
            cv = clause_vars[chosen]
            
            if rng.random() < p_noise:
                # Random flip
                var_idx, sign = rng.choice(cv)
                assignment[var_idx] = (sign == 1)
            else:
                # Damage-minimizing flip
                best_var = -1
                best_score = float('inf')
                for var_idx, sign in cv:
                    new_val = (sign == 1)
                    if assignment[var_idx] == new_val:
                        # Flipping this would make the literal unsatisfied; 
                        # but we're choosing from an unsatisfied clause, so this literal
                        # is currently not satisfying. Actually in an unsatisfied clause,
                        # none of the literals are satisfied, so flipping to satisfy
                        # is always "forward".
                        # But we should consider: maybe flipping away from satisfying
                        # a different clause is worse. Let's just compute damage.
                        pass
                    score = compute_damage(var_idx, new_val, assignment)
                    if score < best_score:
                        best_score = score
                        best_var = var_idx
                        best_flip = new_val
                if best_var >= 0:
                    assignment[best_var] = best_flip
                else:
                    var_idx, sign = rng.choice(cv)
                    assignment[var_idx] = (sign == 1)
            
            # Periodically check if solved
            if step % 100 == 0:
                if count_unsat(assignment) == 0:
                    return assignment, True
        
        return assignment, count_unsat(assignment) == 0
    
    # Main search with restarts
    best_assignment = random_assignment()
    best_unsat = count_unsat(best_assignment)
    
    # Dynamic noise: start with higher noise, decrease over time
    base_noise = 0.4
    min_noise = 0.1
    
    restart = 0
    max_restarts = 200
    steps_per_restart = 3000
    
    while time.time() - start_time < time_limit:
        if restart >= max_restarts:
            break
        
        # Choose initial assignment
        if restart == 0:
            assignment = best_assignment[:]
        else:
            # With some probability, start from best; otherwise random
            if rng.random() < 0.2:
                assignment = best_assignment[:]
            else:
                assignment = random_assignment()
        
        # Noise decreases with restart count (cools down)
        progress = min(restart / max_restarts, 1.0)
        p_noise = base_noise - (base_noise - min_noise) * progress
        
        assignment, solved = walksat(assignment, steps_per_restart, p_noise)
        
        if solved:
            return assignment
        
        cur_unsat = count_unsat(assignment)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        
        restart += 1
    
    return best_assignment
