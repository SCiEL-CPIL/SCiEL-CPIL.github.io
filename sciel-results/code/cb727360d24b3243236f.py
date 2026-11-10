import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290

    # Preprocessing
    clause_vars = [] # list of lists of (var_idx, sign)
    var_occurrences = [[] for _ in range(n)] # list of lists of (clause_idx, sign)
    
    for ci, clause in enumerate(clauses):
        lits = []
        for l in clause:
            vi = abs(l) - 1
            sign = 1 if l > 0 else -1
            lits.append((vi, sign))
            var_occurrences[vi].append((ci, sign))
        clause_vars.append(lits)

    def compute_initial_state(assignment):
        counts = [0] * m
        for ci in range(m):
            cnt = 0
            for vi, sign in clause_vars[ci]:
                if assignment[vi] == (sign == 1):
                    cnt += 1
            counts[ci] = cnt
        unsat = sum(1 for c in counts if c == 0)
        return counts, unsat

    def flip_and_get_delta(assignment, vi, counts, unsat_count):
        """
        Flips variable vi, updates counts and unsat_count.
        Returns the delta in unsat_count.
        """
        delta = 0
        old_val = assignment[vi]
        new_val = not old_val
        
        for ci, sign in var_occurrences[vi]:
            was_sat = old_val == (sign == 1)
            
            if was_sat:
                # Literal becomes unsatisfied
                # counts[ci] decreases by 1
                if counts[ci] == 1:
                    # Clause becomes unsatisfied
                    delta += 1
                counts[ci] -= 1
            else:
                # Literal becomes satisfied
                # counts[ci] increases by 1
                if counts[ci] == 0:
                    # Clause becomes satisfied
                    delta -= 1
                counts[ci] += 1
        
        assignment[vi] = new_val
        return delta

    best_unsat = m + 1
    best_assignment = None

    # Simulated Annealing with smarter variable selection
    # Focus on variables in unsatisfied clauses
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 12345 + 6789)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        counts, unsat_count = compute_initial_state(assignment)
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        # SA Parameters with adaptive cooling
        T0 = 8.0
        T = T0
        # Use geometric cooling with periodic reheating
        cooling_rate = 0.9995
        reheat_interval = 50000
        reheat_temp = 3.0
        
        start_time = time.time()
        steps_in_restart = 0
        max_steps_per_restart = 300000
        
        while time.time() < deadline - 5 and steps_in_restart < max_steps_per_restart:
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                
            # Smarter variable selection: prefer variables in unsatisfied clauses
            # with probability proportional to their contribution
            if unsat_count > 0 and rng.random() < 0.7:
                # Find a random unsatisfied clause and pick a variable from it
                # Build list of unsatisfied clause indices
                unsat_clauses = [ci for ci in range(m) if counts[ci] == 0]
                if unsat_clauses:
                    ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
                    vi = clause_vars[ci][rng.randrange(len(clause_vars[ci]))][0]
                else:
                    vi = rng.randrange(n)
            else:
                # Random variable for exploration
                vi = rng.randrange(n)
            
            # Calculate delta
            delta = 0
            for ci, sign in var_occurrences[vi]:
                was_sat = assignment[vi] == (sign == 1)
                if was_sat:
                    if counts[ci] == 1:
                        delta += 1
                else:
                    if counts[ci] == 0:
                        delta -= 1
            
            # Acceptance with temperature-dependent probability
            if delta < 0 or (T > 1e-6 and rng.random() < math.exp(-delta / T)):
                # Accept move
                for ci, sign in var_occurrences[vi]:
                    was_sat = assignment[vi] == (sign == 1)
                    if was_sat:
                        counts[ci] -= 1
                    else:
                        counts[ci] += 1
                assignment[vi] = not assignment[vi]
                unsat_count += delta
                
            # Cool with periodic reheating
            T *= cooling_rate
            if steps_in_restart % reheat_interval == 0 and T < reheat_temp:
                T = reheat_temp
            
            steps_in_restart += 1
            
        # End of restart loop
        # If we didn't solve, we keep best_assignment

    # Final polish: aggressive local search with multiple passes
    if best_assignment is not None and best_unsat > 0:
        assignment = best_assignment[:]
        counts, unsat_count = compute_initial_state(assignment)
        
        # Multiple passes of local search with different strategies
        for polish_round in range(10):
            if time.time() > deadline - 1:
                break
            if unsat_count == 0:
                return assignment
                
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                
                # Try all variables, but prioritize those in unsatisfied clauses
                if unsat_count > 0:
                    # Get variables from unsatisfied clauses first
                    unsat_clauses = [ci for ci in range(m) if counts[ci] == 0]
                    candidate_vars = set()
                    for ci in unsat_clauses:
                        for vi, sign in clause_vars[ci]:
                            candidate_vars.add(vi)
                    
                    # Try candidate variables first, then others
                    var_order = list(candidate_vars) + [vi for vi in range(n) if vi not in candidate_vars]
                    rng.shuffle(var_order)
                else:
                    var_order = list(range(n))
                    rng.shuffle(var_order)
                
                for vi in var_order:
                    if time.time() > deadline - 1:
                        break
                    
                    # Calculate delta
                    delta = 0
                    for ci, sign in var_occurrences[vi]:
                        was_sat = assignment[vi] == (sign == 1)
                        if was_sat:
                            if counts[ci] == 1:
                                delta += 1
                        else:
                            if counts[ci] == 0:
                                delta -= 1
                    
                    # Accept if it improves or is neutral (with some probability)
                    if delta < 0 or (delta == 0 and rng.random() < 0.3):
                        for ci, sign in var_occurrences[vi]:
                            was_sat = assignment[vi] == (sign == 1)
                            if was_sat:
                                counts[ci] -= 1
                            else:
                                counts[ci] += 1
                        assignment[vi] = not assignment[vi]
                        unsat_count += delta
                        improved = True
                        if unsat_count == 0:
                            return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
        
        if unsat_count == 0:
            return assignment

    if best_assignment is not None:
        return best_assignment
    return [False] * n
