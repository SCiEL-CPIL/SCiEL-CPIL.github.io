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

    # Simulated Annealing Parameters
    # We will try multiple restarts with different cooling rates or initial temps
    # Or just one long SA run with periodic reheating?
    # Given n=150, a single SA run with many steps is likely sufficient if tuned well.
    # Let's do restarts to diversify.
    
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

        # SA Parameters
        # Initial temperature T0 should be high enough to allow exploration.
        # Delta U is at most the number of clauses containing v (usually small, e.g., < 10).
        # So T0 ~ 5-10 is a reasonable start.
        T0 = 5.0
        T = T0
        cooling_rate = 0.999 # Slower cooling for more exploration
        # Or adaptive: cool based on steps
        # Let's use a fixed number of steps per restart or time-based.
        # Time-based is safer.
        
        start_time = time.time()
        steps_in_restart = 0
        max_steps_per_restart = 200000 # Adjust based on speed
        
        while time.time() < deadline - 5 and steps_in_restart < max_steps_per_restart:
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                
            # Pick random variable
            vi = rng.randrange(n)
            
            # Calculate delta without actually flipping yet?
            # Or flip, calculate, and unflip if rejected?
            # Flipping and checking is easier with the incremental logic.
            # But we need to revert if rejected.
            
            # Let's calculate delta first to avoid modifying state unnecessarily.
            # Delta calculation:
            delta = 0
            for ci, sign in var_occurrences[vi]:
                was_sat = assignment[vi] == (sign == 1)
                if was_sat:
                    if counts[ci] == 1:
                        delta += 1
                else:
                    if counts[ci] == 0:
                        delta -= 1
            
            # Acceptance
            if delta < 0 or (T > 1e-6 and rng.random() < math.exp(-delta / T)):
                # Accept move
                # Apply the flip
                for ci, sign in var_occurrences[vi]:
                    was_sat = assignment[vi] == (sign == 1)
                    if was_sat:
                        counts[ci] -= 1
                    else:
                        counts[ci] += 1
                assignment[vi] = not assignment[vi]
                unsat_count += delta
                
            # Cool
            T *= cooling_rate
            steps_in_restart += 1
            
        # End of restart loop
        # If we didn't solve, we keep best_assignment

    if best_assignment is not None:
        return best_assignment
    return [False] * n
