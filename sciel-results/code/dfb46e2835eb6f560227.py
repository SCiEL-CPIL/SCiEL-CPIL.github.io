import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-compute variable to clause mapping
    # var_clauses[i] is a list of clause indices containing variable i+1
    var_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_clauses[var].append(i)
            
    # We need to check satisfaction quickly.
    # is_satisfied(c, assignment)
    def is_satisfied(c_idx, assign):
        clause = clauses[c_idx]
        for lit in clause:
            if assign[abs(lit)-1] == (lit > 0):
                return True
        return False

    # Initialize
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Initial unsat set
    unsat = set()
    for i in range(len(clauses)):
        if not is_satisfied(i, assignment):
            unsat.add(i)
            
    if not unsat:
        return assignment

    start_time = time.time()
    deadline = start_time + 290.0 # Leave 10s margin
    
    # Parameters
    walk_prob = 0.4
    max_steps_per_restart = 200000 # High limit, time is the real limit
    
    best_assignment = assignment[:]
    best_unsat_count = len(unsat)
    
    while time.time() < deadline:
        # Restart logic: if we've been going for a while, or if unsat count is high, maybe restart?
        # Simple restart: every N steps, or if no improvement in M steps.
        # Let's do a simple step limit per restart.
        steps_in_restart = 0
        while steps_in_restart < max_steps_per_restart:
            if time.time() > deadline:
                return best_assignment
                
            if not unsat:
                return assignment
                
            # Check if we found a better assignment (fewer unsat)
            current_unsat_count = len(unsat)
            if current_unsat_count < best_unsat_count:
                best_unsat_count = current_unsat_count
                best_assignment = assignment[:]
                
            # Pick a random unsatisfied clause
            c_idx = rng.choice(list(unsat))
            clause = clauses[c_idx]
            
            # Find best variable to flip
            # Cost: number of satisfied clauses containing var (excluding c_idx)
            min_cost = float('inf')
            best_vars = []
            
            for lit in clause:
                var = abs(lit) - 1
                cost = 0
                for c2_idx in var_clauses[var]:
                    if c2_idx == c_idx:
                        continue
                    # If c2 is satisfied, flipping var might break it
                    # We need to check if c2 is satisfied.
                    # Optimization: We can check if c2 is in unsat. If not, it is satisfied.
                    if c2_idx not in unsat:
                        cost += 1
                        
                if cost < min_cost:
                    min_cost = cost
                    best_vars = [var]
                elif cost == min_cost:
                    best_vars.append(var)
                    
            # Decide which variable to flip
            if rng.random() < walk_prob:
                # Random walk
                lit = rng.choice(clause)
                var = abs(lit) - 1
            else:
                # Min conflicts
                var = rng.choice(best_vars)
                
            # Flip variable
            assignment[var] = not assignment[var]
            
            # Update unsat set
            # We only need to check clauses containing var
            for c2_idx in var_clauses[var]:
                # Determine if c2 was satisfied before flip
                # We don't store previous state easily, but we can infer:
                # If c2 is in unsat, it was unsat.
                # If c2 is not in unsat, it was sat.
                was_unsat = c2_idx in unsat
                
                # Check if c2 is satisfied now
                is_sat_now = is_satisfied(c2_idx, assignment)
                
                if was_unsat and is_sat_now:
                    unsat.remove(c2_idx)
                elif not was_unsat and not is_sat_now:
                    unsat.add(c2_idx)
                    
            steps_in_restart += 1
            
    return best_assignment
