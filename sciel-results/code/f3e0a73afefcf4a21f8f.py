import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Data structures for faster access
    # vars_to_clauses: list of lists of clause indices
    vars_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            vars_to_clauses[abs(lit) - 1].append(i)
    
    # State: 
    # assignment: list of None, True, False
    # None means unassigned
    assignment = [None] * n
    # active_clauses: set of clause indices that are not yet satisfied and not falsified
    active_clauses = set(range(len(clauses)))
    
    def is_satisfied(clause_idx):
        c = clauses[clause_idx]
        for lit in c:
            val = assignment[abs(lit) - 1]
            if val is None:
                continue
            if val == (lit > 0):
                return True
        return False

    def is_falsified(clause_idx):
        c = clauses[clause_idx]
        for lit in c:
            val = assignment[abs(lit) - 1]
            if val is None:
                return False
            if val != (lit > 0):
                return False
        return True

    def get_unassigned_lits(clause_idx):
        c = clauses[clause_idx]
        return [lit for lit in c if assignment[abs(lit) - 1] is None]

    # Step 1: Unit Propagation
    # Queue of unit clauses
    unit_queue = []
    for i in active_clauses:
        unassigned = get_unassigned_lits(i)
        if len(unassigned) == 1:
            unit_queue.append(i)
            
    while unit_queue:
        if time.time() > deadline:
            break
        ci = unit_queue.pop()
        if ci not in active_clauses:
            continue
        unassigned = get_unassigned_lits(ci)
        if len(unassigned) == 1:
            lit = unassigned[0]
            var_idx = abs(lit) - 1
            val = (lit > 0)
            if assignment[var_idx] is not None and assignment[var_idx] != val:
                # Conflict
                break 
            assignment[var_idx] = val
            # Update active clauses
            # Remove satisfied clauses, add new unit clauses
            new_units = []
            for cj in vars_to_clauses[var_idx]:
                if cj in active_clauses:
                    if is_satisfied(cj):
                        active_clauses.remove(cj)
                    elif len(get_unassigned_lits(cj)) == 1:
                        new_units.append(cj)
            unit_queue.extend(new_units)
            
    # Check if solved
    if all(assignment[i] is not None for i in range(n)):
        if all(is_satisfied(i) for i in range(len(clauses))):
            return assignment
    
    # If not solved, run WalkSAT on remaining unassigned vars
    # We will treat unassigned vars as random in the search, but effectively
    # we need to assign all vars.
    
    # Initialize unassigned vars randomly
    rng = random.Random(42)
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = rng.random() < 0.5
            
    # WalkSAT Parameters
    p = 0.5  # Random flip probability
    max_steps = 10000
    max_restarts = 50
    steps_per_restart = 5000
    
    best_assignment = assignment[:]
    best_unsat_count = sum(1 for i in range(len(clauses)) if not is_satisfied(i))
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Reset unassigned vars to random if we are restarting? 
        # Actually, for WalkSAT, we usually keep the current state and just continue,
        # or restart from scratch. Let's restart from scratch for diversity.
        for i in range(n):
            if assignment[i] is None or restart > 0:
                assignment[i] = rng.random() < 0.5
                
        # Re-apply unit propagation? No, let's just run WalkSAT.
        # But we should respect the previously fixed variables from unit prop?
        # The unit prop variables are "forced". If we flip them, we might break them.
        # Better approach: Fix the unit-propagated variables and only search on the rest.
        
        fixed_vars = set(i for i in range(n) if assignment[i] is not None and i in [abs(lit)-1 for lit in clauses[0] if False]) # This logic is flawed.
        
        # Let's identify which variables were fixed by unit propagation in the first pass.
        # We didn't store them separately. Let's just run WalkSAT on ALL variables.
        # The unit propagation result is a good initial assignment.
        
        # If we restart, we should probably re-randomize only the non-fixed ones?
        # Or just randomize all. Given n=50, randomizing all is cheap.
        
        # Let's store the "hard" assignments from unit prop separately to keep them fixed?
        # Actually, if unit propagation led to a conflict or didn't solve it,
        # the remaining subproblem is the hard part.
        # Let's just run WalkSAT on the full instance, starting from the current assignment.
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return best_assignment
                
            # Count unsatisfied clauses
            unsat_clauses = [i for i in range(len(clauses)) if not is_satisfied(i)]
            if not unsat_clauses:
                return assignment
                
            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                
            if step % 100 == 0 and current_unsat == best_unsat_count and step > 1000:
                # Stagnation, break early to restart
                break
                
            clause_idx = rng.choice(unsat_clauses)
            clause = clauses[clause_idx]
            
            # Choose literal to flip
            if rng.random() < p:
                lit = rng.choice(clause)
            else:
                # Min-conflicts: pick lit that minimizes number of unsatisfied clauses after flip
                best_lit = None
                best_score = float('inf')
                for lit in clause:
                    var_idx = abs(lit) - 1
                    # Simulate flip
                    old_val = assignment[var_idx]
                    assignment[var_idx] = not old_val
                    # Count unsat
                    count = 0
                    for cj in vars_to_clauses[var_idx]:
                        if not is_satisfied(cj):
                            count += 1
                    # Restore
                    assignment[var_idx] = old_val
                    
                    if count < best_score:
                        best_score = count
                        best_lit = lit
                        
                if best_lit is not None:
                    lit = best_lit
                else:
                    lit = rng.choice(clause)
                    
            var_idx = abs(lit) - 1
            assignment[var_idx] = not assignment[var_idx]
            
    return best_assignment
