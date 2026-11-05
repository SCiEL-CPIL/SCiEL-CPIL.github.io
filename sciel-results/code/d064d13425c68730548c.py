import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-process clauses to 0-based indices and signs
    # Sign: 1 for positive, -1 for negative
    # Variable: 0 to n-1
    processed_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            pc.append((v, s))
        processed_clauses.append(pc)
        
    rng = random.Random(0)
    
    # Time management
    start_time = time.time()
    deadline = 290.0  # 10s buffer
    
    # Search parameters
    # Restart every `steps_per_restart` steps
    steps_per_restart = 500
    # Probability of making a random flip within the clause (otherwise greedy)
    p_random = 0.5
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    def count_unsatisfied(assign):
        count = 0
        for c in processed_clauses:
            sat = False
            for v, s in c:
                if assign[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    # Main search loop
    while time.time() < deadline:
        # Initialize or Restart
        # We use a flag to indicate if we just restarted
        # Simpler: track steps_done
        # But we need to handle the first iteration specially or just let it run
        
        # To avoid complex state, we will do a batch of steps, then restart
        # However, the time check is inside the step loop.
        
        # Let's use a structure:
        # while time < deadline:
        #    if restart needed:
        #       init random
        #       update best
        #    if solved: return
        #    find unsat clauses
        #    if none: return
        #    pick clause
        #    pick var
        #    flip
        #    update best
        #    steps++
        
        # We need a step counter that resets on restart
        # And a flag for first iteration
        
        if not hasattr(solve, '_current_state') or solve._current_state is None:
             # This approach is stateful and bad for multiple calls.
             # Let's just use local variables.
             pass

    # The above stateful idea is bad. Let's stick to local variables.
    
    steps_done = 0
    assignment = None
    
    while time.time() < deadline:
        # Check if we need to restart
        # We restart if assignment is None (first time) or steps_done >= steps_per_restart
        if assignment is None or steps_done >= steps_per_restart:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            steps_done = 0
            unsat_count = count_unsatisfied(assignment)
            if unsat_count == 0:
                return assignment
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                
        # Find all unsatisfied clauses
        unsat_clauses = []
        for c in processed_clauses:
            sat = False
            for v, s in c:
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(c)
                
        if not unsat_clauses:
            return assignment
            
        # Pick a random unsatisfied clause
        c = rng.choice(unsat_clauses)
        
        # Pick a variable to flip from c
        if rng.random() < p_random:
            # Random choice
            v, s = rng.choice(c)
        else:
            # Greedy choice: variable in c that appears in the most unsat clauses
            best_v = c[0][0]
            max_count = -1
            for v, s in c:
                count = 0
                for uc in unsat_clauses:
                    for uv, us in uc:
                        if uv == v:
                            count += 1
                            break
                if count > max_count:
                    max_count = count
                    best_v = v
            v = best_v
            
        # Flip
        assignment[v] = not assignment[v]
        
        # Update unsat count
        # Optimization: We could update incrementally, but full count is safe and fast enough
        unsat_count = count_unsatisfied(assignment)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        steps_done += 1
        
    return best_assignment
