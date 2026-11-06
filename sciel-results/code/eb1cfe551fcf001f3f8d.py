import random
import time

def solve(instance):
    """
    Solves a planted 3-SAT instance using a robust WalkSAT algorithm with restarts and a deadline guard.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Safety margin: finish before 300s. We use 290s to be safe.
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Parameters
    # p is the probability of making a greedy move (minimizing unsat count)
    # vs a random move.
    p = 0.5 
    # Max steps per restart. For 50 vars, 500*50 = 25000 steps is a reasonable initial budget.
    # We can scale this slightly if n is larger, but n=50 is fixed in the task description.
    max_steps_per_restart = 500 * n
    
    rng = random.Random(0) # Fixed seed for reproducibility if needed, but random is fine too. 
    # Actually, using a fixed seed might be bad if the evaluator expects different behavior on different runs? 
    # The task says "score is the fraction solved". Using a fixed seed makes the solver deterministic.
    # If the instances are fixed, determinism is fine. If instances vary, random seed is better.
    # Let's use a seed based on time or just 0. The seed program used 0. I'll stick with 0 for consistency with "seed program" style, 
    # but maybe randomize it to avoid systematic failures on specific hard instances if the seed 0 is bad?
    # Let's use random.SystemRandom() or just random.Random() without seed to get variety.
    rng = random.Random()

    best_assignment = None
    min_unsat_count = float('inf')

    while time.time() < deadline:
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Pre-compute clause literals indices for speed? 
        # The clause structure is fixed. We can pre-process clauses to store (var_idx, is_positive)
        # But n is small (50), so direct lookup is fast enough.
        
        current_steps = 0
        solved = False
        
        while current_steps < max_steps_per_restart and time.time() < deadline:
            # Find unsatisfied clauses
            unsat_clauses = []
            for c in clauses:
                sat = False
                for l in c:
                    val = assignment[abs(l) - 1]
                    if (l > 0 and val) or (l < 0 and not val):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(c)
            
            if not unsat_clauses:
                return assignment
            
            # Track best assignment found so far in this restart (or globally)
            # Actually, for SAT, any solution is a solution. We don't need "best" in terms of minimization, 
            # but if we don't find a solution, we return the one with fewest unsat clauses?
            # The task asks for a "satisfying assignment". If we fail, we should probably return the best effort?
            # The contract says "Return a satisfying assignment". If we can't, what do we return?
            # "Score is the fraction solved". So if we don't solve it, we score 0 for that instance.
            # So we just need to try hard to solve it.
            
            # Pick a random unsatisfied clause
            clause = rng.choice(unsat_clauses)
            
            # Determine which variable to flip
            # Calculate score for each literal in the clause
            # Score = number of clauses that would become satisfied (or stay satisfied) if we flip this var?
            # Standard WalkSAT: 
            # For each literal l in clause:
            #   Let v = var(l)
            #   Let count = number of clauses containing v (negated or not) that are currently unsatisfied?
            #   Actually, the "gain" is: how many other unsatisfied clauses contain this variable?
            #   If we flip v, we satisfy the current clause. 
            #   We might break other clauses that contain v and are currently satisfied.
            #   So we want to minimize the number of satisfied clauses that contain v.
            
            # Efficient way:
            # For each literal in the clause, count how many *satisfied* clauses contain that variable.
            # We want to flip the variable with the *minimum* such count.
            
            # Since n is small (50) and clauses are 3-SAT, we can afford O(M * 3) per step.
            # M is number of clauses.
            
            min_count = float('inf')
            best_literal = None
            candidates = []
            
            for l in clause:
                var_idx = abs(l) - 1
                # Count satisfied clauses that contain this variable
                count = 0
                for c2 in clauses:
                    # Check if c2 is satisfied
                    # If c2 is satisfied, does it contain var_idx?
                    # This is O(M) per literal, so O(3*M) per step.
                    # M is likely ~200-300 for 50 vars (ratio 4-6). 3*300 = 900 ops.
                    # 25000 steps * 900 ops = 22.5M ops. Python might be slow for this.
                    # Let's optimize.
                    pass
            
            # Optimized WalkSAT step:
            # Pre-compute for each variable, which clauses it appears in?
            # Let's do that outside the loop.
            
            break # Placeholder for optimization logic
            
        # If loop finished without solving, we restart
        # We don't explicitly track "best" unsat count for return if not solved, 
        # because the score is binary (solved or not).
        # However, if we run out of time, we should return the current assignment? 
        # The contract says "Return a satisfying assignment". 
        # If we return an unsatisfying one, it's scored 0.
        # So we just need to loop until solved or time out.

    # If we reach here, we ran out of time.
    # Return the last assignment (which is likely unsatisfying).
    # It's better to return the assignment from the last restart that had the fewest unsat clauses?
    # No, just return the current one. It doesn't matter for the score if it's not solved.
    return assignment

# Wait, the above implementation has a broken inner loop (the `break` placeholder).
# I need to rewrite the inner loop properly with the optimization.

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-process clauses to speed up variable occurrence checks
    # var_to_clauses[var_idx] = list of clause indices
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l) - 1].append(i)
            
    start_time = time.time()
    deadline = start_time + 290.0
    
    p = 0.5  # Probability of greedy move
    max_steps_per_restart = 500 * n
    
    rng = random.Random()
    
    # We will keep the best assignment (min unsat) in case we don't find a solution, 
    # though for scoring it only matters if it's fully satisfied.
    best_assignment = None
    best_unsat_count = float('inf')
    
    while time.time() < deadline:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Helper to check if clause is satisfied
        def is_satisfied(c):
            for l in c:
                val = assignment[abs(l) - 1]
                if (l > 0 and val) or (l < 0 and not val):
                    return True
            return False
        
        # Initial unsat count
        unsat_count = 0
        for c in clauses:
            if not is_satisfied(c):
                unsat_count += 1
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        steps = 0
        while steps < max_steps_per_restart and time.time() < deadline:
            # Find an unsatisfied clause
            # To be faster, we can maintain a list of unsat clauses, but updating it is complex.
            # Given n=50, M~250, scanning all clauses is O(250*3) = 750 ops.
            # This is fast enough in Python if we do it efficiently.
            
            unsat_clause_idx = -1
            for i, c in enumerate(clauses):
                if not is_satisfied(c):
                    unsat_clause_idx = i
                    break
            
            if unsat_clause_idx == -1:
                return assignment
            
            clause = clauses[unsat_clause_idx]
            
            # Choose a variable to flip
            # Greedy: minimize number of satisfied clauses that contain the variable
            # Random: pick random literal in clause
            
            if rng.random() < p:
                # Greedy
                min_breaks = float('inf')
                best_literal = None
                for l in clause:
                    var_idx = abs(l) - 1
                    breaks = 0
                    # Count how many satisfied clauses contain this variable
                    # We can iterate through var_to_clauses[var_idx]
                    for c_idx in var_to_clauses[var_idx]:
                        if c_idx == unsat_clause_idx:
                            continue # We are already breaking this one, or satisfying it?
                            # Actually, this clause is currently UNSAT.
                            # If we flip the variable, this clause becomes SAT (assuming no other literal in it is sat, which is true for unsat clause).
                            # So this clause is "fixed".
                            # We care about other clauses.
                        if is_satisfied(clauses[c_idx]):
                            breaks += 1
                    if breaks < min_breaks:
                        min_breaks = breaks
                        best_literal = l
                
                if best_literal is not None:
                    assignment[abs(best_literal) - 1] = not assignment[abs(best_literal) - 1]
            else:
                # Random
                l = rng.choice(clause)
                assignment[abs(l) - 1] = not assignment[abs(l) - 1]
            
            # Update unsat_count? 
            # Recalculating full unsat_count is O(M). 
            # Doing it every step is O(Steps * M).
            # 25000 steps * 250 clauses = 6.25M checks. Each check is O(3). ~18M ops.
            # This might be too slow in Python for 300s if we do many restarts.
            # But we only do 1 restart if it fails.
            # If it takes 25000 steps and fails, we restart.
            # How many restarts can we do in 300s?
            # 18M ops per restart. Python does ~10-50M simple ops/sec?
            # So ~1-2 restarts per second? 
            # 300 seconds -> 300-600 restarts.
            # This seems reasonable.
            
            # For tracking best, we need unsat_count.
            # Let's just calculate it every step for simplicity, but optimize the `is_satisfied` call.
            # Actually, we can update incrementally, but that's complex.
            # Let's stick to full recalculation but make `is_satisfied` as fast as possible.
            
            unsat_count = 0
            for c in clauses:
                # Inline is_satisfied for speed
                sat = False
                for l in c:
                    val = assignment[abs(l) - 1]
                    if (l > 0 and val) or (l < 0 and not val):
                        sat = True
                        break
                if not sat:
                    unsat_count += 1
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                
            steps += 1
    
    return best_assignment if best_assignment is not None else [False] * n
