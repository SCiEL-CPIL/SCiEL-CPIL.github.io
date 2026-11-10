import random
import time
import math
from collections import defaultdict

def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    
    Strategy: Optimized WalkSAT with incremental updates, adaptive noise, and restarts.
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 320.0
    
    # Pre-process clauses for faster access
    # Convert to 0-based indices and signs.
    # Store as (var_idx, is_positive)
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)
    
    # Build variable-to-clauses index
    var_to_clauses = defaultdict(list)
    for i, fc in enumerate(fast_clauses):
        for var_idx, _ in fc:
            var_to_clauses[var_idx].append(i)
    
    def is_satisfied(clause_idx, assignment):
        fc = fast_clauses[clause_idx]
        for var_idx, required in fc:
            if assignment[var_idx] == required:
                return True
        return False

    def walk_sat(max_steps, initial_p=0.5, final_p=0.05, start_assignment=None):
        """
        Run WalkSAT with adaptive noise.
        Returns (assignment, unsat_count)
        """
        if start_assignment is None:
            assignment = [random.random() < 0.5 for _ in range(n)]
        else:
            assignment = start_assignment[:]
        
        # Initialize status and counts
        clause_sat = [False] * num_clauses
        unsat_count = 0
        for i in range(num_clauses):
            if is_satisfied(i, assignment):
                clause_sat[i] = True
            else:
                unsat_count += 1
        
        if unsat_count == 0:
            return assignment, 0
        
        # Frequency counts for heuristic
        # unsat_freq[v] = number of unsat clauses containing v
        unsat_freq = [0] * n
        for i in range(num_clauses):
            if not clause_sat[i]:
                for var_idx, _ in fast_clauses[i]:
                    unsat_freq[var_idx] += 1
        
        # Adaptive noise parameters
        # We will decay p linearly from initial_p to final_p over max_steps
        p_decay = (initial_p - final_p) / max_steps if max_steps > 0 else 0
        
        best_unsat = unsat_count
        best_assignment = assignment[:]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assignment, 0
            
            # Update best
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            # Select a random unsatisfied clause
            # Optimization: Keep a list of unsat clause indices for fast selection
            # But maintaining a list with deletions is tricky. 
            # For 150 vars, num_clauses is likely ~400-500. 
            # random.randint(0, num_clauses-1) and retrying is O(1) on average if unsat ratio is high.
            # If unsat ratio is low, it's still fast enough.
            while True:
                c_idx = random.randint(0, num_clauses - 1)
                if not clause_sat[c_idx]:
                    break
            
            clause = fast_clauses[c_idx]
            
            # Calculate scores for literals in this clause
            # Score is unsat_freq[var_idx]
            max_score = -1
            best_var_idx = -1
            best_required = False
            for var_idx, required in clause:
                score = unsat_freq[var_idx]
                if score > max_score:
                    max_score = score
                    best_var_idx = var_idx
                    best_required = required
            
            # Adaptive noise
            p = initial_p - p_decay * step
            if p < final_p:
                p = final_p
            
            if random.random() < (1 - p):
                var_idx = best_var_idx
            else:
                # Pick random literal from clause
                var_idx, _ = random.choice(clause)
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause statuses and frequencies
            # Iterate through all clauses containing var_idx
            for c_idx2 in var_to_clauses[var_idx]:
                was_sat = clause_sat[c_idx2]
                is_now_sat = is_satisfied(c_idx2, assignment)
                
                if was_sat == is_now_sat:
                    continue
                
                # Update unsat_count
                if is_now_sat:
                    unsat_count -= 1
                    # Update unsat_freq for all vars in this clause
                    for v, _ in fast_clauses[c_idx2]:
                        unsat_freq[v] -= 1
                else:
                    unsat_count += 1
                    for v, _ in fast_clauses[c_idx2]:
                        unsat_freq[v] += 1
                
                clause_sat[c_idx2] = is_now_sat
                
                if unsat_count == 0:
                    return assignment, 0
        
        return best_assignment, best_unsat

    # Main loop: Restart strategy
    # We will run multiple attempts.
    # Each attempt has a budget of steps.
    # We can increase steps per attempt or just restart quickly.
    # Quick restarts are often better for WalkSAT.
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    start_time = time.time()
    
    # Phase 1: Many short restarts
    # 150 variables, ~420 clauses.
    # 1000 steps per restart is very fast.
    num_restarts = 5000  # Adjust based on speed
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Adaptive restarts: 
        # Start with random.
        # Later, we can try perturbing the best solution.
        
        if i < 100:
            # Pure random
            start_assign = None
            max_steps = 2000
            p_start = 0.5
            p_end = 0.1
        elif i < 300:
            # Slightly longer, lower noise
            start_assign = None
            max_steps = 5000
            p_start = 0.4
            p_end = 0.05
        else:
            # If we have a good best, perturb it? 
            # Or just keep random with tuned params.
            # Let's try restarting from best with small perturbations
            if best_assignment is not None and random.random() < 0.5:
                start_assign = best_assignment[:]
                # Flip a few random variables
                for _ in range(random.randint(1, 5)):
                    v = random.randint(0, n-1)
                    start_assign[v] = not start_assign[v]
            else:
                start_assign = None
            
            max_steps = 5000
            p_start = 0.3
            p_end = 0.02
        
        assignment, unsat = walk_sat(max_steps, p_start, p_end, start_assign)
        
        if unsat == 0:
            return assignment
        
        if unsat < best_unsat_count:
            best_unsat_count = unsat
            best_assignment = assignment
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    else:
        return [random.random() < 0.5 for _ in range(n)]
