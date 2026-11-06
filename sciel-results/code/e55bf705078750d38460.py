import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 300s limit, stop at 295s to be safe.
    start_time = time.time()
    deadline = start_time + 295.0
    
    rng = random.Random(42)
    
    # Assignment as list of bools for simplicity, or bitmask?
    # List of bools is easy to work with.
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Preprocess clauses to 0-based indices and signs
    # literal l: var = abs(l)-1, is_pos = l > 0
    pre_clauses = []
    for c in clauses:
        pre_c = [(abs(l)-1, l > 0) for l in c]
        pre_clauses.append(pre_c)
        
    def is_satisfied(clause_idx):
        for var, is_pos in pre_clauses[clause_idx]:
            if assignment[var] == is_pos:
                return True
        return False
        
    def get_unsat_clauses():
        return [i for i in range(len(pre_clauses)) if not is_satisfied(i)]
        
    # WalkSAT parameters
    p = 0.5 # probability of random flip
    max_steps = 10000 # per restart
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Try a few restarts if stuck
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Re-randomize on restart? Or continue?
        # Standard WalkSAT often restarts with new random assignment.
        if restart > 0:
             assignment = [rng.random() < 0.5 for _ in range(n)]
             
        for step in range(max_steps):
            if time.time() > deadline:
                return best_assignment if best_assignment else assignment
                
            unsat_indices = get_unsat_clauses()
            unsat_count = len(unsat_indices)
            
            if unsat_count == 0:
                return assignment
                
            # Track best seen
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                
            # Select clause
            # Uniform selection from unsat clauses
            target_clause_idx = rng.choice(unsat_indices)
            target_clause = pre_clauses[target_clause_idx]
            
            # Calculate gain for each literal in target clause
            # Gain = number of OTHER unsat clauses that contain this literal
            gains = []
            for var, is_pos in target_clause:
                gain = 0
                # Literal is (var, is_pos)
                # Check other unsat clauses
                for u_idx in unsat_indices:
                    if u_idx == target_clause_idx:
                        continue
                    u_clause = pre_clauses[u_idx]
                    # Does u_clause contain the literal (var, is_pos)?
                    for u_var, u_is_pos in u_clause:
                        if u_var == var and u_is_pos == is_pos:
                            gain += 1
                            break # One match per clause is enough
                gains.append(gain)
                
            # Choose literal to flip
            if rng.random() < p:
                # Random literal from target clause
                idx = rng.randint(0, len(target_clause)-1)
            else:
                # Max gain literal
                max_gain = max(gains)
                # Find all with max gain
                candidates = [i for i, g in enumerate(gains) if g == max_gain]
                idx = rng.choice(candidates)
                
            var, is_pos = target_clause[idx]
            # Flip variable
            assignment[var] = not assignment[var]
            
    return best_assignment if best_assignment else assignment
