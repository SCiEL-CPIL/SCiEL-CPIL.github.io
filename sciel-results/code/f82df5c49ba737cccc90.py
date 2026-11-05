"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Precompute clause data for speed
    # For each clause, store the variable indices and the required sign (True if literal > 0)
    clause_data = []
    for c in clauses:
        clause_vars = []
        clause_sigs = []
        for l in c:
            var_idx = abs(l) - 1
            clause_vars.append(var_idx)
            clause_sigs.append(l > 0)
        clause_data.append((clause_vars, clause_sigs))
    
    num_clauses = len(clauses)
    
    # Deadline guard: 300s limit, leave 5s margin
    deadline = time.time() + 295.0
    
    def is_satisfied(c_idx, assignment):
        vars_, sigs_ = clause_data[c_idx]
        for v, s in zip(vars_, sigs_):
            if assignment[v] == s:
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                cnt += 1
        return cnt
    
    def walksat_start(assignment, steps, p):
        """Perform WalkSAT steps. Returns (solved, new_assignment)."""
        for _ in range(steps):
            if time.time() > deadline:
                return False, assignment
            unsat_clauses = []
            for i in range(num_clauses):
                if not is_satisfied(i, assignment):
                    unsat_clauses.append(i)
            if not unsat_clauses:
                return True, assignment
            # Pick random unsat clause
            c_idx = random.choice(unsat_clauses)
            vars_, sigs_ = clause_data[c_idx]
            
            # Count how many literals would be satisfied if flipped
            # For each literal in clause, if we flip var, does it satisfy the clause?
            # Actually, standard WalkSAT: pick a literal from the unsat clause.
            # With prob p, pick a literal that minimizes breakage (or random if tie).
            # With prob 1-p, pick random literal.
            
            # Let's stick to simple random literal flip for speed, or weighted.
            # Simple random is often sufficient for planted 3-SAT.
            lit_idx = random.randrange(3)
            var_to_flip = vars_[lit_idx]
            assignment[var_to_flip] = not assignment[var_to_flip]
            
        # Check if solved
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                return False, assignment
        return True, assignment
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Strategy: Multiple restarts with increasing step limits
    # Planted 3-SAT is usually easy.
    # Try a few short runs, then longer ones.
    
    rng = random.Random()
    
    # Initial random assignment
    current_assign = [rng.random() < 0.5 for _ in range(n)]
    
    # Try WalkSAT with various parameters
    # 1. Short runs to find quick solutions
    # 2. If not solved, try longer runs
    
    max_restarts = 50
    steps_per_restart = 1000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # New random assignment for each restart? Or continue?
        # Restarting helps escape local minima.
        if restart > 0:
            current_assign = [rng.random() < 0.5 for _ in range(n)]
        
        solved, current_assign = walksat_start(current_assign, steps_per_restart, 0.5)
        
        unsat_count = count_unsat(current_assign)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = current_assign[:]
        
        if solved:
            return current_assign
    
    # If we ran out of time or restarts, try one final long run with best assignment
    if time.time() < deadline - 1.0 and best_assignment:
        # Continue from best assignment
        remaining_time = deadline - time.time()
        # Estimate steps: ~1000 steps per 0.1s? Depends on n and clauses.
        # For n=50, clauses ~100-300. Checking all clauses is O(C*3).
        # Let's just do a fixed number of steps based on time.
        # Safe bet: 5000 steps
        solved, current_assign = walksat_start(best_assignment, 5000, 0.5)
        if solved:
            return current_assign
        unsat_count = count_unsat(current_assign)
        if unsat_count < best_unsat_count:
            best_assignment = current_assign[:]
    
    return best_assignment
# EVOLVE-BLOCK-END
