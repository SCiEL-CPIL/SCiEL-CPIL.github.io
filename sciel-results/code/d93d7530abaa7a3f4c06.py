"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 300s limit, use 290s to be safe
    deadline = time.time() + 290.0
    
    # Parameters
    restart_threshold = 1000  # Restart if no solution in this many steps
    max_restarts = 10000
    decay_factor = 0.95
    weight_increase = 1.0
    
    # Precompute clause indices for variables for faster conflict checking
    # var_to_clauses[v] = list of indices of clauses containing variable v
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(i)
    
    def is_satisfied(clause_idx, assignment):
        clause = clauses[clause_idx]
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False

    def count_unsat_clauses(assignment):
        count = 0
        for i in range(len(clauses)):
            if not is_satisfied(i, assignment):
                count += 1
        return count

    def get_unsat_clauses(assignment):
        unsat = []
        for i in range(len(clauses)):
            if not is_satisfied(i, assignment):
                unsat.append(i)
        return unsat

    def find_best_flip(clause_idx, assignment):
        """
        Finds the literal in the clause that, when flipped, 
        results in the fewest newly unsatisfied clauses.
        """
        clause = clauses[clause_idx]
        best_lit = None
        min_cost = float('inf')
        
        for lit in clause:
            var = abs(lit) - 1
            # Simulate flipping var
            # Cost = number of clauses containing var that become unsatisfied
            # Note: The current clause is already unsatisfied, so flipping any lit in it
            # will satisfy the current clause (since it's 3-SAT and we pick from unsat clause).
            # So we only need to check the *other* clauses containing var.
            
            cost = 0
            for c_idx in var_to_clauses[var]:
                if c_idx == clause_idx:
                    continue
                # Check if this clause is satisfied by the *new* value
                # New value of var is NOT assignment[var]
                new_val = not assignment[var]
                # Check if clause c_idx is satisfied with new_val
                satisfied = False
                for l in clauses[c_idx]:
                    v = abs(l) - 1
                    if v == var:
                        if new_val == (l > 0):
                            satisfied = True
                            break
                    else:
                        if assignment[v] == (l > 0):
                            satisfied = True
                            break
                if not satisfied:
                    cost += 1
            
            if cost < min_cost:
                min_cost = cost
                best_lit = lit
        
        return best_lit

    def run_search(rng):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        weights = [1.0] * len(clauses)
        
        for step in range(restart_threshold):
            if time.time() > deadline:
                return assignment
            
            unsat_indices = get_unsat_clauses(assignment)
            if not unsat_indices:
                return assignment
            
            # Update weights
            for i in range(len(clauses)):
                if i in unsat_indices:
                    weights[i] += weight_increase
                else:
                    weights[i] *= decay_factor
            
            # Select a clause to work on, weighted by 'weights'
            # We only pick from unsat clauses
            # To do this efficiently, we can pick randomly from unsat_indices,
            # but bias towards higher weights.
            # Simple approach: pick random unsat clause. 
            # Better: pick with probability proportional to weight.
            # Since number of unsat clauses might be small, we can just build a list of (weight, idx)
            
            # For efficiency with n=50 and ~100 clauses, this is fine.
            # Let's just pick a random unsat clause for now, or the one with max weight?
            # Max weight is deterministic. Random is stochastic.
            # Let's use weighted random selection.
            
            total_w = sum(weights[i] for i in unsat_indices)
            if total_w == 0:
                clause_idx = rng.choice(unsat_indices)
            else:
                r = rng.random() * total_w
                cumulative = 0
                clause_idx = unsat_indices[0]
                for i in unsat_indices:
                    cumulative += weights[i]
                    if r <= cumulative:
                        clause_idx = i
                        break
            
            # Find best flip in this clause
            lit = find_best_flip(clause_idx, assignment)
            
            # Flip
            var = abs(lit) - 1
            assignment[var] = not assignment[var]
            
        return assignment

    # Main loop with restarts
    rng = random.Random(0)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        result = run_search(rng)
        
        # Check if result is valid
        if all(is_satisfied(i, result) for i in range(len(clauses))):
            return result
            
        # If not solved, restart with new random assignment
        # Optionally, we could keep the best partial assignment, but full reset is standard for WalkSAT restarts.
        
    # If we run out of restarts or time, return the last result
    return result

# EVOLVE-BLOCK-END
