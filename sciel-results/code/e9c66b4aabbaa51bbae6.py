"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict


def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 325.0
    
    # Pre-process clauses for faster access
    # Each clause is a list of signed literals (1-based).
    # Convert to 0-based indices and signs.
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)
    
    # Build variable-to-clauses index for efficient updates
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
    
    # WalkSAT with restarts, clause weighting, and adaptive walk probability
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters
    num_restarts = 300
    max_steps = 5000
    base_walk_prob = 0.2
    max_walk_prob = 0.5
    
    # Clause activity weights
    clause_weights = [1.0] * num_clauses
    decay_rate = 0.99
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment randomly
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Track current unsat count and unsat clause set
        unsat_set = set()
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        # Pre-compute per-variable unsat clause count for fast heuristic
        var_unsat_count = [0] * n
        for c_idx in unsat_set:
            fc = fast_clauses[c_idx]
            for var_idx, _ in fc:
                var_unsat_count[var_idx] += 1
        
        current_unsat = len(unsat_set)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # Adaptive walk probability: starts low, increases if stuck
        current_walk_prob = base_walk_prob
        last_improvement_step = 0
        min_unsat_in_restart = current_unsat
        
        # WalkSAT loop
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_set:
                return assignment
            
            if len(unsat_set) < best_unsat_count:
                best_unsat_count = len(unsat_set)
                best_assignment = assignment[:]
                last_improvement_step = step
                current_walk_prob = base_walk_prob # Reset walk prob on improvement
            
            # Increase walk probability if stuck
            if step - last_improvement_step > 500:
                current_walk_prob = min(max_walk_prob, current_walk_prob + 0.05)
            
            # Select clause using weighted random selection
            # For efficiency, if unsat_set is small, just pick uniformly or use weights
            # Weights are maintained in clause_weights
            # To avoid O(N) sum every step, we can approximate or just pick uniformly if weights are similar
            # However, weighted selection is better. Let's do a simple weighted pick.
            # Optimization: If the number of unsat clauses is small, summing weights is fast.
            total_weight = 0.0
            weights_list = []
            for c_idx in unsat_set:
                w = clause_weights[c_idx]
                weights_list.append((c_idx, w))
                total_weight += w
            
            if total_weight == 0:
                clause_idx = random.choice(list(unsat_set))
            else:
                r = random.random() * total_weight
                cumulative = 0.0
                clause_idx = weights_list[0][0]
                for c_idx, w in weights_list:
                    cumulative += w
                    if r <= cumulative:
                        clause_idx = c_idx
                        break
            
            clause = fast_clauses[clause_idx]
            
            # Select literal
            if random.random() < current_walk_prob:
                var_idx, required = random.choice(clause)
            else:
                # Heuristic: prefer literals that appear in more unsat clauses
                scores = []
                for var_idx, required in clause:
                    score = var_unsat_count[var_idx]
                    scores.append(score)
                
                total_score = sum(scores)
                if total_score == 0:
                    var_idx, required = random.choice(clause)
                else:
                    # Weighted selection
                    r = random.random() * total_score
                    cumulative = 0
                    var_idx, required = clause[0]
                    for i, score in enumerate(scores):
                        cumulative += score
                        if r <= cumulative:
                            var_idx, required = clause[i]
                            break
            
            # Flip the variable
            old_val = assignment[var_idx]
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat_set and var_unsat_count incrementally
            for c_idx in var_to_clauses[var_idx]:
                was_unsat = c_idx in unsat_set
                is_sat = is_satisfied(c_idx, assignment)
                
                if was_unsat and is_sat:
                    # Clause became satisfied: remove from unsat_set, decrement counts
                    unsat_set.discard(c_idx)
                    fc = fast_clauses[c_idx]
                    for v_idx, _ in fc:
                        var_unsat_count[v_idx] -= 1
                elif not was_unsat and not is_sat:
                    # Clause became unsatisfied: add to unsat_set, increment counts
                    unsat_set.add(c_idx)
                    fc = fast_clauses[c_idx]
                    for v_idx, _ in fc:
                        var_unsat_count[v_idx] += 1
            
            # Update clause weights
            # Boost the weight of the clause we just worked on
            clause_weights[clause_idx] *= 1.5
            # Decay all weights slightly to prevent saturation
            # Doing this for all clauses is O(N). We can do it less frequently or just decay the selected one relative to others?
            # Standard activity-based: decay all, boost selected.
            # To keep it fast, we only decay every 100 steps or so? Or just boost selected and let them grow, then normalize?
            # Let's just boost the selected one and decay the global scale occasionally?
            # Actually, for simplicity and speed, let's just boost the selected clause's weight.
            # The relative importance will shift. To prevent overflow, we can scale down all weights if they get too large.
            if clause_weights[clause_idx] > 1000:
                # Scale down all weights
                for i in range(num_clauses):
                    clause_weights[i] /= 10.0
            
            # Early exit if solved
            if not unsat_set:
                return assignment
        
        # End of restart
        # Reset clause weights for next restart? Or keep them?
        # Keeping them might help if the hard clauses are consistent across restarts.
        # But random restarts change the landscape. Let's reset to uniform to avoid bias from previous failed attempts.
        clause_weights = [1.0] * num_clauses

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: return a random assignment
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
