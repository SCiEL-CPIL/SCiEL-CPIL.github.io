"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute variable to clauses mapping for faster access
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for l in clause:
            var_to_clauses[abs(l) - 1].append(i)
    
    # Phase saving: track the last value that made each variable true
    # Initialize phases randomly
    rng = random.Random(42)
    phase = [rng.random() < 0.5 for _ in range(n)]
    
    def count_violated(assignment):
        """Return list of violated clause indices."""
        violated = []
        for i, clause in enumerate(clauses):
            sat = False
            for l in clause:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                violated.append(i)
        return violated
    
    def flip_cost(assignment, var_idx, new_val):
        """Estimate cost of setting var_idx to new_val (delta in #violated clauses)."""
        # Count clauses containing this variable
        delta = 0
        for clause_idx in var_to_clauses[var_idx]:
            clause = clauses[clause_idx]
            currently_sat = False
            other_sat = False
            for l in clause:
                if abs(l) - 1 == var_idx:
                    if assignment[var_idx] == (l > 0):
                        currently_sat = True
                else:
                    if assignment[abs(l) - 1] == (l > 0):
                        other_sat = True
            
            if currently_sat:
                # Currently satisfied by this var; flipping may unsatisfy
                if not other_sat:
                    delta += 1  # This clause becomes violated
            else:
                # Currently not satisfied by this var; flipping may satisfy
                if other_sat:
                    pass  # Already satisfied, no change
                else:
                    delta -= 1  # This clause becomes satisfied
        return delta
    
    max_restarts = 50
    flips_per_restart = 100000
    
    for restart in range(max_restarts):
        seed = restart * 7919 + 13
        rng = random.Random(seed)
        
        # Initialize assignment using phase saving
        assignment = [phase[i] for i in range(n)]
        
        for _ in range(flips_per_restart):
            violated = count_violated(assignment)
            if not violated:
                return assignment
            
            # Pick a random violated clause
            clause_idx = rng.choice(violated)
            clause = clauses[clause_idx]
            
            # Choose best flip: minimize delta in violated clauses
            best_var = -1
            best_delta = float('inf')
            best_new_val = None
            
            candidates = []
            for l in clause:
                var_idx = abs(l) - 1
                new_val = not assignment[var_idx]
                delta = flip_cost(assignment, var_idx, new_val)
                candidates.append((delta, var_idx, new_val))
            
            # Weighted random selection favoring lower delta
            deltas = [c[0] for c in candidates]
            min_d = min(deltas)
            # Normalize: probability proportional to exp(-delta/temperature)
            temperature = 2.0
            weights = [1.0 / (1.0 + d - min_d) for d in deltas]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen = i
                    break
            
            _, var_idx, new_val = candidates[chosen]
            
            # Apply flip and update phase
            assignment[var_idx] = new_val
            phase[var_idx] = new_val
        
        # If we didn't solve, try again with restart
        # Keep phase information for next restart
    
    # Final attempt: return last assignment
    return assignment
# EVOLVE-BLOCK-END
