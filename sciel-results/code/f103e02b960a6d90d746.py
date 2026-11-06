"""Improved WalkSAT with enhanced phase saving and adaptive heuristics for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    timeout = 320.0
    
    m = len(clauses)
    if m == 0:
        return [True] * n
    
    # Preprocess clauses into numpy arrays
    clause_data = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_data[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    sign_bool = (clause_data == 1)
    
    # Build variable-to-clauses mapping
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clauses[v].append(i)
    
    # Convert to numpy arrays for faster access
    var_clauses_np = [np.array(vc, dtype=np.int32) if vc else np.array([], dtype=np.int32) for vc in var_clauses]
    
    def compute_satisfaction(assignment):
        vals = assignment[clause_vars]
        literal_true = (vals == sign_bool)
        return np.any(literal_true, axis=1)
    
    def walk_sat(restart_assignment, max_flips, seed, phase_save=None, p=0.55):
        """WalkSAT with improved heuristics."""
        rng = random.Random(seed)
        assignment = restart_assignment.copy()
        
        sat = compute_satisfaction(assignment)
        unsat_count = int(np.sum(~sat))
        
        if unsat_count == 0:
            return assignment.tolist()
        
        unsat_set = set(np.where(~sat)[0].tolist())
        
        best_unsat = unsat_count
        best_assignment = assignment.copy()
        
        # Track clause satisfaction counts for difficulty estimation
        clause_sat_counts = np.zeros(m, dtype=np.int32)
        for i in range(m):
            if sat[i]:
                clause_sat_counts[i] = 1
        
        for step in range(max_flips):
            if time.time() - start_time > timeout - 2:
                break
            
            if unsat_count == 0:
                return assignment.tolist()
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            # Pick a random unsatisfied clause
            clause_idx = list(unsat_set)[rng.randint(0, len(unsat_set) - 1)]
            
            clause_v = clause_vars[clause_idx]
            
            # Compute weights: for each variable in the clause, count how many
            # other unsat clauses it appears in (flipping it would satisfy more)
            weights = np.zeros(3, dtype=np.float64)
            for k, v in enumerate(clause_v):
                vclauses = var_clauses_np[v]
                if len(vclauses) > 0:
                    # Count unsat clauses containing this variable
                    weights[k] = np.sum([1 for ci in vclauses if ci in unsat_set])
            
            # Add small random noise to break ties and explore
            weights += 0.1 * rng.random(3)
            
            # Normalize and sample
            total_w = np.sum(weights)
            if total_w < 1e-10:
                var_to_flip = clause_v[rng.randint(0, 2)]
            else:
                probs = weights / total_w
                r = rng.random()
                cumulative = 0.0
                var_to_flip = clause_v[0]
                for i, prob in enumerate(probs):
                    cumulative += prob
                    if r <= cumulative:
                        var_to_flip = clause_v[i]
                        break
            
            # Flip the variable
            old_val = assignment[var_to_flip]
            new_val = not old_val
            assignment[var_to_flip] = new_val
            
            # Incrementally update satisfaction
            for ci in var_clauses_np[var_to_flip]:
                was_sat = sat[ci]
                v_vals = assignment[clause_vars[ci]]
                is_now_sat = bool(np.any(v_vals == sign_bool[ci]))
                sat[ci] = is_now_sat
                
                if was_sat and not is_now_sat:
                    unsat_count += 1
                    unsat_set.add(ci)
                    clause_sat_counts[ci] = 0
                elif not was_sat and is_now_sat:
                    unsat_count -= 1
                    unsat_set.discard(ci)
                    clause_sat_counts[ci] = 1
        
        if unsat_count == 0:
            return assignment.tolist()
        
        return None
    
    # Main search loop
    rng = random.Random(42)
    restart_count = 0
    
    # Build a good initial phase by running several short walks
    phase_assignment = None
    best_phase_unsat = float('inf')
    
    for _ in range(5):
        initial = np.array([rng.random() < 0.5 for _ in range(n)])
        sat = compute_satisfaction(initial)
        unsat = int(np.sum(~sat))
        if unsat < best_phase_unsat:
            best_phase_unsat = unsat
            phase_assignment = initial.copy()
    
    # If we have a good phase, refine it with a short walk
    if phase_assignment is not None:
        refined = walk_sat(phase_assignment, 2000, seed=99999)
        if refined is not None:
            return refined
        # Keep the best state from the refined walk as new phase
        # (walk_sat returns None if unsolved, but we can't easily get best state)
    
    while time.time() - start_time < timeout - 15:
        restart_count += 1
        
        # Create initial assignment
        if phase_assignment is not None and restart_count > 1:
            initial = phase_assignment.copy()
            # Perturb a fraction of variables
            num_flips = max(1, int(n * 0.1))
            flip_indices = rng.sample(range(n), min(num_flips, n))
            for idx in flip_indices:
                initial[idx] = not initial[idx]
        else:
            initial = np.array([rng.random() < 0.5 for _ in range(n)])
        
        # Adaptive flip budget
        base_flips = 10000
        adaptive_factor = 1.0 + 0.2 * min(restart_count, 15)
        max_flips = int(base_flips * adaptive_factor)
        
        result = walk_sat(initial, max_flips, seed=restart_count * 7919 + 104729)
        
        if result is not None:
            return result
        
        # After failed walk, try to update phase with best state
        # We can't easily get the best state from walk_sat, so use a heuristic:
        # If we've done several restarts, try a fresh random walk as phase
        if restart_count % 5 == 0:
            trial = np.array([rng.random() < 0.5 for _ in range(n)])
            sat = compute_satisfaction(trial)
            unsat = int(np.sum(~sat))
            if phase_assignment is None or unsat < best_phase_unsat:
                best_phase_unsat = unsat
                phase_assignment = trial.copy()
    
    # Fallback: return best known assignment
    if phase_assignment is not None:
        return phase_assignment.tolist()
    
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
