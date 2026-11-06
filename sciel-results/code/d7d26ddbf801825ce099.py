"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = 315.0  # safety margin before 330s kill
    
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
    
    # Build variable-to-clause incidence
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clauses[v].append(i)
    
    var_clauses_arr = [np.array(vc, dtype=np.int32) for vc in var_clauses]
    
    # For each variable, precompute the position of that variable in each clause it appears in
    # var_pos[v] = list of (clause_idx, pos_in_clause) pairs
    var_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_pos[v].append((i, j))
    var_pos_arr = [np.array(vp, dtype=np.int32) for vp in var_pos]  # shape (k, 2)
    
    # Precompute variable degrees for weighted selection
    var_degree = np.array([len(vp) for vp in var_pos], dtype=np.int32)
    
    # Clause weights: start at 1.0, decay over time
    clause_weights = np.ones(m, dtype=np.float32)
    
    def full_evaluate(assignment):
        """Full vectorized evaluation. Returns (unsat_count, unsat_indices, num_true_per_clause)."""
        assign_int = (2 * assignment.astype(np.int8) - 1)
        lit_vals = clause_data * assign_int[clause_vars]
        clause_satisfied = np.max(lit_vals, axis=1) == 1
        unsat_count = int(np.sum(~clause_satisfied))
        unsat_indices = np.where(~clause_satisfied)[0]
        # Count true literals per clause for incremental updates
        num_true = np.sum(lit_vals == 1, axis=1)
        return unsat_count, unsat_indices, num_true
    
    def unit_propagate(assignment, num_true, unsat_count):
        """Apply unit propagation. Returns (new_unsat_count, new_unsat_indices, propagated)."""
        changed = True
        while changed:
            changed = False
            # Find unit clauses (num_true == 1)
            unit_mask = (num_true == 1)
            if not np.any(unit_mask):
                break
            
            unit_indices = np.where(unit_mask)[0]
            for ci in unit_indices:
                # Find the true literal in this clause
                assign_int = (2 * assignment.astype(np.int8) - 1)
                lit_vals = clause_data[ci] * assign_int[clause_vars[ci]]
                true_pos = np.where(lit_vals == 1)[0]
                if len(true_pos) == 0:
                    continue
                pos = true_pos[0]
                v = int(clause_vars[ci, pos])
                
                # Check if flipping v would satisfy this clause
                # The literal is currently true, so v is already set correctly
                # No change needed
                pass
            
            # Find empty clauses (num_true == 0) - these are unsat
            # We can't propagate from them directly in WalkSAT
            # But we can use them to guide variable selection
            empty_mask = (num_true == 0)
            if np.any(empty_mask):
                empty_indices = np.where(empty_mask)[0]
                # For each empty clause, one of its variables must be flipped
                # We'll handle this in the main search loop
                pass
        
        # Recompute unsat count
        unsat_count = int(np.sum(num_true == 0))
        unsat_indices = np.where(num_true == 0)[0]
        return unsat_count, unsat_indices, True
    
    def compute_flip_delta(v, assignment, num_true, clause_weights):
        """Compute the weighted change in unsat count if we flip variable v.
        Returns the delta (new_weighted_unsat - old_weighted_unsat). Negative means improvement."""
        vp = var_pos_arr[v]
        if len(vp) == 0:
            return 0
        
        clause_idxs = vp[:, 0]
        positions = vp[:, 1]
        
        # Current sign of variable v in each clause
        signs = clause_data[clause_idxs, positions]
        
        # Current value of v: True -> 1, False -> -1
        v_val = 1 if assignment[v] else -1
        
        # Clauses where v's literal is currently TRUE (sign * v_val == 1)
        v_lit_true = (signs * v_val) == 1
        
        # Clauses where v's literal is currently FALSE
        v_lit_false = ~v_lit_true
        
        nt = num_true[clause_idxs]
        
        # Weighted delta
        w = clause_weights[clause_idxs]
        
        delta = 0.0
        # Clauses that become unsat: v_lit_true AND nt == 1
        delta += float(np.sum(w[v_lit_true & (nt == 1)]))
        # Clauses that become sat: v_lit_false AND nt == 0
        delta -= float(np.sum(w[v_lit_false & (nt == 0)]))
        
        return delta
    
    def apply_flip(v, assignment, num_true):
        """Apply flip to variable v and update num_true incrementally."""
        vp = var_pos_arr[v]
        if len(vp) == 0:
            return
        
        clause_idxs = vp[:, 0]
        positions = vp[:, 1]
        signs = clause_data[clause_idxs, positions]
        
        v_val = 1 if assignment[v] else -1
        
        # v's literal is true where sign * v_val == 1
        v_lit_true = (signs * v_val) == 1
        
        # Where v's lit was true, after flip it's false: num_true decreases by 1
        num_true[clause_idxs[v_lit_true]] -= 1
        # Where v's lit was false, after flip it's true: num_true increases by 1
        num_true[clause_idxs[~v_lit_true]] += 1
        
        assignment[v] = not assignment[v]
    
    # Main search loop with restarts
    best_assignment = None
    best_unsat = m + 1
    
    rng = random.Random(42)
    
    # Adaptive parameters
    base_max_steps = 5000
    restart_count = 0
    phase = 0  # 0: random init, 1: best+noise, 2: pure best
    
    while time.time() - start_time < deadline:
        restart_count += 1
        
        # Adaptive initialization strategy
        if best_assignment is None:
            assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        elif restart_count <= 5:
            # Early restarts: mostly random
            assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        elif restart_count <= 15:
            # Middle restarts: mix of best and random
            if rng.random() < 0.5:
                assignment = best_assignment.copy()
                # Flip a small number of random variables
                num_flips = max(1, n // 20)
                flip_vars = rng.sample(range(n), num_flips)
                for v in flip_vars:
                    assignment[v] = not assignment[v]
            else:
                assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        else:
            # Late restarts: mostly based on best
            if rng.random() < 0.7:
                assignment = best_assignment.copy()
                num_flips = max(1, n // 50)
                flip_vars = rng.sample(range(n), num_flips)
                for v in flip_vars:
                    assignment[v] = not assignment[v]
            else:
                assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        unsat_count, unsat_indices, num_true = full_evaluate(assignment)
        
        # Apply unit propagation
        unsat_count, unsat_indices, _ = unit_propagate(assignment, num_true, unsat_count)
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
        
        # Reset clause weights for this restart
        clause_weights.fill(1.0)
        
        # WalkSAT loop with incremental updates
        # Adaptive step limit
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            max_steps = 500
        elif remaining < 30.0:
            max_steps = 2000
        else:
            max_steps = base_max_steps + restart_count * 100
        
        step = 0
        no_improve_steps = 0
        local_best_unsat = unsat_count
        
        while step < max_steps and time.time() - start_time < deadline:
            if unsat_count == 0:
                return assignment.tolist()
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            if unsat_count < local_best_unsat:
                local_best_unsat = unsat_count
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            # Adaptive: if stuck for too long, break out early
            if no_improve_steps > 2000:
                break
            
            # Pick a weighted random unsatisfied clause
            if len(unsat_indices) > 0:
                # Weight by clause weight (heavier clauses are more likely to be picked)
                ci_weights = clause_weights[unsat_indices]
                total_w = float(np.sum(ci_weights))
                if total_w > 0:
                    probs = ci_weights / total_w
                    ci_idx = rng.choices(range(len(unsat_indices)), weights=probs, k=1)[0]
                else:
                    ci_idx = rng.randrange(len(unsat_indices))
                ci = int(unsat_indices[ci_idx])
            else:
                break
            
            # For each of the 3 variables in this clause, compute weighted flip delta
            deltas = []
            for j in range(3):
                v = int(clause_vars[ci, j])
                d = compute_flip_delta(v, assignment, num_true, clause_weights)
                # Add noise for exploration, scaled by degree
                noise_scale = 0.1 + 0.05 * min(var_degree[v], 10)
                d += rng.random() * noise_scale
                deltas.append((d, v))
            
            # Pick the variable with minimum delta
            deltas.sort()
            best_var = deltas[0][1]
            
            # With some probability, pick a random variable for exploration
            # Higher exploration probability when stuck
            explore_prob = 0.05 + 0.02 * min(no_improve_steps / 500.0, 1.0)
            if rng.random() < explore_prob:
                best_var = int(clause_vars[ci, rng.randrange(3)])
            
            # Apply the flip
            old_unsat = unsat_count
            apply_flip(best_var, assignment, num_true)
            
            # Update unsat count incrementally
            unsat_count = int(np.sum(num_true == 0))
            
            # Update unsat_indices
            if unsat_count != old_unsat:
                unsat_indices = np.where(num_true == 0)[0]
            
            # Update clause weights: increase weight of unsat clauses, decay sat ones
            unsat_mask = (num_true == 0)
            sat_mask = ~unsat_mask
            clause_weights[unsat_mask] *= 1.05  # boost unsat
            clause_weights[sat_mask] *= 0.95    # decay sat
            # Normalize to prevent overflow
            max_w = float(np.max(clause_weights))
            if max_w > 100.0:
                clause_weights /= max_w
            
            step += 1
        
        # Check if we solved it
        if unsat_count == 0:
            return assignment.tolist()
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [True] * n
# EVOLVE-BLOCK-END
