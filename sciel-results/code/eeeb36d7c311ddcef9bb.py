"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause structure as arrays
    clause_vars = np.zeros((m, 3), dtype=np.int32)  # variable indices (0-based)
    clause_polarity = np.zeros((m, 3), dtype=np.int8)  # 1 for positive, -1 for negative
    for ci, c in enumerate(clauses):
        for li, l in enumerate(c):
            clause_vars[ci, li] = abs(l) - 1
            clause_polarity[ci, li] = 1 if l > 0 else -1
    
    # For each variable, store list of (clause_idx, polarity)
    var_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for li in range(3):
            vi = clause_vars[ci, li]
            var_clauses[vi].append((ci, clause_polarity[ci, li]))
    
    # Convert var_clauses to numpy arrays for speed
    var_clause_indices = []
    var_clause_polarities = []
    for vi in range(n):
        if var_clauses[vi]:
            var_clause_indices.append(np.array([x[0] for x in var_clauses[vi]], dtype=np.int32))
            var_clause_polarities.append(np.array([x[1] for x in var_clauses[vi]], dtype=np.int8))
        else:
            var_clause_indices.append(np.array([], dtype=np.int32))
            var_clause_polarities.append(np.array([], dtype=np.int8))
    
    def init_state(rng_np):
        """Initialize random assignment and compute clause unsat counts."""
        # Random assignment: 0 or 1
        assignment = rng_np.random(n).astype(np.int8)
        
        # Compute clause satisfaction: for each clause, count falsified literals
        # A literal is falsified if (assignment[vi] == 0 and polarity == 1) or (assignment[vi] == 1 and polarity == -1)
        # i.e., assignment[vi] * polarity == -1 (if we treat 0 as -1 and 1 as +1)
        # Let's use: lit_val = 2*assignment - 1 (so 0->-1, 1->1)
        # Literal is satisfied if lit_val * polarity > 0
        # Literal is falsified if lit_val * polarity < 0
        # For 3-SAT, unsat_count = 3 - (number of satisfied literals)
        
        lit_vals = (2 * assignment.astype(np.int8) - 1)  # -1 or 1
        # For each clause, compute product of lit_val * polarity for each literal
        # clause_sat[ci, li] = lit_vals[clause_vars[ci,li]] * clause_polarity[ci,li]
        # If product > 0, literal is satisfied; if < 0, falsified
        
        # Vectorized: gather lit_vals for all clause positions
        clause_lit_vals = lit_vals[clause_vars]  # shape (m, 3)
        clause_products = clause_lit_vals * clause_polarity  # shape (m, 3)
        
        # Count falsified literals (product < 0)
        clause_unsat_count = (clause_products < 0).sum(axis=1).astype(np.int8)
        
        return assignment, clause_unsat_count
    
    def flip_var(vi, assignment, clause_unsat_count):
        """Flip variable vi and update clause_unsat_count incrementally using numpy."""
        old_val = assignment[vi]
        assignment[vi] = 1 - old_val  # Flip
        
        # Get clauses containing this variable
        cidx = var_clause_indices[vi]
        cpol = var_clause_polarities[vi]
        
        if len(cidx) == 0:
            return
        
        # For each clause containing vi, check if the literal was satisfied before
        # old_val: 0 or 1. polarity: 1 or -1
        # Literal was satisfied if (old_val == 0 and pol == -1) or (old_val == 1 and pol == 1)
        # i.e., old_val * pol > 0 if we map 0->-1, 1->1
        # Let's compute: old_lit_val = 2*old_val - 1
        old_lit_val = 2 * old_val - 1  # -1 or 1
        was_sat = (old_lit_val * cpol) > 0  # boolean array
        
        # After flip, new_lit_val = -old_lit_val
        # New literal is satisfied if (-old_lit_val) * pol > 0, i.e., old_lit_val * pol < 0
        # So if was_sat, it becomes unsat (count +1)
        # If not was_sat, it becomes sat (count -1)
        
        clause_unsat_count[cidx[was_sat]] += 1
        clause_unsat_count[cidx[~was_sat]] -= 1
    
    def get_unsat_count(clause_unsat_count):
        """Return total number of unsatisfied clauses."""
        return int((clause_unsat_count >= 3).sum())
    
    def find_unsat_clause(clause_unsat_count, rng_np):
        """Find an unsatisfied clause index."""
        unsat_mask = clause_unsat_count >= 3
        unsat_indices = np.where(unsat_mask)[0]
        if len(unsat_indices) == 0:
            return -1
        return int(rng_np.choice(unsat_indices))
    
    def compute_gain(vi, assignment, clause_unsat_count, unsat_idx):
        """Compute the gain (change in number of unsatisfied clauses) if we flip vi."""
        cidx = var_clause_indices[vi]
        cpol = var_clause_polarities[vi]
        
        if len(cidx) == 0:
            return 0
        
        old_val = assignment[vi]
        old_lit_val = 2 * old_val - 1  # -1 or 1
        
        # For each clause containing vi, determine if flipping changes satisfaction
        # was_sat: literal was satisfied before flip
        was_sat = (old_lit_val * cpol) > 0
        
        # Gain = -1 for each clause that was sat and becomes unsat
        #        +1 for each clause that was unsat and becomes sat
        # A clause becomes unsat if it was sat (was_sat=True) and had exactly 2 unsat literals (count==2)
        # A clause becomes sat if it was unsat (was_sat=False) and had exactly 3 unsat literals (count==3)
        
        gain = 0
        # Clauses that were satisfied and might become unsatisfied
        sat_clauses = cidx[was_sat]
        if len(sat_clauses) > 0:
            # These clauses had count < 3. They become unsat if count == 2
            counts = clause_unsat_count[sat_clauses]
            gain -= int((counts == 2).sum())
        
        # Clauses that were unsatisfied and might become satisfied
        unsat_clauses = cidx[~was_sat]
        if len(unsat_clauses) > 0:
            # These clauses had count >= 3 (actually count == 3 for 3-SAT). They become sat if count == 3
            counts = clause_unsat_count[unsat_clauses]
            gain += int((counts == 3).sum())
        
        return gain
    
    # Main search loop
    best_assignment = None
    best_unsat = m + 1
    
    rng_np = np.random.default_rng(42)
    
    # Parameters
    max_restarts = 200
    base_steps = 50000
    max_steps = 500000
    walk_prob_base = 0.2
    walk_prob_max = 0.8
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        assignment, clause_unsat_count = init_state(rng_np)
        current_unsat = get_unsat_count(clause_unsat_count)
        
        if current_unsat == 0:
            return assignment.tolist()
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
        
        steps_in_restart = base_steps + restart * 5000
        steps_in_restart = min(steps_in_restart, max_steps)
        
        local_best = current_unsat
        steps_since_improve = 0
        walk_prob = walk_prob_base
        
        for step in range(steps_in_restart):
            if time.time() > deadline:
                break
            
            # Find an unsatisfied clause
            unsat_idx = find_unsat_clause(clause_unsat_count, rng_np)
            if unsat_idx == -1:
                return assignment.tolist()
            
            # Get the three variables in this clause
            cv = clause_vars[unsat_idx]
            
            # Decide: greedy or random walk
            if rng_np.random() < walk_prob:
                # Random: pick a random variable from the clause
                vi = int(rng_np.choice(cv))
            else:
                # Greedy: pick the variable with best gain
                best_gain = -10
                best_vi = -1
                for li in range(3):
                    vi_candidate = int(cv[li])
                    g = compute_gain(vi_candidate, assignment, clause_unsat_count, unsat_idx)
                    # Add small random tiebreaker
                    g += rng_np.random() * 0.01
                    if g > best_gain:
                        best_gain = g
                        best_vi = vi_candidate
                vi = best_vi
            
            # Flip the variable
            flip_var(vi, assignment, clause_unsat_count)
            
            # Update unsat count
            current_unsat = get_unsat_count(clause_unsat_count)
            
            if current_unsat == 0:
                return assignment.tolist()
            
            if current_unsat < local_best:
                local_best = current_unsat
                steps_since_improve = 0
                # Reduce walk probability on improvement
                walk_prob = max(walk_prob_base, walk_prob * 0.9)
            else:
                steps_since_improve += 1
                # Increase walk probability on stagnation
                walk_prob = min(walk_prob_max, walk_prob * 1.05)
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
            
            # Periodic reset of walk probability to prevent it from getting too high
            if steps_since_improve > 10000:
                walk_prob = walk_prob_base
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    
    # Fallback: random assignment
    rng2 = np.random.default_rng(42)
    return (rng2.random(n) < 0.5).tolist()
# EVOLVE-BLOCK-END
