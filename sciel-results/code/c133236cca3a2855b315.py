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
        assignment = rng_np.random(n).astype(np.int8)
        
        lit_vals = (2 * assignment.astype(np.int8) - 1)  # -1 or 1
        clause_lit_vals = lit_vals[clause_vars]  # shape (m, 3)
        clause_products = clause_lit_vals * clause_polarity  # shape (m, 3)
        
        clause_unsat_count = (clause_products < 0).sum(axis=1).astype(np.int8)
        
        return assignment, clause_unsat_count
    
    def flip_var(vi, assignment, clause_unsat_count):
        """Flip variable vi and update clause_unsat_count incrementally using numpy."""
        old_val = assignment[vi]
        assignment[vi] = 1 - old_val  # Flip
        
        cidx = var_clause_indices[vi]
        cpol = var_clause_polarities[vi]
        
        if len(cidx) == 0:
            return
        
        old_lit_val = 2 * old_val - 1  # -1 or 1
        was_sat = (old_lit_val * cpol) > 0  # boolean array
        
        clause_unsat_count[cidx[was_sat]] += 1
        clause_unsat_count[cidx[~was_sat]] -= 1
    
    def get_unsat_count(clause_unsat_count):
        """Return total number of unsatisfied clauses."""
        return int((clause_unsat_count >= 3).sum())
    
    def compute_soft_unsat(clause_unsat_count, temperature=0.5):
        """Compute a smooth surrogate for unsatisfied clause count using log-sum-exp."""
        excess = np.maximum(clause_unsat_count - 3, 0).astype(np.float64)
        soft_excess = np.where(excess > 0, np.exp(excess / temperature), 0.0)
        return float(np.sum(soft_excess))
    
    def find_unsat_clause_weighted(clause_unsat_count, rng_np, temperature=1.0):
        """Find an unsatisfied clause with probability proportional to its violation severity."""
        unsat_mask = clause_unsat_count >= 3
        if not np.any(unsat_mask):
            return -1
        weights = np.where(unsat_mask, clause_unsat_count.astype(np.float64), 0.0)
        total_w = weights.sum()
        if total_w == 0:
            unsat_indices = np.where(unsat_mask)[0]
            return int(rng_np.choice(unsat_indices))
        probs = weights / total_w
        return int(rng_np.choice(len(clause_unsat_count), p=probs))
    
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
        
        was_sat = (old_lit_val * cpol) > 0
        
        gain = 0
        sat_clauses = cidx[was_sat]
        if len(sat_clauses) > 0:
            counts = clause_unsat_count[sat_clauses]
            gain -= int((counts == 2).sum())
        
        unsat_clauses = cidx[~was_sat]
        if len(unsat_clauses) > 0:
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
            
            unsat_idx = find_unsat_clause_weighted(clause_unsat_count, rng_np)
            if unsat_idx == -1:
                return assignment.tolist()
            
            cv = clause_vars[unsat_idx]
            
            if rng_np.random() < walk_prob:
                vi = int(rng_np.choice(cv))
            else:
                best_gain = -10
                best_vi = -1
                for li in range(3):
                    vi_candidate = int(cv[li])
                    g = compute_gain(vi_candidate, assignment, clause_unsat_count, unsat_idx)
                    g += rng_np.random() * 0.01
                    if g > best_gain:
                        best_gain = g
                        best_vi = vi_candidate
                vi = best_vi
            
            flip_var(vi, assignment, clause_unsat_count)
            
            current_unsat = get_unsat_count(clause_unsat_count)
            
            if current_unsat == 0:
                return assignment.tolist()
            
            if current_unsat < local_best:
                local_best = current_unsat
                steps_since_improve = 0
                walk_prob = max(walk_prob_base, walk_prob * 0.9)
            else:
                steps_since_improve += 1
                walk_prob = min(walk_prob_max, walk_prob * 1.05)
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
            
            if steps_since_improve > 10000:
                walk_prob = walk_prob_base
        
        # Polish pass on best assignment
        if best_assignment is not None and time.time() < deadline - 30:
            polish_assignment = best_assignment.copy()
            polish_unsat_count = clause_unsat_count.copy()
            
            lit_vals = (2 * polish_assignment.astype(np.int8) - 1)
            clause_lit_vals = lit_vals[clause_vars]
            clause_products = clause_lit_vals * clause_polarity
            polish_unsat_count = (clause_products < 0).sum(axis=1).astype(np.int8)
            
            polish_unsat = get_unsat_count(polish_unsat_count)
            
            polish_steps = 0
            polish_local_best = polish_unsat
            polish_walk_prob = 0.3
            
            while polish_steps < 10000 and time.time() < deadline - 10:
                pidx = find_unsat_clause_weighted(polish_unsat_count, rng_np)
                if pidx == -1:
                    break
                
                pcv = clause_vars[pidx]
                
                if rng_np.random() < polish_walk_prob:
                    pvi = int(rng_np.choice(pcv))
                else:
                    pbest_gain = -10
                    pbest_vi = -1
                    for pli in range(3):
                        pvi_candidate = int(pcv[pli])
                        pg = compute_gain(pvi_candidate, polish_assignment, polish_unsat_count, pidx)
                        pg += rng_np.random() * 0.01
                        if pg > pbest_gain:
                            pbest_gain = pg
                            pbest_vi = pvi_candidate
                    pvi = pbest_vi
                
                flip_var(pvi, polish_assignment, polish_unsat_count)
                polish_unsat = get_unsat_count(polish_unsat_count)
                polish_steps += 1
                
                if polish_unsat == 0:
                    return polish_assignment.tolist()
                
                if polish_unsat < polish_local_best:
                    polish_local_best = polish_unsat
                    polish_walk_prob = max(0.2, polish_walk_prob * 0.95)
                else:
                    polish_walk_prob = min(0.8, polish_walk_prob * 1.02)
                
                if polish_unsat < best_unsat:
                    best_unsat = polish_unsat
                    best_assignment = polish_assignment.copy()
    
    if best_assignment is not None:
        return best_assignment.tolist()
    
    rng2 = np.random.default_rng(42)
    return (rng2.random(n) < 0.5).tolist()
