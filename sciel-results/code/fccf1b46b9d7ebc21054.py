"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Precompute clause literals as (var_index, polarity) for speed
    # var_index is 0-based, polarity is +1 or -1
    cl = []
    for c in clauses:
        cl.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])
    
    # Precompute variable to clauses mapping
    # var_to_clauses[var] = list of (clause_idx, polarity_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(cl):
        for var, pol in clause:
            var_to_clauses[var].append((i, pol))
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s/330s limit
    
    # Best known assignment (by number of satisfied clauses)
    best_sat_count = -1
    best_assign = None
    
    rng = random.Random(0)
    
    # WalkSAT parameters
    max_restarts = 1000
    base_steps = 10000
    noise_schedule = [0.25, 0.5, 0.75, 0.35, 0.65, 0.45, 0.55, 0.3, 0.7, 0.4]
    
    def init_state(assign):
        """Initialize clause satisfaction counts and unsat list."""
        clause_sat_count = [0] * m
        unsat_indices = []
        
        for i in range(m):
            sat_count = 0
            for var, pol in cl[i]:
                if assign[var] == (pol > 0):
                    sat_count += 1
            clause_sat_count[i] = sat_count
            if sat_count == 0:
                unsat_indices.append(i)
                
        return clause_sat_count, unsat_indices

    def flip_variable(assign, var, target_val, clause_sat_count, unsat_indices):
        """Flip variable to target_val and update state."""
        old_val = assign[var]
        if old_val == target_val:
            return
        
        assign[var] = target_val
        
        for c_idx, pol in var_to_clauses[var]:
            new_lit_sat = (pol > 0) == target_val
            old_lit_sat = (pol > 0) == old_val
            
            if new_lit_sat != old_lit_sat:
                old_count = clause_sat_count[c_idx]
                new_count = old_count + (1 if new_lit_sat else -1)
                clause_sat_count[c_idx] = new_count
                
                old_unsat = (old_count == 0)
                new_unsat = (new_count == 0)
                
                if old_unsat and not new_unsat:
                    # Removed from unsat
                    try:
                        unsat_indices.remove(c_idx)
                    except ValueError:
                        pass
                elif not old_unsat and new_unsat:
                    # Added to unsat
                    unsat_indices.append(c_idx)

    def dpll_unit_propagation(assign, clause_sat_count, unsat_indices):
        """
        Perform unit propagation. Returns (success, assign, clause_sat_count, unsat_indices).
        success=True if no contradiction and all clauses satisfied or no more units.
        """
        # We need to track which variables are unassigned
        unassigned = [True] * n
        
        # Build a list of clauses that are currently unit (exactly one unassigned literal)
        # We'll use a worklist approach
        
        # First, identify all unit clauses
        unit_clauses = []
        for i in range(m):
            if clause_sat_count[i] > 0:
                continue  # Already satisfied
            # Count unassigned literals
            unassigned_count = 0
            for var, pol in cl[i]:
                if unassigned[var]:
                    unassigned_count += 1
            if unassigned_count == 1:
                unit_clauses.append(i)
        
        # Process unit clauses
        while unit_clauses:
            if time.time() > deadline:
                return False, assign, clause_sat_count, unsat_indices
            
            ci = unit_clauses.pop()
            
            # Find the unassigned literal in this clause
            for var, pol in cl[ci]:
                if unassigned[var]:
                    # This is the unit literal; assign it
                    target_val = (pol > 0)
                    unassigned[var] = False
                    flip_variable(assign, var, target_val, clause_sat_count, unsat_indices)
                    
                    # After flipping, some clauses may become unit
                    # Check all clauses containing var
                    for c_idx, c_pol in var_to_clauses[var]:
                        if clause_sat_count[c_idx] > 0:
                            continue  # Already satisfied
                        # Count remaining unassigned literals
                        unassigned_count = 0
                        for v2, p2 in cl[c_idx]:
                            if unassigned[v2]:
                                unassigned_count += 1
                        if unassigned_count == 1:
                            unit_clauses.append(c_idx)
                        elif unassigned_count == 0:
                            # Contradiction: clause is unsatisfied and no unassigned literals
                            return False, assign, clause_sat_count, unsat_indices
                    break
        
        # Check if all clauses are satisfied
        if not unsat_indices:
            return True, assign, clause_sat_count, unsat_indices
        
        return False, assign, clause_sat_count, unsat_indices

    def spectral_init():
        """
        Use SVD of the clause-variable incidence matrix to get an initial assignment.
        For planted instances, the true assignment may create a low-rank structure.
        """
        try:
            import numpy as np
            
            # Build incidence matrix A[m, n]
            # A[i, j] = 1 if clause i has literal +j, -1 if -j, 0 otherwise
            A = np.zeros((m, n))
            for i, clause in enumerate(cl):
                for var, pol in clause:
                    A[i, var] = pol
            
            # SVD
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            
            # The idea: for a planted instance, there exists a vector x (the true assignment)
            # such that A @ x >= 0 (all clauses satisfied).
            # If we project onto the dominant right singular vectors, we might recover structure.
            
            # Use the first few right singular vectors
            # Vt has shape (min(m,n), n), so Vt.T has shape (n, min(m,n))
            # The columns of Vt.T are the right singular vectors
            
            # Try different numbers of components
            best_assign = None
            best_sat = -1
            
            for k in [1, 2, 3, 5, 10]:
                if k > Vt.shape[0]:
                    break
                # Project onto first k right singular vectors
                # Vt[:k, :] has shape (k, n)
                # For each variable, compute a score based on its projection
                scores = Vt[:k, :] @ (s[:k] ** 2)  # Weighted by singular values squared
                # Or just use the first singular vector
                # scores = Vt[0, :]
                
                # Convert scores to boolean assignment
                # If score > 0, assign True; else False
                assign_k = [score > 0 for score in scores]
                
                # Count satisfied clauses
                sat_count = 0
                for i in range(m):
                    for var, pol in cl[i]:
                        if assign_k[var] == (pol > 0):
                            sat_count += 1
                            break
                
                if sat_count > best_sat:
                    best_sat = sat_count
                    best_assign = list(assign_k)
            
            if best_assign is not None and best_sat > m * 0.5:
                return best_assign
            
        except Exception:
            pass
        
        return None

    def walksat(assign, noise, max_steps, rng):
        """Run WalkSAT, return (satisfied, assign, sat_count)"""
        clause_sat_count, unsat_indices = init_state(assign)
        
        for step in range(max_steps):
            if not unsat_indices:
                return True, assign, m
            
            # Pick a random unsatisfied clause
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause = cl[ci]
            
            # Find best variable to flip
            best_var = None
            best_conflicts = float('inf')
            best_pol = None
            
            for var, pol in clause:
                target_val = (pol > 0)
                
                # Calculate conflicts
                conflicts = 0
                
                for c_idx, c_pol in var_to_clauses[var]:
                    if c_idx == ci:
                        continue
                    
                    if clause_sat_count[c_idx] > 0:
                        if (c_pol > 0) == target_val:
                            pass
                        else:
                            if clause_sat_count[c_idx] == 1:
                                conflicts += 1
                    
                if conflicts < best_conflicts or (conflicts == best_conflicts and rng.random() < 0.5):
                    best_conflicts = conflicts
                    best_var = var
                    best_pol = pol
            
            # With probability noise, flip a random variable in the clause
            if rng.random() < noise:
                var, pol = clause[rng.randrange(len(clause))]
            else:
                var = best_var
                pol = best_pol
            
            target_val = (pol > 0)
            flip_variable(assign, var, target_val, clause_sat_count, unsat_indices)
            
            if step % 100 == 0 and time.time() > deadline:
                break
        
        sat_count = sum(1 for c in range(m) if clause_sat_count[c] > 0)
        return False, assign, sat_count

    # Try spectral initialization first
    spectral_assign = spectral_init()
    
    # Main loop with restarts
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Choose noise level
        noise = noise_schedule[restart % len(noise_schedule)]
        
        # Initialize: use spectral init for first few restarts, then random
        if restart < 3 and spectral_assign is not None:
            assign = list(spectral_assign)
        else:
            assign = [rng.random() < 0.5 for _ in range(n)]
        
        # Try unit propagation first
        clause_sat_count, unsat_indices = init_state(assign)
        solved, assign, clause_sat_count, unsat_indices = dpll_unit_propagation(assign, clause_sat_count, unsat_indices)
        
        if solved:
            return assign
        
        # Run WalkSAT
        solved, assign, sat_count = walksat(assign, noise, base_steps, rng)
        
        if sat_count > best_sat_count:
            best_sat_count = sat_count
            best_assign = list(assign)
        
        if solved:
            return assign
    
    # If we didn't solve, return the best assignment found
    if best_assign is not None:
        return best_assign
    
    # Fallback: random assignment
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
