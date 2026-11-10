"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from collections import defaultdict


def solve(instance):
    """WalkSAT with clause weighting, adaptive walk probability, and SA-style polish for 3-SAT."""
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # safety margin before 330s kill

    # Precompute literal indices (0-based) and sign for each clause
    clause_lits = []
    for c in clauses:
        lits = [(abs(l) - 1, 1 if l > 0 else -1) for l in c]
        clause_lits.append(lits)

    # For each variable, track which clauses contain it (and with what sign)
    pos_clauses = [[] for _ in range(n)]
    neg_clauses = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_lits):
        for var, sign in lits:
            if sign == 1:
                pos_clauses[var].append(ci)
            else:
                neg_clauses[var].append(ci)

    # Variable occurrence counts for weighted selection
    var_occurrence = [len(pos_clauses[v]) + len(neg_clauses[v]) for v in range(n)]

    def compute_clause_sat(assignment):
        """Compute satisfaction count for each clause."""
        clause_sat = [0] * m
        for ci, lits in enumerate(clause_lits):
            sat_count = 0
            for var, sign in lits:
                if assignment[var] == (sign == 1):
                    sat_count += 1
            clause_sat[ci] = sat_count
        return clause_sat

    def spectral_init(rng):
        """Initialize assignment using spectral method on literal co-occurrence matrix."""
        A = np.zeros((n, n), dtype=np.float64)
        
        for ci, lits in enumerate(clause_lits):
            for i in range(len(lits)):
                for j in range(i+1, len(lits)):
                    v1, s1 = lits[i]
                    v2, s2 = lits[j]
                    if s1 == s2:
                        A[v1][v2] += 1
                        A[v2][v1] += 1
                    else:
                        A[v1][v2] -= 1
                        A[v2][v1] -= 1
        
        # Add self-loops based on occurrence frequency
        for v in range(n):
            A[v][v] += 0.1 * var_occurrence[v]
        
        # Compute principal eigenvector
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            idx = np.argmax(eigenvalues)
            v_vec = eigenvectors[:, idx]
            assignment = [bool(x > 0) for x in v_vec]
        except Exception:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        return assignment

    def init_assignment(rng, use_spectral=True):
        """Initialize assignment with spectral method or clause-weighted bias."""
        if use_spectral:
            return spectral_init(rng)
        else:
            # Clause-weighted bias initialization
            assignment = []
            for v in range(n):
                if rng.random() < 0.3:
                    pos_count = len(pos_clauses[v])
                    neg_count = len(neg_clauses[v])
                    if pos_count > neg_count:
                        assignment.append(True)
                    elif neg_count > pos_count:
                        assignment.append(False)
                    else:
                        assignment.append(rng.random() < 0.5)
                else:
                    assignment.append(rng.random() < 0.5)
            return assignment

    def weighted_clause_select(clause_sat, clause_weights, rng):
        """Select an unsatisfied clause using weighted random selection."""
        # Build list of unsat clauses with weights
        unsat_clauses = []
        total_weight = 0.0
        for ci in range(m):
            if clause_sat[ci] == 0:
                w = clause_weights[ci]
                unsat_clauses.append((ci, w))
                total_weight += w
        
        if not unsat_clauses:
            return None
        
        if total_weight <= 0:
            return random.choice(unsat_clauses)[0]
        
        r = rng.random() * total_weight
        cumulative = 0.0
        for ci, w in unsat_clauses:
            cumulative += w
            if cumulative >= r:
                return ci
        return unsat_clauses[-1][0]

    def min_conflicts_var_select(clause_idx, assignment, clause_sat, var_unsat_count, rng, walk_prob):
        """Select a variable from the clause using min-conflicts heuristic or random walk."""
        lits = clause_lits[clause_idx]
        
        if rng.random() < walk_prob:
            # Random walk: pick a random variable from the clause
            var, sign = rng.choice(lits)
            return var
        
        # Min-conflicts: prefer variables that appear in fewer unsatisfied clauses
        scores = []
        for var, sign in lits:
            score = var_unsat_count[var]
            scores.append(score)
        
        total_score = sum(scores)
        if total_score == 0:
            var, sign = rng.choice(lits)
            return var
        
        # Weighted selection favoring lower scores (min-conflicts)
        # Use inverse scoring: lower unsat count -> higher probability
        inv_scores = [1.0 / (s + 1.0) for s in scores]
        total_inv = sum(inv_scores)
        r = rng.random() * total_inv
        cumulative = 0.0
        for i, inv_s in enumerate(inv_scores):
            cumulative += inv_s
            if cumulative >= r:
                return lits[i][0]
        return lits[-1][0]

    def update_unsat_counts(var, old_val, clause_sat, var_unsat_count):
        """Update clause satisfaction and variable unsat counts after flipping var."""
        delta_unsat = 0
        
        if old_val:
            # Was True, now False
            for ci in pos_clauses[var]:
                if clause_sat[ci] == 1:
                    # Becomes unsat
                    clause_sat[ci] -= 1
                    delta_unsat += 1
                    # Update var_unsat_count for all vars in this clause
                    for v2, s2 in clause_lits[ci]:
                        var_unsat_count[v2] += 1
                elif clause_sat[ci] == 0:
                    # Stays unsat (no change in count, but var_unsat_count unchanged for this clause)
                    pass
                else:
                    # clause_sat[ci] >= 2, stays sat
                    clause_sat[ci] -= 1
            for ci in neg_clauses[var]:
                if clause_sat[ci] == 1:
                    # Was sat due to neg, now neg is false, but pos might be true
                    # Wait, if old_val=True, then neg literal was false. So this clause was sat only if pos was true.
                    # Actually, let's just recompute or carefully track.
                    # Easier: just recompute clause_sat for affected clauses? No, that's slow.
                    # Let's stick to the delta approach but it's tricky.
                    # Alternative: just update clause_sat directly.
                    pass
                clause_sat[ci] += 1  # neg literal becomes true
                if clause_sat[ci] == 1:
                    # Just became sat
                    delta_unsat -= 1
                    for v2, s2 in clause_lits[ci]:
                        var_unsat_count[v2] -= 1
        else:
            # Was False, now True
            for ci in pos_clauses[var]:
                clause_sat[ci] += 1  # pos literal becomes true
                if clause_sat[ci] == 1:
                    # Just became sat
                    delta_unsat -= 1
                    for v2, s2 in clause_lits[ci]:
                        var_unsat_count[v2] -= 1
            for ci in neg_clauses[var]:
                if clause_sat[ci] == 1:
                    # Becomes unsat
                    clause_sat[ci] -= 1
                    delta_unsat += 1
                    for v2, s2 in clause_lits[ci]:
                        var_unsat_count[v2] += 1
                else:
                    clause_sat[ci] -= 1
        
        return delta_unsat

    def walksat_search(assignment, clause_sat, var_unsat_count, unsat_count, deadline, rng, max_steps):
        """Run WalkSAT with adaptive walk probability and clause weighting."""
        clause_weights = [1.0] * m
        base_walk_prob = 0.2
        max_walk_prob = 0.5
        current_walk_prob = base_walk_prob
        last_improvement_step = 0
        min_unsat_in_restart = unsat_count
        best_unsat_local = unsat_count
        best_assignment_local = assignment[:]
        
        step = 0
        while time.time() < deadline and step < max_steps:
            if unsat_count == 0:
                return assignment, clause_sat, 0, best_unsat_local, best_assignment_local
            
            if unsat_count < best_unsat_local:
                best_unsat_local = unsat_count
                best_assignment_local = assignment[:]
                last_improvement_step = step
                current_walk_prob = base_walk_prob
            
            # Increase walk probability if stuck
            if step - last_improvement_step > 500:
                current_walk_prob = min(max_walk_prob, current_walk_prob + 0.05)
            
            # Select clause
            clause_idx = weighted_clause_select(clause_sat, clause_weights, rng)
            if clause_idx is None:
                break
            
            # Select variable
            var = min_conflicts_var_select(clause_idx, assignment, clause_sat, var_unsat_count, rng, current_walk_prob)
            
            # Flip variable
            old_val = assignment[var]
            assignment[var] = not assignment[var]
            
            # Update counts
            unsat_count += update_unsat_counts(var, old_val, clause_sat, var_unsat_count)
            
            # Update clause weights: boost selected, decay others slightly
            clause_weights[clause_idx] *= 1.5
            # Decay all weights occasionally to prevent saturation
            if step % 100 == 0:
                for i in range(m):
                    clause_weights[i] *= 0.95
                # Normalize to prevent underflow/overflow
                max_w = max(clause_weights)
                if max_w > 0:
                    for i in range(m):
                        clause_weights[i] /= max_w
            
            step += 1
        
        return assignment, clause_sat, unsat_count, best_unsat_local, best_assignment_local

    def polish(assignment, deadline):
        """Polish: try flipping single vars that reduce unsat, with SA-style acceptance for near-zero."""
        polish_assignment = assignment[:]
        polish_clause_sat = compute_clause_sat(polish_assignment)
        polish_unsat = sum(1 for s in polish_clause_sat if s == 0)
        
        # Build var_unsat_count for polish
        var_unsat_count = [0] * n
        for ci in range(m):
            if polish_clause_sat[ci] == 0:
                for var, sign in clause_lits[ci]:
                    var_unsat_count[var] += 1
        
        improved = True
        T = 0.1  # Small temperature for SA-style acceptance
        while improved and time.time() < deadline - 5:
            improved = False
            for v in range(n):
                if polish_unsat == 0:
                    break
                # Compute delta for flipping v
                delta = 0
                if polish_assignment[v]:
                    for ci in pos_clauses[v]:
                        if polish_clause_sat[ci] == 1:
                            delta += 1
                else:
                    for ci in pos_clauses[v]:
                        if polish_clause_sat[ci] == 0:
                            delta -= 1
                if not polish_assignment[v]:
                    for ci in neg_clauses[v]:
                        if polish_clause_sat[ci] == 1:
                            delta += 1
                else:
                    for ci in neg_clauses[v]:
                        if polish_clause_sat[ci] == 0:
                            delta -= 1
                
                # Accept if delta < 0 (improvement) or with probability exp(-delta/T) if delta > 0
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / max(T, 1e-10))):
                    # Apply flip
                    polish_assignment[v] = not polish_assignment[v]
                    if polish_assignment[v]:
                        for ci in pos_clauses[v]:
                            polish_clause_sat[ci] += 1
                            if polish_clause_sat[ci] == 1:
                                for v2, s2 in clause_lits[ci]:
                                    var_unsat_count[v2] -= 1
                        for ci in neg_clauses[v]:
                            polish_clause_sat[ci] -= 1
                            if polish_clause_sat[ci] == 0:
                                for v2, s2 in clause_lits[ci]:
                                    var_unsat_count[v2] += 1
                    else:
                        for ci in neg_clauses[v]:
                            polish_clause_sat[ci] += 1
                            if polish_clause_sat[ci] == 1:
                                for v2, s2 in clause_lits[ci]:
                                    var_unsat_count[v2] -= 1
                        for ci in pos_clauses[v]:
                            polish_clause_sat[ci] -= 1
                            if polish_clause_sat[ci] == 0:
                                for v2, s2 in clause_lits[ci]:
                                    var_unsat_count[v2] += 1
                    
                    polish_unsat += delta
                    improved = True
        
        return polish_assignment, polish_unsat

    # Main loop with restarts
    rng = random.Random(42)
    global_best_unsat = float('inf')
    global_best_assignment = None

    # Restart strategy: mix spectral and random initializations
    num_restarts = 15
    phase_time_budget = 285.0 / num_restarts

    for restart in range(num_restarts):
        phase_deadline = time.time() + phase_time_budget

        # Alternate between spectral and random initialization
        use_spectral = (restart % 3 == 0)  # Spectral every 3rd restart
        assignment = init_assignment(rng, use_spectral=use_spectral)
        clause_sat = compute_clause_sat(assignment)
        unsat_count = sum(1 for s in clause_sat if s == 0)
        
        # Build var_unsat_count
        var_unsat_count = [0] * n
        for ci in range(m):
            if clause_sat[ci] == 0:
                for var, sign in clause_lits[ci]:
                    var_unsat_count[var] += 1

        # Run WalkSAT phase
        assignment, clause_sat, unsat_count, best_unsat_local, best_assignment_local = walksat_search(
            assignment, clause_sat, var_unsat_count, unsat_count, phase_deadline, rng, 300000
        )

        # Update global best
        if best_unsat_local < global_best_unsat:
            global_best_unsat = best_unsat_local
            global_best_assignment = best_assignment_local[:]

        if global_best_unsat == 0:
            return global_best_assignment

        # If we're close to solved, do a focused polish
        if global_best_unsat <= 10 and time.time() < deadline:
            polish_assignment, polish_unsat = polish(global_best_assignment, deadline)

            if polish_unsat < global_best_unsat:
                global_best_unsat = polish_unsat
                global_best_assignment = polish_assignment[:]

            if global_best_unsat == 0:
                return global_best_assignment

    # Final polish if time remains
    if global_best_assignment is not None and global_best_unsat > 0 and time.time() < deadline - 10:
        polish_assignment, polish_unsat = polish(global_best_assignment, deadline)
        if polish_unsat < global_best_unsat:
            global_best_unsat = polish_unsat
            global_best_assignment = polish_assignment[:]

    return global_best_assignment
# EVOLVE-BLOCK-END
