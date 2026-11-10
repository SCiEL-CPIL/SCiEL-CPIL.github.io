"""3-SAT solver with optimized WalkSAT + fast local search + deadline guard."""
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["instance_clauses"] if "instance_clauses" in instance else instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 330s kill

    # Preprocess: remove tautological clauses, dedupe
    seen = set()
    filtered_clauses = []
    for c in clauses:
        lits = set(c)
        if any(-l in lits for l in lits):
            continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            filtered_clauses.append(c)
    clauses = filtered_clauses

    clause_count = len(clauses)
    if clause_count == 0:
        return [True] * n

    # Build clause index for efficiency
    var_pos = [[] for _ in range(n)]
    var_neg = [[] for _ in range(n)]
    clause_vars = []
    for i, c in enumerate(clauses):
        var_list = []
        for l in c:
            idx = abs(l) - 1
            var_list.append((idx, l > 0))
            if l > 0:
                var_pos[idx].append(i)
            else:
                var_neg[idx].append(i)
        clause_vars.append(var_list)

    best_assignment = None
    best_unsat_count = clause_count + 1
    rng = random.Random(42)

    def init_sat_count(assignment):
        """Build sat_count array and unsat_clauses list from scratch."""
        sat_count = [0] * clause_count
        unsat_clauses = []
        for i, cv in enumerate(clause_vars):
            cnt = 0
            for idx, is_pos in cv:
                if assignment[idx] == is_pos:
                    cnt += 1
            sat_count[i] = cnt
            if cnt == 0:
                unsat_clauses.append(i)
        return sat_count, unsat_clauses

    def apply_flip(assignment, var, sat_count, unsat_clauses):
        """Flip variable var, update sat_count and unsat_clauses. Returns new unsat_clauses."""
        old_val = assignment[var]
        assignment[var] = not old_val

        # Track clauses that change sat status
        newly_unsat = []
        newly_sat = []
        for ci in var_pos[var]:
            old_sat = sat_count[ci]
            if old_val:
                sat_count[ci] = old_sat - 1
            else:
                sat_count[ci] = old_sat + 1
            if sat_count[ci] == 0:
                newly_unsat.append(ci)
            elif old_sat == 0:
                newly_sat.append(ci)

        for ci in var_neg[var]:
            old_sat = sat_count[ci]
            if old_val:
                sat_count[ci] = old_sat + 1
            else:
                sat_count[ci] = old_sat - 1
            if sat_count[ci] == 0:
                newly_unsat.append(ci)
            elif old_sat == 0:
                newly_sat.append(ci)

        # Rebuild unsat list efficiently
        sat_set = set(newly_sat)
        updated = []
        for ci in unsat_clauses:
            if ci not in sat_set:
                updated.append(ci)
        # Add newly unsat clauses (avoid duplicates)
        existing = set(updated)
        for ci in newly_unsat:
            if ci not in existing:
                updated.append(ci)
        return updated

    def score_var_flip(var, assignment, sat_count):
        """Score flipping var: return (score, delta) where score = #clauses newly satisfied,
        delta = change in unsat count (negative is good)."""
        old_val = assignment[var]
        target_val = not old_val
        score = 0
        delta = 0
        
        # Check positive literals
        for ci in var_pos[var]:
            sc = sat_count[ci]
            if sc == 0:
                # All false. If we set var to True (target_val), this literal becomes true.
                # Clause becomes satisfied.
                if target_val:
                    score += 1
                    delta -= 1
            elif sc == 1:
                # Exactly one satisfied. Check if positive literal of var is the satisfied one.
                if old_val:
                    # Positive literal is satisfied. Flipping to False breaks it.
                    delta += 1
                else:
                    # Positive literal not satisfied. Flipping to True adds satisfaction (1->2), no status change.
                    pass
        
        # Check negative literals
        for ci in var_neg[var]:
            sc = sat_count[ci]
            if sc == 0:
                # All false. If we set var to False (target_val), this literal becomes true.
                # Clause becomes satisfied.
                if not target_val:
                    score += 1
                    delta -= 1
            elif sc == 1:
                # Exactly one satisfied. Check if negative literal of var is the satisfied one.
                if not old_val:
                    # Negative literal is satisfied. Flipping to True breaks it.
                    delta += 1
                else:
                    # Negative literal not satisfied. Flipping to False adds satisfaction (1->2), no status change.
                    pass
        
        return score, delta

    # Phase 1: WalkSAT with adaptive noise and better restart strategy
    max_restarts = 10000
    base_steps = 4000
    base_noise = 0.25
    noise_decay = 0.00005  # slight decay per restart
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Adaptive noise: start high, decrease slightly
        current_noise = max(0.1, base_noise - restart * noise_decay)
        # Adaptive steps: slightly increase in later restarts if we're making progress
        current_steps = base_steps + min(2000, restart * 10)

        # Initialize assignment
        if restart == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            if best_assignment is not None and rng.random() < 0.7:
                assignment = best_assignment[:]
                # Perturb more as we go deeper
                n_perturb = max(1, min(n, n // 5 + restart // 100))
                for _ in range(n_perturb):
                    assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]

        sat_count, unsat_clauses = init_sat_count(assignment)
        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        for step in range(current_steps):
            if time.time() > deadline:
                break

            if not unsat_clauses:
                return assignment

            if rng.random() < current_noise:
                var = rng.randint(0, n - 1)
                unsat_clauses = apply_flip(assignment, var, sat_count, unsat_clauses)
                continue

            # Choose random unsatisfied clause
            clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            clause = clauses[clause_idx]

            # Score variables in clause: prefer those that satisfy more unsat clauses
            best_var = -1
            best_score = -1
            for l in clause:
                var = abs(l) - 1
                # Quick score: count how many unsat clauses contain this literal
                # This is cheaper than full delta computation for most cases
                target_val = (l > 0)
                if assignment[var] != target_val:
                    affected = var_pos[var] if l > 0 else var_neg[var]
                    score = 0
                    for ci in affected:
                        if sat_count[ci] == 0:
                            score += 1
                    # Bonus if flipping doesn't break any sat clauses
                    if assignment[var]:
                        for ci in affected:
                            if sat_count[ci] == 1:
                                score -= 0.5  # penalty for potentially breaking
                else:
                    # Variable already has target value; flipping would break this clause
                    score = -10  # discourage

                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var >= 0:
                var_to_flip = best_var
            else:
                l = clause[rng.randint(0, len(clause) - 1)]
                var_to_flip = abs(l) - 1

            unsat_clauses = apply_flip(assignment, var_to_flip, sat_count, unsat_clauses)

            if step % 200 == 0:
                current_unsat = len(unsat_clauses)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]

    # Phase 2: Fast local search polish using incremental tracking
    if best_assignment is not None and time.time() < deadline - 5:
        assignment = best_assignment[:]
        sat_count, unsat_clauses = init_sat_count(assignment)

        if len(unsat_clauses) == 0:
            return assignment

        # Iterative improvement: find flips that reduce unsat count
        max_iter = 30000
        stall_count = 0
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break

            if not unsat_clauses:
                return assignment

            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                stall_count = 0
            else:
                stall_count += 1

            # Find best flip: one that reduces unsat count the most
            # Only consider variables in unsat clauses
            candidate_vars = set()
            # Sample up to 200 unsat clauses for candidates
            sample_size = min(200, len(unsat_clauses))
            for _ in range(sample_size):
                ci = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
                for idx, is_pos in clause_vars[ci]:
                    candidate_vars.add(idx)

            if not candidate_vars:
                break

            best_var = -1
            best_delta = 0  # we want negative delta (reduction in unsat)
            best_score = 0

            for var in candidate_vars:
                score, delta = score_var_flip(var, assignment, sat_count)
                # Prefer negative delta (reduces unsat), then higher score
                if delta < best_delta or (delta == best_delta and score > best_score):
                    best_delta = delta
                    best_var = var
                    best_score = score

            if best_var >= 0 and best_delta < 0:
                unsat_clauses = apply_flip(assignment, best_var, sat_count, unsat_clauses)
            elif best_var >= 0 and best_delta == 0 and rng.random() < 0.3:
                # Accept neutral moves occasionally to escape local optima
                unsat_clauses = apply_flip(assignment, best_var, sat_count, unsat_clauses)
            else:
                # No improving flip found; try random perturbation
                if rng.random() < 0.3:
                    # Strong perturbation
                    n_perturb = max(2, n // 10)
                    for _ in range(n_perturb):
                        var = rng.randint(0, n - 1)
                        unsat_clauses = apply_flip(assignment, var, sat_count, unsat_clauses)
                elif rng.random() < 0.5:
                    var = rng.randint(0, n - 1)
                    unsat_clauses = apply_flip(assignment, var, sat_count, unsat_clauses)
                else:
                    # Flip a variable from an unsat clause
                    if unsat_clauses:
                        ci = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
                        cv2 = clause_vars[ci]
                        var = cv2[rng.randint(0, len(cv2) - 1)][0]
                        unsat_clauses = apply_flip(assignment, var, sat_count, unsat_clauses)

            # If stuck for too long, do a bigger restart
            if stall_count > 500:
                # Perturb significantly
                n_perturb = max(3, n // 5)
                for _ in range(n_perturb):
                    var = rng.randint(0, n - 1)
                    unsat_clauses = apply_flip(assignment, var, sat_count, unsat_clauses)
                stall_count = 0

    # Phase 3: Final high-precision polish - try all single flips from best
    if best_assignment is not None and time.time() < deadline - 1:
        assignment = best_assignment[:]
        sat_count, unsat_clauses = init_sat_count(assignment)
        
        if len(unsat_clauses) == 0:
            return assignment
        
        # Try each variable flip once, keep the best
        best_local = assignment[:]
        best_local_unsat = len(unsat_clauses)
        
        for var in range(n):
            if time.time() > deadline - 1:
                break
            # Try flipping var
            old_val = assignment[var]
            assignment[var] = not old_val
            sat_count, unsat_clauses = init_sat_count(assignment)
            current_unsat = len(unsat_clauses)
            
            if current_unsat < best_local_unsat:
                best_local_unsat = current_unsat
                best_local = assignment[:]
            
            # Restore
            assignment[var] = old_val
        
        if best_local_unsat < best_unsat_count:
            best_unsat_count = best_local_unsat
            best_assignment = best_local
        
        if best_unsat_count == 0:
            return best_assignment

    # Final check
    if best_assignment is not None:
        # Verify and return
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
