"""3-SAT solver with vectorized WalkSAT + incremental tracking + restarts."""
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["instance_clauses"] if "instance_clauses" in instance else instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285  # safety margin before 330s kill

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

    # Build data structures
    var_pos = [[] for _ in range(n)]
    var_neg = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            idx = abs(l) - 1
            if l > 0:
                var_pos[idx].append(i)
            else:
                var_neg[idx].append(i)

    # Build numpy arrays for vectorized operations
    # clause_lits: 2D array of shape (clause_count, 3) with signed literals
    clause_lits = np.zeros((clause_count, 3), dtype=np.int32)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            clause_lits[i, j] = l

    # Variable indices for each clause position
    var_indices = np.abs(clause_lits) - 1  # shape (clause_count, 3)
    # Sign: 1 if positive literal, -1 if negative
    signs = np.sign(clause_lits).astype(np.int8)

    # For each clause and position, the variable index
    # lit_satisfied[i,j] = assignment[var_indices[i,j]] == (clause_lits[i,j] > 0)
    # We can compute: if clause_lits[i,j] > 0, we need assignment[var] == True
    #                  if clause_lits[i,j] < 0, we need assignment[var] == False
    # So lit_satisfied = (assignment[var] == (clause_lits > 0))
    # In numpy: satisfied = (assignment[var_indices] == (signs > 0))

    # Precompute for each variable, the list of (clause_idx, position) where it appears
    # var_clause_pos[var] = list of (clause_idx, pos)
    var_clause_pos = [[] for _ in range(n)]
    for i in range(clause_count):
        for j in range(3):
            v = var_indices[i, j]
            var_clause_pos[v].append((i, j))

    # Convert to numpy arrays for faster access
    var_clause_pos_arr = []
    for v in range(n):
        if var_clause_pos[v]:
            arr = np.array(var_clause_pos[v], dtype=np.int32).reshape(-1, 2)
            var_clause_pos_arr.append(arr)
        else:
            var_clause_pos_arr.append(np.empty((0, 2), dtype=np.int32))

    best_assignment = None
    best_unsat_count = clause_count + 1

    def count_unsat_vec(assignment_np):
        """Count unsatisfied clauses using numpy."""
        # assignment_np: boolean array of length n
        # For each clause position, check if the literal is satisfied
        # satisfied[i,j] = assignment_np[var_indices[i,j]] == (signs[i,j] > 0)
        a = assignment_np[var_indices]  # shape (clause_count, 3)
        sat = (a == (signs > 0)).astype(np.int8)  # shape (clause_count, 3)
        sat_count = sat.sum(axis=1)  # shape (clause_count,)
        return int((sat_count == 0).sum())

    def build_state(assignment_np):
        """Build sat_count and unsat_clauses from assignment."""
        a = assignment_np[var_indices]
        sat = (a == (signs > 0)).astype(np.int8)
        sat_count = sat.sum(axis=1)
        unsat_mask = (sat_count == 0)
        unsat_clauses = np.where(unsat_mask)[0]
        return sat_count, unsat_clauses

    def flip_var(var, assignment_np, sat_count, unsat_clauses):
        """Flip variable var and update sat_count, unsat_clauses incrementally."""
        old_val = assignment_np[var]
        assignment_np[var] = not old_val

        # Get all (clause_idx, pos) where this variable appears
        vc = var_clause_pos_arr[var]
        if len(vc) == 0:
            return assignment_np, sat_count, unsat_clauses

        ci_list = vc[:, 0]
        pos_list = vc[:, 1]

        # For each occurrence, determine if the literal was satisfied before flip
        # literal satisfied = assignment[var] == (sign > 0)
        # Before flip: old_val == (signs[ci,pos] > 0)
        old_sat = (old_val == (signs[ci_list, pos_list] > 0))

        # After flip: not old_val == (signs[ci,pos] > 0)
        # If old_sat was True, now it's False: sat_count decreases by 1
        # If old_sat was False, now it's True: sat_count increases by 1
        delta = np.where(old_sat, -1, 1).astype(np.int8)

        # Update sat_count
        # We need to add delta to sat_count[ci_list]
        # But ci_list may have duplicates? No, each (clause,pos) is unique per variable
        np.add.at(sat_count, ci_list, delta)

        # Determine new unsat clauses
        # Clauses that just became unsat: sat_count[ci] == 0 and old_sat was True (was satisfied, now not)
        new_unsat_candidates = ci_list[old_sat]
        # Clauses that were unsat but might now be sat: need to check
        # Previously unsat: sat_count[ci] == 0 before update
        # After update: sat_count[ci] > 0 means it's now sat

        # Build new unsat set
        # Start with old unsat_clauses, remove those that are now sat, add those that became unsat
        old_unsat_set = set(unsat_clauses.tolist())

        # Clauses in old_unsat that are now satisfied
        for ci in old_unsat_set:
            if sat_count[ci] > 0:
                old_unsat_set.discard(ci)

        # Clauses that just became unsat
        for ci in new_unsat_candidates:
            if sat_count[ci] == 0:
                old_unsat_set.add(ci)

        unsat_clauses = np.array(sorted(old_unsat_set), dtype=np.int32)
        return assignment_np, sat_count, unsat_clauses

    rng = random.Random(42)
    max_restarts = 5000
    steps_per_restart = 8000
    noise = 0.25

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize assignment
        if restart == 0:
            assignment_np = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        else:
            if best_assignment is not None and rng.random() < 0.7:
                assignment_np = best_assignment.copy()
                # Flip a few random variables
                for _ in range(n // 4 + 1):
                    v = rng.randint(0, n - 1)
                    assignment_np[v] = not assignment_np[v]
            else:
                assignment_np = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)

        sat_count, unsat_clauses = build_state(assignment_np)
        current_unsat = len(unsat_clauses)

        if current_unsat == 0:
            return assignment_np.tolist()

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment_np.copy()

        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            if len(unsat_clauses) == 0:
                return assignment_np.tolist()

            # With some probability, do a random flip
            if rng.random() < noise:
                var = rng.randint(0, n - 1)
                assignment_np, sat_count, unsat_clauses = flip_var(var, assignment_np, sat_count, unsat_clauses)
                continue

            # Choose a random unsatisfied clause
            clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            clause = clauses[clause_idx]

            # Score each variable in the clause: count how many unsat clauses it would fix
            best_var = -1
            best_score = -1
            for l in clause:
                var = abs(l) - 1
                score = 0
                target_val = (l > 0)
                if not assignment_np[var]:
                    # Flipping would make this literal true (if positive) or false (if negative)
                    # Wait: if l > 0, target_val is True. If assignment is False, flipping to True satisfies it.
                    # if l < 0, target_val is False. If assignment is True, flipping to False satisfies it.
                    # So: if assignment_np[var] != target_val, flipping satisfies this literal.
                    pass
                if assignment_np[var] != target_val:
                    # Flipping this variable would satisfy this literal
                    # Count how many unsat clauses this would fix (where this is the only unsat literal)
                    vc = var_clause_pos_arr[var]
                    if len(vc) > 0:
                        ci_list = vc[:, 0]
                        pos_list = vc[:, 1]
                        # For each clause this var appears in, check if it's unsat and this literal is the only unsat one
                        for k in range(len(vc)):
                            ci = ci_list[k]
                            if sat_count[ci] != 0:
                                continue
                            # This clause is unsat. Check if flipping var would satisfy it
                            # The literal at position pos_list[k] in clause ci involves var
                            # It's satisfied if assignment[var] == (signs[ci, pos_list[k]] > 0)
                            # After flip: not assignment[var] == (signs[ci, pos_list[k]] > 0)
                            # So it would be satisfied if currently not satisfied
                            if not (assignment_np[var] == (signs[ci, pos_list[k]] > 0)):
                                # This literal would become satisfied. Is it the only unsat one?
                                # Check other literals in clause ci
                                only_unsat = True
                                for j in range(3):
                                    if j == pos_list[k]:
                                        continue
                                    v2 = var_indices[ci, j]
                                    if assignment_np[v2] == (signs[ci, j] > 0):
                                        only_unsat = False
                                        break
                                if only_unsat:
                                    score += 1

                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var >= 0:
                var_to_flip = best_var
            else:
                l = clause[rng.randint(0, len(clause) - 1)]
                var_to_flip = abs(l) - 1

            assignment_np, sat_count, unsat_clauses = flip_var(var_to_flip, assignment_np, sat_count, unsat_clauses)

            # Periodically check unsat count and update best
            if step % 500 == 0:
                current_unsat = len(unsat_clauses)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment_np.copy()

    # Phase 2: Local search from best assignment with vectorized scoring
    if best_assignment is not None:
        assignment_np = best_assignment.copy()
        sat_count, unsat_clauses = build_state(assignment_np)
        for _ in range(20000):
            if time.time() > deadline:
                break
            if len(unsat_clauses) == 0:
                return assignment_np.tolist()

            # Greedy: find variable that fixes most unsat clauses
            best_var = -1
            best_gain = 0
            # Sample up to 100 unsat clauses for speed
            sample_size = min(100, len(unsat_clauses))
            sample_indices = rng.sample(range(len(unsat_clauses)), sample_size)
            sample_clauses = [unsat_clauses[i] for i in sample_indices]

            for ci in sample_clauses:
                for j in range(3):
                    var = var_indices[ci, j]
                    if assignment_np[var] == (signs[ci, j] > 0):
                        continue
                    # Count how many unsat clauses this flip would fix
                    gain = 0
                    vc = var_clause_pos_arr[var]
                    if len(vc) == 0:
                        continue
                    ci_list = vc[:, 0]
                    pos_list = vc[:, 1]
                    for k in range(len(vc)):
                        ci2 = ci_list[k]
                        if sat_count[ci2] != 0:
                            continue
                        if not (assignment_np[var] == (signs[ci2, pos_list[k]] > 0)):
                            # Would be satisfied. Check if only unsat
                            only_unsat = True
                        else:
                            continue
                        for j2 in range(3):
                            if j2 == pos_list[k]:
                                continue
                            v2 = var_indices[ci2, j2]
                            if assignment_np[v2] == (signs[ci2, j2] > 0):
                                only_unsat = False
                                break
                        if only_unsat:
                            gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_var = var

            if best_var >= 0:
                assignment_np, sat_count, unsat_clauses = flip_var(best_var, assignment_np, sat_count, unsat_clauses)
                current_unsat = len(unsat_clauses)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment_np.copy()
                if current_unsat == 0:
                    return assignment_np.tolist()

    # Return best found or random
    if best_assignment is not None:
        return best_assignment.tolist()
    return [rng.random() < 0.5 for _ in range(n)]
