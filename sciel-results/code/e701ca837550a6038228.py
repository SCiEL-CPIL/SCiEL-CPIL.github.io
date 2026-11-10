"""3-SAT solver with hybrid WalkSAT + restarts + deadline guard."""
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
        # Check for tautology (both x and -x in clause)
        lits = set(c)
        if any(-l in lits for l in lits):
            continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            filtered_clauses.append(c)
    clauses = filtered_clauses

    # Build clause index for efficiency
    clause_count = len(clauses)
    if clause_count == 0:
        return [True] * n

    # Variable occurrence counts for heuristics
    var_pos = [[] for _ in range(n)]
    var_neg = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            idx = abs(l) - 1
            if l > 0:
                var_pos[idx].append(i)
            else:
                var_neg[idx].append(i)

    best_assignment = None
    best_unsat_count = clause_count + 1

    def clause_sat_count(assignment, c):
        """Count how many literals in clause c are satisfied by assignment."""
        count = 0
        for l in c:
            idx = abs(l) - 1
            if assignment[idx] == (l > 0):
                count += 1
        return count

    def count_unsat(assignment):
        count = 0
        for c in clauses:
            if clause_sat_count(assignment, c) == 0:
                count += 1
        return count

    def satisfied(assignment, c):
        for l in c:
            idx = abs(l) - 1
            if assignment[idx] == (l > 0):
                return True
        return False

    # Precompute literal -> (var_idx, is_positive) for speed
    lit_to_var = {}
    for i, c in enumerate(clauses):
        for l in c:
            lit_to_var[l] = (abs(l) - 1, l > 0)

    # Phase 1: Try random restarts with WalkSAT
    rng = random.Random(42)
    max_restarts = 5000
    steps_per_restart = 8000
    noise = 0.25  # probability of random flip

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize assignment
        if restart == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Perturb from best or random
            if best_assignment is not None and rng.random() < 0.7:
                assignment = best_assignment[:]
                # Flip a few random variables
                for _ in range(n // 4 + 1):
                    assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]

        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # Build live unsat set and sat_count per clause
        unsat_set = set()
        sat_counts = [0] * clause_count
        for i, c in enumerate(clauses):
            sc = clause_sat_count(assignment, c)
            sat_counts[i] = sc
            if sc == 0:
                unsat_set.add(i)

        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            if not unsat_set:
                return assignment

            # With some probability, do a random flip
            if rng.random() < noise:
                var = rng.randint(0, n - 1)
                old_val = assignment[var]
                assignment[var] = not old_val

                # Update sat_counts and unsat_set for affected clauses
                for ci in var_pos[var]:
                    c = clauses[ci]
                    if old_val:
                        sat_counts[ci] -= 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] += 1
                for ci in var_neg[var]:
                    c = clauses[ci]
                    if not old_val:
                        sat_counts[ci] -= 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] += 1
                continue

            # Choose a random unsatisfied clause
            clause_idx = list(unsat_set)[rng.randint(0, len(unsat_set) - 1)]
            clause = clauses[clause_idx]

            # Score each variable in the clause: count how many other clauses it would satisfy
            best_var = -1
            best_score = -1
            for l in clause:
                var = abs(l) - 1
                score = 0
                target_val = (l > 0)
                if assignment[var] == target_val:
                    # Flipping would make this literal false - likely bad, skip
                    pass
                else:
                    # Flipping would make this literal true
                    # Count unsat clauses where this is the only unsat literal
                    if l > 0:
                        affected = var_pos[var]
                    else:
                        affected = var_neg[var]
                    for ci in affected:
                        if sat_counts[ci] != 0:
                            continue
                        # Check if this literal is the only unsat one in c2
                        c2 = clauses[ci]
                        only_unsat = True
                        for l2 in c2:
                            if l2 == l:
                                continue
                            idx2 = abs(l2) - 1
                            if assignment[idx2] == (l2 > 0):
                                only_unsat = False
                                break
                        if only_unsat:
                            score += 1

                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var >= 0:
                var = best_var
            else:
                # Random flip within clause
                l = clause[rng.randint(0, len(clause) - 1)]
                var = abs(l) - 1

            old_val = assignment[var]
            assignment[var] = not old_val

            # Update sat_counts and unsat_set for affected clauses
            for ci in var_pos[var]:
                if old_val:
                    sat_counts[ci] -= 1
                    if sat_counts[ci] == 0:
                        unsat_set.add(ci)
                else:
                    if sat_counts[ci] == 0:
                        unsat_set.discard(ci)
                    sat_counts[ci] += 1
            for ci in var_neg[var]:
                if not old_val:
                    sat_counts[ci] -= 1
                    if sat_counts[ci] == 0:
                        unsat_set.add(ci)
                else:
                    if sat_counts[ci] == 0:
                        unsat_set.discard(ci)
                    sat_counts[ci] += 1

            # Periodically check unsat count
            if step % 500 == 0:
                current_unsat = len(unsat_set)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]

    # Phase 2: Local search from best assignment
    if best_assignment is not None:
        assignment = best_assignment[:]
        
        # Build unsat set and sat_counts
        unsat_set = set()
        sat_counts = [0] * clause_count
        for i, c in enumerate(clauses):
            sc = clause_sat_count(assignment, c)
            sat_counts[i] = sc
            if sc == 0:
                unsat_set.add(i)

        for _ in range(20000):
            if time.time() > deadline:
                break
            if not unsat_set:
                return assignment
            
            # Greedy: find variable that fixes most clauses
            best_var = -1
            best_gain = 0
            unsat_list = list(unsat_set)
            for ci in unsat_list[:100]:  # limit to first 100 for speed
                c = clauses[ci]
                for l in c:
                    var = abs(l) - 1
                    if assignment[var] == (l > 0):
                        continue
                    # Count how many unsat clauses this flip fixes
                    gain = 0
                    if l > 0:
                        affected = var_pos[var]
                    else:
                        affected = var_neg[var]
                    for ci2 in affected:
                        if sat_counts[ci2] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
            if best_var >= 0:
                old_val = assignment[best_var]
                assignment[best_var] = not old_val
                
                # Update sat_counts and unsat_set
                for ci in var_pos[best_var]:
                    if old_val:
                        sat_counts[ci] -= 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] += 1
                for ci in var_neg[best_var]:
                    if not old_val:
                        sat_counts[ci] -= 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] += 1
                        
                current_unsat = len(unsat_set)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                if current_unsat == 0:
                    return assignment

    # Phase 3: Final polish - try all single flips if close
    if best_assignment is not None:
        assignment = best_assignment[:]
        unsat_set = set()
        sat_counts = [0] * clause_count
        for i, c in enumerate(clauses):
            sc = clause_sat_count(assignment, c)
            sat_counts[i] = sc
            if sc == 0:
                unsat_set.add(i)
        
        if len(unsat_set) <= 10:
            # Try flipping each variable and see if we fix all
            for var in range(n):
                if time.time() > deadline:
                    break
                old_val = assignment[var]
                assignment[var] = not old_val
                
                # Update
                for ci in var_pos[var]:
                    if old_val:
                        sat_counts[ci] -= 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] += 1
                for ci in var_neg[var]:
                    if not old_val:
                        sat_counts[ci] -= 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] += 1
                
                if not unsat_set:
                    return assignment
                
                # Revert
                assignment[var] = old_val
                for ci in var_pos[var]:
                    if old_val:
                        sat_counts[ci] += 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] -= 1
                for ci in var_neg[var]:
                    if not old_val:
                        sat_counts[ci] += 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] -= 1

    # Return best found or random
    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
