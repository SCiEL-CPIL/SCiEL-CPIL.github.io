"""3-SAT solver with hybrid WalkSAT + restarts + deadline guard."""
import random
import time


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

    def count_unsat(assignment):
        count = 0
        for c in clauses:
            sat = False
            for l in c:
                idx = abs(l) - 1
                if assignment[idx] == (l > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def satisfied(assignment, c):
        for l in c:
            idx = abs(l) - 1
            if assignment[idx] == (l > 0):
                return True
        return False

    # Phase 1: Try random restarts with WalkSAT using incremental clause tracking
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

        # Build sat_count for each clause and track unsat clauses
        sat_count = [0] * clause_count
        unsat_clauses = []
        for i, c in enumerate(clauses):
            cnt = 0
            for l in c:
                idx = abs(l) - 1
                if assignment[idx] == (l > 0):
                    cnt += 1
            sat_count[i] = cnt
            if cnt == 0:
                unsat_clauses.append(i)

        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            if not unsat_clauses:
                return assignment

            # With some probability, do a random flip
            if rng.random() < noise:
                var = rng.randint(0, n - 1)
                old_val = assignment[var]
                assignment[var] = not old_val

                # Update sat_count and unsat_clauses for affected clauses
                new_unsat = []
                for ci in var_pos[var]:
                    old_sat = sat_count[ci]
                    if old_val:
                        # Flipped from True to False: this literal was satisfying, now it's not
                        new_sat = old_sat - 1
                    else:
                        # Flipped from False to True: this literal was not satisfying, now it is
                        new_sat = old_sat + 1
                    sat_count[ci] = new_sat
                    if new_sat == 0:
                        new_unsat.append(ci)
                
                for ci in var_neg[var]:
                    old_sat = sat_count[ci]
                    if old_val:
                        # Flipped from True to False: this literal (neg) was not satisfying, now it is
                        new_sat = old_sat + 1
                    else:
                        # Flipped from False to True: this literal (neg) was satisfying, now it's not
                        new_sat = old_sat - 1
                    sat_count[ci] = new_sat
                    if new_sat == 0:
                        new_unsat.append(ci)
                
                # Rebuild unsat_clauses: keep old ones that are still unsat, add new ones
                # We need to be careful: some old unsat clauses might have become sat
                new_unsat_set = set(new_unsat)
                updated_unsat = []
                for ci in unsat_clauses:
                    if sat_count[ci] == 0:
                        updated_unsat.append(ci)
                    elif ci not in new_unsat_set:
                        # This clause was unsat but might have been affected
                        pass
                # Add newly unsat clauses
                for ci in new_unsat:
                    if ci not in updated_unsat:
                        updated_unsat.append(ci)
                unsat_clauses = updated_unsat
                continue

            # Choose a random unsatisfied clause
            clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            clause = clauses[clause_idx]

            # Score each variable in the clause: count how many other clauses it would satisfy
            best_var = -1
            best_score = -1
            for l in clause:
                var = abs(l) - 1
                score = 0
                # Flipping this variable: if currently False (for pos) or True (for neg), it becomes True
                target_val = (l > 0)
                if assignment[var] != target_val:
                    # Flipping would make this literal true
                    # Count unsat clauses where this is the only unsat literal
                    affected = var_pos[var] if l > 0 else var_neg[var]
                    for ci in affected:
                        if sat_count[ci] == 0:
                            # This clause is unsat. Check if this literal is the only unsat one
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
                var_to_flip = best_var
            else:
                # Random flip within clause
                l = clause[rng.randint(0, len(clause) - 1)]
                var_to_flip = abs(l) - 1

            # Apply the flip
            old_val = assignment[var_to_flip]
            assignment[var_to_flip] = not old_val

            # Update sat_count and unsat_clauses for affected clauses
            new_unsat = []
            for ci in var_pos[var_to_flip]:
                old_sat = sat_count[ci]
                if old_val:
                    new_sat = old_sat - 1
                else:
                    new_sat = old_sat + 1
                sat_count[ci] = new_sat
                if new_sat == 0:
                    new_unsat.append(ci)
            
            for ci in var_neg[var_to_flip]:
                old_sat = sat_count[ci]
                if old_val:
                    new_sat = old_sat + 1
                else:
                    new_sat = old_sat - 1
                sat_count[ci] = new_sat
                if new_sat == 0:
                    new_unsat.append(ci)
            
            # Rebuild unsat_clauses
            new_unsat_set = set(new_unsat)
            updated_unsat = []
            for ci in unsat_clauses:
                if sat_count[ci] == 0:
                    updated_unsat.append(ci)
            for ci in new_unsat:
                if ci not in updated_unsat:
                    updated_unsat.append(ci)
            unsat_clauses = updated_unsat

            # Periodically check unsat count
            if step % 500 == 0:
                current_unsat = len(unsat_clauses)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]

    # Phase 2: Local search from best assignment
    if best_assignment is not None:
        assignment = best_assignment[:]
        for _ in range(20000):
            if time.time() > deadline:
                break
            unsat_clauses = [c for c in clauses if not satisfied(assignment, c)]
            if not unsat_clauses:
                return assignment
            # Greedy: find variable that fixes most clauses
            best_var = -1
            best_gain = 0
            for c in unsat_clauses[:100]:  # limit to first 100 for speed
                for l in c:
                    var = abs(l) - 1
                    if assignment[var] == (l > 0):
                        continue
                    # Count how many unsat clauses this flip fixes
                    gain = 0
                    for c2 in unsat_clauses[:100]:
                        for l2 in c2:
                            if abs(l2) - 1 == var and assignment[var] != (l2 > 0):
                                gain += 1
                                break
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
                current_unsat = count_unsat(assignment)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                if current_unsat == 0:
                    return assignment

    # Return best found or random
    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
