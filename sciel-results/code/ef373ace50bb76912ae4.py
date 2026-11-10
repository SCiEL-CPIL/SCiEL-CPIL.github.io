"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 300s kill

    # Precompute clause structure for speed
    m = len(clauses)
    # For each clause, store (var_indices, signs) for fast evaluation
    # var_indices: list of 0-based variable indices
    # signs: list of booleans (True = positive literal)
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_ = [abs(l) - 1 for l in c]
        signs = [l > 0 for l in c]
        clause_vars.append(vars_)
        clause_signs.append(signs)

    def check_assignment(assignment):
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[cv[j]] == cs[j]:
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsatisfied(assignment):
        count = 0
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[cv[j]] == cs[j]:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    # Try several initializations
    initializations = []
    # All true
    initializations.append([True] * n)
    # All false
    initializations.append([False] * n)
    # Random
    rng = random.Random(0)
    initializations.append([rng.random() < 0.5 for _ in range(n)])
    # Literal frequency based: if a literal appears more often, set it true
    lit_count = [0] * n
    for c in clauses:
        for l in c:
            lit_count[abs(l) - 1] += 1
    # Set variable true if positive literal count > negative, else false
    # Actually, let's just use: for each var, count pos and neg occurrences
    pos_count = [0] * n
    neg_count = [0] * n
    for c in clauses:
        for l in c:
            if l > 0:
                pos_count[l - 1] += 1
            else:
                neg_count[-l - 1] += 1
    freq_init = []
    for i in range(n):
        freq_init.append(pos_count[i] >= neg_count[i])
    initializations.append(freq_init)

    best_assignment = None
    best_unsat = m + 1

    def walk_sat(assignment, max_steps, seed_offset=0):
        nonlocal best_assignment, best_unsat
        rng_ws = random.Random(seed_offset)
        curr = assignment[:]
        # Clause weights
        weights = [1.0] * m
        curr_unsat = count_unsatisfied(curr)
        if curr_unsat < best_unsat:
            best_unsat = curr_unsat
            best_assignment = curr[:]
        if curr_unsat == 0:
            return curr

        noise = 0.55  # probability of random flip
        for step in range(max_steps):
            if time.time() > deadline:
                return curr
            # Find unsatisfied clauses
            unsat_indices = []
            for i in range(m):
                cv = clause_vars[i]
                cs = clause_signs[i]
                sat = False
                for j in range(3):
                    if curr[cv[j]] == cs[j]:
                        sat = True
                        break
                if not sat:
                    unsat_indices.append(i)
            if not unsat_indices:
                return curr

            # Pick an unsatisfied clause (weighted)
            total_w = sum(weights[i] for i in unsat_indices)
            r = rng_ws.random() * total_w
            acc = 0.0
            chosen_clause = unsat_indices[0]
            for idx in unsat_indices:
                acc += weights[idx]
                if r <= acc:
                    chosen_clause = idx
                    break

            # Increment weight of chosen clause
            weights[chosen_clause] += 1.0

            # Select variable to flip
            cv = clause_vars[chosen_clause]
            cs = clause_signs[chosen_clause]
            if rng_ws.random() < noise:
                # Random variable from the clause
                v = rng_ws.choice(cv)
            else:
                # Weighted by how many unsatisfied clauses the variable appears in
                # Count unsatisfied clauses containing each variable in this clause
                var_scores = {}
                for v in cv:
                    score = 0.0
                    for ui in unsat_indices:
                        if v in clause_vars[ui]:
                            score += weights[ui]
                    var_scores[v] = score
                # Pick variable with highest score (with some randomness)
                max_score = max(var_scores.values())
                candidates = [v for v in cv if var_scores[v] == max_score]
                if len(candidates) > 1:
                    v = rng_ws.choice(candidates)
                else:
                    v = candidates[0]

            # Decide new value: try to satisfy the clause
            # Current value of v
            old_val = curr[v]
            # What value would satisfy this clause for literal v?
            # Find the sign of v in this clause
            v_pos_in_clause = None
            for j in range(3):
                if cv[j] == v:
                    v_pos_in_clause = cs[j]
                    break
            # If we set curr[v] = v_pos_in_clause, the literal is satisfied
            # But flipping might break other clauses
            # WalkSAT: flip to satisfy this clause (i.e., set to v_pos_in_clause)
            # But that's just setting, not necessarily flipping
            # Standard WalkSAT: flip the variable (toggle)
            # But better: set to the value that satisfies this clause
            # Let's do: with prob 0.5, set to satisfy; else flip
            if rng_ws.random() < 0.5:
                new_val = v_pos_in_clause
            else:
                new_val = not old_val
            curr[v] = new_val

            curr_unsat = count_unsatisfied(curr)
            if curr_unsat < best_unsat:
                best_unsat = curr_unsat
                best_assignment = curr[:]
            if curr_unsat == 0:
                return curr

            # Decay weights slightly to prevent stagnation
            if step % 500 == 0:
                for i in range(m):
                    weights[i] = 1.0 + 0.5 * (weights[i] - 1.0)

        return curr

    # Try each initialization with increasing step budgets
    step_budgets = [2000, 5000, 10000, 20000, 40000, 80000]
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline:
            break
        for step_idx, budget in enumerate(step_budgets):
            if time.time() > deadline:
                break
            result = walk_sat(init, budget, seed_offset=init_idx * 1000 + step_idx)
            if best_unsat == 0:
                return best_assignment

    # If we still haven't found a solution, return the best we have
    if best_assignment is not None:
        return best_assignment
    # Fallback: return a random assignment
    rng_final = random.Random(12345)
    return [rng_final.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
