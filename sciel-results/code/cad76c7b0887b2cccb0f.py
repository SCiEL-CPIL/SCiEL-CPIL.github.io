# EVOLVE-BLOCK-START
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    rng = random.Random(0)

    # --- Bitset representation: one 64-bit int per clause ---
    # For each variable j (0-based), store (clause_idx, polarity) where
    # polarity 0 means the literal is positive (+j), 1 means negative (-j).
    var_pos = [[] for _ in range(n)]   # var -> list of (clause_idx, is_neg)
    for ci, cl in enumerate(clauses):
        for lit in cl:
            j = abs(lit) - 1
            var_pos[j].append((ci, lit < 0))

    # --- Weighted selection: count occurrences of each variable in unsat set ---
    # We maintain per-variable "weight" = number of unsat clauses containing it.
    # Instead of recomputing, we do a fast pass over unsat clauses each step.

    # Initial assignment
    assignment = np.zeros(n, dtype=bool)
    assignment[:] = rng.random(n) < 0.5

    # Compute initial unsat count and per-variable weights
    # sat_count[ci] = number of satisfied literals in clause ci
    sat_count = np.zeros(m, dtype=np.int32)
    for j in range(n):
        val = assignment[j]
        for ci, is_neg in var_pos[j]:
            # literal satisfied if (val and not is_neg) or (not val and is_neg)
            if (val and not is_neg) or ((not val) and is_neg):
                sat_count[ci] += 1

    def is_unsat(ci):
        return sat_count[ci] == 0

    # Build list of unsat clause indices
    unsat_indices = [ci for ci in range(m) if sat_count[ci] == 0]

    # If already satisfied
    if not unsat_indices:
        return assignment.tolist()

    # Precompute for each variable the list of (clause_idx, is_neg) for fast flip
    # and for weighted selection we need per-variable weight in unsat set.

    # Weighted flip: pick a variable proportional to its count in unsat clauses.
    # We'll compute weights by scanning unsat clauses.

    for step in range(200000):
        if not unsat_indices:
            break

        # Compute per-variable weight = number of unsat clauses containing it
        # Only for variables that appear in unsat clauses.
        # For efficiency, use a dict.
        var_weight = {}
        for ci in unsat_indices:
            for lit in clauses[ci]:
                j = abs(lit) - 1
                var_weight[j] = var_weight.get(j, 0) + 1

        # Build weighted choice list
        if not var_weight:
            break
        vars_list = list(var_weight.keys())
        weights = [var_weight[v] for v in vars_list]
        total_w = sum(weights)

        # Pick variable with probability proportional to weight
        r = rng.random() * total_w
        acc = 0
        chosen_var = vars_list[-1]
        for v, w in zip(vars_list, weights):
            acc += w
            if r <= acc:
                chosen_var = v
                break

        # Flip chosen variable
        old_val = assignment[chosen_var]
        new_val = not old_val
        assignment[chosen_var] = new_val

        # Update sat_count for all clauses containing chosen_var
        # Determine which clauses become sat/unsat
        new_unsat_indices = []
        for ci, is_neg in var_pos[chosen_var]:
            # Before flip: contribution of this literal to sat_count[ci]
            # old contribution: (old_val and not is_neg) or ((not old_val) and is_neg)
            old_contrib = (old_val and not is_neg) or ((not old_val) and is_neg)
            # new contribution
            new_contrib = (new_val and not is_neg) or ((not new_val) and is_neg)
            if old_contrib and not new_contrib:
                # literal was satisfied, now not -> sat_count decreases
                sat_count[ci] -= 1
            elif (not old_contrib) and new_contrib:
                # literal was not satisfied, now satisfied -> sat_count increases
                sat_count[ci] += 1
            # else no change

        # Rebuild unsat_indices
        unsat_indices = [ci for ci in range(m) if sat_count[ci] == 0]

    return assignment.tolist()
# EVOLVE-BLOCK-END
