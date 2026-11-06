"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Trivial cases
    if m == 0:
        return [True] * n

    # Deadline guard: kill well before the 330s wall-clock limit.
    start_time = time.time()
    deadline = start_time + 320.0

    # ---- Preprocess into vectorized arrays ---------------------------------
    # vars_arr: (m, 3) 0-based variable indices
    # signs_arr: (m, 3) +1 if positive literal, -1 if negative
    vars_arr = np.empty((m, 3), dtype=np.int32)
    signs_arr = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            if lit > 0:
                vars_arr[i, j] = lit - 1
                signs_arr[i, j] = 1
            else:
                vars_arr[i, j] = -lit - 1
                signs_arr[i, j] = -1

    # Clause index of every variable occurrence (var -> list of (clause, pos))
    var_occ = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_occ[vars_arr[i, j]].append((i, j))

    # ---- Core vectorized evaluation -----------------------------------------
    # state: (n,) +1 / -1
    # sat_mask: (m,) bool, True if clause satisfied
    # unsat_count: (m,) int, number of unsatisfied literals in clause (0..3)
    def evaluate(state):
        vals = state[vars_arr] * signs_arr          # (m,3)
        sat_mask = (vals > 0).any(axis=1)           # (m,)
        unsat_count = (vals <= 0).sum(axis=1)       # (m,)
        return sat_mask, unsat_count

    # ---- Restart-based WalkSAT ----------------------------------------------
    rng = np.random.default_rng(0)

    # Best assignment seen so far (lowest unsat count)
    best_state = None
    best_unsat = m + 1

    # Number of restarts allowed within the time budget
    max_restarts = 5000

    for restart in range(max_restarts):
        # Random initial assignment
        state = rng.choice([1, -1], size=n).astype(np.int8)
        sat_mask, unsat_count = evaluate(state)
        total_unsat = int(unsat_count.sum())

        # Track best
        if total_unsat < best_unsat:
            best_unsat = total_unsat
            best_state = state.copy()
            if total_unsat == 0:
                return (state > 0).tolist()

        # If we already have a solution, return it
        if best_unsat == 0:
            return (best_state > 0).tolist()

        # Adaptive flip budget: shorter for later restarts
        # Base budget ~ 3*n, grows a bit for early restarts
        base_budget = 40 * n
        budget = base_budget + (max_restarts - restart) * 2

        # WalkSAT loop
        for step in range(budget):
            if total_unsat == 0:
                return (state > 0).tolist()

            # Check deadline periodically (every 256 steps) to avoid
            # the overhead of a time check every step
            if (step & 255) == 0 and time.time() > deadline:
                # Return best found
                if best_state is not None:
                    return (best_state > 0).tolist()
                return (state > 0).tolist()

            # Pick an unsatisfied clause: prefer the one with fewest unsat literals
            unsat_indices = np.where(~sat_mask)[0]
            if len(unsat_indices) == 0:
                return (state > 0).tolist()

            # Choose clause: with some probability pick the most constrained
            # (fewest unsat literals), else random among unsat
            if rng.random() < 0.5:
                # Most constrained: min unsat_count among unsat
                local_unsat = unsat_count[unsat_indices]
                min_val = local_unsat.min()
                candidates = unsat_indices[local_unsat == min_val]
                clause_idx = int(rng.choice(candidates))
            else:
                clause_idx = int(rng.choice(unsat_indices))

            # Score-guided flip: for each literal in the clause, compute gain
            # gain = (# clauses that become satisfied) - (# clauses that become unsatisfied)
            # We flip the variable if it currently makes its literal unsatisfied
            var = int(vars_arr[clause_idx, 0])  # we'll iterate over all 3

            # Evaluate all 3 literals in the clause
            best_gain = -10**9
            best_var = -1
            for pos in range(3):
                v = int(vars_arr[clause_idx, pos])
                # Current contribution of this literal: is it satisfied?
                # If the literal is already satisfied, flipping would hurt
                # (we'd make it unsat), so skip unless all are satisfied (impossible
                # since clause is unsat)
                lit_val = state[v] * signs_arr[clause_idx, pos]
                if lit_val > 0:
                    # This literal is satisfied; flipping makes it unsat.
                    # Only consider if we must (but in an unsat clause at least
                    # one literal is unsat, so prefer those)
                    continue

                # v is currently making this literal unsat. Flipping v will
                # satisfy this clause. Compute net gain across all clauses
                # containing v.
                # Before flip: for each occurrence (ci, pj), the literal value is
                #   state[v] * signs_arr[ci, pj]
                # After flip: sign flips.
                # A clause transitions from unsat->sat if its only unsat literal
                # is this one. And sat->unsat if its only sat literal is this one.

                occ = var_occ[v]
                if not occ:
                    gain = 1  # just this clause
                else:
                    occ_clauses = np.array([oc[0] for oc in occ], dtype=np.int32)
                    occ_pos = np.array([oc[1] for oc in occ], dtype=np.int32)
                    # Current literal values for these occurrences
                    cur_vals = state[v] * signs_arr[occ_clauses, occ_pos]  # (k,)
                    # Clauses currently unsat where this literal is the only unsat
                    # (i.e., clause unsat and this literal unsat)
                    # unsat_count[ci] == 1 and cur_vals == -1 (literal unsat)
                    become_sat = int(((unsat_count[occ_clauses] == 1) & (cur_vals < 0)).sum())
                    # Clauses currently sat where this literal is the only sat
                    # (i.e., clause sat and this literal sat)
                    # sat_mask[ci] and cur_vals > 0 and (unsat_count[ci] == 2)
                    # because if clause is sat and only one literal is sat, unsat_count==2
                    become_unsat = int(((unsat_count[occ_clauses] == 2) & (cur_vals > 0)).sum())
                    gain = become_sat - become_unsat

                if gain > best_gain:
                    best_gain = gain
                    best_var = v

            # Fallback: if no unsat literal found (shouldn't happen), flip first
            if best_var < 0:
                best_var = int(vars_arr[clause_idx, 0])

            # Flip best_var
            state[best_var] = -state[best_var]

            # Update sat_mask and unsat_count for clauses containing best_var
            # Only recompute for those clauses
            occ = var_occ[best_var]
            if occ:
                occ_clauses = np.array([oc[0] for oc in occ], dtype=np.int32)
                occ_pos = np.array([oc[1] for oc in occ], dtype=np.int32)
                # Recompute values for these clauses
                new_vals = state[vars_arr[occ_clauses]] * signs_arr[occ_clauses]  # (k,3)
                new_sat = (new_vals > 0).any(axis=1)
                new_unsat = (new_vals <= 0).sum(axis=1)
                sat_mask[occ_clauses] = new_sat
                unsat_count[occ_clauses] = new_unsat
                total_unsat = int(unsat_count.sum())

            # Track best
            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_state = state.copy()
                if total_unsat == 0:
                    return (state > 0).tolist()

    # Deadline exceeded or restarts exhausted: return best found
    if best_state is not None:
        return (best_state > 0).tolist()
    return (state > 0).tolist()
# EVOLVE-BLOCK-END
