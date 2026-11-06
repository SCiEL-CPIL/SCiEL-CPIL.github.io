"""Synthetic planted 3-SAT solver (150 vars). Incremental WalkSAT with restarts."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 275.0  # safety margin under 300s

    rng = random.Random(0)
    if m == 0:
        return [True] * n

    # Precompute per-variable clause lists (1-based literal -> 0-based var).
    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for l in cl:
            var_clauses[abs(l) - 1].append(ci)

    # Track which literal in each clause is satisfied, so a flip is O(1).
    # sat_idx[c] = position in clauses[c] that is currently satisfied, else -1.
    # For each clause we also store the literal signs as booleans (l>0).
    lit_pos = [l > 0 for cl in clauses for l in cl]  # flattened
    # Build flattened arrays for speed
    flat_var = [abs(l) - 1 for cl in clauses for l in cl]
    flat_sign = [l > 0 for cl in clauses for l in cl]
    clause_start = [0]
    for cl in clauses:
        clause_start.append(clause_start[-1] + len(cl))

    best_assign = None
    best_unsat = m + 1

    # Restart loop with adaptive step budget.
    restart = 0
    base_steps = 3000
    while True:
        # Check deadline before starting a restart.
        if time.time() > deadline:
            break

        # Random initial assignment.
        assign = [rng.random() < 0.5 for _ in range(n)]

        # Compute satisfied literal per clause.
        sat_idx = [-1] * m
        unsat = 0
        for c in range(m):
            s = clause_start[c]
            e = clause_start[c + 1]
            found = -1
            for k in range(s, e):
                if assign[flat_var[k]] == flat_sign[k]:
                    found = k
                    break
            sat_idx[c] = found
            if found == -1:
                unsat += 1
        if unsat == 0:
            return assign
        if unsat < best_unsat:
            best_unsat = unsat
            best_assign = assign[:]

        # Adaptive step budget: shrink as restarts accumulate.
        steps = max(400, base_steps // (1 + restart // 3))
        restart += 1

        # WalkSAT steps with incremental tracking.
        for _ in range(steps):
            if unsat == 0:
                return assign
            if time.time() > deadline:
                if unsat < best_unsat:
                    best_unsat = unsat
                    best_assign = assign[:]
                break

            # Choose an unsatisfied clause.
            c = rng.randrange(m)
            if sat_idx[c] != -1:
                # resample a few times for an unsat clause
                for _t in range(8):
                    c = rng.randrange(m)
                    if sat_idx[c] == -1:
                        break

            s = clause_start[c]
            e = clause_start[c + 1]

            # Damage-minimizing flip: for each literal in the clause, compute
            # how many currently-satisfied clauses it would break (damage).
            # Prefer the literal with least damage; random tie-break.
            best_var = -1
            best_damage = None
            best_rand = None
            for k in range(s, e):
                v = flat_var[k]
                # damage: number of satisfied clauses containing var v that
                # would become unsat if we flip v.
                damage = 0
                for cj in var_clauses[v]:
                    si = sat_idx[cj]
                    if si != -1:
                        # the satisfying literal is at si; check if it is var v
                        if flat_var[si] == v:
                            damage += 1
                # We want to flip v to satisfy literal k. Flipping v to sign
                # flat_sign[k]. After flip, clause c becomes satisfied.
                # damage counts clauses that lose their only satisfying literal.
                if best_damage is None or damage < best_damage or (
                    damage == best_damage and rng.random() < 0.5
                ):
                    best_damage = damage
                    best_var = v
                    best_rand = flat_sign[k]

            # WalkSAT random component: sometimes pick a random literal in clause.
            if rng.random() < 0.3:
                k = rng.randrange(s, e)
                best_var = flat_var[k]
                best_rand = flat_sign[k]

            # Flip best_var to satisfy literal (set to best_rand).
            v = best_var
            newval = best_rand
            if assign[v] == newval:
                continue
            assign[v] = newval

            # Update clauses containing v.
            for cj in var_clauses[v]:
                si = sat_idx[cj]
                if si != -1:
                    # was satisfied; check if the satisfying literal is v
                    if flat_var[si] == v:
                        # it may still be satisfied by another literal
                        s2 = clause_start[cj]
                        e2 = clause_start[cj + 1]
                        found = -1
                        for kk in range(s2, e2):
                            if assign[flat_var[kk]] == flat_sign[kk]:
                                found = kk
                                break
                        if found == -1:
                            sat_idx[cj] = -1
                            unsat += 1
                        else:
                            sat_idx[cj] = found
                else:
                    # was unsat; check if now satisfied
                    s2 = clause_start[cj]
                    e2 = clause_start[cj + 1]
                    found = -1
                    for kk in range(s2, e2):
                        if assign[flat_var[kk]] == flat_sign[kk]:
                            found = kk
                            break
                    if found != -1:
                        sat_idx[cj] = found
                        unsat -= 1

        if unsat == 0:
            return assign
        if unsat < best_unsat:
            best_unsat = unsat
            best_assign = assign[:]

    # Deadline reached: return best found (or last assignment).
    if best_assign is not None:
        return best_assign
    return assign
# EVOLVE-BLOCK-END
