"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]

    # Deadline guard (explicit)
    start_time = time.time()
    timeout = 320.0  # Safety margin well before the 330s kill

    # ---- Precompute clause structure (0-based) ----
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vs, ss = [], []
        for lit in clause:
            vs.append(abs(lit) - 1)
            ss.append(lit > 0)
        clause_vars.append(vs)
        clause_signs.append(ss)

    m = len(clauses)

    # Variable -> list of (clause_index, position_in_clause) for O(1) updates
    var_to_clauses = [[] for _ in range(n)]
    for i, vs in enumerate(clause_vars):
        for j, v in enumerate(vs):
            var_to_clauses[v].append((i, j))

    # ---- State helpers ----
    def init_state(assignment):
        csc = np.zeros(m, dtype=np.int32)
        for i in range(m):
            c = 0
            vs = clause_vars[i]
            ss = clause_signs[i]
            for j in range(3):
                if assignment[vs[j]] == ss[j]:
                    c += 1
            csc[i] = c
        return csc

    def flip_var(v, assignment, csc):
        assignment[v] = not assignment[v]
        nv = assignment[v]
        for ci, j in var_to_clauses[v]:
            if nv == clause_signs[ci][j]:
                csc[ci] += 1
            else:
                csc[ci] -= 1

    def polish(assignment, csc):
        """Final high-precision polish: greedily flip any var that strictly
        reduces the unsat count until no such move remains (local optimum)."""
        deadline = timeout - 3.0
        changed = True
        while changed and (time.time() - start_time) < deadline:
            changed = False
            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                break
            # Variables in at least one unsat clause
            cand = set()
            for ci in unsat:
                cand.update(clause_vars[ci])
            base = len(unsat)
            for v in cand:
                # delta = (new_unsat - base); flip if it strictly helps
                delta = 0
                for ci, j in var_to_clauses[v]:
                    if csc[ci] == 0:
                        if assignment[v] == clause_signs[ci][j]:
                            delta -= 1  # currently satisfied -> becomes unsat
                        else:
                            delta += 1  # currently unsat -> becomes satisfied
                    else:
                        if assignment[v] == clause_signs[ci][j]:
                            delta += 1  # satisfied -> becomes unsat (bad)
                        else:
                            delta -= 1  # unsat -> becomes satisfied (good)
                if delta > 0:
                    flip_var(v, assignment, csc)
                    base += delta
                    changed = True
            if len(np.where(csc == 0)[0]) == 0:
                break
        return len(np.where(csc == 0)[0])

    # ---- Main WalkSAT search with restarts + adaptive noise + backtracking ----
    best_assignment = None
    best_unsat = m + 1

    rng = random.Random(42)
    max_flips_per_restart = 10000
    noise = 0.25
    backtrack_prob = 0.05

    while (time.time() - start_time) < timeout - 5.0:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        csc = init_state(assignment)

        current_unsat = int(np.sum(csc == 0))
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        for _ in range(max_flips_per_restart):
            if (time.time() - start_time) > timeout - 5.0:
                break

            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return assignment

            ci = int(rng.choice(unsat))
            vs = clause_vars[ci]

            if rng.random() < noise:
                v = vs[rng.randint(0, 2)]
            elif rng.random() < backtrack_prob:
                # Deliberate worsen: pick the var that hurts the least
                worst_var = -1
                worst_score = 10**9
                for j, v in enumerate(vs):
                    score = 0
                    for oci, _ in var_to_clauses[v]:
                        if oci != ci and csc[oci] == 0:
                            score += 1
                    if score < worst_score:
                        worst_score = score
                        worst_var = v
                v = worst_var
            else:
                # Greedy: satisfy most other unsat clauses (plus tiny jitter)
                best_var = -1
                best_score = -1
                for j, v in enumerate(vs):
                    score = 0
                    for oci, _ in var_to_clauses[v]:
                        if oci != ci and csc[oci] == 0:
                            score += 1
                    score += rng.random() * 0.1
                    if score > best_score:
                        best_score = score
                        best_var = v
                v = best_var

            if v >= 0:
                flip_var(v, assignment, csc)

            current_unsat = int(np.sum(csc == 0))
            if current_unsat == 0:
                return assignment
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()

            # Adaptive noise: widen if stuck, tighten if improving
            if current_unsat > best_unsat:
                noise = min(0.5, noise + 0.05)
            else:
                noise = max(0.1, noise - 0.02)

    # Final polish on the best assignment found
    if best_assignment is not None:
        csc = init_state(best_assignment)
        polish(best_assignment, csc)
        if int(np.sum(csc == 0)) == 0:
            return best_assignment
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
