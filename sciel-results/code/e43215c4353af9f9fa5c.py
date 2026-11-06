"""Planted 3-SAT solver (50 vars) — robust WalkSAT with time guard, smart init, restarts, polish."""
import time
import random
import math

# EVOLVE-BLOCK-START


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290.0  # safety margin under 300s per-instance budget

    # Precompute clause literal indices for speed
    clause_vars = [[abs(l) - 1 for l in c] for c in clauses]
    clause_signs = [[(1 if l > 0 else -1) for l in c] for c in clauses]

    def unsat_count(assign):
        cnt = 0
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            for j in range(3):
                if assign[cv[j]] == (cs[j] > 0):
                    cnt += 1
                    break
        return m - cnt

    def is_satisfied(assign):
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            ok = False
            for j in range(3):
                if assign[cv[j]] == (cs[j] > 0):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def polish(assign):
        """Deterministic sweep: flip any variable that strictly reduces unsat count."""
        changed = True
        while changed:
            changed = False
            base = unsat_count(assign)
            for v in range(n):
                assign[v] = not assign[v]
                c = unsat_count(assign)
                if c < base:
                    base = c
                    changed = True
                else:
                    assign[v] = not assign[v]
        return assign

    def walk_sat(assign, rng, flip_budget):
        """Run WalkSAT; return (assign, solved_bool)."""
        for _ in range(flip_budget):
            if time.time() > deadline - 5.0:
                return assign, False
            # find unsat clauses
            unsat_idx = []
            for i in range(m):
                cv = clause_vars[i]
                cs = clause_signs[i]
                ok = False
                for j in range(3):
                    if assign[cv[j]] == (cs[j] > 0):
                        ok = True
                        break
                if not ok:
                    unsat_idx.append(i)
            if not unsat_idx:
                return assign, True
            # pick most-unsatisfied clause (tie-broken randomly)
            # score = number of unsat clauses that contain each literal
            # choose clause with fewest literals that appear in many other unsat clauses
            # Simpler: pick a random unsat clause, then pick literal that satisfies most other unsat clauses
            ci = unsat_idx[rng.randrange(len(unsat_idx))]
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            # For each literal in this clause, count how many other unsat clauses it would satisfy
            best_score = -1
            best_j = -1
            for j in range(3):
                v = cv[j]
                s = cs[j]
                # flipping v to satisfy this literal: new value = (s > 0)
                new_val = (s > 0)
                score = 0
                for ui in unsat_idx:
                    if ui == ci:
                        continue
                    uc = clause_vars[ui]
                    us = clause_signs[ui]
                    for k in range(3):
                        if uc[k] == v:
                            if new_val == (us[k] > 0):
                                score += 1
                            break
                # add small random noise
                score += rng.random() * 0.01
                if score > best_score:
                    best_score = score
                    best_j = j
            v = cv[best_j]
            assign[v] = (clause_signs[ci][best_j] > 0)
        return assign, is_satisfied(assign)

    # --- Initialization: try several seeds, keep best ---
    rng = random.Random(12345)
    best_assign = None
    best_unsat = m + 1

    # Generate candidate initial assignments
    candidates = []
    # random
    for _ in range(8):
        a = [rng.random() < 0.5 for _ in range(n)]
        candidates.append(a)
    # all-true / all-false / alternating
    candidates.append([True] * n)
    candidates.append([False] * n)
    candidates.append([i % 2 == 0 for i in range(n)])

    for a in candidates:
        u = unsat_count(a)
        if u < best_unsat:
            best_unsat = u
            best_assign = list(a)
        if u == 0:
            return a

    # --- Main search: WalkSAT with restarts ---
    current = list(best_assign)
    restarts = 0
    max_restarts = 5000
    while time.time() < deadline - 10.0 and restarts < max_restarts:
        restarts += 1
        # Adaptive flip budget
        flip_budget = 500 + rng.randrange(200)
        res, solved = walk_sat(current, rng, flip_budget)
        u = unsat_count(res)
        if u < best_unsat:
            best_unsat = u
            best_assign = list(res)
        if solved:
            return res
        # Occasionally re-randomize a few variables to escape local minima
        if restarts % 10 == 0:
            for _ in range(3):
                v = rng.randrange(n)
                current[v] = not current[v]
        current = list(res)

    # Final polish on best found
    final = list(best_assign)
    polish(final)
    if is_satisfied(final):
        return final
    # Return best (fewest unsat) — for planted SAT, this is usually the solution or very close
    return final


# EVOLVE-BLOCK-END
