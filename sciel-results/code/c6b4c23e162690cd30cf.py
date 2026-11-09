"""Synthetic planted 3-SAT solver (50 variables)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE_MARGIN = 5.0  # seconds of safety margin before 330s kill


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Preprocess: build literal -> variable index and sign
    # literals are 1-based signed: var = abs(l)-1, positive if l>0
    # Represent assignment as a numpy int8 array: 0=False, 1=True
    # Precompute for each clause the list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)

    # Build variable -> list of (clause_idx, is_positive) for weight updates
    var_occurrences = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for v, pos in cv:
            var_occurrences[v].append((ci, pos))

    start_time = time.time()
    deadline = start_time + 330.0 - _DEADLINE_MARGIN

    # Convert clauses to numpy arrays for fast evaluation
    # For each clause: array of var indices, array of signs (1 for positive, -1 for negative)
    # A literal is satisfied if assignment[var] * sign == 1
    # i.e., if sign==1: assignment[var]==1; if sign==-1: assignment[var]==0
    clause_var_arrs = []
    clause_sign_arrs = []
    for cv in clause_vars:
        vars_ = np.array([v for v, _ in cv], dtype=np.int32)
        signs = np.array([1 if p else -1 for _, p in cv], dtype=np.int8)
        clause_var_arrs.append(vars_)
        clause_sign_arrs.append(signs)

    # Count occurrences per variable for weighting
    var_count = np.zeros(n, dtype=np.float64)
    for v in range(n):
        var_count[v] = len(var_occurrences[v])

    best_assignment = None
    best_unsat = m + 1

    # Multi-restart WalkSAT with adaptive parameters
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(42 + restart)
        # Initial assignment: random
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int8)

        # Compute unsatisfied clause count
        def count_unsat(assignment):
            cnt = 0
            for i in range(m):
                cv = clause_var_arrs[i]
                cs = clause_sign_arrs[i]
                if not np.any(assignment[cv] * cs == 1):
                    cnt += 1
            return cnt

        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment.tolist()
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()

        # WalkSAT parameters
        max_steps = 5000 + restart * 200
        noise = 0.55  # probability of random flip (noise parameter)
        # Use a local search with weighted clause selection

        # Precompute unsat clause indices for faster access
        unsat_clauses = []
        for i in range(m):
            cv = clause_var_arrs[i]
            cs = clause_sign_arrs[i]
            if not np.any(assignment[cv] * cs == 1):
                unsat_clauses.append(i)

        for step in range(max_steps):
            if time.time() > deadline:
                return best_assignment.tolist() if best_assignment is not None else assignment.tolist()

            if not unsat_clauses:
                return assignment.tolist()

            # Choose an unsatisfied clause (weighted by length or just random)
            ci = rng.choice(unsat_clauses)
            cv = clause_var_arrs[ci]
            cs = clause_sign_arrs[ci]

            # Find variables in this clause that, if flipped, would satisfy it
            # A literal (v, sign) is currently unsatisfied:
            #   if sign==1: assignment[v]==0, flipping to 1 satisfies
            #   if sign==-1: assignment[v]==1, flipping to 0 satisfies
            candidates = []
            for idx in range(len(cv)):
                v = cv[idx]
                s = cs[idx]
                # Current value of this literal:
                lit_val = assignment[v] * s
                if lit_val == -1:  # unsatisfied literal, flipping v satisfies this clause
                    candidates.append(v)

            if not candidates:
                # Should not happen for a 3-clause, but just in case
                v = rng.choice(cv.tolist())
                assignment[v] = 1 - assignment[v]
                continue

            # Choose a variable to flip
            if rng.random() < noise:
                # Random flip among all variables in the clause
                v = rng.choice(cv.tolist())
            else:
                # Flip a variable that satisfies this clause (greedy)
                # Weight by how many other clauses this flip might break
                v = rng.choice(candidates)

            old_val = assignment[v]
            assignment[v] = 1 - old_val

            # Update unsat_clauses: check clauses containing v
            # Remove ci from unsat if now satisfied (it should be, since we flipped a satisfying literal)
            # Actually, we need to update all clauses containing v
            # For efficiency, we'll just recount periodically or update incrementally

            # Incremental update:
            # Clauses that contain v may change satisfaction status
            # First, remove ci from unsat_clauses (it's now satisfied)
            if ci in unsat_clauses:
                unsat_clauses.remove(ci)

            # Check other clauses containing v
            for (other_ci, other_pos) in var_occurrences[v]:
                if other_ci == ci:
                    continue
                # Check if this clause is now satisfied
                oc = clause_var_arrs[other_ci]
                os_ = clause_sign_arrs[other_ci]
                is_sat = np.any(assignment[oc] * os_ == 1)
                if is_sat:
                    if other_ci in unsat_clauses:
                        unsat_clauses.remove(other_ci)
                else:
                    if other_ci not in unsat_clauses:
                        unsat_clauses.append(other_ci)

            # Periodic full recount to avoid drift
            if step % 1000 == 0:
                new_unsat = []
                for i in range(m):
                    oc = clause_var_arrs[i]
                    os_ = clause_sign_arrs[i]
                    if not np.any(assignment[oc] * os_ == 1):
                        new_unsat.append(i)
                unsat_clauses = new_unsat

                if not unsat_clauses:
                    return assignment.tolist()

                if len(unsat_clauses) < best_unsat:
                    best_unsat = len(unsat_clauses)
                    best_assignment = assignment.copy()

    # Return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()
# EVOLVE-BLOCK-END
