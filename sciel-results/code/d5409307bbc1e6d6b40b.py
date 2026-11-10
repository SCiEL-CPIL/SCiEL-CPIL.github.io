"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # safety margin before 330s kill

    # Pre-compute clause structure for fast access
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_in_c = [abs(l) - 1 for l in c]
        signs = [l > 0 for l in c]
        clause_vars.append(vars_in_c)
        clause_signs.append(signs)

    m = len(clauses)

    # Build variable -> clause index map for efficient delta computation
    var_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        seen = set()
        for l in c:
            v = abs(l) - 1
            if v not in seen:
                var_clauses[v].append(ci)
                seen.add(v)

    def is_satisfied(assignment, clause_idx):
        idxs = clause_vars[clause_idx]
        signs = clause_signs[clause_idx]
        for i, s in zip(idxs, signs):
            if assignment[i] == s:
                return True
        return False

    def count_unsat_fast(assignment):
        count = 0
        for ci in range(m):
            idxs = clause_vars[ci]
            signs = clause_signs[ci]
            sat = False
            for i, s in zip(idxs, signs):
                if assignment[i] == s:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def get_unsat_indices(assignment):
        unsat = []
        for ci in range(m):
            idxs = clause_vars[ci]
            signs = clause_signs[ci]
            sat = False
            for i, s in zip(idxs, signs):
                if assignment[i] == s:
                    sat = True
                    break
            if not sat:
                unsat.append(ci)
        return unsat

    # Build conflict graph for spectral initialization
    A = np.zeros((n, n))
    for c in clauses:
        if len(c) < 2:
            continue
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                li, lj = c[i], c[j]
                ui, uj = abs(li) - 1, abs(lj) - 1
                if ui == uj:
                    continue
                if (li > 0) == (lj > 0):
                    A[ui][uj] += 1.0
                    A[uj][ui] += 1.0
                else:
                    A[ui][uj] -= 1.0
                    A[uj][ui] -= 1.0

    # Build Laplacian
    D = np.diag(np.sum(A, axis=1))
    L = D - A

    # Compute Fiedler vector
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
    except Exception:
        fiedler = np.zeros(n)

    spectral_assignment = [fiedler[i] > 0 for i in range(n)]

    # Prepare diverse initial assignments
    spectral_assignments = []
    spectral_assignments.append(list(spectral_assignment))
    spectral_assignments.append([not x for x in spectral_assignment])

    for _ in range(4):
        perturbed = list(spectral_assignment)
        for i in range(n):
            if random.random() < 0.1:
                perturbed[i] = not perturbed[i]
        spectral_assignments.append(perturbed)

    def walksat_efficient(assignment, max_steps, p=0.5):
        current_unsat = count_unsat_fast(assignment)
        if current_unsat == 0:
            return True, assignment

        unsat_indices = get_unsat_indices(assignment)

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if current_unsat == 0:
                return True, assignment

            # Pick a random unsat clause
            if not unsat_indices:
                unsat_indices = get_unsat_indices(assignment)
                if not unsat_indices:
                    return True, assignment

            ci = random.choice(unsat_indices)
            clause = clauses[ci]
            vars_in_clause = clause_vars[ci]

            # Compute deltas for each variable in the clause
            min_delta = float('inf')
            best_vars = []

            for v in vars_in_clause:
                old_val = assignment[v]
                assignment[v] = not old_val
                new_unsat = count_unsat_fast(assignment)
                assignment[v] = old_val

                delta = new_unsat - current_unsat

                if delta < min_delta:
                    min_delta = delta
                    best_vars = [v]
                elif delta == min_delta:
                    best_vars.append(v)

            if random.random() < p:
                v_to_flip = random.choice(vars_in_clause)
            else:
                v_to_flip = random.choice(best_vars)

            assignment[v_to_flip] = not assignment[v_to_flip]
            current_unsat = count_unsat_fast(assignment)

            # Update unsat indices incrementally
            if current_unsat > 0:
                unsat_indices = get_unsat_indices(assignment)

        return False, assignment

    def min_conflicts_polish(assignment, max_steps):
        """Min-conflicts style: flip variable appearing in most unsat clauses."""
        for step in range(max_steps):
            if time.time() > deadline:
                break

            unsat_indices = get_unsat_indices(assignment)
            if not unsat_indices:
                return True, assignment

            # Count how many unsat clauses each variable appears in
            var_count = [0] * n
            for ci in unsat_indices:
                for v in clause_vars[ci]:
                    var_count[v] += 1

            max_count = max(var_count)
            if max_count == 0:
                break

            candidates = [i for i in range(n) if var_count[i] == max_count]
            v_to_flip = random.choice(candidates)
            assignment[v_to_flip] = not assignment[v_to_flip]

        final_unsat = count_unsat_fast(assignment)
        if final_unsat == 0:
            return True, assignment
        return False, assignment

    # Main loop with restarts
    start_time = time.time()

    restart = 0
    best_unsat_count = float('inf')
    best_assignment = None

    while time.time() < deadline:
        restart += 1

        # Adaptive step budget: grow with restarts but cap
        steps = min(500 * (2 ** (min(restart - 1, 10))), 50000)

        # Choose initial assignment
        if restart <= len(spectral_assignments):
            assignment = list(spectral_assignments[restart - 1])
        elif restart <= len(spectral_assignments) + 5:
            assignment = list(spectral_assignment)
            for i in range(n):
                if random.random() < 0.15:
                    assignment[i] = not assignment[i]
        else:
            assignment = [random.random() < 0.5 for _ in range(n)]

        init_unsat = count_unsat_fast(assignment)
        if init_unsat < best_unsat_count:
            best_unsat_count = init_unsat
            best_assignment = list(assignment)

        if init_unsat == 0:
            return assignment

        # Phase 1: WalkSAT
        solved, final_assignment = walksat_efficient(assignment, steps, p=0.5)

        if solved:
            return final_assignment

        # Phase 2: Min-conflicts polish on the result
        solved2, final_assignment = min_conflicts_polish(final_assignment, 2000)

        if solved2:
            return final_assignment

        final_unsat = count_unsat_fast(final_assignment)
        if final_unsat < best_unsat_count:
            best_unsat_count = final_unsat
            best_assignment = list(final_assignment)

        # If we're close, try a short WalkSAT from the best so far
        if best_unsat_count <= 3 and time.time() < deadline:
            polished, final_assignment = walksat_efficient(list(best_assignment), 3000, p=0.3)
            if polished:
                return final_assignment
            final_unsat = count_unsat_fast(final_assignment)
            if final_unsat < best_unsat_count:
                best_unsat_count = final_unsat
                best_assignment = list(final_assignment)

    if best_assignment is not None:
        # Final verification
        if count_unsat_fast(best_assignment) == 0:
            return best_assignment
        return best_assignment

    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
