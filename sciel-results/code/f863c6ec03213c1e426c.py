"""Improved WalkSAT with spectral initialization, incremental tracking, and deadline guard."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    # var_idx is 0-based, sign is +1 for positive, -1 for negative
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Build literal co-occurrence matrix for spectral initialization
    # Each literal is indexed as: var_idx * 2 + (0 for positive, 1 for negative)
    # M[a][b] = number of clauses containing both literal a and literal b (a != b)
    num_literals = 2 * n
    M = np.zeros((num_literals, num_literals), dtype=np.int32)
    for clause in clause_data:
        lit_indices = [var_idx * 2 + (0 if sign == 1 else 1) for var_idx, sign in clause]
        for i in range(len(lit_indices)):
            for j in range(i + 1, len(lit_indices)):
                a, b = lit_indices[i], lit_indices[j]
                M[a][b] += 1
                M[b][a] += 1

    # Compute principal eigenvector via power iteration
    def spectral_init():
        """Use principal eigenvector of co-occurrence matrix to initialize assignment."""
        # Normalize rows for stability
        row_sums = M.sum(axis=1).astype(float)
        row_sums[row_sums == 0] = 1
        M_norm = M.astype(float) / row_sums[:, np.newaxis]

        # Power iteration
        v = np.random.RandomState(42).rand(num_literals)
        v /= np.linalg.norm(v)
        for _ in range(200):
            v_new = M_norm @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-10:
                break
            v_new /= norm
            if np.linalg.norm(v_new - v) < 1e-8:
                v = v_new
                break
            v = v_new

        # Convert eigenvector to assignment:
        # For each variable, compare v[2*var] (positive literal) vs v[2*var+1] (negative literal)
        assignment = [False] * n
        for var_idx in range(n):
            pos_score = v[2 * var_idx]
            neg_score = v[2 * var_idx + 1]
            assignment[var_idx] = pos_score > neg_score
        return assignment

    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def walk_one(assignment, rng, max_steps, p_flip):
        """Run WalkSAT with incremental unsatisfied clause tracking."""
        # Build initial unsatisfied clause set
        unsat_set = set()
        for i, clause in enumerate(clause_data):
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                unsat_set.add(i)

        # Track which clauses each variable appears in
        var_to_clauses = [[] for _ in range(n)]
        for i, clause in enumerate(clause_data):
            for var_idx, sign in clause:
                var_to_clauses[var_idx].append(i)

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if not unsat_set:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clause_data[ci]

            # Score each literal in the clause
            # Count how many unsatisfied clauses contain each variable
            var_unsat_count = {}
            for uc in unsat_set:
                for var_idx, sign in clause_data[uc]:
                    var_unsat_count[var_idx] = var_unsat_count.get(var_idx, 0) + 1

            # Weighted selection with noise
            weights = []
            for var_idx, sign in clause:
                base_w = var_unsat_count.get(var_idx, 0) + 1
                noise = rng.random() * p_flip * 5
                weights.append(base_w + noise)

            # Normalize and pick
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_idx = 0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = idx
                    break

            var_idx, sign = clause[chosen_idx]
            assignment[var_idx] = not assignment[var_idx]

            # Update unsatisfied set incrementally
            # Only clauses containing var_idx may change status
            for ci2 in var_to_clauses[var_idx]:
                cl = clause_data[ci2]
                sat = False
                for vi, s in cl:
                    if (assignment[vi] and s == 1) or (not assignment[vi] and s == -1):
                        sat = True
                        break
                if sat:
                    unsat_set.discard(ci2)
                else:
                    unsat_set.add(ci2)

        return assignment, False

    def polish(assignment, rng, max_iters=2000):
        """High-precision polish: try to reduce remaining unsat clauses to zero."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            unsat_clauses = []
            for i, clause in enumerate(clause_data):
                sat = False
                for var_idx, sign in clause:
                    if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
            if not unsat_clauses:
                return assignment, True
            # Try flipping each variable in each unsat clause
            current_unsat = len(unsat_clauses)
            best_flip = None
            best_new_unsat = current_unsat
            for uc in unsat_clauses:
                for var_idx, sign in clause_data[uc]:
                    assignment[var_idx] = not assignment[var_idx]
                    new_unsat = 0
                    for j, cl in enumerate(clause_data):
                        sat = False
                        for vi, s in cl:
                            if (assignment[vi] and s == 1) or (not assignment[vi] and s == -1):
                                sat = True
                                break
                        if not sat:
                            new_unsat += 1
                    assignment[var_idx] = not assignment[var_idx]
                    if new_unsat < best_new_unsat:
                        best_new_unsat = new_unsat
                        best_flip = var_idx
            if best_flip is not None:
                assignment[best_flip] = not assignment[best_flip]
            else:
                break
        unsat_final = count_unsat(assignment)
        return assignment, unsat_final == 0

    # Multi-restart strategy with spectral initialization
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Compute spectral initialization once
    spec_assignment = spectral_init()
    spec_unsat = count_unsat(spec_assignment)

    # Parameters tuned for planted 3-SAT with 150 variables
    max_steps_per_restart = 30000
    num_restarts = 2000

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Alternate between spectral and random initialization
        if restart % 3 == 0:
            # Use spectral initialization (possibly perturbed)
            assignment = spec_assignment[:]
            if restart > 0:
                # Perturb a few random variables to escape local minima
                num_perturb = min(5, n // 10)
                for _ in range(num_perturb):
                    vi = rng.randint(0, n - 1)
                    assignment[vi] = not assignment[vi]
        else:
            # Random initial assignment
            assignment = [rng.random() < 0.5 for _ in range(n)]

        # Vary the flip probability
        p_flip = 0.2 + 0.3 * (restart % 5) / 4.0

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart, p_flip)

        if solved:
            return assignment

        # If close to solved, try a polish pass
        uc = count_unsat(assignment)
        if uc <= 3 and time.time() < deadline - 10:
            assignment, solved = polish(assignment, rng, max_iters=1500)
            if solved:
                return assignment

        # Track best
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Final polish on best assignment if we have time
    if best_assignment is not None and best_unsat_count <= 5 and time.time() < deadline - 5:
        best_assignment, solved = polish(best_assignment, rng, max_iters=3000)
        if solved:
            return best_assignment

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: use spectral assignment
    return spec_assignment

# EVOLVE-BLOCK-END
