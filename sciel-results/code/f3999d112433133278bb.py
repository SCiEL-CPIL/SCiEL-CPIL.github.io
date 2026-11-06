"""Rewritten program with a different search strategy: multi-start local search with
incremental square counting and a richer move set (including adding/removing terms
while preserving 12 nonzero terms, and coefficient perturbation)."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill

    N = 41  # max length
    NZ_TARGET = 12

    def square_nonzero_count(poly):
        """Compute the number of nonzero coefficients in poly^2 (exact integer)."""
        nz = [(i, poly[i]) for i in range(len(poly)) if poly[i] != 0]
        n = len(nz)
        if n == 0:
            return 0
        sq = [0] * (2 * len(poly) - 1)
        for ii in range(n):
            i, ai = nz[ii]
            for jj in range(ii, n):
                j, aj = nz[jj]
                prod = ai * aj
                if ii == jj:
                    sq[i + j] += prod
                else:
                    sq[i + j] += 2 * prod
        return sum(1 for c in sq if c != 0)

    def local_search(poly, deadline, max_iter=20000):
        """Local search with restarts inside. Returns best poly and score found."""
        current = poly[:]
        current_score = square_nonzero_count(current)
        best = current[:]
        best_score = current_score

        for it in range(max_iter):
            if time.time() > deadline:
                break

            nz = [i for i in range(len(current)) if current[i] != 0]
            if len(nz) != NZ_TARGET:
                # Repair
                break

            move_type = random.random()

            if move_type < 0.45:
                # Perturb a coefficient
                idx = random.choice(nz)
                old_val = current[idx]
                # Try small changes first, then larger
                r = random.random()
                if r < 0.5:
                    delta = random.randint(-5, 5)
                    new_val = old_val + delta
                elif r < 0.8:
                    new_val = random.randint(1, 64)
                else:
                    new_val = random.randint(1, 64)
                if random.random() < 0.5:
                    new_val = -new_val
                if new_val == 0:
                    new_val = 1 if new_val <= 0 else -1
                if abs(new_val) > 64:
                    new_val = 64 if new_val > 0 else -64
                if new_val == old_val:
                    continue
                current[idx] = new_val
                new_score = square_nonzero_count(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best = current[:]
                else:
                    current[idx] = old_val

            elif move_type < 0.70:
                # Move a term to a new position
                idx = random.choice(nz)
                coeff = current[idx]
                # Find empty slot
                empty = [i for i in range(len(current)) if current[i] == 0]
                if not empty:
                    continue
                new_pos = random.choice(empty)
                current[idx] = 0
                current[new_pos] = coeff
                new_score = square_nonzero_count(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best = current[:]
                else:
                    current[new_pos] = 0
                    current[idx] = coeff

            elif move_type < 0.85:
                # Swap two coefficients
                i1, i2 = random.sample(nz, 2)
                v1, v2 = current[i1], current[i2]
                current[i1], current[i2] = v2, v1
                new_score = square_nonzero_count(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best = current[:]
                else:
                    current[i1], current[i2] = v1, v2

            else:
                # Add a new term at random empty position, remove one at random nz
                empty = [i for i in range(len(current)) if current[i] == 0]
                if not empty:
                    continue
                new_pos = random.choice(empty)
                new_coeff = random.randint(1, 64)
                if random.random() < 0.5:
                    new_coeff = -new_coeff
                rem_idx = random.choice(nz)
                old_coeff = current[rem_idx]
                current[rem_idx] = 0
                current[new_pos] = new_coeff
                new_score = square_nonzero_count(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best = current[:]
                else:
                    current[new_pos] = 0
                    current[rem_idx] = old_coeff

        return best, best_score

    # Generate diverse initial candidates
    best_poly = None
    best_score = float('inf')

    def evaluate_and_update(poly):
        nonlocal best_poly, best_score
        # Trim trailing zeros
        last = -1
        for i in range(len(poly) - 1, -1, -1):
            if poly[i] != 0:
                last = i
                break
        if last == -1:
            return
        trimmed = poly[:last + 1]
        nz_count = sum(1 for c in trimmed if c != 0)
        if nz_count != NZ_TARGET:
            return
        if len(trimmed) > N:
            return
        if any(abs(c) > 64 for c in trimmed):
            return
        score = square_nonzero_count(trimmed)
        if score < best_score:
            best_score = score
            best_poly = trimmed[:]

    # Candidate 1: Alternating signs at consecutive positions
    p = [0] * N
    for i in range(NZ_TARGET):
        p[i] = 1 if i % 2 == 0 else -1
    evaluate_and_update(p)

    # Candidate 2: Alternating signs at even positions (gaps of 2)
    p = [0] * N
    for i in range(NZ_TARGET):
        p[2 * i] = 1 if i % 2 == 0 else -1
    evaluate_and_update(p)

    # Candidate 3: All 1s
    p = [0] * N
    for i in range(NZ_TARGET):
        p[i] = 1
    evaluate_and_update(p)

    # Candidate 4: Fibonacci-like coefficients
    p = [0] * N
    fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    fibs = [min(f, 64) for f in fibs]
    for i in range(NZ_TARGET):
        p[i] = fibs[i] if i % 2 == 0 else -fibs[i]
    evaluate_and_update(p)

    # Random restarts
    while time.time() < deadline - 5:
        positions = random.sample(range(N), NZ_TARGET)
        poly = [0] * N
        for pos in positions:
            c = random.randint(1, 64)
            if random.random() < 0.5:
                c = -c
            poly[pos] = c
        evaluate_and_update(poly)

        # Run local search
        poly, score = local_search(poly, deadline - 3, max_iter=15000)
        evaluate_and_update(poly)

        if time.time() > deadline - 10:
            break

    # Final refinement on best found
    if best_poly is not None and time.time() < deadline - 2:
        best_poly, best_score = local_search(best_poly, deadline, max_iter=30000)
        # Re-trim
        last = -1
        for i in range(len(best_poly) - 1, -1, -1):
            if best_poly[i] != 0:
                last = i
                break
        if last >= 0:
            best_poly = best_poly[:last + 1]

    # Fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # Pad to have 12 nonzero terms
        best_poly = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # Need exactly 12 nonzero; add more
        best_poly = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        best_poly = best_poly + [0] * (N - len(best_poly))

    # Final validation
    last = -1
    for i in range(len(best_poly) - 1, -1, -1):
        if best_poly[i] != 0:
            last = i
            break
    if last == -1:
        return [1] + [0] * 10 + [1]

    result = best_poly[:last + 1]
    nz_count = sum(1 for c in result if c != 0)
    if nz_count != NZ_TARGET:
        # Force fix: take first 12 nonzero positions
        nz_positions = [i for i in range(len(result)) if result[i] != 0]
        if len(nz_positions) >= NZ_TARGET:
            fixed = [0] * len(result)
            for i in nz_positions[:NZ_TARGET]:
                fixed[i] = result[i]
            result = fixed
        else:
            return [1] * NZ_TARGET

    return result

# EVOLVE-BLOCK-END
