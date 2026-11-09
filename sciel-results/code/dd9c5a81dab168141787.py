"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # wall-clock limit with margin

    base = 51975  # 3^3 * 5^2 * 7 * 11

    # Precompute all divisors > 1 of base
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    n_divs = len(divisors)
    div_arr = np.array(divisors, dtype=np.int64)

    # Precompute residue indices for each divisor
    # For divisor m, residues are 0, m, 2m, ..., base-1
    # We'll store the count of positions each class covers: base // m
    cover_counts = np.array([base // m for m in divisors], dtype=np.int64)

    def evaluate(classes):
        """Count uncovered residues mod base. classes is list of [m_idx, r]."""
        covered = np.zeros(base, dtype=np.uint8)
        for m_idx, r in classes:
            m = divisors[m_idx]
            covered[r::m] = 1
        return int(base - np.sum(covered))

    def evaluate_fast(m_indices, residues):
        """Fast evaluation using numpy. m_indices: array of indices into divisors, residues: array."""
        covered = np.zeros(base, dtype=np.uint8)
        for i in range(len(m_indices)):
            m = divisors[m_indices[i]]
            r = residues[i]
            covered[r::m] = 1
        return int(base - np.sum(covered))

    # Greedy initialization with multiple orderings
    def greedy_init(order):
        """Greedy: process moduli in given order, each picks best residue."""
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m_idx in order:
            m = divisors[m_idx]
            best_r = 0
            best_count = -1
            for r in range(m):
                idx = np.arange(r, base, m)
                count = np.sum(covered[idx] == 0)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m_idx, best_r])
            covered[best_r::m] = 1
        return classes, int(base - np.sum(covered))

    # Generate multiple initial solutions
    best_classes = None
    best_score = base

    # Try several deterministic orderings
    orderings = []
    orderings.append(list(range(n_divs)))  # increasing
    orderings.append(list(range(n_divs - 1, -1, -1)))  # decreasing
    orderings.append(sorted(range(n_divs), key=lambda i: -divisors[i]))  # largest m first
    orderings.append(sorted(range(n_divs), key=lambda i: (base // divisors[i], divisors[i])))

    # Add some random orderings
    for _ in range(10):
        perm = list(range(n_divs))
        random.shuffle(perm)
        orderings.append(perm)

    for i, order in enumerate(orderings):
        if time.time() - start_time > deadline - 30:
            break
        try:
            classes, sc = greedy_init(order)
            if sc < best_score:
                best_score = sc
                best_classes = [c[:] for c in classes]
        except Exception:
            pass

    # Simulated annealing for improvement
    if best_classes is not None and time.time() - start_time < deadline - 5:
        import math

        # Work with the best solution
        curr_classes = [c[:] for c in best_classes]
        curr_score = best_score

        # Precompute for fast evaluation
        # We'll maintain a coverage array and update incrementally

        def compute_coverage(classes):
            """Compute coverage array from classes."""
            covered = np.zeros(base, dtype=np.uint8)
            for m_idx, r in classes:
                m = divisors[m_idx]
                covered[r::m] = 1
            return covered

        def delta_score_remove_add(covered, m_idx_old, r_old, m_idx_new, r_new):
            """Compute score change when replacing class (m_idx_old, r_old) with (m_idx_new, r_new)."""
            m_old = divisors[m_idx_old]
            m_new = divisors[m_idx_new]

            # Remove old class: positions r_old, r_old+m_old, ... become potentially uncovered
            # But they might still be covered by other classes
            # This is complex, so we'll do a full recompute for simplicity in SA
            # But we can optimize by only checking affected positions

            # For now, use full recompute but with a fast path
            # Actually, let's just do full recompute - it's O(base) per eval
            # With ~100 iterations/sec, that's fine for 60 seconds

            return None  # We'll just recompute

        # Simulated annealing
        sa_start = time.time()
        sa_deadline = deadline - 5
        T = 10.0
        T_min = 0.1
        cooling = 0.999

        # Use a different approach: maintain coverage array
        covered = compute_coverage(curr_classes)
        curr_uncovered = int(base - np.sum(covered))

        # Precompute for each divisor index, the list of positions it covers
        # positions_for_m[m_idx] = array of positions covered by class with that modulus at residue 0
        # For residue r, positions are r, r+m, r+2m, ...

        # SA move: pick a random class, change its residue
        best_sa_classes = [c[:] for c in curr_classes]
        best_sa_score = curr_uncovered

        n_iterations = 0
        while time.time() - sa_start < sa_deadline - sa_start and T > T_min:
            n_iterations += 1

            # Pick a random class to modify
            idx = random.randint(0, len(curr_classes) - 1)
            m_idx_old, r_old = curr_classes[idx]
            m_old = divisors[m_idx_old]

            # Generate new residue
            r_new = random.randint(0, m_old - 1)
            if r_new == r_old:
                continue

            # Compute delta: remove old, add new
            # Positions affected by old class: r_old, r_old+m_old, ..., < base
            # Positions affected by new class: r_new, r_new+m_old, ..., < base

            # Count how many positions are currently covered ONLY by this class
            # This is complex. Let's just do full recompute for the affected region.

            # Actually, for efficiency, let's compute the score change directly:
            # 1. Remove old class: for each position p in old class's AP, if it's not covered by any other class, uncovered increases
            # 2. Add new class: for each position p in new class's AP, if it's not covered by any other class, uncovered decreases

            # This requires knowing which positions are covered by other classes.
            # Let's precompute a "coverage count" array.

            # For simplicity and correctness, do full recompute every few steps
            # and use a cheaper heuristic most of the time

            # Cheap heuristic: just try the move and recompute
            new_classes = [c[:] for c in curr_classes]
            new_classes[idx] = [m_idx_old, r_new]

            # Fast recompute
            new_covered = np.zeros(base, dtype=np.uint8)
            for i, (mi, ri) in enumerate(new_classes):
                m = divisors[mi]
                new_covered[ri::m] = 1
            new_score = int(base - np.sum(new_covered))

            delta = new_score - curr_uncovered

            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                curr_classes = new_classes
                curr_uncovered = new_score
                covered = new_covered

                if new_score < best_sa_score:
                    best_sa_score = new_score
                    best_sa_classes = [c[:] for c in curr_classes]

            T *= cooling

            # Check time
            if time.time() - start_time > deadline - 5:
                break

        if best_sa_score < best_score:
            best_score = best_sa_score
            best_classes = [c[:] for c in best_sa_classes]

    # Final local search: try changing each class's residue to the best one
    if best_classes is not None and time.time() - start_time < deadline - 2:
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline - 2:
                break
            for i in range(len(best_classes)):
                if time.time() - start_time > deadline - 2:
                    break
                m_idx, r = best_classes[i]
                m = divisors[m_idx]
                # Try all residues for this class
                best_r = r
                best_sc = best_score
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Compute score with this class at new_r
                    new_classes = [c[:] for c in best_classes]
                    new_classes[i] = [m_idx, new_r]

                    covered = np.zeros(base, dtype=np.uint8)
                    for mi, ri in new_classes:
                        mm = divisors[mi]
                        covered[ri::mm] = 1
                    sc = int(base - np.sum(covered))

                    if sc < best_sc:
                        best_sc = sc
                        best_r = new_r

                if best_r != r:
                    best_classes[i] = [m_idx, best_r]
                    best_score = best_sc
                    improved = True

    # Convert indices back to actual moduli
    result = [[divisors[m_idx], r] for m_idx, r in best_classes]
    return result
# EVOLVE-BLOCK-END
