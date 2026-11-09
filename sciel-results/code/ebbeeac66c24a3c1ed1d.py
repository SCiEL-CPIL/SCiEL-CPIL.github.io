"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
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

    # Precompute position arrays for each divisor
    # pos_arrays[m_idx] = numpy array of positions covered when residue=0
    # For residue r, positions are pos_arrays[m_idx] + r (mod base)
    pos_arrays = []
    for d in divisors:
        pos_arrays.append(np.arange(0, base, d, dtype=np.int32))

    # Precompute for each divisor, the number of positions it covers
    div_sizes = [len(pa) for pa in pos_arrays]

    # Greedy initialization with coverage-count approach
    def greedy_init(order):
        """Greedy: process moduli in given order, each picks best residue."""
        cover_count = np.zeros(base, dtype=np.int16)
        classes = []
        for m_idx in order:
            m = divisors[m_idx]
            pos0 = pos_arrays[m_idx]
            best_r = 0
            best_count = -1
            # Try each residue; count how many new positions we cover
            for r in range(m):
                positions = (pos0 + r) % base
                new_cover = np.sum(cover_count[positions] == 0)
                if new_cover > best_count:
                    best_count = new_cover
                    best_r = r
            classes.append([m_idx, best_r])
            positions = (pos0 + best_r) % base
            cover_count[positions] += 1
        uncovered = int(np.sum(cover_count == 0))
        return classes, uncovered

    # Fast evaluation using coverage count
    def evaluate_classes(classes):
        """Count uncovered residues. classes is list of [m_idx, r]."""
        cover_count = np.zeros(base, dtype=np.int16)
        for m_idx, r in classes:
            positions = (pos_arrays[m_idx] + r) % base
            cover_count[positions] += 1
        return int(np.sum(cover_count == 0))

    # Generate multiple initial solutions
    best_classes = None
    best_score = base

    # Try several deterministic orderings
    orderings = []
    orderings.append(list(range(n_divs)))  # increasing
    orderings.append(list(range(n_divs - 1, -1, -1)))  # decreasing
    orderings.append(sorted(range(n_divs), key=lambda i: -divisors[i]))  # largest m first
    orderings.append(sorted(range(n_divs), key=lambda i: (base // divisors[i], divisors[i])))
    orderings.append(sorted(range(n_divs), key=lambda i: divisors[i] % 10))

    # Add random orderings
    for _ in range(15):
        perm = list(range(n_divs))
        random.shuffle(perm)
        orderings.append(perm)

    for i, order in enumerate(orderings):
        if time.time() - start_time > deadline - 25:
            break
        try:
            classes, sc = greedy_init(order)
            if sc < best_score:
                best_score = sc
                best_classes = [c[:] for c in classes]
        except Exception:
            pass

    # Simulated annealing with fast delta computation
    if best_classes is not None and time.time() - start_time < deadline - 3:
        curr_classes = [c[:] for c in best_classes]
        curr_score = best_score

        # Build coverage count array
        cover_count = np.zeros(base, dtype=np.int16)
        for m_idx, r in curr_classes:
            positions = (pos_arrays[m_idx] + r) % base
            cover_count[positions] += 1

        def compute_delta(m_idx, r_old, r_new):
            """Compute score change when changing residue of class m_idx from r_old to r_new."""
            pos0 = pos_arrays[m_idx]
            old_positions = (pos0 + r_old) % base
            new_positions = (pos0 + r_new) % base
            old_only = np.sum((cover_count[old_positions] == 1))
            new_uncovered = np.sum((cover_count[new_positions] == 0))
            return int(old_only - new_uncovered)

        def compute_swap_delta(i1, r1_old, r1_new, i2, r2_old, r2_new):
            """Compute score change when swapping residues of two classes."""
            pos0_1 = pos_arrays[i1]
            pos0_2 = pos_arrays[i2]
            old1 = (pos0_1 + r1_old) % base
            new1 = (pos0_1 + r1_new) % base
            old2 = (pos0_2 + r2_old) % base
            new2 = (pos0_2 + r2_new) % base

            # Simulate: remove old1, old2; add new1, new2
            # Uncovered increases by: positions where removing makes them uncovered
            # Uncovered decreases by: positions where adding makes them covered
            # Need to be careful about interactions

            # Count how many positions become uncovered after removing old1 and old2
            # and how many become covered after adding new1 and new2

            # For positions in old1 only (not in old2): if cover_count was 1, becomes 0
            # For positions in old2 only (not in old1): if cover_count was 1, becomes 0
            # For positions in both old1 and old2: if cover_count was 2, becomes 0
            # For positions in new1 only (not in new2): if cover_count was 0, becomes 1
            # For positions in new2 only (not in new1): if cover_count was 0, becomes 1
            # For positions in both new1 and new2: if cover_count was 0, becomes 1 (only need one)

            # This is complex; let's just compute it by tracking changes
            # Actually, let's compute the net effect more carefully

            # Positions that lose coverage:
            # old1 \ old2: lose 1 coverage
            # old2 \ old1: lose 1 coverage
            # old1 ∩ old2: lose 2 coverage

            # Positions that gain coverage:
            # new1 \ new2: gain 1 coverage
            # new2 \ new1: gain 1 coverage
            # new1 ∩ new2: gain 2 coverage (but if already covered, doesn't matter)

            # Let's compute using numpy
            # Create a temporary view of cover_count
            # But we can't modify it, so let's compute the delta analytically

            # For each position, track the net change in cover_count
            # This is expensive, so let's use a simpler approach:
            # Just compute the delta by checking specific conditions

            # Positions that become uncovered:
            # - In old1 but not old2: if cover_count == 1
            # - In old2 but not old1: if cover_count == 1
            # - In both old1 and old2: if cover_count == 2

            # Positions that become covered:
            # - In new1 but not new2: if cover_count == 0
            # - In new2 but not new1: if cover_count == 0
            # - In both new1 and new2: if cover_count == 0

            # But we need to account for overlaps between old and new sets
            # This is getting complex. Let's just do a full recomputation for swaps
            # since they should be rare

            # Actually, let's compute it step by step:
            # 1. Remove old1: positions in old1 lose 1 coverage
            # 2. Remove old2: positions in old2 lose 1 coverage
            # 3. Add new1: positions in new1 gain 1 coverage
            # 4. Add new2: positions in new2 gain 1 coverage

            # Net effect on each position:
            # Let's create a diff array
            # But that's expensive. Let's just compute the uncovered count change directly

            # Alternative: compute the number of positions that transition from covered to uncovered
            # and from uncovered to covered

            # Let's use a different approach: compute the cover_count after the swap
            # for the affected positions only

            # Affected positions are in old1 ∪ old2 ∪ new1 ∪ new2
            # Let's get the union
            affected = np.union1d(np.union1d(old1, old2), np.union1d(new1, new2))

            # For each affected position, compute the net change in cover_count
            # and determine if it transitions between covered/uncovered

            # This is still expensive. Let's just do a full evaluation for swaps
            # and use them sparingly

            # For now, let's return a conservative estimate
            # We'll compute it by actually simulating the change on a copy
            # But that's too slow. Let's just skip swaps in SA and use them in local search

            # Return a rough estimate: assume no interaction
            old1_only = np.sum((cover_count[old1] == 1) & ~np.isin(old1, old2))
            old2_only = np.sum((cover_count[old2] == 1) & ~np.isin(old2, old1))
            old_both = np.sum((cover_count[np.intersect1d(old1, old2)] == 2))

            new1_only = np.sum((cover_count[new1] == 0) & ~np.isin(new1, new2))
            new2_only = np.sum((cover_count[new2] == 0) & ~np.isin(new2, new1))
            new_both = np.sum((cover_count[np.intersect1d(new1, new2)] == 0))

            return int(old1_only + old2_only + old_both - new1_only - new2_only - new_both)

        # SA parameters
        T = float(max(curr_score, 5) + 1)
        T_min = 0.01
        cooling = 0.9998

        best_sa_classes = [c[:] for c in curr_classes]
        best_sa_score = curr_score

        n_iter = 0
        while time.time() - start_time < deadline - 3 and T > T_min:
            n_iter += 1

            # Pick move type: 70% single change, 30% swap
            if random.random() < 0.7:
                # Single residue change
                idx = random.randint(0, len(curr_classes) - 1)
                m_idx_old, r_old = curr_classes[idx]
                m_old = divisors[m_idx_old]
                r_new = random.randint(0, m_old - 1)
                if r_new == r_old:
                    continue

                delta = compute_delta(m_idx_old, r_old, r_new)

                if delta < 0 or random.random() < math.exp(-delta / T):
                    pos0 = pos_arrays[m_idx_old]
                    old_positions = (pos0 + r_old) % base
                    new_positions = (pos0 + r_new) % base
                    cover_count[old_positions] -= 1
                    cover_count[new_positions] += 1

                    curr_classes[idx] = [m_idx_old, r_new]
                    curr_score += delta

                    if curr_score < best_sa_score:
                        best_sa_score = curr_score
                        best_sa_classes = [c[:] for c in curr_classes]
            else:
                # Swap residues between two classes
                if len(curr_classes) < 2:
                    continue
                i1, i2 = random.sample(range(len(curr_classes)), 2)
                m_idx_1, r1_old = curr_classes[i1]
                m_idx_2, r2_old = curr_classes[i2]
                m1 = divisors[m_idx_1]
                m2 = divisors[m_idx_2]
                r1_new = random.randint(0, m1 - 1)
                r2_new = random.randint(0, m2 - 1)
                if r1_new == r1_old and r2_new == r2_old:
                    continue

                delta = compute_swap_delta(m_idx_1, r1_old, r1_new, m_idx_2, r2_old, r2_new)

                if delta < 0 or random.random() < math.exp(-delta / T):
                    pos0_1 = pos_arrays[m_idx_1]
                    pos0_2 = pos_arrays[m_idx_2]
                    old1 = (pos0_1 + r1_old) % base
                    new1 = (pos0_1 + r1_new) % base
                    old2 = (pos0_2 + r2_old) % base
                    new2 = (pos0_2 + r2_new) % base
                    cover_count[old1] -= 1
                    cover_count[old2] -= 1
                    cover_count[new1] += 1
                    cover_count[new2] += 1

                    curr_classes[i1] = [m_idx_1, r1_new]
                    curr_classes[i2] = [m_idx_2, r2_new]
                    curr_score += delta

                    if curr_score < best_sa_score:
                        best_sa_score = curr_score
                        best_sa_classes = [c[:] for c in curr_classes]

            T *= cooling

        if best_sa_score < best_score:
            best_score = best_sa_score
            best_classes = [c[:] for c in best_sa_classes]

    # Final local search: try changing each class's residue to the best one
    if best_classes is not None and time.time() - start_time < deadline - 1:
        # Rebuild cover_count
        cover_count = np.zeros(base, dtype=np.int16)
        for m_idx, r in best_classes:
            positions = (pos_arrays[m_idx] + r) % base
            cover_count[positions] += 1
        curr_score = int(np.sum(cover_count == 0))

        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline - 1:
                break
            for i in range(len(best_classes)):
                if time.time() - start_time > deadline - 1:
                    break
                m_idx, r = best_classes[i]
                m = divisors[m_idx]
                # Try all residues for this class
                best_r = r
                best_sc = curr_score
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Compute delta
                    pos0 = pos_arrays[m_idx]
                    old_positions = (pos0 + r) % base
                    new_positions = (pos0 + new_r) % base
                    old_only = np.sum(cover_count[old_positions] == 1)
                    new_uncovered = np.sum(cover_count[new_positions] == 0)
                    delta = int(old_only - new_uncovered)
                    sc = curr_score + delta
                    if sc < best_sc:
                        best_sc = sc
                        best_r = new_r

                if best_r != r:
                    # Apply the best change
                    pos0 = pos_arrays[m_idx]
                    old_positions = (pos0 + r) % base
                    new_positions = (pos0 + best_r) % base
                    cover_count[old_positions] -= 1
                    cover_count[new_positions] += 1
                    best_classes[i] = [m_idx, best_r]
                    curr_score = best_sc
                    improved = True

            # Also try pair swaps in local search
            if time.time() - start_time > deadline - 1:
                break
            for i1 in range(len(best_classes)):
                if time.time() - start_time > deadline - 1:
                    break
                for i2 in range(i1 + 1, len(best_classes)):
                    if time.time() - start_time > deadline - 1:
                        break
                    m_idx_1, r1 = best_classes[i1]
                    m_idx_2, r2 = best_classes[i2]
                    m1 = divisors[m_idx_1]
                    m2 = divisors[m_idx_2]

                    # Try swapping: give i1 a random residue and i2 a random residue
                    # But this is expensive. Instead, try a few random swaps
                    for _ in range(3):
                        if time.time() - start_time > deadline - 1:
                            break
                        r1_new = random.randint(0, m1 - 1)
                        r2_new = random.randint(0, m2 - 1)
                        if r1_new == r1 and r2_new == r2:
                            continue

                        # Compute delta for the swap
                        pos0_1 = pos_arrays[m_idx_1]
                        pos0_2 = pos_arrays[m_idx_2]
                        old1 = (pos0_1 + r1) % base
                        new1 = (pos0_1 + r1_new) % base
                        old2 = (pos0_2 + r2) % base
                        new2 = (pos0_2 + r2_new) % base

                        # Compute the delta by simulating
                        # Positions that become uncovered:
                        old1_only = np.sum((cover_count[old1] == 1) & ~np.isin(old1, old2))
                        old2_only = np.sum((cover_count[old2] == 1) & ~np.isin(old2, old1))
                        old_both = np.sum((cover_count[np.intersect1d(old1, old2)] == 2))

                        # Positions that become covered:
                        new1_only = np.sum((cover_count[new1] == 0) & ~np.isin(new1, new2))
                        new2_only = np.sum((cover_count[new2] == 0) & ~np.isin(new2, new1))
                        new_both = np.sum((cover_count[np.intersect1d(new1, new2)] == 0))

                        delta = int(old1_only + old2_only + old_both - new1_only - new2_only - new_both)
                        sc = curr_score + delta

                        if sc < curr_score:
                            # Apply the swap
                            cover_count[old1] -= 1
                            cover_count[old2] -= 1
                            cover_count[new1] += 1
                            cover_count[new2] += 1
                            best_classes[i1] = [m_idx_1, r1_new]
                            best_classes[i2] = [m_idx_2, r2_new]
                            curr_score = sc
                            improved = True

        best_score = curr_score

    # Convert indices back to actual moduli
    result = [[divisors[m_idx], r] for m_idx, r in best_classes]
    return result
# EVOLVE-BLOCK-END
