"""Pairwise disjoint residue classes - greedy + add-remove-swap local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # For each modulus, precompute residue compatibility masks
    # compat_mask[i][j][r] = bitmask of residues of moduli[j] compatible with residue r of moduli[i]
    # This is expensive to precompute fully, so we compute on-demand and cache
    compat_cache = {}

    def get_compatible_mask(m1, r1, m2):
        """Get bitmask of residues of m2 compatible with (m1, r1)."""
        key = (m1, r1, m2)
        if key in compat_cache:
            return compat_cache[key]
        g = gcd(m1, m2)
        if g == 1:
            mask = (1 << m2) - 1
        else:
            # r2 must satisfy r2 % g != r1 % g
            bad = r1 % g
            mask = 0
            for r2 in range(m2):
                if r2 % g != bad:
                    mask |= (1 << r2)
        compat_cache[key] = mask
        return mask

    def can_add(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True

    def find_valid_residue(classes, m):
        """Find a valid residue for modulus m given existing classes.
        Returns the residue with maximum slack, or None."""
        mi = mod_idx[m]
        best_r = None
        best_slack = -1

        for r in range(m):
            valid = True
            min_dist = m
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                if (r - r2) % g == 0:
                    valid = False
                    break
                if g > 1:
                    diff = (r - r2) % g
                    dist = min(diff, g - diff)
                    if dist < min_dist:
                        min_dist = dist
            if valid:
                if min_dist > best_slack:
                    best_slack = min_dist
                    best_r = r

        return best_r

    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        total = 0.0
        for (m, r) in classes:
            total += 1.0 / m
        return total

    def try_add_all(classes):
        """Try to add all remaining moduli, starting with smallest."""
        used_moduli = set(m for m, r in classes)
        remaining = [m for m in sorted(moduli) if m not in used_moduli]

        for m in remaining:
            if time.time() - start_time > deadline - 3:
                break
            r = find_valid_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)

        return classes

    def add_remove_swap(classes, max_iter=300):
        """Local search with add, remove, and swap moves.
        The swap move replaces a low-value modulus with a high-value one if packing permits."""
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)

        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break

            improved = False
            n_classes = len(best_classes)

            # Move 1: Add a new modulus (smallest first for best 1/m)
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                r = find_valid_residue(best_classes, m)
                if r is not None:
                    new_classes = best_classes + [[m, r]]
                    new_score = best_score + 1.0 / m
                    if new_score > best_score + 1e-12:
                        best_classes = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break

            # Move 2: Swap - remove a large modulus and try to add a smaller one
            if not improved:
                # Sort current classes by m descending (largest first, lowest value)
                sorted_indices = sorted(range(n_classes), key=lambda i: -best_classes[i][0])
                for i in sorted_indices:
                    if time.time() - start_time > deadline - 3:
                        break
                    m_remove = best_classes[i][0]
                    # Try removing this modulus and adding smaller ones
                    temp_classes = [list(c) for j, c in enumerate(best_classes) if j != i]
                    temp_used = set(m for m, r in temp_classes)
                    temp_score = best_score - 1.0 / m_remove

                    # Try to add smaller moduli
                    added_any = False
                    for m_add in sorted(moduli):
                        if m_add in temp_used:
                            continue
                        if m_add >= m_remove:
                            break  # only add smaller
                        r = find_valid_residue(temp_classes, m_add)
                        if r is not None:
                            temp_classes.append([m_add, r])
                            temp_used.add(m_add)
                            temp_score += 1.0 / m_add
                            added_any = True
                            # Check if we can add even more
                            for m_add2 in sorted(moduli):
                                if m_add2 in temp_used:
                                    continue
                                r2 = find_valid_residue(temp_classes, m_add2)
                                if r2 is not None:
                                    temp_classes.append([m_add2, r2])
                                    temp_used.add(m_add2)
                                    temp_score += 1.0 / m_add2

                    if temp_score > best_score + 1e-12:
                        best_classes = temp_classes
                        best_score = temp_score
                        used_moduli = set(m for m, r in best_classes)
                        improved = True
                        break

            # Move 3: Residue reassignment to free up space
            if not improved:
                for i in range(n_classes):
                    if time.time() - start_time > deadline - 3:
                        break
                    m, r = best_classes[i]
                    mi = mod_idx[m]
                    # Try all other residues for this modulus
                    best_r = r
                    best_slack = -1
                    for r2 in range(m):
                        if r2 == r:
                            continue
                        valid = True
                        min_dist = m
                        for j in range(n_classes):
                            if i == j:
                                continue
                            m2, r2_other = best_classes[j]
                            mj = mod_idx[m2]
                            g = gcd_table[mi][mj]
                            if (r2 - r2_other) % g == 0:
                                valid = False
                                break
                            if g > 1:
                                diff = (r2 - r2_other) % g
                                dist = min(diff, g - diff)
                                if dist < min_dist:
                                    min_dist = dist
                        if valid and min_dist > best_slack:
                            best_slack = min_dist
                            best_r = r2

                    if best_r != r:
                        # Swapping might allow adding more moduli
                        temp_classes = [list(c) for c in best_classes]
                        temp_classes[i][1] = best_r
                        temp_classes = try_add_all(temp_classes)
                        if score(temp_classes) > best_score + 1e-12:
                            best_classes = temp_classes
                            best_score = score(best_classes)
                            used_moduli = set(m for m, r in best_classes)
                            improved = True
                            break

            if not improved:
                break

        return best_classes

    def greedy_construct(order=None):
        """Greedy construction: add classes in the given order if possible."""
        if order is None:
            order = sorted(moduli, key=lambda m: 1.0 / m)  # smallest moduli first
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            r = find_valid_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        return classes

    # Multi-start approach
    best_classes = None
    best_score = 0.0

    # Start 1: Greedy with smallest moduli first
    classes = greedy_construct()
    classes = try_add_all(classes)
    classes = add_remove_swap(classes, max_iter=500)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]

    # Start 2: Multiple random starts with add-remove-swap
    random.seed(42)
    num_starts = 40

    for start_idx in range(num_starts):
        if time.time() - start_time > deadline - 15:
            break

        # Create a random ordering biased toward small moduli
        remaining = list(moduli)
        order = []
        while remaining:
            alpha = 1.0 + start_idx * 0.1
            weights = [1.0 / (m ** alpha) for m in remaining]
            total_w = sum(weights)
            if total_w <= 0:
                order.append(remaining[0])
                remaining.pop(0)
            else:
                r = random.random() * total_w
                cum = 0
                chosen_idx = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        chosen_idx = i
                        break
                order.append(remaining[chosen_idx])
                remaining.pop(chosen_idx)

        classes = greedy_construct(order)
        classes = try_add_all(classes)

        if time.time() - start_time > deadline - 8:
            s = score(classes)
            if s > best_score:
                best_score = s
                best_classes = [list(c) for c in classes]
            continue

        classes = add_remove_swap(classes, max_iter=200)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]

    # Final polish: intensive local search on best solution
    if best_classes is not None:
        for round_num in range(4):
            if time.time() - start_time > deadline - 2:
                break
            max_iter = 400 + round_num * 200
            improved_classes = add_remove_swap(best_classes, max_iter=max_iter)
            improved_score = score(improved_classes)
            if improved_score > best_score + 1e-12:
                best_classes = improved_classes
                best_score = improved_score
            else:
                break

    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
