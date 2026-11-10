import math
import random
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # Precompute gcd for all pairs to speed up compatibility checks
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    gcd_cache = {}
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            m1, m2 = moduli[i], moduli[j]
            gcd_cache[(m1, m2)] = math.gcd(m1, m2)

    def is_compatible_fast(m1, r1, m2, r2):
        if m1 == m2:
            return r1 != r2
        key = (min(m1, m2), max(m1, m2))
        g = gcd_cache[key]
        return (r1 % g) != (r2 % g)

    def is_compatible_with_list_fast(m, r, sel):
        for m2, r2 in sel:
            if not is_compatible_fast(m, r, m2, r2):
                return False
        return True

    def find_valid_residue_full_fast(m, sel):
        """Find any valid residue for m given sel. Returns residue or None."""
        for r in range(m):
            if is_compatible_with_list_fast(m, r, sel):
                return r
        return None

    def count_blocked(m, sel):
        """Count how many residues of m are blocked by sel."""
        blocked = 0
        for r in range(m):
            if not is_compatible_with_list_fast(m, r, sel):
                blocked += 1
        return blocked

    def calc_score(sel):
        return sum(1.0 / m for m, r in sel)

    best_selected = []
    best_score = 0.0

    # 1. Greedy Initialization with multiple restarts
    moduli_sorted = sorted(moduli, key=lambda m: 1.0/m, reverse=True)

    for restart in range(50):
        selected = []
        for m in moduli_sorted:
            r = find_valid_residue_full_fast(m, selected)
            if r is not None:
                selected.append((m, r))

        score = calc_score(selected)
        if score > best_score:
            best_score = score
            best_selected = list(selected)

    # 2. Local Search
    current = list(best_selected)
    current_score = best_score
    used_moduli = set(m for m, r in current)
    skipped = [m for m in moduli if m not in used_moduli]
    skipped.sort(key=lambda m: 1.0/m, reverse=True)

    def try_add_any(current, skipped, used_moduli):
        """Try to add any compatible skipped modulus. Returns (added, new_score)."""
        for m in list(skipped):
            r = find_valid_residue_full_fast(m, current)
            if r is not None:
                current.append((m, r))
                used_moduli.add(m)
                skipped.remove(m)
                return True, 1.0 / m
        return False, 0.0

    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False

        # Phase 1: Add any compatible skipped modulus (greedy)
        new_skipped = []
        for m in skipped:
            r = find_valid_residue_full_fast(m, current)
            if r is not None:
                current.append((m, r))
                used_moduli.add(m)
                current_score += 1.0 / m
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(current)
            else:
                new_skipped.append(m)
        skipped = new_skipped

        # Phase 2: Exhaustive residue re-optimization
        # For each selected modulus, try ALL residues and keep the one that
        # maximizes the number of addable skipped moduli (or at least unblocks one)
        if skipped and current:
            n_current = len(current)
            found_improvement = False
            for i in range(n_current):
                if time.time() - start_time > time_limit:
                    break
                if found_improvement:
                    break
                m_i, r_old = current[i]

                best_r = r_old
                best_addable = -1

                # Try all residues for modulus i
                for r_new in range(m_i):
                    # Check compatibility with all other selected moduli
                    compatible = True
                    for j in range(n_current):
                        if j == i:
                            continue
                        m2, r2 = current[j]
                        if not is_compatible_fast(m_i, r_new, m2, r2):
                            compatible = False
                            break
                    if not compatible:
                        continue

                    # Save old state
                    current[i] = (m_i, r_new)

                    # Count how many skipped moduli can be added
                    addable = 0
                    for m_add in skipped:
                        if find_valid_residue_full_fast(m_add, current) is not None:
                            addable += 1

                    if addable > best_addable:
                        best_addable = addable
                        best_r = r_new

                # Restore best residue
                current[i] = (m_i, best_r)

                # If we found a better residue that unblocks additions, apply it
                if best_r != r_old and best_addable > 0:
                    # Now actually add the best skipped modulus
                    added, delta = try_add_any(current, skipped, used_moduli)
                    if added:
                        current_score += delta
                        improved = True
                        found_improvement = True
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = list(current)

        # Phase 3: 2-Opt Swap with residue optimization
        if not improved and skipped and current and time.time() - start_time < time_limit - 15:
            best_swap_score = -1
            best_swap_idx = -1
            best_swap_m_in = -1
            best_swap_r_in = -1
            best_swap_r_out_new = None

            for i, (m_out, r_out) in enumerate(current):
                for m_in in skipped:
                    if m_in >= m_out:
                        continue
                    # Find a valid residue for m_in in current \ {m_out}
                    found_r_in = None
                    for r_test in range(m_in):
                        compatible = True
                        for j, (m2, r2) in enumerate(current):
                            if j == i:
                                continue
                            if not is_compatible_fast(m_in, r_test, m2, r2):
                                compatible = False
                                break
                        if compatible:
                            found_r_in = r_test
                            break

                    if found_r_in is not None:
                        new_score = current_score - 1.0 / m_out + 1.0 / m_in
                        if new_score > best_swap_score:
                            best_swap_score = new_score
                            best_swap_idx = i
                            best_swap_m_in = m_in
                            best_swap_r_in = found_r_in

            if best_swap_idx != -1 and best_swap_score > current_score + 1e-12:
                m_out, r_out = current[best_swap_idx]
                current[best_swap_idx] = (best_swap_m_in, best_swap_r_in)
                used_moduli.remove(m_out)
                used_moduli.add(best_swap_m_in)
                skipped.remove(best_swap_m_in)
                skipped.append(m_out)
                skipped.sort(key=lambda m: 1.0/m, reverse=True)
                current_score = best_swap_score
                best_score = current_score
                best_selected = list(current)
                improved = True

        # Phase 4: Multi-residue perturbation - try changing 2 residues at once
        if not improved and current and skipped and time.time() - start_time < time_limit - 10:
            n_current = len(current)
            if n_current >= 2:
                found = False
                for _ in range(50):
                    if time.time() - start_time > time_limit:
                        break
                    i1 = random.randint(0, n_current - 1)
                    i2 = random.randint(0, n_current - 1)
                    if i1 == i2:
                        continue
                    m1, r1_old = current[i1]
                    m2, r2_old = current[i2]
                    r1_new = random.randint(0, m1 - 1)
                    r2_new = random.randint(0, m2 - 1)
                    if r1_new == r1_old and r2_new == r2_old:
                        continue

                    # Check compatibility of new residues with all others
                    compatible = True
                    for j in range(n_current):
                        if j == i1 or j == i2:
                            continue
                        mj, rj = current[j]
                        if not is_compatible_fast(m1, r1_new, mj, rj):
                            compatible = False
                            break
                        if not is_compatible_fast(m2, r2_new, mj, rj):
                            compatible = False
                            break
                    if not compatible:
                        continue
                    # Check mutual compatibility
                    if not is_compatible_fast(m1, r1_new, m2, r2_new):
                        continue

                    current[i1] = (m1, r1_new)
                    current[i2] = (m2, r2_new)

                    added, delta = try_add_any(current, skipped, used_moduli)
                    if added:
                        current_score += delta
                        improved = True
                        found = True
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = list(current)
                    else:
                        current[i1] = (m1, r1_old)
                        current[i2] = (m2, r2_old)
                    if found:
                        break

        # Phase 5: Single residue perturbation (random fallback)
        if not improved and current and skipped and time.time() - start_time < time_limit - 5:
            idx = random.randint(0, len(current) - 1)
            m, r_old = current[idx]
            for _ in range(20):
                r_new = random.randint(0, m - 1)
                if r_new == r_old:
                    continue
                compatible = True
                for j, (m2, r2) in enumerate(current):
                    if j == idx:
                        continue
                    if not is_compatible_fast(m, r_new, m2, r2):
                        compatible = False
                        break
                if not compatible:
                    continue

                current[idx] = (m, r_new)
                added, delta = try_add_any(current, skipped, used_moduli)
                if added:
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(current)
                    break
                else:
                    current[idx] = (m, r_old)

    # Final polish: try to add any remaining skipped moduli one more time
    if time.time() - start_time < time_limit - 2:
        for m in list(skipped):
            r = find_valid_residue_full_fast(m, best_selected)
            if r is not None:
                best_selected.append((m, r))
                best_score += 1.0 / m

    return [[m, r] for m, r in best_selected]
