import math
import random
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def is_compatible(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)

    def is_compatible_with_list(m, r, sel):
        for m2, r2 in sel:
            if not is_compatible(m, r, m2, r2):
                return False
        return True

    def calc_score(sel):
        return sum(1.0 / m for m, r in sel)

    def find_valid_residue(m, sel):
        # Returns a valid residue for m given sel, or None
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if is_compatible_with_list(m, r, sel):
                return r
        return None

    def find_valid_residue_full(m, sel):
        # Exhaustive search for a valid residue
        for r in range(m):
            if is_compatible_with_list(m, r, sel):
                return r
        return None

    def try_add_skipped(current, skipped, used_moduli):
        """Try to add any compatible skipped modulus. Returns (current, skipped, used_moduli, score_delta, added_m)."""
        for m in list(skipped):
            r = find_valid_residue_full(m, current)
            if r is not None:
                current.append((m, r))
                used_moduli.add(m)
                skipped.remove(m)
                return current, skipped, used_moduli, 1.0 / m, m
        return current, skipped, used_moduli, 0.0, None

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

    def find_valid_residue_fast(m, sel):
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if is_compatible_with_list_fast(m, r, sel):
                return r
        return None

    def find_valid_residue_full_fast(m, sel):
        for r in range(m):
            if is_compatible_with_list_fast(m, r, sel):
                return r
        return None

    best_selected = []
    best_score = 0.0

    # 1. Greedy Initialization with multiple restarts
    # Try adding moduli in order of decreasing 1/m (smallest m first)
    moduli_sorted = sorted(moduli, key=lambda m: 1.0/m, reverse=True)
    
    for _ in range(30):
        selected = []
        for m in moduli_sorted:
            r = find_valid_residue_fast(m, selected)
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
    
    # Sort skipped by 1/m descending to try best additions first
    skipped.sort(key=lambda m: 1.0/m, reverse=True)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Add any compatible skipped modulus
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
        
        # Phase 2: Systematic residue re-optimization
        # For each selected modulus, try all its residues to see if any unblocks a skipped modulus
        if skipped and current:
            n_current = len(current)
            found_improvement = False
            for i in range(n_current):
                if time.time() - start_time > time_limit:
                    break
                if found_improvement:
                    break
                m_i, r_old = current[i]
                # Try each possible residue for modulus i
                for r_new in range(m_i):
                    if r_new == r_old:
                        continue
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
                    
                    # Try to add any skipped modulus
                    added = False
                    for m_add in list(skipped):
                        r_add = find_valid_residue_full_fast(m_add, current)
                        if r_add is not None:
                            current.append((m_add, r_add))
                            used_moduli.add(m_add)
                            skipped.remove(m_add)
                            new_score = current_score + 1.0 / m_add
                            if new_score > best_score:
                                best_score = new_score
                                best_selected = list(current)
                            current_score = new_score
                            improved = True
                            found_improvement = True
                            added = True
                            break
                    
                    if not added:
                        # Revert residue change
                        current[i] = (m_i, r_old)
                    else:
                        break  # Stop trying other residues for this modulus

        # Phase 3: 2-Opt Swap (optimized: only try promising swaps)
        if not improved and skipped and current and time.time() - start_time < time_limit - 10:
            best_swap_score = -1
            best_swap_idx = -1
            best_swap_m_in = -1
            best_swap_r_in = -1
            
            # Only consider swaps where 1/m_in > 1/m_out (improvement possible)
            for i, (m_out, r_out) in enumerate(current):
                for m_in in skipped:
                    if m_in >= m_out:  # 1/m_in <= 1/m_out, no improvement
                        continue
                    # Find a valid residue for m_in in current \ {m_out}
                    found_r = None
                    for r_test in range(m_in):
                        compatible = True
                        for j, (m2, r2) in enumerate(current):
                            if j == i:
                                continue
                            if not is_compatible_fast(m_in, r_test, m2, r2):
                                compatible = False
                                break
                        if compatible:
                            found_r = r_test
                            break
                    
                    if found_r is not None:
                        new_score = current_score - 1.0 / m_out + 1.0 / m_in
                        if new_score > best_swap_score:
                            best_swap_score = new_score
                            best_swap_idx = i
                            best_swap_m_in = m_in
                            best_swap_r_in = found_r
            
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

        # Phase 4: Residue Perturbation (random fallback) - fixed version
        if not improved and current and skipped and time.time() - start_time < time_limit - 5:
            idx = random.randint(0, len(current) - 1)
            m, r_old = current[idx]
            for _ in range(10):
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
                added = False
                for m_add in list(skipped):
                    r_add = find_valid_residue_full_fast(m_add, current)
                    if r_add is not None:
                        current.append((m_add, r_add))
                        used_moduli.add(m_add)
                        skipped.remove(m_add)
                        new_score = current_score + 1.0 / m_add
                        if new_score > best_score:
                            current_score = new_score
                            best_score = new_score
                            best_selected = list(current)
                            improved = True
                            added = True
                        break
                
                if not added:
                    current[idx] = (m, r_old)
                else:
                    break

    return [[m, r] for m, r in best_selected]
