"""Rewritten program for the suite task; using a direct heuristic approach."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table for speed
    gcd_table = {}
    for m1 in all_moduli:
        for m2 in all_moduli:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def find_residue_for(m, selected):
        """Find a valid residue for modulus m given already selected classes."""
        for r in range(m):
            valid = True
            for r_prime, m_prime in selected:
                g = gcd_table.get((m, m_prime), math.gcd(m, m_prime))
                if (r - r_prime) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def try_add(m, selected):
        """Try to add modulus m to selected. Returns True if added."""
        r = find_residue_for(m, selected)
        if r is not None:
            selected.append([r, m])
            return True
        return False

    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    # Multiple greedy initializations
    orders = [
        sorted(all_moduli, key=num_prime_factors),
        sorted(all_moduli, key=lambda m: -num_prime_factors(m)),
        sorted(all_moduli, reverse=True),
        sorted(all_moduli),
    ]
    for _ in range(8):
        shuffled = all_moduli[:]
        random.shuffle(shuffled)
        orders.append(shuffled)

    best_set = []
    best_len = 0

    for order in orders:
        if time.time() > deadline - 40:
            break
        current = []
        for m in order:
            try_add(m, current)
        if len(current) > best_len:
            best_len = len(current)
            best_set = [list(c) for c in current]

    # Local search with multiple move types
    used_moduli = set(m for _, m in best_set)
    unused_moduli = [m for m in all_moduli if m not in used_moduli]
    random.shuffle(unused_moduli)

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False

        # Move 1: Try to add unused moduli
        for i in range(len(unused_moduli)):
            if time.time() > deadline - 5.0:
                break
            m_new = unused_moduli[i]
            if try_add(m_new, best_set):
                unused_moduli.pop(i)
                improved = True
                break

        if not improved:
            # Move 2: Try 1-swap (remove one, add one) to enable future additions
            if unused_moduli:
                for m_new in list(unused_moduli):
                    if time.time() > deadline - 5.0:
                        break
                    for idx in range(len(best_set)):
                        m_old = best_set[idx][1]
                        others = [best_set[j] for j in range(len(best_set)) if j != idx]
                        r_new = find_residue_for(m_new, others)
                        if r_new is not None:
                            # Check if this swap might help: compare "constraint tightness"
                            # Skip for now, just do perturbation
                            pass

            # Move 3: Perturbation - change residue to enable adding a new modulus
            if unused_moduli:
                for _ in range(40):
                    if time.time() > deadline - 5.0:
                        break
                    idx = random.randint(0, len(best_set) - 1)
                    r_old, m_old = best_set[idx]
                    target_m = unused_moduli[random.randint(0, len(unused_moduli) - 1)]
                    
                    candidates = [r for r in range(m_old) if r != r_old]
                    random.shuffle(candidates)
                    
                    for r_try in candidates[:25]:
                        valid = True
                        for j in range(len(best_set)):
                            if j == idx:
                                continue
                            r_j, m_j = best_set[j]
                            if (r_try - r_j) % gcd_table.get((m_old, m_j), math.gcd(m_old, m_j)) == 0:
                                valid = False
                                break
                        if valid:
                            best_set[idx][0] = r_try
                            r_t = find_residue_for(target_m, best_set)
                            if r_t is not None:
                                best_set.append([r_t, target_m])
                                unused_moduli.remove(target_m)
                                improved = True
                                break
                            else:
                                best_set[idx][0] = r_old
                    if improved:
                        break

    # Final verification
    n = len(best_set)
    valid = True
    for i in range(n):
        r1, m1 = best_set[i]
        for j in range(i + 1, n):
            r2, m2 = best_set[j]
            g = gcd_table.get((m1, m2), math.gcd(m1, m2))
            if (r1 - r2) % g == 0:
                valid = False
                break
        if not valid:
            break

    if not valid:
        valid_set = []
        for c in best_set:
            try_add(c[1], valid_set)
        if valid_set:
            best_set = valid_set
        else:
            best_set = [[1, 2]]

    return best_set
# EVOLVE-BLOCK-END
