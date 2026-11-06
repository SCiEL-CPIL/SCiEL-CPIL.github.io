import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)

    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Index for modulus value
    mod_to_idx = {m: i for i, m in enumerate(moduli)}

    best_result = [[1, 2]]
    best_count = 1

    def is_bad_triangle(i, j, k):
        """A triangle is bad if all three pair gcds equal 2 (need 3 distinct residues mod 2)."""
        return (gcd_table[i][j] == 2 and
                gcd_table[i][k] == 2 and
                gcd_table[j][k] == 2)

    def try_find_residue(m_idx, selected, deterministic=True):
        """
        Find a residue r in [0, m) such that for every selected (r_i, m_i)
        with gcd(m, m_i) > 1, we have (r - r_i) % g != 0 where g = gcd(m, m_i).
        """
        m = moduli[m_idx]
        constraints = []
        for (r_i, m_i) in selected:
            m_i_idx = mod_to_idx[m_i]
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                constraints.append((g, r_i))

        constraints.sort(key=lambda x: x[0])

        candidates = list(range(m))
        if not deterministic:
            random.shuffle(candidates)

        for r in candidates:
            valid = True
            for (g, r_i) in constraints:
                if (r - r_i) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def can_add(m_idx, selected):
        """Check if modulus m_idx can be added given current selection."""
        for (r_i, m_i) in selected:
            m_i_idx = mod_to_idx[m_i]
            if gcd_table[m_idx][m_i_idx] == 1:
                return False

        if len(selected) >= 2:
            for a in range(len(selected)):
                for b in range(a + 1, len(selected)):
                    idx_a = mod_to_idx[selected[a][1]]
                    idx_b = mod_to_idx[selected[b][1]]
                    if is_bad_triangle(idx_a, idx_b, m_idx):
                        return False
        return True

    def greedy_pass(order, use_random=False):
        selected = []
        used = set()
        for idx in order:
            if time.time() > deadline:
                break
            m = moduli[idx]
            if m in used:
                continue
            if not can_add(idx, selected):
                continue
            r = try_find_residue(idx, selected, deterministic=not use_random)
            if r is not None:
                selected.append([r, m])
                used.add(m)
        return selected

    def local_search_add(current):
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            used = set(m for _, m in current)
            for idx in range(n_mod):
                if time.time() > deadline - 2.0:
                    break
                m = moduli[idx]
                if m in used:
                    continue
                if not can_add(idx, current):
                    continue
                r = try_find_residue(idx, current, deterministic=True)
                if r is not None:
                    current.append([r, m])
                    used.add(m)
                    improved = True
        return current

    # --- Multi-start greedy with various orderings ---

    # Ascending modulus
    result = greedy_pass(list(range(n_mod)), use_random=False)
    if len(result) > best_count:
        best_count = len(result)
        best_result = result

    # Descending modulus
    result = greedy_pass(list(range(n_mod - 1, -1, -1)), use_random=False)
    if len(result) > best_count:
        best_count = len(result)
        best_result = result

    # Sort by number of divisors (fewer first)
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
        return count

    order_div = sorted(range(n_mod), key=lambda i: num_divisors(moduli[i]))
    result = greedy_pass(order_div, use_random=False)
    if len(result) > best_count:
        best_count = len(result)
        best_result = result

    # Sort by number of divisors (more first)
    order_div_desc = sorted(range(n_mod), key=lambda i: -num_divisors(moduli[i]))
    result = greedy_pass(order_div_desc, use_random=False)
    if len(result) > best_count:
        best_count = len(result)
        best_result = result

    # Randomized restarts with different biases
    base_primes = [2, 3, 5, 7]
    for bp in base_primes:
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            multiples = [i for i in range(n_mod) if moduli[i] % bp == 0]
            others = [i for i in range(n_mod) if moduli[i] % bp != 0]
            random.shuffle(multiples)
            random.shuffle(others)
            order = multiples + others
            result = greedy_pass(order, use_random=True)
            if len(result) > best_count:
                best_count = len(result)
                best_result = result

    # Pure random restarts
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        order = list(range(n_mod))
        random.shuffle(order)
        result = greedy_pass(order, use_random=True)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result

    # Local search on best
    if time.time() < deadline - 3.0:
        current = local_search_add([[r, m] for r, m in best_result])
        if len(current) > best_count:
            best_count = len(current)
            best_result = current

    # More diverse random restarts with mixed residue strategies
    remaining = deadline - time.time()
    if remaining > 5.0:
        for _ in range(40):
            if time.time() > deadline - 2.0:
                break
            order = list(range(n_mod))
            random.shuffle(order)
            result = greedy_pass(order, use_random=random.random() < 0.5)
            if len(result) > best_count:
                best_count = len(result)
                best_result = result

    # Final local search
    if time.time() < deadline - 1.0:
        current = local_search_add([[r, m] for r, m in best_result])
        if len(current) > best_count:
            best_count = len(current)
            best_result = current

    return best_result
