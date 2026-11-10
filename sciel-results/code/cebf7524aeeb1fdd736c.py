import math
import time

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))

    # Precompute gcd table
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Greedy heuristic: sort by number of divisors descending (more constraints first)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    div_counts = {m: num_divisors(m) for m in moduli}

    # Try multiple orderings
    orderings = [
        sorted(moduli, key=lambda x: -div_counts[x]),
        sorted(moduli, key=lambda x: -x),
        sorted(moduli, key=lambda x: x),
        sorted(moduli, key=lambda x: -div_counts[x], reverse=True),
    ]

    best_result = None
    best_count = 0

    def greedy_solve(order, seed_offset=0):
        chosen = []
        chosen_moduli = set()
        for m in order:
            if m in chosen_moduli:
                continue
            # Try to find a valid residue
            found = False
            # Try residues in a deterministic but varied order
            # Use seed_offset to vary the starting residue
            start_r = (seed_offset + m) % m
            for offset in range(m):
                r = (start_r + offset) % m
                ok = True
                for (r2, m2) in chosen:
                    g = gcd_table[m][m2]
                    if (r - r2) % g == 0:
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    chosen_moduli.add(m)
                    found = True
                    break
        return chosen

    # Try each ordering with multiple seed offsets
    for oi, order in enumerate(orderings):
        if time.time() > deadline:
            break
        for seed_off in range(10):
            if time.time() > deadline:
                break
            result = greedy_solve(order, seed_off + oi)
            if len(result) > best_count:
                best_count = len(result)
                best_result = [[r, m] for r, m in result]

    # Fallback
    if best_result is None:
        best_result = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]

    return best_result
