import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Compute compatibility count: number of other moduli sharing gcd > 1
    compat = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and gcd_tab[i][j] > 1:
                compat[i] += 1

    def try_greedy(order, residue_strategy='first'):
        """
        order: list of indices into moduli
        residue_strategy: 'first', 'last', or 'random'
        Returns list of (idx, r, m)
        """
        selected = []  # list of (idx, r, m)
        selected_set = set()
        for idx in order:
            m = moduli[idx]
            found = False
            if residue_strategy == 'first':
                residues = range(m)
            elif residue_strategy == 'last':
                residues = range(m - 1, -1, -1)
            else:  # random
                residues = list(range(m))
                random.shuffle(residues)

            for r in residues:
                ok = True
                for (idx2, r2, m2) in selected:
                    g = gcd_tab[idx][idx2]
                    if (r - r2) % g == 0:
                        ok = False
                        break
                if ok:
                    selected.append((idx, r, m))
                    selected_set.add(idx)
                    found = True
                    break
            # If not found, skip this modulus
        return selected

    best = []

    # Deterministic orderings
    orderings = [
        sorted(range(n), key=lambda i: compat[i]),       # least compatible first
        sorted(range(n), key=lambda i: -compat[i]),      # most compatible first
        sorted(range(n), key=lambda i: moduli[i]),       # ascending modulus
        sorted(range(n), key=lambda i: -moduli[i]),      # descending modulus
        list(range(n)),
        list(reversed(range(n))),
    ]

    for order in orderings:
        for strat in ['first', 'last']:
            res = try_greedy(order, strat)
            if len(res) > len(best):
                best = res

    # Random restarts with varied strategies
    random.seed(42)
    for _ in range(500):
        if time.time() > deadline - 15:
            break
        order = list(range(n))
        random.shuffle(order)
        strat = random.choice(['first', 'last', 'random'])
        res = try_greedy(order, strat)
        if len(res) > len(best):
            best = res

    # Local search phase 1: try to add more moduli
    if time.time() < deadline - 5:
        selected_idx = set(idx for (idx, _, _) in best)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for idx in range(n):
                if idx in selected_idx:
                    continue
                m = moduli[idx]
                for r in range(m):
                    ok = True
                    for (idx2, r2, m2) in best:
                        g = gcd_tab[idx][idx2]
                        if (r - r2) % g == 0:
                            ok = False
                            break
                    if ok:
                        best.append((idx, r, m))
                        selected_idx.add(idx)
                        improved = True
                        break
                if improved:
                    break

    # Local search phase 2: try swapping residues to free up space
    if time.time() < deadline - 5:
        selected_idx = set(idx for (idx, _, _) in best)
        swap_improved = True
        while swap_improved and time.time() < deadline - 5:
            swap_improved = False
            # Try each selected modulus: can we change its residue to allow adding a new one?
            for new_idx in range(n):
                if new_idx in selected_idx:
                    continue
                m_new = moduli[new_idx]
                # For each existing modulus, try changing its residue
                for pos in range(len(best)):
                    old_idx, old_r, old_m = best[pos]
                    # Try all residues for old_idx
                    for new_r in range(old_m):
                        if new_r == old_r:
                            continue
                        # Check if new_r is compatible with all other selected (excluding old_idx)
                        ok_old = True
                        for qpos in range(len(best)):
                            if qpos == pos:
                                continue
                            other_idx, other_r, other_m = best[qpos]
                            g = gcd_tab[old_idx][other_idx]
                            if (new_r - other_r) % g == 0:
                                ok_old = False
                                break
                        if not ok_old:
                            continue
                        # Now check if new_idx can be added with some residue
                        found_new = False
                        for r_new in range(m_new):
                            ok_new = True
                            for qpos in range(len(best)):
                                if qpos == pos:
                                    # Use new_r for old_idx
                                    if (r_new - new_r) % gcd_tab[new_idx][old_idx] == 0:
                                        ok_new = False
                                        break
                                else:
                                    other_idx, other_r, other_m = best[qpos]
                                    if (r_new - other_r) % gcd_tab[new_idx][other_idx] == 0:
                                        ok_new = False
                                        break
                            if ok_new:
                                found_new = True
                                break
                        if found_new:
                            # Perform the swap
                            best[pos] = (old_idx, new_r, old_m)
                            best.append((new_idx, r_new, m_new))
                            selected_idx.add(new_idx)
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                if swap_improved:
                    break

    return [[r, m] for (_, r, m) in best]
