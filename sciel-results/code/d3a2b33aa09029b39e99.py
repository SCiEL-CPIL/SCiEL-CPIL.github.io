import time
import math
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n_mod = len(moduli)  # 127

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    # Precompute gcd matrix
    gcd_mat = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute conflict masks
    conflict = [[None] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        mi = moduli[i]
        for j in range(n_mod):
            if i == j:
                continue
            mj = moduli[j]
            g = gcd_mat[i][j]
            masks = [0] * mi
            for r_i in range(mi):
                target = r_i % g
                mask = 0
                for r_j in range(mj):
                    if r_j % g == target:
                        mask |= (1 << r_j)
                masks[r_i] = mask
            conflict[i][j] = masks

    def popcount(x):
        return bin(x).count('1')

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

    def is_valid_solution(chosen):
        items = list(chosen.items())
        for i in range(len(items)):
            idx_i, r_i = items[i]
            mi = moduli[idx_i]
            for j in range(i + 1, len(items)):
                idx_j, r_j = items[j]
                mj = moduli[idx_j]
                g = gcd_mat[idx_i][idx_j]
                if r_i % g == r_j % g:
                    return False
        return True

    def backtrack_csp(order, time_limit):
        """
        CSP backtracking with forward checking.
        order: list of modulus indices to try in this order
        Returns: dict of idx -> residue, or None if failed
        """
        chosen = {}
        # avail[idx] = bitmask of available residues for modulus idx
        avail = {}
        for idx in order:
            avail[idx] = (1 << moduli[idx]) - 1

        # Track which moduli have been assigned
        assigned = set()

        def backtrack(pos):
            if time.time() > time_limit:
                return None

            if pos >= len(order):
                return dict(chosen)

            idx = order[pos]
            mi = moduli[idx]
            mask = avail[idx]

            if mask == 0:
                return None

            # Try residues in order of least impact on future variables
            # For each candidate residue, count how many future residues it blocks
            candidates = []
            for r in range(mi):
                if mask & (1 << r):
                    # Count blocked future residues
                    block_count = 0
                    for j in order[pos + 1:]:
                        if j in assigned:
                            continue
                        if avail[j] == 0:
                            continue
                        m = conflict[idx][j][r]
                        block_count += popcount(avail[j] & m)
                    candidates.append((block_count, r))

            if not candidates:
                return None

            # Sort by least blocking first
            candidates.sort()

            for _, r in candidates:
                if time.time() > time_limit:
                    return None

                # Assign
                chosen[idx] = r
                assigned.add(idx)

                # Forward check: update avail for future unassigned
                saved_avail = {}
                feasible = True
                for j in order[pos + 1:]:
                    if j in assigned:
                        continue
                    old_avail = avail[j]
                    m = conflict[idx][j][r]
                    new_avail = old_avail & ~m
                    saved_avail[j] = old_avail
                    avail[j] = new_avail
                    if new_avail == 0:
                        feasible = False
                        break

                if feasible:
                    result = backtrack(pos + 1)
                    if result is not None:
                        return result

                # Undo
                del chosen[idx]
                assigned.discard(idx)
                for j, old in saved_avail.items():
                    avail[j] = old

            return None

        return backtrack(0)

    def local_search_add(chosen, time_limit):
        """Try to add missing moduli by finding valid residues."""
        chosen = dict(chosen)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            missing = [i for i in range(n_mod) if i not in chosen]
            for idx in missing:
                if time.time() > time_limit:
                    break
                mi = moduli[idx]
                # Find valid residues
                valid_r = []
                for r in range(mi):
                    ok = True
                    for j, rj in chosen.items():
                        g = gcd_mat[idx][j]
                        if r % g == rj % g:
                            ok = False
                            break
                    if ok:
                        valid_r.append(r)
                if valid_r:
                    # Pick residue that blocks fewest other missing
                    best_r = valid_r[0]
                    best_cost = float('inf')
                    for r in valid_r:
                        cost = 0
                        for j in missing:
                            if j == idx:
                                continue
                            mj = moduli[j]
                            g = gcd_mat[idx][j]
                            target = r % g
                            blocked = sum(1 for rj in range(mj) if rj % g == target)
                            cost += blocked
                        if cost < best_cost:
                            best_cost = cost
                            best_r = r
                    chosen[idx] = best_r
                    improved = True
                    break
        return chosen

    def swap_improve(chosen, time_limit):
        """Try swapping residues to free up space for more moduli."""
        chosen = dict(chosen)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            missing = [i for i in range(n_mod) if i not in chosen]
            if not missing:
                break
            # Try to add each missing by swapping
            for m_idx in missing:
                if time.time() > time_limit:
                    break
                mi = moduli[m_idx]
                # Try each residue for m_idx
                for r_m in range(mi):
                    # Check if r_m conflicts with any chosen
                    conflicts = []
                    for j, rj in chosen.items():
                        g = gcd_mat[m_idx][j]
                        if r_m % g == rj % g:
                            conflicts.append(j)
                    if not conflicts:
                        chosen[m_idx] = r_m
                        improved = True
                        break
                    # Try to reassign conflicting moduli
                    # For each conflicting j, find alternative residue
                    reassign_ok = True
                    saved = {}
                    for j in conflicts:
                        mj = moduli[j]
                        old_rj = chosen[j]
                        # Find new residue for j that doesn't conflict with m_idx or other chosen
                        found = False
                        for rj_new in range(mj):
                            if rj_new == old_rj:
                                continue
                            # Check against m_idx
                            g_mj = gcd_mat[m_idx][j]
                            if rj_new % g_mj == r_m % g_mj:
                                continue
                            # Check against other chosen (excluding those being reassigned)
                            ok = True
                            for k, rk in chosen.items():
                                if k == j or k == m_idx or k in conflicts:
                                    continue
                                g_kj = gcd_mat[j][k]
                                if rj_new % g_kj == rk % g_kj:
                                    ok = False
                                    break
                            if ok:
                                saved[j] = old_rj
                                chosen[j] = rj_new
                                found = True
                                break
                        if not found:
                            reassign_ok = False
                            break
                    if reassign_ok:
                        chosen[m_idx] = r_m
                        improved = True
                        break
                if improved:
                    break
        return chosen

    best_sol = {}
    best_len = 0

    # Orderings for CSP backtracking
    # Sort by descending prime factors (more constrained first)
    order_pf_desc = sorted(range(n_mod), key=lambda i: -num_prime_factors(moduli[i]))
    # Sort by descending modulus
    order_m_desc = sorted(range(n_mod), key=lambda i: -moduli[i])
    # Sort by ascending modulus
    order_m_asc = sorted(range(n_mod), key=lambda i: moduli[i])
    # Natural order
    order_nat = list(range(n_mod))

    orderings = [order_pf_desc, order_m_desc, order_m_asc, order_nat]

    # Try CSP backtracking with each ordering
    for order in orderings:
        if time.time() > deadline - 15.0:
            break
        # Give each CSP run a reasonable time budget
        t_limit = min(deadline - 10.0, start_time + 20.0)
        result = backtrack_csp(order, t_limit)
        if result and is_valid_solution(result):
            if len(result) > best_len:
                best_len = len(result)
                best_sol = dict(result)

    # Local search to add more
    if best_sol and time.time() < deadline - 10.0:
        best_sol = local_search_add(best_sol, deadline - 8.0)
        if is_valid_solution(best_sol):
            best_len = len(best_sol)

    # Swap improvement
    if best_sol and time.time() < deadline - 8.0:
        best_sol = swap_improve(best_sol, deadline - 5.0)
        if is_valid_solution(best_sol):
            best_len = len(best_sol)
        else:
            # Rebuild valid subset
            valid_subset = {}
            for idx in sorted(best_sol.keys()):
                r = best_sol[idx]
                ok = True
                for j, rj in valid_subset.items():
                    g = gcd_mat[idx][j]
                    if r % g == rj % g:
                        ok = False
                        break
                if ok:
                    valid_subset[idx] = r
            best_sol = valid_subset
            best_len = len(best_sol)

    # Randomized restarts with CSP
    random.seed(42)
    trials = 0
    while time.time() < deadline - 10.0 and trials < 200:
        trials += 1
        order = list(range(n_mod))
        random.shuffle(order)
        t_limit = min(deadline - 10.0, start_time + 15.0)
        result = backtrack_csp(order, t_limit)
        if result and is_valid_solution(result):
            # Try to extend
            result = local_search_add(result, min(deadline - 8.0, time.time() + 5.0))
            if is_valid_solution(result):
                if len(result) > best_len:
                    best_len = len(result)
                    best_sol = dict(result)

    # Final polish
    if best_sol and time.time() < deadline - 5.0:
        best_sol = local_search_add(best_sol, deadline - 3.0)
        if is_valid_solution(best_sol):
            best_len = len(best_sol)
        else:
            valid_subset = {}
            for idx in sorted(best_sol.keys()):
                r = best_sol[idx]
                ok = True
                for j, rj in valid_subset.items():
                    g = gcd_mat[idx][j]
                    if r % g == rj % g:
                        ok = False
                        break
                if ok:
                    valid_subset[idx] = r
            best_sol = valid_subset
            best_len = len(best_sol)

    if not best_sol:
        best_sol = {0: 0}
        best_len = 1

    return [[r, moduli[idx]] for idx, r in best_sol.items()]
