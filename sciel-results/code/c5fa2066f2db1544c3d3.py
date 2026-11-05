import time
import math
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        gcd_tab[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute conflict masks: for each pair (i,j), which residues of i conflict with which of j
    # conflict_mask[i][j] = list of bitmasks, one per residue of i, where bit r is set if residue r of i
    # conflicts with residue r of j
    # Actually, let's use a different approach: precompute for each pair (i,j) and each residue r_i,
    # the set of residues r_j that are compatible
    # compat[i][j][r_i] = bitmask of r_j values that work with r_i

    # For speed, use integer bitmasks for compatibility
    # compat_mask[i][j][r_i] = bitmask where bit r_j is 1 if (r_i, r_j) is compatible
    compat_mask = [[[0] * moduli[i] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g = gcd_tab[i][j]
            m_i = moduli[i]
            m_j = moduli[j]
            for r_i in range(m_i):
                mask = 0
                for r_j in range(m_j):
                    if (r_i - r_j) % g != 0:
                        mask |= (1 << r_j)
                compat_mask[i][j][r_i] = mask

    # For a selected set, compute the allowed residues for each unselected modulus
    # allowed_mask[idx] = bitmask of valid residues for modulus idx given current selection
    def compute_allowed(selected, remaining_set):
        """
        selected: list of (idx, r)
        remaining_set: set of idx values still to place
        Returns dict: idx -> bitmask of allowed residues
        """
        allowed = {}
        for idx in remaining_set:
            m = moduli[idx]
            # Start with all residues allowed
            mask = (1 << m) - 1
            for (sidx, sr) in selected:
                # AND with compatibility mask
                mask &= compat_mask[idx][sidx][sr]
                if mask == 0:
                    break
            allowed[idx] = mask
        return allowed

    def popcount(x):
        return bin(x).count('1')

    # Greedy with MRV and least-constraining value
    def try_greedy(order, time_budget=0.5):
        selected = []
        remaining = set(order)
        start_t = time.time()

        while remaining:
            if time.time() - start_t > time_budget:
                break
            # MRV: pick the modulus with fewest valid residues
            best_idx = -1
            best_count = float('inf')
            best_mask = None
            for idx in remaining:
                m = moduli[idx]
                mask = (1 << m) - 1
                for (sidx, sr) in selected:
                    mask &= compat_mask[idx][sidx][sr]
                    if mask == 0:
                        break
                cnt = popcount(mask)
                if cnt < best_count:
                    best_count = cnt
                    best_idx = idx
                    best_mask = mask
                if cnt == 0:
                    break

            if best_count == 0 or best_idx == -1:
                break

            # Least-constraining: pick residue that forbids fewest on remaining
            m = moduli[best_idx]
            remaining2 = remaining - {best_idx}
            best_r = -1
            best_cost = float('inf')

            # Iterate over set bits in best_mask
            mask = best_mask
            while mask:
                # Get lowest set bit
                lb = mask & (-mask)
                r = lb.bit_length() - 1
                mask ^= lb

                # Compute cost: how many residues this forbids on remaining moduli
                cost = 0
                for j in remaining2:
                    # Count how many residues of j are forbidden by (best_idx, r)
                    # i.e., total - compatible
                    total = moduli[j]
                    compat = compat_mask[j][best_idx][r]
                    cost += (total - popcount(compat))

                if cost < best_cost:
                    best_cost = cost
                    best_r = r

            if best_r >= 0:
                selected.append((best_idx, best_r))
                remaining.discard(best_idx)

        return selected

    # Backtracking with constraint propagation
    def backtrack(selected, remaining, allowed, depth=0, max_depth=8):
        if time.time() > deadline - 5:
            return False, selected
        if not remaining:
            return True, selected
        if depth >= max_depth:
            return False, selected

        # MRV: pick modulus with fewest valid residues
        best_idx = -1
        best_count = float('inf')
        best_mask = None
        for idx in remaining:
            cnt = popcount(allowed[idx])
            if cnt < best_count:
                best_count = cnt
                best_idx = idx
                best_mask = allowed[idx]
            if cnt == 0:
                break

        if best_count == 0 or best_idx == -1:
            return False, selected

        m = moduli[best_idx]
        remaining2 = [idx for idx in remaining if idx != best_idx]

        # Try residues in order of least constraining
        candidates = []
        mask = best_mask
        while mask:
            lb = mask & (-mask)
            r = lb.bit_length() - 1
            mask ^= lb
            # Cost: how many it forbids on remaining
            cost = 0
            for j in remaining2:
                total = moduli[j]
                compat = compat_mask[j][best_idx][r]
                cost += (total - popcount(compat))
            candidates.append((cost, r))

        candidates.sort()

        for cost, r in candidates:
            new_selected = selected + [(best_idx, r)]
            # Update allowed for remaining
            new_allowed = {}
            valid = True
            for j in remaining2:
                new_allowed[j] = allowed[j] & compat_mask[j][best_idx][r]
                if popcount(new_allowed[j]) == 0:
                    valid = False
                    break
            if not valid:
                continue
            success, result = backtrack(new_selected, remaining2, new_allowed, depth + 1, max_depth)
            if success:
                return True, result

        return False, selected

    # Add one more modulus to a solution
    def add_one(best):
        selected_idx = set(idx for (idx, _) in best)
        for idx in range(n):
            if idx in selected_idx:
                continue
            m = moduli[idx]
            mask = (1 << m) - 1
            for (sidx, sr) in best:
                mask &= compat_mask[idx][sidx][sr]
                if mask == 0:
                    break
            if mask:
                # Find a residue
                lb = mask & (-mask)
                r = lb.bit_length() - 1
                best.append((idx, r))
                return True
        return False

    # Local search: remove one, add two
    def local_search_remove_one_add_two(best):
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for pos in range(len(best)):
                if time.time() > deadline - 10:
                    break
                old_idx, old_r = best[pos]
                base = best[:pos] + best[pos+1:]
                base_set = set(idx for (idx, _) in base)

                for i in range(n):
                    if i in base_set:
                        continue
                    m_i = moduli[i]
                    mask_i = (1 << m_i) - 1
                    for (sidx, sr) in base:
                        mask_i &= compat_mask[i][sidx][sr]
                        if mask_i == 0:
                            break
                    if not mask_i:
                        continue

                    for j in range(n):
                        if j in base_set or j == i:
                            continue
                        m_j = moduli[j]
                        mask_j = (1 << m_j) - 1
                        for (sidx, sr) in base:
                            mask_j &= compat_mask[j][sidx][sr]
                            if mask_j == 0:
                                break
                        if not mask_j:
                            continue

                        # Check if i and j can coexist with some residues
                        # Try all pairs
                        found = False
                        mask_i_tmp = mask_i
                        while mask_i_tmp and not found:
                            lb_i = mask_i_tmp & (-mask_i_tmp)
                            r_i = lb_i.bit_length() - 1
                            mask_i_tmp ^= lb_i

                            # Check r_i with base
                            # Already checked via mask_i

                            # Now check which residues of j are compatible with r_i
                            mask_j_compat = mask_j & compat_mask[j][i][r_i]
                            if mask_j_compat:
                                lb_j = mask_j_compat & (-mask_j_compat)
                                r_j = lb_j.bit_length() - 1
                                best[pos] = (i, r_i)
                                best.append((j, r_j))
                                improved = True
                                found = True

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return best

    # Swap: change residue of one selected to allow adding a new one
    def swap_and_add(best):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            selected_idx = set(idx for (idx, _) in best)

            for new_idx in range(n):
                if new_idx in selected_idx:
                    continue
                m_new = moduli[new_idx]

                for pos in range(len(best)):
                    old_idx, old_r = best[pos]
                    m_old = moduli[old_idx]

                    # Try all new residues for old_idx
                    for new_r in range(m_old):
                        if new_r == old_r:
                            continue
                        # Check if new_r is compatible with all other selected
                        ok = True
                        for qpos in range(len(best)):
                            if qpos == pos:
                                continue
                            other_idx, other_r = best[qpos]
                            if (new_r - other_r) % gcd_tab[old_idx][other_idx] == 0:
                                ok = False
                                break
                        if not ok:
                            continue

                        # Check if we can add new_idx with some residue
                        # Given: best with pos changed to (old_idx, new_r)
                        # Need: r_new such that compatible with all
                        mask_new = (1 << m_new) - 1
                        for qpos in range(len(best)):
                            if qpos == pos:
                                # Compatible with (old_idx, new_r)
                                mask_new &= compat_mask[new_idx][old_idx][new_r]
                            else:
                                other_idx, other_r = best[qpos]
                                mask_new &= compat_mask[new_idx][other_idx][other_r]
                            if mask_new == 0:
                                break

                        if mask_new:
                            lb = mask_new & (-mask_new)
                            r_new = lb.bit_length() - 1
                            best[pos] = (old_idx, new_r)
                            best.append((new_idx, r_new))
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best

    best = []
    best_len = 0

    # Deterministic orderings
    orderings = [
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        sorted(range(n), key=lambda i: (-len([p for p in [2, 3, 5, 7, 11, 13, 17, 19, 23] if moduli[i] % p == 0]), moduli[i])),
        sorted(range(n), key=lambda i: (len([p for p in [2, 3, 5, 7, 11, 13, 17, 19, 23] if moduli[i] % p == 0]), moduli[i])),
        sorted(range(n), key=lambda i: -sum(gcd_tab[i][j] for j in range(n) if j != i)),
        sorted(range(n), key=lambda i: sum(gcd_tab[i][j] for j in range(n) if j != i)),
    ]

    for order in orderings:
        res = try_greedy(order, time_budget=1.0)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Random restarts
    random.seed(42)
    n_restarts = 5000
    for restart in range(n_restarts):
        if time.time() > deadline - 25:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order, time_budget=0.3)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Improve the best solution
    if time.time() < deadline - 15:
        # Add loop
        changed = True
        while changed and time.time() < deadline - 15:
            changed = add_one(best)
        best_len = len(best)

    if time.time() < deadline - 10:
        best = local_search_remove_one_add_two(best)
        best_len = len(best)

    if time.time() < deadline - 5:
        best = swap_and_add(best)
        best_len = len(best)

    # Final add loop
    if time.time() < deadline - 2:
        changed = True
        while changed and time.time() < deadline - 2:
            changed = add_one(best)
        best_len = len(best)

    return [[r, moduli[idx]] for (idx, r) in best]
