# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses bitmask-based conflict detection with multi-phase search:
      1. Deterministic greedy + local search
      2. Randomized greedy (many shuffles) + local search
      3. Structural greedy (prefer low-conflict elements) + local search
      4. Perturbation of best set + local search
      5. Random restarts with small seeds + greedy + local search
      6. Final polish: 2-swap and 3-swap refinement on best set
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 570  # safety margin before 630s kill

    def time_left():
        return deadline - time.time()

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = sorted(forbidden)
    n_forbid = len(forbidden_list)

    # For each element i, compute bitmask of elements that conflict with i
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for f in forbidden_list:
            j = (i - f) % m
            mask |= (1 << j)
            j2 = (i + f) % m
            if j2 != j:
                mask |= (1 << j2)
        conflict_mask[i] = mask

    # Precompute degree (number of conflicts) for each element
    degree = [bin(conflict_mask[i]).count('1') for i in range(m)]

    def is_valid(s):
        """Check if set s has no forbidden pairs using bitmasks."""
        mask = 0
        for a in s:
            if mask & conflict_mask[a]:
                return False
            mask |= (1 << a)
        return True

    def greedy_random(order):
        """Greedy: process candidates in given order, add if no conflict with selected."""
        selected = set()
        mask = 0
        for cand in order:
            if not (mask & conflict_mask[cand]):
                selected.add(cand)
                mask |= (1 << cand)
        return selected

    def greedy_structural():
        """Greedy: prefer elements with fewer conflicts (lower degree first)."""
        selected = set()
        mask = 0
        # Sort by degree (fewer conflicts first), break ties by value
        order = sorted(range(m), key=lambda x: (degree[x], x))
        for cand in order:
            if not (mask & conflict_mask[cand]):
                selected.add(cand)
                mask |= (1 << cand)
        return selected

    def local_search(s, max_iters=2000):
        """Improve set via add, 1-swap, and 2-swap operations using bitmasks."""
        s = set(s)
        mask = 0
        for a in s:
            mask |= (1 << a)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            # Try adding any element
            for cand in range(m):
                if not (mask >> cand) & 1:
                    if (mask & conflict_mask[cand]) == 0:
                        s.add(cand)
                        mask |= (1 << cand)
                        improved = True
                        break
            if improved:
                continue
            # Try 1-swap: remove one, add another
            s_list = sorted(s)
            for rem in s_list:
                if time.time() > deadline:
                    break
                mask_without = mask & ~(1 << rem)
                for cand in range(m):
                    if not (mask_without >> cand) & 1:
                        if (mask_without & conflict_mask[cand]) == 0:
                            s.discard(rem)
                            s.add(cand)
                            mask = mask_without | (1 << cand)
                            improved = True
                            break
                if improved:
                    break
            # Try 2-swap: remove two, add two
            if not improved and len(s) >= 2:
                s_list = sorted(s)
                found = False
                for i_idx in range(len(s_list)):
                    if time.time() > deadline:
                        break
                    for j_idx in range(i_idx + 1, len(s_list)):
                        rem1, rem2 = s_list[i_idx], s_list[j_idx]
                        mask_without = mask & ~(1 << rem1) & ~(1 << rem2)
                        for cand1 in range(m):
                            if (mask_without >> cand1) & 1:
                                continue
                            if (mask_without & conflict_mask[cand1]) != 0:
                                continue
                            mask_with1 = mask_without | (1 << cand1)
                            for cand2 in range(m):
                                if (mask_with1 >> cand2) & 1:
                                    continue
                                if (mask_with1 & conflict_mask[cand2]) != 0:
                                    continue
                                s.discard(rem1)
                                s.discard(rem2)
                                s.add(cand1)
                                s.add(cand2)
                                mask = mask_with1 | (1 << cand2)
                                improved = True
                                found = True
                                break
                            if found:
                                break
                    if found:
                        break
        return s

    def polish(s, time_budget=30):
        """Final polish: aggressive 2-swap and 3-swap refinement with a time budget."""
        s = set(s)
        mask = 0
        for a in s:
            mask |= (1 << a)
        best_s = set(s)
        best_mask = mask
        best_size = len(s)
        polish_deadline = time.time() + time_budget
        global_deadline = deadline

        while time.time() < min(polish_deadline, global_deadline):
            improved = False
            s_list = sorted(s)
            # Try 2-swap
            for i_idx in range(len(s_list)):
                if time.time() > min(polish_deadline, global_deadline):
                    break
                for j_idx in range(i_idx + 1, len(s_list)):
                    rem1, rem2 = s_list[i_idx], s_list[j_idx]
                    mask_without = mask & ~(1 << rem1) & ~(1 << rem2)
                    for cand1 in range(m):
                        if (mask_without >> cand1) & 1:
                            continue
                        if (mask_without & conflict_mask[cand1]) != 0:
                            continue
                        mask_with1 = mask_without | (1 << cand1)
                        for cand2 in range(m):
                            if (mask_with1 >> cand2) & 1:
                                continue
                            if (mask_with1 & conflict_mask[cand2]) != 0:
                                continue
                            new_size = len(s)  # same size
                            if new_size > best_size:
                                s.discard(rem1)
                                s.discard(rem2)
                                s.add(cand1)
                                s.add(cand2)
                                mask = mask_with1 | (1 << cand2)
                                best_s = set(s)
                                best_mask = mask
                                best_size = new_size
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Try 3-swap: remove three, add three (can enable size increase via intermediate states)
            if len(s) >= 3:
                found = False
                for i_idx in range(len(s_list)):
                    if time.time() > min(polish_deadline, global_deadline):
                        break
                    for j_idx in range(i_idx + 1, len(s_list)):
                        if time.time() > min(polish_deadline, global_deadline):
                            break
                        for k_idx in range(j_idx + 1, len(s_list)):
                            rem1, rem2, rem3 = s_list[i_idx], s_list[j_idx], s_list[k_idx]
                            mask_without = mask & ~(1 << rem1) & ~(1 << rem2) & ~(1 << rem3)
                            # Try to add 3 elements
                            for cand1 in range(m):
                                if (mask_without >> cand1) & 1:
                                    continue
                                if (mask_without & conflict_mask[cand1]) != 0:
                                    continue
                                mask_with1 = mask_without | (1 << cand1)
                                for cand2 in range(m):
                                    if (mask_with1 >> cand2) & 1:
                                        continue
                                    if (mask_with1 & conflict_mask[cand2]) != 0:
                                        continue
                                    mask_with2 = mask_with1 | (1 << cand2)
                                    for cand3 in range(m):
                                        if (mask_with2 >> cand3) & 1:
                                            continue
                                        if (mask_with2 & conflict_mask[cand3]) != 0:
                                            continue
                                        # Valid 3-swap
                                        s.discard(rem1)
                                        s.discard(rem2)
                                        s.discard(rem3)
                                        s.add(cand1)
                                        s.add(cand2)
                                        s.add(cand3)
                                        mask = mask_with2 | (1 << cand3)
                                        best_s = set(s)
                                        best_mask = mask
                                        best_size = len(s)
                                        improved = True
                                        found = True
                                        break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if improved:
                    continue
            # No improvement found, try adding
            for cand in range(m):
                if not (mask >> cand) & 1:
                    if (mask & conflict_mask[cand]) == 0:
                        s.add(cand)
                        mask |= (1 << cand)
                        best_s = set(s)
                        best_mask = mask
                        best_size = len(s)
                        improved = True
                        break
            if not improved:
                break

        return best_s

    # Strategy 1: deterministic greedy
    best_set = set()
    mask = 0
    for cand in range(m):
        if not (mask & conflict_mask[cand]):
            best_set.add(cand)
            mask |= (1 << cand)
    best_set = local_search(best_set, max_iters=3000)
    best_size = len(best_set)

    # Strategy 2: structural greedy (prefer low-conflict elements)
    if time_left() > 30:
        s_struct = greedy_structural()
        s_struct = local_search(s_struct, max_iters=2000)
        if len(s_struct) > best_size:
            best_size = len(s_struct)
            best_set = s_struct

    # Strategy 3: randomized greedy with many shuffles
    rng = random.Random(12345)
    for trial in range(50000):
        if time_left() < 60:
            break
        order = list(range(m))
        rng.shuffle(order)
        s = greedy_random(order)
        s = local_search(s, max_iters=1000)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Strategy 4: perturbation of best set
    for trial in range(20000):
        if time_left() < 60:
            break
        s = set(best_set)
        n_remove = min(len(s), rng.randint(1, 4))
        to_remove = rng.sample(sorted(s), n_remove)
        for r in to_remove:
            s.discard(r)
        s = local_search(s, max_iters=2000)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Strategy 5: random restarts with small seeds
    for trial in range(30000):
        if time_left() < 60:
            break
        seed_size = rng.randint(1, 6)
        seed = set(rng.sample(range(m), seed_size))
        valid_seed = set()
        smask = 0
        for x in sorted(seed):
            if not (smask & conflict_mask[x]):
                valid_seed.add(x)
                smask |= (1 << x)
        s = set(valid_seed)
        mask = smask
        for cand in range(m):
            if not (mask >> cand) & 1:
                if not (mask & conflict_mask[cand]):
                    s.add(cand)
                    mask |= (1 << cand)
        s = local_search(s, max_iters=1500)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Strategy 6: Final polish with 2-swap and 3-swap
    if time_left() > 40:
        best_set = polish(best_set, time_budget=min(30, time_left() - 10))
        best_size = len(best_set)

    return np.array(sorted(best_set), dtype=np.int64)


# EVOLVE-BLOCK-END
