# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def _compute_forbidden(m):
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    return forbidden


def _is_valid_set(selected, forbidden, m):
    sset = set(selected)
    for a in selected:
        for b in selected:
            if a == b:
                continue
            if (a - b) % m in forbidden:
                return False
    return True


def _build_conflict_masks(forbidden, m):
    """For each residue r, build a bitmask of residues s such that (r-s)%m is in forbidden.
       conflict_mask[r] has bit s set iff r cannot coexist with s."""
    conflict_mask = [0] * m
    for r in range(m):
        mask = 0
        for s in range(m):
            if (r - s) % m in forbidden:
                mask |= (1 << s)
        conflict_mask[r] = mask
    return conflict_mask


def _greedy_build(order, conflict_mask, m, selected_set=None, selected_list=None, selected_mask=0):
    """Build an independent set by scanning `order` and adding candidates that don't conflict
       with already-selected elements."""
    if selected_set is None:
        selected_set = set()
    if selected_list is None:
        selected_list = []
    for cand in order:
        # Check if cand conflicts with any selected: cand is blocked if selected_mask & conflict_mask[cand] != 0
        if (selected_mask & conflict_mask[cand]) == 0:
            selected_set.add(cand)
            selected_list.append(cand)
            selected_mask |= (1 << cand)
    return selected_set, selected_list, selected_mask


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 600.0
    forbidden = _compute_forbidden(m)
    conflict_mask = _build_conflict_masks(forbidden, m)

    best_set = set()
    best_list = []
    best_size = 0

    def try_update(sset, slist):
        nonlocal best_set, best_list, best_size
        if len(slist) > best_size:
            best_set = set(sset)
            best_list = list(slist)
            best_size = len(slist)

    # --- Phase 1: greedy with multiple orderings ---
    # Natural order
    sset, slist, smask = _greedy_build(list(range(m)), conflict_mask, m)
    try_update(sset, slist)

    # Random shuffles
    random.seed(12345)
    n_shuffles = 400
    for i in range(n_shuffles):
        if time.time() > deadline - 60.0:
            break
        order = list(range(m))
        random.shuffle(order)
        sset, slist, smask = _greedy_build(order, conflict_mask, m)
        try_update(sset, slist)

    # --- Phase 2: enhanced local search on best set ---
    def local_search(sset, slist, max_iter=50000):
        """Local search with add, remove-one-add-two, and swap moves."""
        sset = set(sset)
        slist = list(sset)
        smask = 0
        for s in sset:
            smask |= (1 << s)

        for it in range(max_iter):
            if time.time() > deadline - 30.0:
                break
            improved = False

            # Try pure add
            for cand in range(m):
                if cand in sset:
                    continue
                if (smask & conflict_mask[cand]) == 0:
                    sset.add(cand)
                    slist.append(cand)
                    smask |= (1 << cand)
                    improved = True
                    break

            if not improved:
                # Try remove-one-add-two
                found = False
                for s in list(sset):
                    # Remove s
                    new_mask = smask & ~(1 << s)
                    new_set = sset - {s}
                    # Try to add two elements greedily
                    added = []
                    for cand in range(m):
                        if len(added) == 2:
                            break
                        if cand in new_set:
                            continue
                        if (new_mask & conflict_mask[cand]) == 0:
                            new_set.add(cand)
                            new_mask |= (1 << cand)
                            added.append(cand)
                    if len(added) == 2:
                        sset = new_set
                        slist = list(new_set)
                        smask = new_mask
                        improved = True
                        found = True
                        break

            if not improved:
                # Try swap: remove one, add one (net 0, but may enable future adds)
                # Only try if we can then add at least one more
                found_swap = False
                for s in list(sset):
                    for cand in range(m):
                        if cand in sset:
                            continue
                        # Check if cand conflicts with sset - {s}
                        new_mask = smask & ~(1 << s)
                        if (new_mask & conflict_mask[cand]) != 0:
                            continue
                        # Swap: remove s, add cand
                        new_set = (sset - {s}) | {cand}
                        new_mask = (smask & ~(1 << s)) | (1 << cand)
                        # Check if we can add at least one more
                        extra_added = 0
                        for extra in range(m):
                            if extra in new_set:
                                continue
                            if (new_mask & conflict_mask[extra]) == 0:
                                extra_added += 1
                                if extra_added >= 1:
                                    break
                        if extra_added >= 1:
                            sset = new_set
                            slist = list(new_set)
                            smask = new_mask
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break

            if not improved:
                break

        return sset, slist

    sset, slist = local_search(best_set, best_list)
    try_update(sset, slist)

    # --- Phase 3: multi-start local search from random greedy sets ---
    random.seed(67890)
    n_starts = 80
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        order = list(range(m))
        random.shuffle(order)
        sset, slist, smask = _greedy_build(order, conflict_mask, m)
        sset, slist = local_search(sset, slist, max_iter=20000)
        try_update(sset, slist)

    return np.array(sorted(best_list), dtype=np.int64)


# EVOLVE-BLOCK-END
